"""Minimum-viable inspection harness over the bridged HookedTransformer checkpoints.

Three first-pass probes per the chunk-6 plan:

1. attention_at_result_positions: for each result-bit predictor position (18..25
   in the 28-token sequence, i.e., 17..24 in the autoregressive 27-token input),
   show the attention pattern across input positions, averaged or per-head.

2. direct_layer_attribution: at each result position, decompose the logit into
   contributions from the embedding + each transformer block. Tells us which
   layer is doing the work for each result bit.

3. equality_preservation_check: Park Thm 5 sanity check on doubling inputs
   (a == b). The strict claim ("residuals at column-identical positions are
   identical") won't hold because learned PE breaks input-column equality.
   But: does the model correct for PE and converge toward identical residuals
   at corresponding operand-a / operand-b positions across layers? Plus
   commutativity: f(a,b) == f(b,a).

Usage:
    python -m errorboard.inspection --checkpoint runs/sweep-L4-E044-s0/checkpoint_020000.pt
    python -m errorboard.inspection --checkpoint <path> --probe attention --regime cancellation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from transformer_lens import HookedTransformer

from .hooked_bridge import load_hooked
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES
from .tokenizer import (
    POS_A_START, POS_A_END, POS_B_START, POS_B_END,
    POS_PLUS, POS_EQ, POS_C_START, POS_C_END, SEQ_LEN,
    encode_batch,
)


# Result-bit names for human-readable output.
RESULT_BIT_NAMES = ["sign", "e3", "e2", "e1", "e0", "m2", "m1", "m0"]
INPUT_POS_NAMES = (
    ["BOS"]
    + [f"a:{n}" for n in RESULT_BIT_NAMES]
    + ["+"]
    + [f"b:{n}" for n in RESULT_BIT_NAMES]
    + ["="]
    + [f"c:{n}" for n in RESULT_BIT_NAMES]
    + ["EOS"]
)
# The autoregressive input has block_size = SEQ_LEN - 1 = 27 tokens.
# Result-predicting positions in the input (i.e., the positions whose next-token
# prediction is a result bit) are POS_C_START - 1 .. POS_C_END - 1 - 1 inclusive,
# i.e., 18..25. We use 18..25 as "result predictor positions" throughout.
RESULT_PREDICTOR_POSITIONS = list(range(POS_C_START - 1, POS_C_END - 1))


def _make_input_target(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode a structured-array slice (with a_bits, b_bits, result_bits) into
    autoregressive (input, target) pairs of shape (N, SEQ_LEN-1)."""
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seq = encode_batch(triples.astype(np.uint8))   # (N, SEQ_LEN)
    return seq[:, :-1].astype(np.int64), seq[:, 1:].astype(np.int64)


def _sample_regime(regime: str, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pull n samples from a specific regime in the holdout split. Returns (input, target)."""
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=seed)
    if regime not in REGIME_NAMES:
        raise ValueError(f"Unknown regime {regime!r}. Options: {REGIME_NAMES}")
    regime_id = REGIME_NAMES.index(regime)
    rows_all = table[holdout_idx]
    mask = rows_all["regime_id"] == regime_id
    rows = rows_all[mask][:n]
    if len(rows) < n:
        raise ValueError(f"Regime {regime!r} only has {len(rows)} holdout rows (need {n})")
    return _make_input_target(rows)


# ---- Probe 1: attention pattern at result positions ----------------------

@torch.no_grad()
def attention_at_result_positions(
    hooked: HookedTransformer,
    inputs: np.ndarray,
) -> np.ndarray:
    """Compute attention patterns at result-predicting query positions.

    Returns shape (n_layer, n_head, n_result_predictor=8, key_len=27).
    Averaged over batch."""
    idx = torch.from_numpy(inputs).long().to(next(hooked.parameters()).device)
    _, cache = hooked.run_with_cache(idx, return_type="logits")
    n_layer = hooked.cfg.n_layers
    n_head = hooked.cfg.n_heads
    T = hooked.cfg.n_ctx
    out = np.zeros((n_layer, n_head, len(RESULT_PREDICTOR_POSITIONS), T))
    for L in range(n_layer):
        pat = cache[f"blocks.{L}.attn.hook_pattern"]   # (B, n_head, T, T)
        for qi, qpos in enumerate(RESULT_PREDICTOR_POSITIONS):
            out[L, :, qi, :] = pat[:, :, qpos, :].mean(dim=0).cpu().numpy()
    return out


def mantissa_attention_breakdown(att: np.ndarray) -> dict[str, np.ndarray]:
    """For each mantissa-bit predictor (c:m2, c:m1, c:m0 → idx 5, 6, 7 in the
    result-predictor axis), report attention weight at each operand bit position
    plus aux positions (BOS, +, =), aggregated across heads and layers.

    Input: att of shape (n_layer, n_head, 8, T) from attention_at_result_positions.
    Returns:
      'per_predictor': (3, 8) — for each of c:m2,c:m1,c:m0, mean attention weight
                        on each of [a:m2, a:m1, a:m0, b:m2, b:m1, b:m0, ctx_other,
                        aux]. ctx_other = sign+exponent operand bits and already-
                        emitted result bits; aux = BOS+`+`+`=`.
      'per_layer':     (n_layer, 3, 8) — same but per layer (averaged across heads).
      'op_bit_dist':   (3, 6) — fraction of total attention budget that lands on
                        each of [a:m2, a:m1, a:m0, b:m2, b:m1, b:m0]. Useful for
                        the "which operand bit does each result bit look at most"
                        question. Heads + layers averaged.
    """
    n_layer, n_head, n_pred, T = att.shape
    assert n_pred == 8, f"expected 8 result-predictors, got {n_pred}"

    # Positions in the 27-token autoregressive input.
    # 0=BOS; 1..8=a (sign,e3,e2,e1,e0,m2,m1,m0); 9=+; 10..17=b; 18==;
    # 19..26 are c-bits already emitted (auto-regressive predictions).
    A_M2, A_M1, A_M0 = 6, 7, 8           # operand a mantissa positions
    B_M2, B_M1, B_M0 = 15, 16, 17        # operand b mantissa positions
    AUX = [0, 9, 18]                      # BOS, +, =
    OP_OTHER = [1, 2, 3, 4, 5,            # a sign + exponent
                10, 11, 12, 13, 14]       # b sign + exponent
    RESULT_EMITTED = [19, 20, 21, 22, 23, 24, 25]  # already-emitted c-bits
    MANT_POSITIONS = [A_M2, A_M1, A_M0, B_M2, B_M1, B_M0]

    # Pred indices: 5=c:m2, 6=c:m1, 7=c:m0.
    pred_idx = [5, 6, 7]

    # Average across heads first (gives per-layer per-predictor attention).
    # Shape (n_layer, 3, T).
    per_layer = att[:, :, pred_idx, :].mean(axis=1)

    # Aggregate buckets.
    def buckets(arr):
        # arr shape (..., T). Returns (..., 8) corresponding to [a:m2, a:m1, a:m0,
        # b:m2, b:m1, b:m0, ctx_other, aux].
        a_m2 = arr[..., A_M2]
        a_m1 = arr[..., A_M1]
        a_m0 = arr[..., A_M0]
        b_m2 = arr[..., B_M2]
        b_m1 = arr[..., B_M1]
        b_m0 = arr[..., B_M0]
        ctx = arr[..., OP_OTHER + RESULT_EMITTED].sum(axis=-1)
        aux = arr[..., AUX].sum(axis=-1)
        return np.stack([a_m2, a_m1, a_m0, b_m2, b_m1, b_m0, ctx, aux], axis=-1)

    per_layer_bk = buckets(per_layer)              # (n_layer, 3, 8)
    per_predictor = per_layer_bk.mean(axis=0)      # (3, 8) averaged across layers

    # Mantissa-only distribution (just the 6 mantissa positions, renormalized).
    mant_attn = buckets(per_layer)[..., :6]        # (n_layer, 3, 6)
    op_bit_dist = mant_attn.mean(axis=0)           # (3, 6) layer-averaged

    return {
        "per_predictor": per_predictor,
        "per_layer": per_layer_bk,
        "op_bit_dist": op_bit_dist,
    }


def print_mantissa_breakdown(breakdown: dict[str, np.ndarray]) -> None:
    """Pretty-print the mantissa attention breakdown."""
    BUCKET_NAMES = ["a:m2", "a:m1", "a:m0", "b:m2", "b:m1", "b:m0", "ctx_other", "aux"]
    pred_names = ["c:m2", "c:m1", "c:m0"]
    print("\nMantissa-bit attention breakdown (mean attention weight, averaged "
          "across batch, heads, and layers).")
    print(f"  {'predict':<8}  " + "  ".join(f"{n:>9}" for n in BUCKET_NAMES))
    for pi, pn in enumerate(pred_names):
        row = "  ".join(f"{breakdown['per_predictor'][pi, bi]:>9.4f}"
                        for bi in range(len(BUCKET_NAMES)))
        print(f"  {pn:<8}  {row}")
    print("\nMantissa-only weight (normalized over the 6 operand mantissa positions):")
    print(f"  {'predict':<8}  " + "  ".join(f"{n:>9}" for n in BUCKET_NAMES[:6]))
    for pi, pn in enumerate(pred_names):
        total = breakdown['op_bit_dist'][pi].sum()
        if total > 0:
            normalized = breakdown['op_bit_dist'][pi] / total
            row = "  ".join(f"{x:>9.3f}" for x in normalized)
        else:
            row = "  ".join("    nan  " for _ in range(6))
        print(f"  {pn:<8}  {row}")


def print_attention_summary(att: np.ndarray, top_k: int = 4) -> None:
    """Pretty-print: for each (layer, head, result-bit-being-predicted), show the
    top-k input positions attended to."""
    n_layer, n_head, n_results, T = att.shape
    print(f"\nAttention from result-predictor positions (averaged over batch).")
    print(f"For each (layer, head, predicting result-bit) → top-{top_k} attended input positions:\n")
    for L in range(n_layer):
        print(f"--- layer {L} ---")
        for h in range(n_head):
            head_rows = []
            for ri, pred_bit_name in enumerate(["c:sign", "c:e3", "c:e2", "c:e1", "c:e0", "c:m2", "c:m1", "c:m0"]):
                row = att[L, h, ri, :]
                top_idx = np.argsort(row)[::-1][:top_k]
                tops = "  ".join(
                    f"{INPUT_POS_NAMES[i]}={row[i]:.2f}"
                    for i in top_idx
                )
                head_rows.append(f"   pred {pred_bit_name}:  {tops}")
            print(f"  head {h}:")
            print("\n".join(head_rows))


# ---- Probe 2: direct per-layer logit attribution -------------------------

@torch.no_grad()
def direct_layer_attribution(
    hooked: HookedTransformer,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    """For each result-predictor position, attribute the *correct-class* logit
    to (embedding) + each transformer block's contribution.

    Method: read `blocks.{i}.hook_resid_pre` (i.e., the residual stream entering
    block i), apply the model's final norm + unembed to it, get logits-if-we-stopped-
    here. The difference `logit_post_i - logit_post_{i-1}` is block i's contribution.
    Index 0 ('embed') is the pure-embedding logit.

    Returns:
      'contributions': (n_layer + 1, B, n_result_predictor=8) — per-(stage, sample,
                      result-bit) correct-class logit contribution.
      'final_logits':  (B, n_result_predictor=8) — the actual logit at the correct
                      class for each result-predictor position.
    """
    idx = torch.from_numpy(inputs).long().to(next(hooked.parameters()).device)
    tgt = torch.from_numpy(targets).long().to(idx.device)
    logits, cache = hooked.run_with_cache(idx, return_type="logits")
    B = idx.shape[0]
    n_layer = hooked.cfg.n_layers
    pos = torch.tensor(RESULT_PREDICTOR_POSITIONS, device=idx.device)
    correct_tokens = tgt[:, pos]  # (B, 8) — correct result-bit token IDs

    def logit_at_correct(resid_stream: torch.Tensor) -> torch.Tensor:
        """resid_stream: (B, T, d_model). Returns logit at correct class for
        result-predictor positions only, shape (B, 8)."""
        normed = hooked.ln_final(resid_stream)
        logits_full = hooked.unembed(normed)               # (B, T, vocab)
        logits_pred = logits_full[:, pos, :]                # (B, 8, vocab)
        return torch.gather(logits_pred, 2, correct_tokens.unsqueeze(-1)).squeeze(-1)

    cum_logits = []
    # Stage 0: pure embedding contribution.
    cum_logits.append(logit_at_correct(cache["blocks.0.hook_resid_pre"]))
    # Stage i+1: after block i.
    for L in range(n_layer):
        cum_logits.append(logit_at_correct(cache[f"blocks.{L}.hook_resid_post"]))

    cum = torch.stack(cum_logits, dim=0)  # (n_layer+1, B, 8)
    # Contributions: stage 0 is embedding; stage i+1 is "block i added this much".
    contrib = torch.zeros_like(cum)
    contrib[0] = cum[0]
    contrib[1:] = cum[1:] - cum[:-1]

    final = cum[-1]  # (B, 8)
    return {
        "contributions": contrib.cpu().numpy(),
        "final_logits": final.cpu().numpy(),
        "cumulative": cum.cpu().numpy(),
    }


def print_attribution_summary(att: dict[str, np.ndarray]) -> None:
    contrib = att["contributions"]      # (n_layer+1, B, 8)
    final = att["final_logits"]         # (B, 8)
    n_stages, B, n_results = contrib.shape
    stage_names = ["embed"] + [f"block{i}" for i in range(n_stages - 1)]
    print(f"\nDirect logit attribution at correct class (avg over batch of {B}).")
    print(f"Each row = predicting result bit; cols = (stage contributions, final).\n")
    header = f"  {'predict':<8}  " + "  ".join(f"{s:>8}" for s in stage_names) + "    final"
    print(header)
    bit_names = ["c:sign", "c:e3", "c:e2", "c:e1", "c:e0", "c:m2", "c:m1", "c:m0"]
    for ri, name in enumerate(bit_names):
        contribs_mean = contrib[:, :, ri].mean(axis=1)
        final_mean = final[:, ri].mean()
        row = "  ".join(f"{v:+8.2f}" for v in contribs_mean)
        print(f"  {name:<8}  {row}    {final_mean:+6.2f}")


# ---- Probe 3: equality / commutativity ----------------------------------

@torch.no_grad()
def commutativity_check(
    hooked: HookedTransformer,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Check f(a, b) == f(b, a) by constructing swapped inputs.

    For each input sequence, swap tokens at positions [1..8] with [10..17] (operand
    a and b regions). Run both versions; compare top-1 predicted token at result
    positions.
    """
    idx = torch.from_numpy(inputs).long().to(next(hooked.parameters()).device)
    swapped = idx.clone()
    swapped[:, POS_A_START:POS_A_END] = idx[:, POS_B_START:POS_B_END]
    swapped[:, POS_B_START:POS_B_END] = idx[:, POS_A_START:POS_A_END]

    logits_orig = hooked(idx, return_type="logits")
    logits_swap = hooked(swapped, return_type="logits")

    pos = torch.tensor(RESULT_PREDICTOR_POSITIONS, device=idx.device)
    pred_orig = logits_orig[:, pos, :].argmax(dim=-1)   # (B, 8)
    pred_swap = logits_swap[:, pos, :].argmax(dim=-1)

    # Per-position agreement rate.
    agreement_per_pos = (pred_orig == pred_swap).float().mean(dim=0).cpu().numpy()
    full_agreement = (pred_orig == pred_swap).all(dim=-1).float().mean().item()
    return {
        "per_position_agreement": agreement_per_pos.tolist(),
        "full_result_agreement": full_agreement,
    }


@torch.no_grad()
def operand_residual_similarity(
    hooked: HookedTransformer,
    inputs: np.ndarray,
) -> np.ndarray:
    """For doubling inputs (where a's bits equal b's bits), measure cosine similarity
    between residual streams at operand-a positions vs operand-b positions, per layer.

    Park Thm 5 strict reading would predict 1.0 at every layer (modulo PE). PE breaks
    the strict claim — but we can ask whether the model *learns to recover* equality
    across layers (similarity → 1.0).

    Returns shape (n_layer + 1, 8) — similarity for each operand bit position across
    the n_layer + 1 residual-stream snapshots (pre-block-0 through post-final-block).
    """
    idx = torch.from_numpy(inputs).long().to(next(hooked.parameters()).device)
    _, cache = hooked.run_with_cache(idx, return_type="logits")
    n_layer = hooked.cfg.n_layers

    snapshots = [cache[f"blocks.0.hook_resid_pre"]]
    for L in range(n_layer):
        snapshots.append(cache[f"blocks.{L}.hook_resid_post"])

    sims = np.zeros((len(snapshots), POS_A_END - POS_A_START))
    for si, snap in enumerate(snapshots):
        for bi, (pa, pb) in enumerate(zip(
            range(POS_A_START, POS_A_END),
            range(POS_B_START, POS_B_END),
        )):
            va = snap[:, pa, :]           # (B, d_model)
            vb = snap[:, pb, :]
            cos = torch.nn.functional.cosine_similarity(va, vb, dim=-1)  # (B,)
            sims[si, bi] = cos.mean().item()
    return sims


def print_equality_summary(
    comm: dict[str, float],
    sim: np.ndarray,
) -> None:
    n_stages, n_bits = sim.shape
    stage_names = ["pre-blk0"] + [f"post-blk{i}" for i in range(n_stages - 1)]
    print("\nCommutativity check  f(a, b) == f(b, a):")
    print(f"  per-position agreement: {[f'{x:.3f}' for x in comm['per_position_agreement']]}")
    print(f"  full-result agreement:  {comm['full_result_agreement']:.4f}")
    print("\nOperand-a vs operand-b residual cosine similarity on doubling (a==b) inputs:")
    print("  (Park Thm 5 strict: should be 1.00 if PE didn't break equality)")
    print(f"  {'stage':<12}  " + "  ".join(f"bit{i}" for i in range(n_bits)))
    bit_names = ["sign", "e3", "e2", "e1", "e0", "m2", "m1", "m0"]
    print(f"  {'':<12}  " + "  ".join(f"{n:>5}" for n in bit_names))
    for si, stage in enumerate(stage_names):
        row = "  ".join(f"{sim[si, bi]:+.2f}" for bi in range(n_bits))
        print(f"  {stage:<12}  {row}")


def _doubling_holdout(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pull n doubling samples (a_bits == b_bits) from the holdout split."""
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=seed)
    rows = table[holdout_idx]
    mask = rows["a_bits"] == rows["b_bits"]
    rows = rows[mask][:n]
    if len(rows) < n:
        print(f"  warning: only {len(rows)} doubling samples in holdout (requested {n})")
    return _make_input_target(rows)


# ---- CLI driver --------------------------------------------------------

def run_all_probes(
    checkpoint: Path,
    regime: str = "cancellation",
    n_samples: int = 64,
    device: str = "cpu",
) -> None:
    print(f"=== Inspection on {checkpoint} ===")
    hooked, gpt = load_hooked(checkpoint, device=device)
    print(f"   params: {sum(p.numel() for p in gpt.parameters()):,}, "
          f"L={gpt.config.n_layer}, E={gpt.config.n_embd}")

    inputs, targets = _sample_regime(regime, n_samples)
    print(f"\nRunning probes on {n_samples} samples from regime {regime!r}...")

    att = attention_at_result_positions(hooked, inputs)
    print_attention_summary(att, top_k=4)

    attribution = direct_layer_attribution(hooked, inputs, targets)
    print_attribution_summary(attribution)

    dbl_in, dbl_tgt = _doubling_holdout(n_samples)
    comm = commutativity_check(hooked, inputs, targets)
    sim = operand_residual_similarity(hooked, dbl_in)
    print_equality_summary(comm, sim)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--regime", default="cancellation",
                   help=f"Which regime to sample from. Options: {REGIME_NAMES}")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    run_all_probes(args.checkpoint, args.regime, args.n_samples, args.device)


if __name__ == "__main__":
    main()
