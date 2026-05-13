"""Four-arm reciprocal failure-consensus comparison.

Compares L4-E048 reciprocal arms across four input representations:
  - learned-PE bit-level    (`bit-recip-L4-E048-s{0..19}`)
  - learned-PE SEM          (`sem-recip-L4-E048-s{0..19}`)
  - learned-PE FoNE F1      (`fone-recip-L4-E048-s{0..19}`)
  - learned-PE FoNE F2      (`fone-f2-recip-L4-E048-s{0..19}`)

Same probe shape as failure_consensus_mul.py. The pair table is 256
entries with b=0x00 padding; holdout is ~52 entries (20% × 256).
Statistics are intrinsically noisier than the mult arc's; per-regime
counts are also small.

Usage:
    python -m errorboard.failure_consensus_recip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .failure_consensus_mul import (
    _arm_stats, _bit_correctness_mul, _comparison_table,
    _fone_predict_on_mult_holdout, _jaccard_table,
)
from .pentagon import _repo_root
from .preprocess_recip import build_table, split_train_holdout
from .recip_regimes import REGIME_NAMES
from . import tokenizer as _bit_tokenizer
from . import sem_tokenizer as _sem_tokenizer


@torch.no_grad()
def _bit_correctness_recip(checkpoint_path: Path, tokenizer) -> np.ndarray:
    """Per-pair correctness on the RECIPROCAL holdout split."""
    from .hooked_bridge import load_hooked
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    hooked, _gpt = load_hooked(checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seqs = tokenizer.encode_batch(triples.astype(np.uint8))
    inputs = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    targets = torch.from_numpy(seqs[:, 1:].astype(np.int64))
    pos_c_start = tokenizer.POS_C_START
    pos_c_end = tokenizer.POS_C_END
    n = inputs.shape[0]
    bsz = 256
    correct = np.zeros(n, dtype=bool)
    for i in range(0, n, bsz):
        idx = inputs[i:i+bsz].to(next(hooked.parameters()).device)
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)
        tgt = targets[i:i+bsz].to(idx.device)
        pred_slice = preds[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        tgt_slice = tgt[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
    return correct


def _fone_predict_on_recip_holdout(checkpoint_path: Path, arm: str):
    """Run a FoNE checkpoint on the RECIPROCAL holdout split."""
    if arm == "f1":
        from .fone_model import load_fone, fone_correct
        from .fone_tokenizer import (
            encode_batch, encode_targets, POS_SIGN_C, POS_NUM_C,
        )
        load_fn = load_fone
        correct_fn = fone_correct
        digit_dim = 6
    else:
        from .fone_f2_model import load_fone_f2, fone_f2_correct
        from .fone_f2_tokenizer import (
            encode_batch, encode_targets, POS_SIGN_C, POS_NUM_C,
        )
        load_fn = load_fone_f2
        correct_fn = fone_f2_correct
        digit_dim = 18

    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    model = load_fn(checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    tokens, num_values = encode_batch(triples)
    sign_target, digit_target, is_nan = encode_targets(triples)

    n = len(rows)
    bsz = 256
    correct = np.zeros(n, dtype=bool)
    sign_pred = np.zeros(n, dtype=np.int64)
    digit_pred = np.zeros((n, digit_dim), dtype=np.int64)
    sign_pos = POS_SIGN_C - 1
    digit_pos = POS_NUM_C - 1
    for i in range(0, n, bsz):
        input_ids = torch.from_numpy(tokens[i:i+bsz, :-1]).long()
        nv = torch.from_numpy(num_values[i:i+bsz, :-1]).float()
        st = torch.from_numpy(sign_target[i:i+bsz]).long()
        dt = torch.from_numpy(digit_target[i:i+bsz]).long()
        nan_t = torch.from_numpy(is_nan[i:i+bsz])
        out = model(input_ids, nv)
        sp = out["vocab_logits"][:, sign_pos, :].argmax(dim=-1)
        dp = out["digit_logits"][:, digit_pos, :, :].argmax(dim=-1)
        c = correct_fn(out, st, dt, nan_t, sign_pos, digit_pos)
        sign_pred[i:i+bsz] = sp.cpu().numpy()
        digit_pred[i:i+bsz] = dp.cpu().numpy()
        correct[i:i+bsz] = c.cpu().numpy()
    return {
        "rows": rows,
        "correct": correct,
        "sign_pred": sign_pred,
        "digit_pred": digit_pred,
        "sign_target": sign_target,
        "digit_target": digit_target,
        "is_nan": is_nan,
    }


def _load_bit_arm(prefix: str, n_seeds: int, repo: Path) -> dict[int, np.ndarray]:
    corr = {}
    for s in range(n_seeds):
        rel = f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        full = repo / rel
        if not full.exists():
            print(f"  {prefix} s{s}: SKIP (not found)", flush=True)
            continue
        print(f"  {prefix} s{s}: {rel}", flush=True)
        corr[s] = _bit_correctness_recip(full, _bit_tokenizer)
    return corr


def _load_sem_arm(prefix: str, n_seeds: int, repo: Path) -> dict[int, np.ndarray]:
    corr = {}
    for s in range(n_seeds):
        rel = f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        full = repo / rel
        if not full.exists():
            print(f"  {prefix} s{s}: SKIP (not found)", flush=True)
            continue
        print(f"  {prefix} s{s}: {rel}", flush=True)
        corr[s] = _bit_correctness_recip(full, _sem_tokenizer)
    return corr


def _load_fone_arm(prefix: str, n_seeds: int, repo: Path,
                   arm: str = "f1") -> dict[int, np.ndarray]:
    corr = {}
    for s in range(n_seeds):
        rel = f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        full = repo / rel
        if not full.exists():
            print(f"  {prefix} s{s}: SKIP (not found)", flush=True)
            continue
        print(f"  {prefix} s{s}: {rel}", flush=True)
        out = _fone_predict_on_recip_holdout(full, arm)
        corr[s] = out["correct"]
    return corr


def _regime_table(arms, regime_ids):
    lines = ["## Per-regime mean fail rate", ""]
    header = "| regime | n |"
    for name, st in arms:
        header += f" {name} |"
    lines.append(header)
    lines.append("|--------|------|" + "------|" * len(arms))
    for ri, rname in enumerate(REGIME_NAMES):
        m = (regime_ids == ri)
        n_tot = int(m.sum())
        if n_tot == 0:
            continue
        row = f"| {rname} | {n_tot} |"
        for _, st in arms:
            if st is None:
                row += " — |"
                continue
            rate = float(st["fail_matrix"][:, m].mean()) * 100
            row += f" {rate:.2f}% |"
        lines.append(row)
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", default="notes/recip_arm_comparison.md")
    args = p.parse_args()
    repo = _repo_root()

    print("Loading bit-level recip arm...")
    corr_bit = _load_bit_arm("bit-recip-L4-E048", args.n_seeds, repo)
    print("Loading SEM recip arm...")
    corr_sem = _load_sem_arm("sem-recip-L4-E048", args.n_seeds, repo)
    print("Loading FoNE F1 recip arm...")
    corr_f1 = _load_fone_arm("fone-recip-L4-E048", args.n_seeds, repo, arm="f1")
    print("Loading FoNE F2 recip arm...")
    corr_f2 = _load_fone_arm("fone-f2-recip-L4-E048", args.n_seeds, repo, arm="f2")

    arms = [
        ("bit (recip)", _arm_stats(corr_bit)),
        ("SEM (recip)", _arm_stats(corr_sem)),
        ("FoNE F1 (recip)", _arm_stats(corr_f1)),
        ("FoNE F2 (recip)", _arm_stats(corr_f2)),
    ]

    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    regime_ids = rows["regime_id"]

    sections = [
        "# Four-arm reciprocal comparison (L4-E048, 20 seeds, iter 20k)",
        "",
        f"Reciprocal pair table (256 entries with b=0x00 padding); holdout split "
        f"seed=0, n={len(holdout_idx)}.",
        "",
        f"dayval format-intrinsic ε_floor for FP8 E4M3 reciprocal: 0.1875 "
        f"(worst input 0x75). No algorithm can do better than this floor on "
        f"the worst-case input.",
        "",
    ]
    sections += _comparison_table(arms)
    sections += _regime_table(arms, regime_ids)
    sections += _jaccard_table(arms)

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
