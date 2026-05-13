"""Four-arm multiplication failure-consensus comparison.

Compares L4-E048 multiplication arms across four input representations:
  - learned-PE bit-level   (`bit-mul-L4-E048-s{0..19}`)
  - learned-PE SEM         (`sem-mul-L4-E048-s{0..19}`)
  - learned-PE FoNE F1     (`fone-mul-L4-E048-s{0..19}`)
  - learned-PE FoNE F2     (`fone-f2-mul-L4-E048-s{0..19}`)

Reports the same metrics as `failure_consensus_fone.py` but on the
multiplication pair table and using mult-regime classifications.

Usage:
    python -m errorboard.failure_consensus_mul
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .failure_consensus_sem import correctness_one as tokenizer_aware_correctness
from .fone_model import fone_predict_on_holdout
from .fone_f2_model import fone_f2_predict_on_holdout
from .hooked_bridge import load_hooked
from .pentagon import _repo_root
from .preprocess_mult import build_table, split_train_holdout
from .mult_regimes import REGIME_NAMES
from . import tokenizer as _bit_tokenizer
from . import sem_tokenizer as _sem_tokenizer
from . import fone_tokenizer as _fone_tokenizer
from . import fone_f2_tokenizer as _fone_f2_tokenizer


@torch.no_grad()
def _bit_correctness_mul(checkpoint_path: Path, tokenizer) -> np.ndarray:
    """Per-pair correctness on the MULTIPLICATION holdout split."""
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    hooked, gpt = load_hooked(checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seqs = tokenizer.encode_batch(triples.astype(np.uint8))
    inputs = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    targets = torch.from_numpy(seqs[:, 1:].astype(np.int64))
    pos_c_start = tokenizer.POS_C_START
    pos_c_end = tokenizer.POS_C_END
    n = inputs.shape[0]
    bsz = 512
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


@torch.no_grad()
def _fone_correctness_mul(checkpoint_path: Path, predict_fn) -> np.ndarray:
    """Per-pair correctness for FoNE arms (F1 or F2) on the mult holdout split.

    The FoNE predict functions internally use preprocess.build_table (addition);
    we need the mult version. Override by passing rows explicitly.
    """
    # The FoNE predict functions handle their own forward pass but use
    # preprocess.split_train_holdout internally. For mult we need the mult
    # table's holdout split. The cleanest fix: monkey-patch by setting
    # `preprocess.build_table` in the FoNE module to the mult version
    # before calling predict_fn. Cleaner: re-implement the predict here.
    return _fone_predict_on_mult_holdout(checkpoint_path, predict_fn)


def _fone_predict_on_mult_holdout(checkpoint_path: Path, arm: str):
    """Run a FoNE checkpoint on the MULTIPLICATION holdout split.

    Returns a dict matching fone_predict_on_holdout's output shape.
    """
    if arm == "f1":
        from .fone_model import load_fone, fone_correct
        from .fone_tokenizer import (
            encode_batch, encode_targets, POS_SIGN_C, POS_NUM_C,
        )
        load_fn = load_fone
        correct_fn = fone_correct
        digit_dim = 6
    else:  # f2
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
    bsz = 512
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
        corr[s] = _bit_correctness_mul(full, _bit_tokenizer)
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
        corr[s] = _bit_correctness_mul(full, _sem_tokenizer)
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
        out = _fone_predict_on_mult_holdout(full, arm)
        corr[s] = out["correct"]
    return corr


def _arm_stats(correctness: dict[int, np.ndarray]) -> dict:
    seeds_loaded = sorted(correctness.keys())
    if not seeds_loaded:
        return None
    fail_matrix = np.stack([~correctness[s] for s in seeds_loaded], axis=0)
    fail_count = fail_matrix.sum(axis=0)
    return dict(
        seeds=seeds_loaded,
        fail_matrix=fail_matrix,
        fail_count=fail_count,
        n_seeds=len(seeds_loaded),
        n_pairs=fail_matrix.shape[1],
        p_mean=float(fail_matrix.mean()),
        var_obs=float(fail_count.var()),
    )


def _comparison_table(arms):
    lines = ["## Cross-arm headline table", ""]
    lines.append("| arm | mean fail % | var ratio | core % | lottery % | easy % | "
                 "p̂ ≥ 0.8 | p̂ = 1.0 |")
    lines.append("|-----|------:|------:|------:|------:|------:|------:|------:|")
    for name, st in arms:
        if st is None:
            lines.append(f"| {name} | — | — | — | — | — | — | — |")
            continue
        p = st["p_mean"]
        iid_var = st["n_seeds"] * p * (1 - p)
        var_ratio = st["var_obs"] / iid_var if iid_var > 0 else float("nan")
        n_seeds = st["n_seeds"]
        core = int((st["fail_count"] == n_seeds).sum())
        lot = int(((st["fail_count"] >= 1) & (st["fail_count"] <= n_seeds - 1)).sum())
        easy = int((st["fail_count"] == 0).sum())
        n_high = int((st["fail_count"] >= 0.8 * n_seeds).sum())
        lines.append(f"| {name} | {p*100:.2f}% | {var_ratio:.2f} | "
                     f"{core/st['n_pairs']*100:.2f}% | {lot/st['n_pairs']*100:.1f}% | "
                     f"{easy/st['n_pairs']*100:.1f}% | {n_high} | {core} |")
    lines.append("")
    return lines


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


def _jaccard_table(arms):
    lines = ["## Pairwise lottery-zone Jaccard", "",
             "Lottery zone = pairs failed by 1..n_seeds-1 of the seeds.", ""]
    masks = {}
    for name, st in arms:
        if st is None:
            masks[name] = None
            continue
        n_seeds = st["n_seeds"]
        masks[name] = (st["fail_count"] >= 1) & (st["fail_count"] <= n_seeds - 1)
    names = [name for name, _ in arms]
    header = "| | " + " | ".join(names) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(names))
    for a in names:
        row = f"| {a} |"
        if masks[a] is None:
            row += " — |" * len(names)
            lines.append(row)
            continue
        for b in names:
            if a == b:
                row += " 1.000 |"
                continue
            if masks[b] is None:
                row += " — |"
                continue
            inter = int((masks[a] & masks[b]).sum())
            union = int((masks[a] | masks[b]).sum())
            jacc = inter / union if union > 0 else float("nan")
            row += f" {jacc:.3f} |"
        lines.append(row)
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", default="notes/mul_arm_comparison.md")
    args = p.parse_args()
    repo = _repo_root()

    print("Loading bit-level mult arm...")
    corr_bit = _load_bit_arm("bit-mul-L4-E048", args.n_seeds, repo)
    print("Loading SEM mult arm...")
    corr_sem = _load_sem_arm("sem-mul-L4-E048", args.n_seeds, repo)
    print("Loading FoNE F1 mult arm...")
    corr_f1 = _load_fone_arm("fone-mul-L4-E048", args.n_seeds, repo, arm="f1")
    print("Loading FoNE F2 mult arm...")
    corr_f2 = _load_fone_arm("fone-f2-mul-L4-E048", args.n_seeds, repo, arm="f2")

    arms = [
        ("bit (mul)", _arm_stats(corr_bit)),
        ("SEM (mul)", _arm_stats(corr_sem)),
        ("FoNE F1 (mul)", _arm_stats(corr_f1)),
        ("FoNE F2 (mul)", _arm_stats(corr_f2)),
    ]

    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    regime_ids = rows["regime_id"]

    sections = [
        "# Four-arm multiplication comparison (L4-E048, 20 seeds, iter 20k)",
        "",
        f"Multiplication pair table; holdout split seed=0, n={len(holdout_idx)}.",
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
