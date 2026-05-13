"""Four-arm failure-consensus comparison incorporating FoNE.

Compares L4-E048 with four input representations:
  - learned-PE bit-level
  - RoPE        bit-level
  - learned-PE SEM       (3-token sign/exp/mant)
  - learned-PE FoNE      (1 [NUM] token + Fourier features + per-digit decoder)

For each arm: per-pair failure count across seeds, p̂ binning, per-regime
fail rates, and cross-arm Jaccards on the lottery zones.

Same holdout split (seed=0). Per-pair correctness definitions:
  bit / SEM : token-by-token match
  FoNE      : sign class match AND (NaN OR digit-by-digit match)

All four definitions identify the same underlying notion ("did the model
emit the correct FP8 value?") — the spot check in fone_tokenizer asserts
no two non-NaN FP8 values share a (sign, 6-digit) representation.

Usage:
    python -m errorboard.failure_consensus_fone
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .failure_consensus import correctness_one as bit_correctness_one
from .failure_consensus_sem import correctness_one as tokenizer_aware_correctness
from .fone_model import fone_predict_on_holdout
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES
from . import tokenizer as _bit_tokenizer
from . import sem_tokenizer as _sem_tokenizer


def _load_bit_arm(prefix: str, n_seeds: int, repo: Path) -> dict[int, np.ndarray]:
    corr = {}
    for s in range(n_seeds):
        rel = f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        full = repo / rel
        if not full.exists():
            print(f"  {prefix} s{s}: SKIP (not found)", flush=True)
            continue
        print(f"  {prefix} s{s}: {rel}", flush=True)
        corr[s] = bit_correctness_one(full)
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
        corr[s] = tokenizer_aware_correctness(full, _sem_tokenizer)
    return corr


def _load_fone_arm(prefix: str, n_seeds: int, repo: Path) -> dict[int, np.ndarray]:
    corr = {}
    for s in range(n_seeds):
        rel = f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        full = repo / rel
        if not full.exists():
            print(f"  {prefix} s{s}: SKIP (not found)", flush=True)
            continue
        print(f"  {prefix} s{s}: {rel}", flush=True)
        out = fone_predict_on_holdout(full, device="cpu")
        corr[s] = out["correct"]
    return corr


def _arm_stats(correctness: dict[int, np.ndarray]) -> dict:
    seeds_loaded = sorted(correctness.keys())
    fail_matrix = np.stack([~correctness[s] for s in seeds_loaded], axis=0)
    fail_count = fail_matrix.sum(axis=0)
    n_seeds = len(seeds_loaded)
    n_pairs = fail_matrix.shape[1]
    return dict(
        seeds=seeds_loaded,
        fail_matrix=fail_matrix,
        fail_count=fail_count,
        n_seeds=n_seeds,
        n_pairs=n_pairs,
        p_mean=float(fail_matrix.mean()),
        var_obs=float(fail_count.var()),
    )


def _comparison_table(arms: list[tuple[str, dict]]) -> list[str]:
    lines = ["## Cross-arm headline table", ""]
    lines.append("| arm | mean fail % | var ratio | core % | lottery % | easy % | "
                 "p̂ ≥ 0.8 | p̂ = 1.0 |")
    lines.append("|-----|------:|------:|------:|------:|------:|------:|------:|")
    for name, st in arms:
        p = st["p_mean"]
        iid_var = st["n_seeds"] * p * (1 - p)
        var_ratio = st["var_obs"] / iid_var if iid_var > 0 else float("nan")
        n_seeds = st["n_seeds"]
        core = int((st["fail_count"] == n_seeds).sum())
        lot = int(((st["fail_count"] >= 1) & (st["fail_count"] <= n_seeds - 1)).sum())
        easy = int((st["fail_count"] == 0).sum())
        n_high = int((st["fail_count"] >= 0.8 * n_seeds).sum())
        n_full = int((st["fail_count"] == n_seeds).sum())
        lines.append(f"| {name} | {p*100:.2f}% | {var_ratio:.2f} | "
                     f"{core/st['n_pairs']*100:.2f}% | {lot/st['n_pairs']*100:.1f}% | "
                     f"{easy/st['n_pairs']*100:.1f}% | {n_high} | {n_full} |")
    lines.append("")
    return lines


def _phat_distribution(arms: list[tuple[str, dict]]) -> list[str]:
    lines = ["## Per-pair p̂ distribution (20 seeds)", ""]
    bins = [(0, 0), (1, 1), (2, 5), (6, 10), (11, 15), (16, 19), (20, 20)]
    header = "| bin | " + " | ".join(name for name, _ in arms) + " |"
    sep = "|-----" + "|------" * len(arms) + "|"
    lines.append(header)
    lines.append(sep)
    for lo, hi in bins:
        label = f"{lo}/20" if lo == hi else f"{lo}-{hi}/20"
        row = f"| {label} |"
        for _, st in arms:
            cnt = int(((st["fail_count"] >= lo) & (st["fail_count"] <= hi)).sum())
            row += f" {cnt} |"
        lines.append(row)
    lines.append("")
    return lines


def _regime_table(arms: list[tuple[str, dict]], regime_ids: np.ndarray) -> list[str]:
    lines = ["## Per-regime mean fail rate", ""]
    header = "| regime | n |"
    for name, _ in arms:
        header += f" {name} |"
    lines.append(header)
    lines.append("|-----|------|" + "------|" * len(arms))
    for ri, rname in enumerate(REGIME_NAMES):
        m = (regime_ids == ri)
        n_tot = int(m.sum())
        if n_tot == 0:
            continue
        row = f"| {rname} | {n_tot} |"
        for _, st in arms:
            rate = float(st["fail_matrix"][:, m].mean()) * 100
            row += f" {rate:.2f}% |"
        lines.append(row)
    lines.append("")
    return lines


def _jaccard_table(arms: list[tuple[str, dict]]) -> list[str]:
    lines = ["## Pairwise lottery-zone Jaccard", "",
             "Lottery zone = pairs failed by 1..n_seeds-1 of the seeds.", ""]
    masks = {}
    for name, st in arms:
        n_seeds = st["n_seeds"]
        masks[name] = (st["fail_count"] >= 1) & (st["fail_count"] <= n_seeds - 1)
    names = [name for name, _ in arms]
    header = "| | " + " | ".join(names) + " |"
    lines.append(header)
    lines.append("|---|" + "---|" * len(names))
    for a in names:
        row = f"| {a} |"
        for b in names:
            if a == b:
                row += " 1.000 |"
                continue
            inter = int((masks[a] & masks[b]).sum())
            union = int((masks[a] | masks[b]).sum())
            jacc = inter / union if union > 0 else float("nan")
            row += f" {jacc:.3f} |"
        lines.append(row)
    lines.append("")
    return lines


def _heavy_tail_overlap(arms: list[tuple[str, dict]]) -> list[str]:
    lines = ["## Heavy-tail (p̂ ≥ 0.8) overlap across arms", ""]
    masks = {}
    for name, st in arms:
        n_seeds = st["n_seeds"]
        masks[name] = st["fail_count"] >= 0.8 * n_seeds
    names = [name for name, _ in arms]
    for a in names:
        n_a = int(masks[a].sum())
        if n_a == 0:
            lines.append(f"- **{a}** heavy tail empty.")
            continue
        lines.append(f"- **{a}** heavy tail = {n_a} pairs; of those:")
        for b in names:
            if b == a:
                continue
            inter = int((masks[a] & masks[b]).sum())
            lines.append(f"  - {inter} ({inter/n_a*100:.1f}%) are also heavy-tail in **{b}**")
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", default="notes/fone_arm_comparison.md")
    args = p.parse_args()
    repo = _repo_root()

    print("Loading learned-PE bit arm...")
    corr_learned = _load_bit_arm("sweep-L4-E048", args.n_seeds, repo)
    print("Loading RoPE bit arm...")
    corr_rope = _load_bit_arm("rope-L4-E048", args.n_seeds, repo)
    print("Loading SEM arm...")
    corr_sem = _load_sem_arm("sem-L4-E048", args.n_seeds, repo)
    print("Loading FoNE arm...")
    corr_fone = _load_fone_arm("fone-L4-E048", args.n_seeds, repo)

    arms = [
        ("learned-PE bit", _arm_stats(corr_learned)),
        ("RoPE bit", _arm_stats(corr_rope)),
        ("learned-PE SEM", _arm_stats(corr_sem)),
        ("learned-PE FoNE", _arm_stats(corr_fone)),
    ]

    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    regime_ids = rows["regime_id"]

    sections = [
        "# Four-arm comparison: bit vs RoPE-bit vs SEM vs FoNE",
        "",
        f"All L4-E048, iter 20k, {args.n_seeds} seeds each. Same holdout split.",
        "",
    ]
    sections += _comparison_table(arms)
    sections += _phat_distribution(arms)
    sections += _regime_table(arms, regime_ids)
    sections += _jaccard_table(arms)
    sections += _heavy_tail_overlap(arms)

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
