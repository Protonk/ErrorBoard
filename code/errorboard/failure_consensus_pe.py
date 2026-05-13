"""Two-arm failure-consensus comparison: learned-PE vs RoPE at L4-E048, 20 seeds each.

Pairs with failure_consensus.py (which canonically runs the 5-seed L4-E044 sweep).
This script:
  - loads correctness for two run-name prefixes (default learned-PE and RoPE at L4-E048)
  - emits the same lottery-zone / variance / regime / m_c breakdown for each arm
  - emits a side-by-side diff section so the PE-driven differences are visible

Usage:
    python -m errorboard.failure_consensus_pe
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .failure_consensus import binomial_expected, correctness_one
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES


def _checkpoints(prefix: str, n_seeds: int, ckpt: str = "checkpoint_020000.pt") -> dict[int, str]:
    return {s: f"runs/{prefix}-s{s}/{ckpt}" for s in range(n_seeds)}


def _load_arm(prefix: str, n_seeds: int, repo: Path) -> dict[int, np.ndarray]:
    correctness = {}
    for seed, rel in _checkpoints(prefix, n_seeds).items():
        full = repo / rel
        if not full.exists():
            print(f"  {prefix} s{seed}: SKIP (not found at {full})", flush=True)
            continue
        print(f"  {prefix} s{seed}: {rel}", flush=True)
        correctness[seed] = correctness_one(full)
    return correctness


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
        hist=[int((fail_count == k).sum()) for k in range(n_seeds + 1)],
        var_obs=float(fail_count.var()),
    )


def _arm_section(name: str, st: dict) -> list[str]:
    lines = [f"## {name} — single-arm stats", ""]
    lines.append(f"n_seeds = {st['n_seeds']}, n_holdout = {st['n_pairs']}, "
                 f"mean per-(seed,pair) fail rate = {st['p_mean']*100:.2f}%.")
    lines.append("")

    iid = binomial_expected(st["n_seeds"], st["p_mean"], st["n_pairs"])
    iid_var = st["n_seeds"] * st["p_mean"] * (1 - st["p_mean"])

    lines.append(f"**Variance of per-pair fail count**: observed {st['var_obs']:.3f}, "
                 f"i.i.d. predicted {iid_var:.3f} (ratio {st['var_obs']/iid_var:.2f}). "
                 "Higher ⟹ heavier-tailed per-pair difficulty distribution.")
    lines.append("")

    core_mask = st["fail_count"] == st["n_seeds"]
    easy_mask = st["fail_count"] == 0
    lottery_mask = (st["fail_count"] >= 1) & (st["fail_count"] <= st["n_seeds"] - 1)
    lines.append("| stratum | n_pairs | fraction |")
    lines.append("|---------|--------:|---------:|")
    lines.append(f"| structural core ({st['n_seeds']}/{st['n_seeds']} fail) | "
                 f"{int(core_mask.sum())} | {core_mask.mean()*100:.1f}% |")
    lines.append(f"| lottery zone (1..{st['n_seeds']-1}/{st['n_seeds']} fail) | "
                 f"{int(lottery_mask.sum())} | {lottery_mask.mean()*100:.1f}% |")
    lines.append(f"| structural easy (0/{st['n_seeds']} fail) | "
                 f"{int(easy_mask.sum())} | {easy_mask.mean()*100:.1f}% |")
    lines.append("")
    return lines


def _diff_section(name_a: str, st_a: dict, name_b: str, st_b: dict,
                  regime_ids: np.ndarray) -> list[str]:
    lines = [f"## Side-by-side diff: {name_a} vs {name_b}", ""]

    # Top-level summary table.
    lines.append("| metric | " + name_a + " | " + name_b + " | Δ (B − A) |")
    lines.append("|--------|------:|------:|-----:|")

    p_a, p_b = st_a["p_mean"], st_b["p_mean"]
    lines.append(f"| mean fail rate | {p_a*100:.2f}% | {p_b*100:.2f}% | "
                 f"{(p_b - p_a)*100:+.2f}pp |")
    lines.append(f"| variance ratio (obs/iid) | "
                 f"{st_a['var_obs'] / (st_a['n_seeds']*p_a*(1-p_a)):.2f} | "
                 f"{st_b['var_obs'] / (st_b['n_seeds']*p_b*(1-p_b)):.2f} | "
                 f"{st_b['var_obs'] / (st_b['n_seeds']*p_b*(1-p_b)) - st_a['var_obs'] / (st_a['n_seeds']*p_a*(1-p_a)):+.2f} |")

    core_a = (st_a["fail_count"] == st_a["n_seeds"]).mean() * 100
    core_b = (st_b["fail_count"] == st_b["n_seeds"]).mean() * 100
    lot_a = ((st_a["fail_count"] >= 1) & (st_a["fail_count"] <= st_a["n_seeds"] - 1)).mean() * 100
    lot_b = ((st_b["fail_count"] >= 1) & (st_b["fail_count"] <= st_b["n_seeds"] - 1)).mean() * 100
    easy_a = (st_a["fail_count"] == 0).mean() * 100
    easy_b = (st_b["fail_count"] == 0).mean() * 100
    lines.append(f"| structural core % | {core_a:.1f}% | {core_b:.1f}% | {core_b - core_a:+.1f}pp |")
    lines.append(f"| lottery zone % | {lot_a:.1f}% | {lot_b:.1f}% | {lot_b - lot_a:+.1f}pp |")
    lines.append(f"| structural easy % | {easy_a:.1f}% | {easy_b:.1f}% | {easy_b - easy_a:+.1f}pp |")
    lines.append("")

    # Per-pair p estimate distribution: at 20 seeds, p_hat ∈ {0/20, 1/20, ..., 20/20}.
    lines.append(f"### Per-pair p̂ distribution (binned)\n")
    lines.append(f"| bin | {name_a} pairs | {name_b} pairs |")
    lines.append("|-----|-------:|-------:|")
    bins = [(0, 0), (1, 1), (2, 5), (6, 10), (11, 15), (16, 19), (20, 20)]
    for lo, hi in bins:
        a_cnt = int(((st_a["fail_count"] >= lo) & (st_a["fail_count"] <= hi)).sum())
        b_cnt = int(((st_b["fail_count"] >= lo) & (st_b["fail_count"] <= hi)).sum())
        label = f"{lo}/{st_a['n_seeds']}" if lo == hi else f"{lo}-{hi}/{st_a['n_seeds']}"
        lines.append(f"| {label} | {a_cnt} | {b_cnt} |")
    lines.append("")

    # Per-regime fail-rate diff.
    lines.append(f"### Per-regime mean fail rate\n")
    lines.append(f"| regime | n_total | {name_a} fail% | {name_b} fail% | Δ |")
    lines.append("|--------|--------:|------:|------:|------:|")
    for ri, rname in enumerate(REGIME_NAMES):
        m = (regime_ids == ri)
        n_tot = int(m.sum())
        if n_tot == 0:
            continue
        # Mean fail rate across (seeds × pairs in this regime).
        a_rate = float(st_a["fail_matrix"][:, m].mean()) * 100
        b_rate = float(st_b["fail_matrix"][:, m].mean()) * 100
        lines.append(f"| {rname:<18} | {n_tot:>7} | {a_rate:>5.2f}% | {b_rate:>5.2f}% | "
                     f"{b_rate - a_rate:+.2f}pp |")
    lines.append("")

    # Lottery-zone overlap: of pairs in A's lottery zone, how many are in B's lottery zone?
    lot_mask_a = (st_a["fail_count"] >= 1) & (st_a["fail_count"] <= st_a["n_seeds"] - 1)
    lot_mask_b = (st_b["fail_count"] >= 1) & (st_b["fail_count"] <= st_b["n_seeds"] - 1)
    inter = int((lot_mask_a & lot_mask_b).sum())
    union = int((lot_mask_a | lot_mask_b).sum())
    jaccard = inter / union if union else float("nan")
    n_a, n_b = int(lot_mask_a.sum()), int(lot_mask_b.sum())
    lines.append(f"### Lottery-zone overlap\n")
    lines.append(f"- {name_a} lottery: {n_a} pairs")
    lines.append(f"- {name_b} lottery: {n_b} pairs")
    lines.append(f"- intersection: {inter} pairs")
    lines.append(f"- union: {union} pairs")
    lines.append(f"- Jaccard: {jaccard:.3f}")
    lines.append("")
    if n_a and n_b:
        lines.append(f"Of {name_a}'s lottery, {inter / n_a * 100:.1f}% is also "
                     f"{name_b}'s lottery; of {name_b}'s lottery, "
                     f"{inter / n_b * 100:.1f}% is also {name_a}'s. "
                     "High overlap ⟹ both PE arms struggle on the same pairs (format-driven). "
                     "Low overlap ⟹ each PE arm has its own lottery (arch-driven).")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prefix-a", default="sweep-L4-E048")
    p.add_argument("--prefix-b", default="rope-L4-E048")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", default="notes/pe_arm_comparison.md")
    args = p.parse_args()
    repo = _repo_root()

    print(f"Loading arm A: {args.prefix_a} (n_seeds={args.n_seeds})")
    corr_a = _load_arm(args.prefix_a, args.n_seeds, repo)
    print(f"Loading arm B: {args.prefix_b} (n_seeds={args.n_seeds})")
    corr_b = _load_arm(args.prefix_b, args.n_seeds, repo)

    if not corr_a or not corr_b:
        print("Missing seeds. Exiting.")
        return

    st_a = _arm_stats(corr_a)
    st_b = _arm_stats(corr_b)

    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    regime_ids = rows["regime_id"]

    sections = [
        f"# PE arm comparison: {args.prefix_a} vs {args.prefix_b}",
        "",
        f"Both at L4-E048, iter 20k, {args.n_seeds} seeds each. Same holdout split (seed=0).",
        "",
    ]
    sections += _arm_section(args.prefix_a, st_a)
    sections += _arm_section(args.prefix_b, st_b)
    sections += _diff_section(args.prefix_a, st_a, args.prefix_b, st_b, regime_ids)

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
