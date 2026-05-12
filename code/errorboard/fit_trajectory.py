"""Probe 1: regime-trajectory ordering vs per-pair training density.

Tests whether the order in which regimes saturate during training matches
per-pair training density (fit hypothesis) or structural complexity (abstraction
hypothesis).

Setup:
  - StratifiedSampler distributes equal mass across active regimes per batch.
  - Per-pair training density = mass_per_regime / regime_size.
  - Smaller regimes → higher per-pair density → fit hypothesis predicts earlier
    saturation.

Compare two orderings:
  (a) per-pair density rank order (smallest regime first)
  (b) saturation rank order (which regime crossed 95% accuracy first during
       training, from L4-E044-s0's per-iter trajectory)

A high Spearman correlation between (a) and (b) supports the fit hypothesis.
A low or negative correlation supports the structural hypothesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES, NUM_REGIMES


# Cells whose trajectories we'll inspect. L4-E044-s0 is the V1-V3 anchor.
# Use a few seeds to check that the saturation order is stable across seeds.
TRAJECTORY_CELLS = [
    ("L4-E044", 0),
    ("L4-E044", 1),
    ("L4-E044", 2),
    ("L4-E044", 3),
    ("L4-E044", 4),
]


def compute_regime_sizes(seed: int = 0) -> dict[str, dict[str, int]]:
    """Compute pair counts in train and holdout splits per regime."""
    table = build_table()
    train_idx, holdout_idx = split_train_holdout(table, seed=seed)
    train_sizes = {}
    holdout_sizes = {}
    for i, name in enumerate(REGIME_NAMES):
        train_sizes[name] = int((table["regime_id"][train_idx] == i).sum())
        holdout_sizes[name] = int((table["regime_id"][holdout_idx] == i).sum())
    return {"train": train_sizes, "holdout": holdout_sizes}


def saturation_iter(metrics_path: Path, regime: str, threshold: float = 0.95) -> int | None:
    """Return the first iter where holdout_acc[regime] >= threshold."""
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ha = row.get("holdout_acc", {})
            if regime in ha and ha[regime] >= threshold:
                return row["iter"]
    return None  # Never saturated


def spearman_rank(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=0.95)
    p.add_argument("--max-iters", type=int, default=20000,
                   help="Total training iters (for mass-per-pair calculation)")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    repo = _repo_root()

    # 1. Regime sizes.
    sizes_by_split = compute_regime_sizes()
    train_sizes = sizes_by_split["train"]

    # 2. Per-pair training density: mass-per-regime / train_size.
    #    Mass-per-regime = total_samples / n_active_regimes.
    total_samples = args.batch_size * args.max_iters
    active_regimes = [r for r in REGIME_NAMES if train_sizes[r] > 0]
    n_active = len(active_regimes)
    mass_per_regime = total_samples / n_active
    density = {}
    for r in REGIME_NAMES:
        if train_sizes[r] > 0:
            density[r] = mass_per_regime / train_sizes[r]
        else:
            density[r] = float("nan")

    # 3. Saturation iter per regime, averaged across seeds.
    sat_per_cell: dict[str, dict[tuple, int | None]] = {r: {} for r in REGIME_NAMES}
    for cell, seed in TRAJECTORY_CELLS:
        run_name = f"sweep-{cell}-s{seed}"
        metrics_path = repo / f"runs/{run_name}/metrics.jsonl"
        if not metrics_path.exists():
            print(f"  warning: {metrics_path} not found, skipping")
            continue
        for r in REGIME_NAMES:
            if train_sizes[r] == 0:
                continue
            sat_per_cell[r][(cell, seed)] = saturation_iter(metrics_path, r, args.threshold)

    # Aggregate: median across seeds, ignore None.
    sat_median: dict[str, float | None] = {}
    sat_min: dict[str, float | None] = {}
    sat_max: dict[str, float | None] = {}
    for r in REGIME_NAMES:
        sats = [v for v in sat_per_cell[r].values() if v is not None]
        if not sats:
            sat_median[r] = None
            sat_min[r] = None
            sat_max[r] = None
        else:
            sat_median[r] = float(np.median(sats))
            sat_min[r] = float(min(sats))
            sat_max[r] = float(max(sats))

    # 4. Table.
    lines = [
        f"# Probe 1 — regime saturation order vs per-pair training density",
        "",
        f"Threshold: first iter where holdout accuracy ≥ {args.threshold*100:.0f}%.",
        f"Trajectory data: {len(TRAJECTORY_CELLS)} seeds of L4-E044.",
        f"Total training mass: {total_samples:,} samples; n_active_regimes = {n_active}; "
        f"mass per regime: {mass_per_regime:,.0f}.",
        "",
        "| regime | train_size | per-pair density | density rank | sat median (min..max) | sat rank |",
        "|--------|-----------:|-----------------:|-------------:|----------------------:|---------:|",
    ]

    # Build rank orderings only over regimes that have data (active + saturated).
    ranked_regimes = [r for r in REGIME_NAMES if sat_median[r] is not None]
    # Sort by density descending (highest density = rank 1)
    density_sorted = sorted(ranked_regimes, key=lambda r: -density[r])
    density_rank = {r: i + 1 for i, r in enumerate(density_sorted)}
    # Sort by saturation iter ascending (earliest sat = rank 1)
    sat_sorted = sorted(ranked_regimes, key=lambda r: sat_median[r])
    sat_rank = {r: i + 1 for i, r in enumerate(sat_sorted)}

    for r in REGIME_NAMES:
        if r not in ranked_regimes:
            if train_sizes[r] == 0:
                note = "structurally empty"
            else:
                note = "never saturated"
            lines.append(f"| {r:<18} | {train_sizes[r]:>10} | {'—':>16} | {'—':>12} | {note:>21} | {'—':>8} |")
            continue
        d = density[r]
        sm = sat_median[r]
        sl, sh = sat_min[r], sat_max[r]
        lines.append(
            f"| {r:<18} | {train_sizes[r]:>10} | {d:>16,.1f} | {density_rank[r]:>12} | "
            f"{int(sm):>6} ({int(sl)}..{int(sh)}) | {sat_rank[r]:>8} |"
        )

    # 5. Correlation.
    densities = [density[r] for r in ranked_regimes]
    sat_iters = [sat_median[r] for r in ranked_regimes]
    rho = spearman_rank(densities, [-s for s in sat_iters])
    # We negate sat_iters so positive rho means "high density → low iter (early sat)"
    lines.append("")
    lines.append(
        f"## Spearman rank correlation (density rank, saturation rank): **ρ = {rho:+.3f}**"
    )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- ρ → +1.0: saturation order matches density order (fit hypothesis)")
    lines.append("- ρ → 0:    density and saturation order are unrelated (structural)")
    lines.append("- ρ → −1.0: saturation order is opposite of density order (anti-fit)")

    output = "\n".join(lines)
    print(output)

    out_path = repo / "notes/fit_trajectory_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
