"""Within-cell variance analysis across seeds.

Question: at borderline cells (those that "barely solve" the task), do different
seeds implement the same algorithm with different quality, or do they find
qualitatively different solutions?

If same algorithm: regimes co-vary across seeds (high default-acc seed also has
high canc-acc, high tie-acc, etc.). Cross-regime Spearman correlation should be
near +1.

If different algorithms: regimes don't co-vary (high default seed might be low
on tie, etc.). Cross-regime correlation is near 0 or negative.

Usage:
    python -m errorboard.analyze_variance [--runs-dir runs] [--band 90 98]
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


REGIMES = [
    "special-values", "overflow", "subnormal-result",
    "cancellation", "rounding-tie", "large-dexp", "default",
]
SHORT = {
    "special-values": "spec", "overflow": "ovfl", "subnormal-result": "sub",
    "cancellation": "canc", "rounding-tie": "tie", "large-dexp": "ldex",
    "default": "dflt",
}
CELL_RE = re.compile(r"sweep-L(\d+)-E(\d+)-s(\d+)")


def last_line_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    last = None
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                last = line
    return json.loads(last) if last else None


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. NaN if either is constant."""
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def collect(runs_dir: Path) -> dict[tuple[int, int], list[tuple[int, dict]]]:
    """Group seeds by (L, E). Returns {(L,E): [(seed, holdout_acc_dict), ...]}."""
    cells: dict[tuple[int, int], list[tuple[int, dict]]] = defaultdict(list)
    for d in runs_dir.iterdir():
        m = CELL_RE.fullmatch(d.name)
        if not m:
            continue
        L, E, seed = int(m.group(1)), int(m.group(2)), int(m.group(3))
        last = last_line_json(d / "metrics.jsonl")
        if last is None or "holdout_acc" not in last:
            continue
        cells[(L, E)].append((seed, last["holdout_acc"]))
    return cells


def summarize_cell(seeds: list[tuple[int, dict]]) -> dict:
    """Compute per-regime mean, stddev, and the regime × regime Spearman matrix
    across seeds within one cell."""
    seeds = sorted(seeds, key=lambda t: t[0])
    matrix = np.array([[s[1][r] for r in REGIMES] for s in seeds])  # (n_seeds, n_regimes)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=1) if matrix.shape[0] > 1 else np.zeros(len(REGIMES))
    n = matrix.shape[0]
    corr = np.full((len(REGIMES), len(REGIMES)), np.nan)
    if n >= 3:
        for i in range(len(REGIMES)):
            for j in range(len(REGIMES)):
                corr[i, j] = spearman(matrix[:, i], matrix[:, j])
    return {
        "n": n, "mean": mean, "std": std, "corr": corr,
        "matrix": matrix, "seeds": [s[0] for s in seeds],
    }


def fisher_avg(rs: list[float]) -> float:
    """Average correlations via Fisher z-transform."""
    rs = [r for r in rs if not np.isnan(r) and -0.9999 < r < 0.9999]
    if not rs:
        return float("nan")
    zs = np.arctanh(rs)
    return float(np.tanh(zs.mean()))


def inspect_cell(L: int, E: int, seeds: list[tuple[int, dict]]) -> None:
    """Detailed dump for one cell: raw matrix, z-scored matrix, pairwise
    between-seed correlation across regimes."""
    seeds = sorted(seeds, key=lambda t: t[0])
    matrix = np.array([[s[1][r] for r in REGIMES] for s in seeds])  # (n_seeds, n_regimes)
    n = matrix.shape[0]
    print(f"\n=== Inspecting L{L}-E{E:03d}  (n_seeds={n}) ===\n")

    # 1. Raw accuracy matrix (percent).
    print("Raw accuracies (% correct), rows=seeds, cols=regimes:")
    print(f"{'seed':<5}  " + "  ".join(f"{SHORT[r]:>6}" for r in REGIMES))
    for i, (sd, acc) in enumerate(seeds):
        row = "  ".join(f"{matrix[i, j]*100:6.2f}" for j in range(len(REGIMES)))
        print(f"s{sd:<4}  {row}")
    print(f"{'mean':<5}  " + "  ".join(f"{matrix[:, j].mean()*100:6.2f}" for j in range(len(REGIMES))))
    print(f"{'std':<5}  " + "  ".join(f"{matrix[:, j].std(ddof=1)*100:6.2f}" for j in range(len(REGIMES))))

    # 2. Z-scored (each col centered + scaled within cell). NaN where std=0.
    print("\nZ-scored deviations (per-regime, within cell):")
    print(f"{'seed':<5}  " + "  ".join(f"{SHORT[r]:>6}" for r in REGIMES))
    col_std = matrix.std(axis=0, ddof=1)
    z = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        if col_std[j] > 0:
            z[:, j] = (matrix[:, j] - matrix[:, j].mean()) / col_std[j]
        else:
            z[:, j] = 0.0
    for i, (sd, _) in enumerate(seeds):
        row = "  ".join(
            ("  .   " if col_std[j] == 0 else f"{z[i, j]:+6.2f}")
            for j in range(matrix.shape[1])
        )
        print(f"s{sd:<4}  {row}")

    # 3. Pairwise seed-to-seed Pearson correlation across non-saturated regimes.
    # A 2-cluster pattern shows up as high within-cluster, low/neg between-cluster.
    active = [j for j in range(matrix.shape[1]) if col_std[j] > 0]
    print(f"\nPairwise seed-to-seed Pearson across {len(active)} non-saturated regimes "
          f"({', '.join(SHORT[REGIMES[j]] for j in active)}):")
    print(f"{' ':<5}  " + "  ".join(f"s{sd:<5}" for sd, _ in seeds))
    for i, (sd_i, _) in enumerate(seeds):
        row_parts = []
        for j, (sd_j, _) in enumerate(seeds):
            if i == j:
                row_parts.append("  1.00")
            else:
                vi = matrix[i, active]
                vj = matrix[j, active]
                if vi.std() == 0 or vj.std() == 0:
                    row_parts.append("  nan ")
                else:
                    r = float(np.corrcoef(vi, vj)[0, 1])
                    row_parts.append(f"{r:+6.2f}")
        print(f"s{sd_i:<4}  " + "  ".join(row_parts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs", type=Path)
    p.add_argument("--band", nargs=2, type=float, default=[90, 98],
                   help="Default-accuracy band defining 'borderline' (percent).")
    p.add_argument("--min-seeds", type=int, default=3)
    p.add_argument("--inspect", nargs="*", default=[],
                   help="Cell name(s) like L2-E044 to dump per-seed detail for.")
    args = p.parse_args()

    cells = collect(args.runs_dir)

    # Handle --inspect first; it can stand alone.
    if args.inspect:
        for name in args.inspect:
            m = re.fullmatch(r"L(\d+)-E(\d+)", name)
            if not m:
                print(f"Skipping bad cell name: {name}")
                continue
            L, E = int(m.group(1)), int(m.group(2))
            if (L, E) not in cells:
                print(f"No data for {name}")
                continue
            inspect_cell(L, E, cells[(L, E)])
        return

    multi = {k: v for k, v in cells.items() if len(v) >= args.min_seeds}
    summaries = {k: summarize_cell(v) for k, v in multi.items()}

    # 1. Per-cell summary, sorted by default-acc mean.
    print(f"=== Multi-seed cells ({len(summaries)} with ≥{args.min_seeds} seeds) ===\n")
    print(f"{'cell':<12} {'n':>3}  " + "  ".join(f"{SHORT[r]:>10}" for r in REGIMES))
    print("-" * (16 + 12 * len(REGIMES)))
    for (L, E), s in sorted(summaries.items(), key=lambda kv: kv[1]["mean"][-1]):
        cell = f"L{L}-E{E:03d}"
        row = "  ".join(
            f"{s['mean'][i]*100:5.1f}±{s['std'][i]*100:.1f}"
            for i in range(len(REGIMES))
        )
        print(f"{cell:<12} {s['n']:>3}  {row}")

    # 2. Borderline cells (default acc in band) - cross-regime correlations.
    lo, hi = args.band[0] / 100, args.band[1] / 100
    border = {
        k: s for k, s in summaries.items()
        if lo <= s["mean"][-1] <= hi and s["n"] >= 4
    }
    print(f"\n=== Borderline cells, default acc in [{lo*100:.0f}, {hi*100:.0f}] "
          f"with ≥4 seeds ({len(border)} cells) ===\n")

    if not border:
        return

    # Per-cell: print Spearman of (each regime) vs default.
    print(f"{'cell':<12} {'n':>3}  default_acc  Spearman(regime, default) across seeds")
    print(f"{' ':<12} {' ':>3}  {'mean±std':<12}  " +
          "  ".join(f"{SHORT[r]:>6}" for r in REGIMES[:-1]))
    print("-" * 80)
    for (L, E), s in sorted(border.items(), key=lambda kv: kv[1]["mean"][-1]):
        cell = f"L{L}-E{E:03d}"
        dflt_idx = REGIMES.index("default")
        rho = [s["corr"][i, dflt_idx] for i in range(len(REGIMES) - 1)]
        acc_str = f"{s['mean'][dflt_idx]*100:5.1f}±{s['std'][dflt_idx]*100:.1f}"
        rho_str = "  ".join(f"{r:+.2f}" if not np.isnan(r) else "  --  " for r in rho)
        print(f"{cell:<12} {s['n']:>3}  {acc_str:<12}  {rho_str}")

    # 3. Aggregate cross-regime correlation matrix across borderline cells (Fisher z avg).
    print(f"\n=== Aggregated Spearman across {len(border)} borderline cells "
          f"(Fisher-z average) ===\n")
    n_r = len(REGIMES)
    agg = np.full((n_r, n_r), np.nan)
    for i in range(n_r):
        for j in range(n_r):
            rs = [s["corr"][i, j] for s in border.values()]
            agg[i, j] = fisher_avg(rs)
    print(f"{' ':<6}  " + "  ".join(f"{SHORT[r]:>6}" for r in REGIMES))
    for i, r in enumerate(REGIMES):
        row = "  ".join(
            f"{agg[i,j]:+.2f}" if not np.isnan(agg[i, j]) else "  --  "
            for j in range(n_r)
        )
        print(f"{SHORT[r]:<6}  {row}")

    # 4. Headline: average off-diagonal of borderline-aggregated matrix.
    mask = ~np.eye(n_r, dtype=bool)
    finite = agg[mask][~np.isnan(agg[mask])]
    if finite.size:
        print(f"\nMean off-diagonal Spearman across borderline cells: "
              f"{finite.mean():+.3f}  (n_pairs={finite.size})")
        print("  Interpretation:")
        print("    +1.00  every seed implements the same algorithm at different quality")
        print("     0.00  seeds find unrelated solutions")
        print("    -1.00  seeds trade off across regimes (different algorithms)")


if __name__ == "__main__":
    main()
