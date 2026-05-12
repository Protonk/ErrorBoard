"""Launch L4-E044 seeds 5..19 to bring the lottery-zone characterization to 20 seeds.

Pairs with the failure_consensus analysis: with 20 seeds we can resolve per-pair
failure probability to ±0.11 (at p=0.5), enough to distinguish 2-3 probability
bins within the lottery zone.

Usage:
    python -m errorboard.lottery_seeds [--runs-dir runs]
"""

from __future__ import annotations

import argparse

from .sweep import Cell, run_cells


def lottery_cells(start: int = 5, end: int = 19) -> list[Cell]:
    """L4-E044 at seeds `start`..`end` inclusive."""
    return [Cell(n_layer=4, n_embd=44, seed=s) for s in range(start, end + 1)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--start-seed", type=int, default=5)
    p.add_argument("--end-seed", type=int, default=19)
    args = p.parse_args()

    cells = lottery_cells(args.start_seed, args.end_seed)
    print(f"Lottery expansion: {len(cells)} L4-E044 seeds, "
          f"{args.start_seed}..{args.end_seed}")
    for c in cells:
        print(f"  {c.name}")
    run_cells(cells=cells, runs_dir=args.runs_dir, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
