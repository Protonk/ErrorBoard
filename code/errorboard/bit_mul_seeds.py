"""Launch the bit-level multiplication arm: L4-E048 at 20 seeds.

Mirrors the existing learned-PE L4-E048 addition arm (run names
`sweep-L4-E048-s{N}`) but with `operation="mul"`. Run names are
`bit-mul-L4-E048-s{N}` to keep mult and add results disjoint on disk.

Usage:
    python -m errorboard.bit_mul_seeds [--runs-dir runs]
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from .training import TrainingConfig, train


def _is_completed(run_dir: Path) -> bool:
    status_file = run_dir / "STATUS"
    return status_file.exists() and status_file.read_text().strip() == "completed"


def run_one(seed: int, runs_dir: str, max_iters: int = 20_000) -> None:
    run_name = f"bit-mul-L4-E048-s{seed}"
    run_dir = Path(runs_dir) / run_name
    if _is_completed(run_dir):
        print(f"  {run_name}  SKIP (completed)", flush=True)
        return
    print(f"  === {run_name}  L=4  E=48  seed={seed}  tokenization=bit  operation=mul ===",
          flush=True)
    cfg = TrainingConfig(
        run_name=run_name,
        runs_dir=runs_dir,
        seed=seed,
        n_layer=4,
        n_head=4,
        n_embd=48,
        d_mlp=192,
        pos_encoding="learned",
        tokenization="bit",
        operation="mul",
        max_iters=max_iters,
    )
    try:
        train(cfg)
    except Exception:
        print(f"  {run_name}  FAILED", flush=True)
        traceback.print_exc()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=19)
    args = p.parse_args()

    t_start = time.time()
    print(f"\n=== bit-level mult L4-E048 seeds {args.start}..{args.end} ===\n",
          flush=True)
    for seed in range(args.start, args.end + 1):
        run_one(seed, args.runs_dir, args.max_iters)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
