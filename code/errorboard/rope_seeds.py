"""Launch the RoPE arm: L4-E048 with RoPE at 20 seeds.

Architecture note: L4-E044 (used for learned-PE pentagon work) has d_head=11
which is odd; RoPE requires even d_head. L4-E048 gives d_head=12 and is right
next to L4-E044 in the borderline ladder (96.5% vs 96.2% default at 5 seeds).

This launcher trains:
  - 15 additional L4-E048 learned-PE seeds (seeds 5..19) to fill out matching
    20-seed coverage for the comparison (we already have seeds 0..4 from the
    original sweep).
  - 20 L4-E048 RoPE seeds (seeds 0..19).

Run names: existing learned-PE use `sweep-L4-E048-s{N}`. RoPE uses
`rope-L4-E048-s{N}` to keep them disjoint.

Usage:
    python -m errorboard.rope_seeds [--runs-dir runs]
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


def run_one(run_name: str, seed: int, pos_encoding: str, runs_dir: str,
            max_iters: int = 20_000) -> None:
    run_dir = Path(runs_dir) / run_name
    if _is_completed(run_dir):
        print(f"  {run_name}  SKIP (completed)", flush=True)
        return
    print(f"  === {run_name}  L=4  E=48  seed={seed}  pos_encoding={pos_encoding} ===",
          flush=True)
    cfg = TrainingConfig(
        run_name=run_name,
        runs_dir=runs_dir,
        seed=seed,
        n_layer=4,
        n_head=4,
        n_embd=48,
        d_mlp=192,
        pos_encoding=pos_encoding,
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
    p.add_argument("--learned-start", type=int, default=5,
                   help="Start seed for learned-PE expansion (we already have 0..4)")
    p.add_argument("--learned-end", type=int, default=19)
    p.add_argument("--rope-start", type=int, default=0)
    p.add_argument("--rope-end", type=int, default=19)
    args = p.parse_args()

    t_start = time.time()

    # Phase 1: expand learned-PE L4-E048 from seeds 0..4 to 0..19.
    print(f"\n=== Phase 1: learned-PE L4-E048 seeds "
          f"{args.learned_start}..{args.learned_end} ===\n", flush=True)
    for seed in range(args.learned_start, args.learned_end + 1):
        run_one(f"sweep-L4-E048-s{seed}", seed, "learned", args.runs_dir,
                args.max_iters)

    # Phase 2: train RoPE L4-E048 at seeds 0..19.
    print(f"\n=== Phase 2: RoPE L4-E048 seeds "
          f"{args.rope_start}..{args.rope_end} ===\n", flush=True)
    for seed in range(args.rope_start, args.rope_end + 1):
        run_one(f"rope-L4-E048-s{seed}", seed, "rope", args.runs_dir,
                args.max_iters)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
