"""F2 (binary FoNE) pilot launcher.

Trains L=4 at configurable n_embd with F2 tokenization. Mirrors
`fone_pilot.py` for direct comparison: same architecture, same training
schedule, same 5 seeds, only the encoding differs.

Run names: `fone-f2-L4-E{NNN}-s{N}`.

Usage:
    python -m errorboard.fone_f2_pilot --n-embd 48
    python -m errorboard.fone_f2_pilot --n-embd 128
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from .fone_f2_training import FoneF2TrainingConfig, train


def _is_completed(run_dir: Path) -> bool:
    status_file = run_dir / "STATUS"
    return status_file.exists() and status_file.read_text().strip() == "completed"


def run_one(seed: int, n_embd: int, runs_dir: str, max_iters: int = 20_000) -> None:
    run_name = f"fone-f2-L4-E{n_embd:03d}-s{seed}"
    run_dir = Path(runs_dir) / run_name
    if _is_completed(run_dir):
        print(f"  {run_name}  SKIP (completed)", flush=True)
        return
    print(f"  === {run_name}  L=4  E={n_embd}  seed={seed}  tokenization=fone-f2 ===",
          flush=True)
    cfg = FoneF2TrainingConfig(
        run_name=run_name,
        runs_dir=runs_dir,
        seed=seed,
        n_layer=4,
        n_head=4,
        n_embd=n_embd,
        d_mlp=4 * n_embd,
        pos_encoding="learned",
        max_iters=max_iters,
    )
    try:
        train(cfg)
    except Exception:
        print(f"  {run_name}  FAILED", flush=True)
        traceback.print_exc()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-embd", type=int, default=48)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=4)
    args = p.parse_args()

    if args.n_embd % 4 != 0:
        raise ValueError(f"n_embd must be divisible by 4 (n_head=4), got {args.n_embd}")

    t_start = time.time()
    print(f"\n=== F2 pilot L4-E{args.n_embd:03d} seeds {args.start}..{args.end} ===\n",
          flush=True)
    for seed in range(args.start, args.end + 1):
        run_one(seed, args.n_embd, args.runs_dir, args.max_iters)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
