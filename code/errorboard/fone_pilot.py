"""FoNE pilot at L4-E128: scale-stability test for the anti-ε sign-flip.

The L4-E048 FoNE arm runs at ~112K params — <2% of Zhou's smallest reported
working scale (8.31M-param 1-layer FoNE). At that capacity, FoNE flips the
anti-ε severity correlation positive (+0.42, +0.51) where bit/RoPE/SEM all
sit at -0.7 to -0.92.

This pilot tests whether the sign-flip is operation-specific (would survive
at higher capacity) or a small-model artifact (would soften / reverse with
more params). L4-E128 is the same depth as the existing arms with 7× the
parameter count; matched to pentagon V4 for cross-reference.

5 seeds, 20k iters each, learned PE. Run names: `fone-L4-E128-s{0..4}`.

Usage:
    python -m errorboard.fone_pilot [--runs-dir runs]
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from .fone_training import FoneTrainingConfig, train


def _is_completed(run_dir: Path) -> bool:
    status_file = run_dir / "STATUS"
    return status_file.exists() and status_file.read_text().strip() == "completed"


def run_one(seed: int, runs_dir: str, max_iters: int = 20_000) -> None:
    run_name = f"fone-L4-E128-s{seed}"
    run_dir = Path(runs_dir) / run_name
    if _is_completed(run_dir):
        print(f"  {run_name}  SKIP (completed)", flush=True)
        return
    print(f"  === {run_name}  L=4  E=128  seed={seed}  tokenization=fone ===", flush=True)
    cfg = FoneTrainingConfig(
        run_name=run_name,
        runs_dir=runs_dir,
        seed=seed,
        n_layer=4,
        n_head=4,
        n_embd=128,
        d_mlp=512,
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
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=4)
    args = p.parse_args()

    t_start = time.time()
    print(f"\n=== FoNE pilot L4-E128 seeds {args.start}..{args.end} ===\n", flush=True)
    for seed in range(args.start, args.end + 1):
        run_one(seed, args.runs_dir, args.max_iters)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f}m)", flush=True)


if __name__ == "__main__":
    main()
