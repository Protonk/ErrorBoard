"""Model-size sweep: fix training hyperparameters, vary (n_layer, n_embd).

Per methodology.md "sweep down to frontier": the smallest model where accuracy
stays clean on the hard regimes (default, rounding-tie) is the interp target.

Grid (4x4 = 16 cells):
    n_layer in {1, 2, 3, 4}
    n_embd  in {16, 32, 64, 128}
    n_head = 4 (fixed; n_embd must be a multiple of n_head)
    d_mlp = 4 * n_embd (standard ratio)

Usage:
    python -m errorboard.sweep                          # run all 16 cells
    python -m errorboard.sweep --skip-existing          # skip cells already completed
    python -m errorboard.sweep --summary                # print summary table
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from .regimes import REGIME_NAMES
from .training import TrainingConfig, train

LAYER_GRID: tuple[int, ...] = (1, 2, 3, 4)
EMBD_GRID: tuple[int, ...] = (16, 32, 64, 128)


def cell_name(n_layer: int, n_embd: int) -> str:
    return f"sweep-L{n_layer}-E{n_embd:03d}"


def compute_n_params(
    n_layer: int, n_embd: int, d_mlp: int, vocab: int = 12, block_size: int = 27
) -> int:
    """Match GPT.num_parameters() for the learned-PE arm, computed in closed form."""
    n = vocab * n_embd                # wte
    n += block_size * n_embd          # wpe (learned-absolute)
    block_n = (
        2 * n_embd                    # 2 RMSNorms (gain only)
        + 3 * n_embd * n_embd         # fused QKV
        + n_embd * n_embd             # attn output proj
        + n_embd * d_mlp              # MLP up
        + d_mlp * n_embd              # MLP down
    )
    n += n_layer * block_n
    n += n_embd                       # final RMSNorm
    n += n_embd * vocab               # lm_head (untied)
    return n


def _load_last_metric_row(run_dir: Path) -> dict | None:
    mpath = run_dir / "metrics.jsonl"
    if not mpath.exists():
        return None
    last = None
    with open(mpath) as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else None


def run_sweep(runs_dir: str, seed: int, max_iters: int, skip_existing: bool) -> None:
    cells = [(L, E) for L in LAYER_GRID for E in EMBD_GRID]
    print(f"Sweep: {len(cells)} cells, seed={seed}, max_iters={max_iters}", flush=True)
    t_total_start = time.time()

    for i, (L, E) in enumerate(cells, 1):
        name = cell_name(L, E)
        run_dir = Path(runs_dir) / name
        status_file = run_dir / "STATUS"
        if (
            skip_existing
            and status_file.exists()
            and status_file.read_text().strip() == "completed"
        ):
            print(f"\n[{i}/{len(cells)}] {name}  SKIP (already completed)", flush=True)
            continue

        n_params = compute_n_params(L, E, 4 * E)
        print(
            f"\n[{i}/{len(cells)}] === {name}  L={L}  E={E}  d_mlp={4*E}  "
            f"params={n_params:,} ===",
            flush=True,
        )
        cfg = TrainingConfig(
            run_name=name,
            runs_dir=runs_dir,
            seed=seed,
            n_layer=L,
            n_head=4,
            n_embd=E,
            d_mlp=4 * E,
            max_iters=max_iters,
        )
        try:
            train(cfg)
        except Exception:
            print(f"[{i}/{len(cells)}] {name}  FAILED", flush=True)
            traceback.print_exc()
            # Continue with the next cell so a single failure doesn't kill the sweep.

    elapsed = time.time() - t_total_start
    print(f"\nSweep complete in {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)


def summarize(runs_dir: str) -> None:
    """Print a per-cell summary table of final natural_loss and per-regime accuracy."""
    regime_short = {
        "special-values": "spec",
        "overflow": "ovfl",
        "subnormal-result": "sub",
        "cancellation": "canc",
        "rounding-tie": "tie",
        "large-dexp": "ldex",
        "default": "dflt",
    }
    regime_order = ["special-values", "overflow", "subnormal-result", "cancellation",
                    "rounding-tie", "large-dexp", "default"]

    header = f"{'cell':<18} {'params':>10} {'nat_loss':>10}  "
    header += "  ".join(f"{regime_short[r]:>5}" for r in regime_order)
    print(header)
    print("-" * len(header))

    for L in LAYER_GRID:
        for E in EMBD_GRID:
            name = cell_name(L, E)
            run_dir = Path(runs_dir) / name
            n_params = compute_n_params(L, E, 4 * E)
            last = _load_last_metric_row(run_dir)
            if last is None:
                print(f"{name:<18} {n_params:>10,} {'(no data)':>10}")
                continue
            nl = last.get("natural_loss", float("nan"))
            accs = last.get("holdout_acc", {})
            acc_strs = []
            for r in regime_order:
                v = accs.get(r)
                acc_strs.append(f"{v*100:>5.1f}" if isinstance(v, (int, float)) else f"{'-':>5}")
            print(f"{name:<18} {n_params:>10,} {nl:>10.5f}  " + "  ".join(acc_strs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=20_000)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip cells whose STATUS is 'completed'",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print summary table of completed cells and exit (no training)",
    )
    args = parser.parse_args()

    if args.summary:
        summarize(args.runs_dir)
        return

    run_sweep(
        runs_dir=args.runs_dir,
        seed=args.seed,
        max_iters=args.max_iters,
        skip_existing=args.skip_existing,
    )
    print("\n=== Final summary ===\n")
    summarize(args.runs_dir)


if __name__ == "__main__":
    main()
