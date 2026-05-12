"""Model-size sweep with seed-aware cell naming.

Cell naming convention: ``sweep-L{n_layer}-E{n_embd:03d}-s{seed}``.

Three planned rounds, all sharing the same infrastructure:

* **Round 1** — extend the (L, E) grid sparsely at seed 0 to probe deeper
  models (L ∈ {6, 8}) and finer-grain widths (E ∈ {48, 96}).
* **Round 2** — seed expansion at frontier cells (auto-selected based on
  completed runs' default-regime accuracy in [90%, 99.5%]). Seeds 1..4.
* **Round 3** — iso-parameter sweep at fixed budgets. For each
  (target_params, n_layer) cell, pick the n_embd that hits the budget.

Each round is resumable via the always-on skip-completed logic.

Usage::

    python -m errorboard.sweep --round 1
    python -m errorboard.sweep --round 2
    python -m errorboard.sweep --round 3
    python -m errorboard.sweep --summary
    python -m errorboard.sweep --plan 1     # dry-run: list cells for round 1
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .regimes import REGIME_NAMES
from .training import TrainingConfig, train


# ---- core cell model ----

@dataclass(frozen=True)
class Cell:
    n_layer: int
    n_embd: int
    seed: int

    @property
    def name(self) -> str:
        return f"sweep-L{self.n_layer}-E{self.n_embd:03d}-s{self.seed}"


_NAME_RE = re.compile(r"^sweep-L(\d+)-E(\d+)-s(\d+)$")


def parse_cell_name(name: str) -> Cell | None:
    m = _NAME_RE.match(name)
    if m is None:
        return None
    return Cell(n_layer=int(m.group(1)), n_embd=int(m.group(2)), seed=int(m.group(3)))


def compute_n_params(
    n_layer: int, n_embd: int, d_mlp: int, vocab: int = 12, block_size: int = 27
) -> int:
    """Match GPT.num_parameters() for the learned-PE arm in closed form."""
    n = vocab * n_embd                # wte
    n += block_size * n_embd          # wpe (learned absolute)
    block_n = (
        2 * n_embd                    # 2 RMSNorms
        + 3 * n_embd * n_embd         # fused QKV
        + n_embd * n_embd             # attn output proj
        + n_embd * d_mlp              # MLP up
        + d_mlp * n_embd              # MLP down
    )
    n += n_layer * block_n
    n += n_embd                       # final RMSNorm
    n += n_embd * vocab               # lm_head (untied)
    return n


# ---- cell-list helpers ----

def make_grid(
    layers: list[int], widths: list[int], seeds: list[int]
) -> list[Cell]:
    """Cartesian product of (layers, widths, seeds)."""
    return [Cell(L, E, s) for L in layers for E in widths for s in seeds]


def make_seed_expansion(borderline: list[tuple[int, int]], seeds: list[int]) -> list[Cell]:
    """For each (L, E) in borderline, instantiate one cell per seed."""
    return [Cell(L, E, s) for (L, E) in borderline for s in seeds]


def _n_embd_for_target(
    target: int, n_layer: int, n_head: int = 4, min_e: int = 4, max_e: int = 384
) -> int:
    """Find the multiple-of-n_head n_embd whose param count is closest to target."""
    best_e = min_e
    best_diff = float("inf")
    for e in range(min_e, max_e + 1, n_head):
        p = compute_n_params(n_layer, e, 4 * e)
        diff = abs(p - target)
        if diff < best_diff:
            best_diff = diff
            best_e = e
    return best_e


def make_iso_params(
    targets: list[int], layers: list[int], seeds: list[int]
) -> list[Cell]:
    """For each (target, L), pick n_embd that hits the budget; one cell per seed."""
    cells: list[Cell] = []
    for tgt in targets:
        for L in layers:
            E = _n_embd_for_target(tgt, L)
            for s in seeds:
                cells.append(Cell(L, E, s))
    return cells


def make_union(*cell_lists: list[Cell]) -> list[Cell]:
    """Deduplicating union of multiple cell lists."""
    seen: set[Cell] = set()
    out: list[Cell] = []
    for cl in cell_lists:
        for c in cl:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


# ---- planned rounds ----

def round1_cells() -> list[Cell]:
    """Round 1: extend (L, E) grid sparsely at seed 0."""
    deeper = make_grid(layers=[6, 8], widths=[16, 32, 48, 64, 96, 128], seeds=[0])
    finer_widths = make_grid(layers=[1, 2, 3, 4], widths=[48, 96], seeds=[0])
    return make_union(deeper, finer_widths)


def round2_cells(runs_dir: str, lo: float = 0.95, hi: float = 0.995) -> list[Cell]:
    """Round 2: seed expansion at borderline cells from completed runs.

    A cell (L, E) is borderline if the default-regime holdout accuracy at any
    completed seed is in [lo, hi]. We re-run seeds {1, 2, 3, 4}.
    """
    borderline: list[tuple[int, int]] = []
    seen_le: set[tuple[int, int]] = set()
    for c, row in _iter_completed(runs_dir):
        if c.seed != 0:
            continue  # Round 2 selects based on seed-0 runs (round 0/1).
        acc = row.get("holdout_acc", {}).get("default")
        if not isinstance(acc, (int, float)):
            continue
        if lo <= acc <= hi:
            key = (c.n_layer, c.n_embd)
            if key not in seen_le:
                seen_le.add(key)
                borderline.append(key)
    borderline.sort()
    return make_seed_expansion(borderline, seeds=[1, 2, 3, 4])


def round3_cells() -> list[Cell]:
    """Round 3: iso-parameter sweep — width-vs-depth at fixed capacity."""
    return make_iso_params(
        targets=[25_000, 50_000, 100_000, 200_000],
        layers=[1, 2, 4, 8],
        seeds=[0, 1, 2, 3, 4],
    )


# ---- runner ----

def _is_completed(run_dir: Path) -> bool:
    status_file = run_dir / "STATUS"
    return status_file.exists() and status_file.read_text().strip() == "completed"


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


def _iter_completed(runs_dir: str):
    """Yield (Cell, last_metric_row) for every completed sweep cell in runs_dir."""
    root = Path(runs_dir)
    if not root.exists():
        return
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        c = parse_cell_name(p.name)
        if c is None or not _is_completed(p):
            continue
        row = _load_last_metric_row(p)
        if row is None:
            continue
        yield c, row


def run_cells(
    cells: list[Cell],
    runs_dir: str,
    max_iters: int,
    skip_existing: bool = True,
) -> None:
    """Run a list of cells sequentially. Always skip already-completed cells."""
    print(f"Plan: {len(cells)} cells, max_iters={max_iters}, skip_existing={skip_existing}", flush=True)
    t_start = time.time()

    for i, c in enumerate(cells, 1):
        run_dir = Path(runs_dir) / c.name
        if skip_existing and _is_completed(run_dir):
            print(f"\n[{i}/{len(cells)}] {c.name}  SKIP (completed)", flush=True)
            continue

        n_params = compute_n_params(c.n_layer, c.n_embd, 4 * c.n_embd)
        print(
            f"\n[{i}/{len(cells)}] === {c.name}  L={c.n_layer}  E={c.n_embd}  "
            f"params={n_params:,}  seed={c.seed} ===",
            flush=True,
        )
        cfg = TrainingConfig(
            run_name=c.name,
            runs_dir=runs_dir,
            seed=c.seed,
            n_layer=c.n_layer,
            n_head=4,
            n_embd=c.n_embd,
            d_mlp=4 * c.n_embd,
            max_iters=max_iters,
        )
        try:
            train(cfg)
        except Exception:
            print(f"[{i}/{len(cells)}] {c.name}  FAILED", flush=True)
            traceback.print_exc()
            # Continue with the next cell.

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s ({elapsed / 60:.1f}m)", flush=True)


# ---- summary ----

_REGIME_ORDER = [
    "special-values", "overflow", "subnormal-result", "cancellation",
    "rounding-tie", "large-dexp", "default",
]
_REGIME_SHORT = {
    "special-values": "spec",
    "overflow": "ovfl",
    "subnormal-result": "sub",
    "cancellation": "canc",
    "rounding-tie": "tie",
    "large-dexp": "ldex",
    "default": "dflt",
}


def _fmt_mean_std(vals: list[float], width: int = 5, decimals: int = 1) -> str:
    if not vals:
        return f"{'-':>{width*2+4}}"
    mean = statistics.mean(vals)
    if len(vals) >= 2:
        std = statistics.stdev(vals)
        return f"{mean:>{width}.{decimals}f}±{std:>{width-2}.{decimals}f}"
    return f"{mean:>{width}.{decimals}f}  ({'1s':>{width-2}})"


def summarize(runs_dir: str) -> None:
    """Print summary grouped by (L, E), showing mean ± stddev across seeds."""
    groups: dict[tuple[int, int], list[tuple[Cell, dict]]] = defaultdict(list)
    for c, row in _iter_completed(runs_dir):
        groups[(c.n_layer, c.n_embd)].append((c, row))
    if not groups:
        print(f"No completed sweep cells in {runs_dir}/")
        return

    header_cols = ["cell", "params", "n_seeds", "nat_loss"] + [
        _REGIME_SHORT[r] for r in _REGIME_ORDER
    ]
    print(f"{header_cols[0]:<13} {header_cols[1]:>10} {header_cols[2]:>7} "
          f"{header_cols[3]:>14}  " + "  ".join(f"{h:>11}" for h in header_cols[4:]))
    print("-" * (13 + 1 + 10 + 1 + 7 + 1 + 14 + 2 + 7 * 13))

    for (L, E), cell_rows in sorted(groups.items()):
        n_params = compute_n_params(L, E, 4 * E)
        nls = [r["natural_loss"] for _, r in cell_rows
               if isinstance(r.get("natural_loss"), (int, float))]
        nl_str = _fmt_mean_std(nls, width=5, decimals=4)
        acc_strs: list[str] = []
        for regime in _REGIME_ORDER:
            vals = [
                r["holdout_acc"][regime] * 100
                for _, r in cell_rows
                if isinstance(r.get("holdout_acc", {}).get(regime), (int, float))
            ]
            acc_strs.append(_fmt_mean_std(vals, width=5, decimals=1))

        cell_name = f"L{L}-E{E:03d}"
        print(
            f"{cell_name:<13} {n_params:>10,} {len(cell_rows):>7} {nl_str:>14}  "
            + "  ".join(f"{s:>11}" for s in acc_strs)
        )


def borderline_summary(runs_dir: str, lo: float = 0.95, hi: float = 0.995) -> None:
    """Print which (L, E) cells qualify as borderline (per the round-2 rule)."""
    cells = round2_cells(runs_dir, lo=lo, hi=hi)
    if not cells:
        print(f"No borderline (L, E) cells with default acc in [{lo:.2%}, {hi:.2%}]")
        return
    seen: set[tuple[int, int]] = set()
    for c in cells:
        key = (c.n_layer, c.n_embd)
        if key in seen:
            continue
        seen.add(key)
        n_params = compute_n_params(c.n_layer, c.n_embd, 4 * c.n_embd)
        print(f"  L={c.n_layer} E={c.n_embd:3d} (params={n_params:,})")
    print(f"\n{len(seen)} borderline (L, E) cells → "
          f"{len(seen) * 4} seed-expansion cells (seeds 1..4)")


# ---- CLI ----

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--round", type=int, choices=[1, 2, 3], default=None,
                   help="run a single planned round")
    p.add_argument("--rounds", type=str, default=None,
                   help="run multiple rounds in sequence, e.g. --rounds 1,2 "
                        "(Round 2's borderline detection sees Round 1's results)")
    p.add_argument("--max-iters", type=int, default=20_000)
    p.add_argument("--no-skip", action="store_true",
                   help="do not skip already-completed cells (re-run them)")
    p.add_argument("--plan", type=int, choices=[1, 2, 3], default=None,
                   help="dry-run: print the cell list for a planned round and exit")
    p.add_argument("--summary", action="store_true",
                   help="print summary grouped by (L, E) and exit")
    p.add_argument("--borderline", action="store_true",
                   help="print which (L, E) cells qualify as borderline and exit")
    args = p.parse_args()

    if args.summary:
        summarize(args.runs_dir)
        return
    if args.borderline:
        borderline_summary(args.runs_dir)
        return
    if args.plan is not None:
        cells = _round_cells(args.plan, args.runs_dir)
        print(f"Round {args.plan}: {len(cells)} cells")
        for c in cells:
            n_params = compute_n_params(c.n_layer, c.n_embd, 4 * c.n_embd)
            print(f"  {c.name}  (params={n_params:,})")
        return
    if args.round is None and args.rounds is None:
        p.print_help()
        sys.exit(1)

    rounds: list[int]
    if args.rounds is not None:
        rounds = [int(r) for r in args.rounds.split(",")]
        for r in rounds:
            if r not in (1, 2, 3):
                raise ValueError(f"--rounds: invalid round {r}")
    else:
        rounds = [args.round]

    for rnum in rounds:
        print(f"\n========== ROUND {rnum} ==========\n", flush=True)
        cells = _round_cells(rnum, args.runs_dir)
        if not cells:
            print(f"Round {rnum}: nothing to do (cell list empty)")
            continue
        run_cells(
            cells=cells,
            runs_dir=args.runs_dir,
            max_iters=args.max_iters,
            skip_existing=not args.no_skip,
        )
        print(f"\n=== Summary after Round {rnum} ===\n")
        summarize(args.runs_dir)


def _round_cells(round_num: int, runs_dir: str) -> list[Cell]:
    if round_num == 1:
        return round1_cells()
    if round_num == 2:
        return round2_cells(runs_dir)
    if round_num == 3:
        return round3_cells()
    raise ValueError(f"unknown round {round_num}")


if __name__ == "__main__":
    main()
