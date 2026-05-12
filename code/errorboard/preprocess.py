"""Enumerate, classify, and split the 65,536-pair FP8 add table.

Per task_spec.md §5: enumerate all (a_bits, b_bits) pairs, classify each via
regimes.classify, store as a structured numpy array, and produce stratified
train/holdout indices (10% per regime with a floor of 10 holdout per regime).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .regimes import (
    NUM_REGIMES,
    REGIME_NAMES,
    TAG_NAMES,
    classify,
)

TABLE_DTYPE = np.dtype(
    [
        ("a_bits", np.uint8),
        ("b_bits", np.uint8),
        ("result_bits", np.uint8),
        ("regime_id", np.uint8),
        ("tag_mask", np.uint8),
    ]
)

TABLE_SIZE = 256 * 256  # 65,536


def build_table() -> np.ndarray:
    """Enumerate all 65,536 (a_bits, b_bits) pairs and return the classified table."""
    table = np.zeros(TABLE_SIZE, dtype=TABLE_DTYPE)
    idx = 0
    for a in range(256):
        for b in range(256):
            result, regime, tags = classify(a, b)
            table[idx] = (a, b, result, regime, tags)
            idx += 1
    return table


def regime_counts(table: np.ndarray) -> dict[str, int]:
    return {
        REGIME_NAMES[r]: int((table["regime_id"] == r).sum())
        for r in range(NUM_REGIMES)
    }


def tag_counts(table: np.ndarray) -> dict[str, int]:
    return {
        name: int(((table["tag_mask"] & bit) != 0).sum())
        for bit, name in TAG_NAMES.items()
    }


def split_train_holdout(
    table: np.ndarray,
    holdout_frac: float = 0.10,
    min_holdout: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split: holdout_frac per regime, floor of min_holdout per regime.

    Raises if any regime would be left with zero training samples.
    """
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    holdout_indices: list[int] = []

    for r in range(NUM_REGIMES):
        regime_indices = np.where(table["regime_id"] == r)[0]
        n = len(regime_indices)
        if n == 0:
            # Structurally empty regime (e.g., underflow-to-zero in FP8 E4M3, where all
            # values are integer multiples of 2**-9 and the interval (0, 2**-10) is
            # therefore unreachable). Skip without error.
            continue
        n_holdout = max(int(round(n * holdout_frac)), min_holdout)
        n_holdout = min(n_holdout, n)
        n_train = n - n_holdout
        if n_train < 1:
            raise ValueError(
                f"Regime '{REGIME_NAMES[r]}' has {n} pairs total but holdout would consume "
                f"all of them (n_holdout={n_holdout}). Lower min_holdout for this regime."
            )
        perm = rng.permutation(regime_indices)
        holdout_indices.extend(perm[:n_holdout].tolist())
        train_indices.extend(perm[n_holdout:].tolist())

    return (
        np.array(sorted(train_indices), dtype=np.int64),
        np.array(sorted(holdout_indices), dtype=np.int64),
    )


def save_table(
    path: Path, table: np.ndarray, train_idx: np.ndarray, holdout_idx: np.ndarray
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, table=table, train_idx=train_idx, holdout_idx=holdout_idx
    )


def load_table(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return data["table"], data["train_idx"], data["holdout_idx"]


def _print_summary(
    table: np.ndarray, train_idx: np.ndarray, holdout_idx: np.ndarray
) -> None:
    print("\nRegime distribution (primary, partition):")
    for r in range(NUM_REGIMES):
        n_total = int((table["regime_id"] == r).sum())
        n_train = int((table["regime_id"][train_idx] == r).sum())
        n_holdout = int((table["regime_id"][holdout_idx] == r).sum())
        pct = n_total / TABLE_SIZE * 100
        print(
            f"  {REGIME_NAMES[r]:20s}  total={n_total:5d}  "
            f"train={n_train:5d}  holdout={n_holdout:5d}  ({pct:5.2f}%)"
        )

    print("\nTag distribution (secondary, overlapping):")
    for bit, name in TAG_NAMES.items():
        n = int(((table["tag_mask"] & bit) != 0).sum())
        pct = n / TABLE_SIZE * 100
        print(f"  {name:20s}  {n:5d}  ({pct:5.2f}%)")

    print(f"\nTotals:  table={TABLE_SIZE}  train={len(train_idx)}  holdout={len(holdout_idx)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and (optionally) cache the regime table.")
    parser.add_argument("--save", type=Path, default=None, help="Output .npz path (optional)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.10)
    parser.add_argument("--min-holdout", type=int, default=10)
    args = parser.parse_args()

    print("Building 65,536-pair classified table...")
    table = build_table()
    train_idx, holdout_idx = split_train_holdout(
        table,
        holdout_frac=args.holdout_frac,
        min_holdout=args.min_holdout,
        seed=args.seed,
    )

    _print_summary(table, train_idx, holdout_idx)

    if args.save is not None:
        save_table(args.save, table, train_idx, holdout_idx)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
