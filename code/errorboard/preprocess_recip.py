"""Enumerate, classify, and split the 256-entry FP8 reciprocal table.

Reciprocal is unary — the natural input space is 256, one per FP8 bit
pattern. To keep the training/probe infrastructure roughly aligned with
the add/mul arcs (binary, 65,536-pair tables), each entry is stored as a
"pair" with `b_bits = 0x00`. The model trains on `BOS a + 0 = recip(a) EOS`-
shaped sequences (using whichever tokenizer) and learns to ignore the
constant b operand.

The b=0x00 padding is a known scaffold cost (see `notes/option_b_recip.md`
or the commit history). For comparison probes the holdout pair-set is
26 pairs (vs 6,554 for add/mul) — statistics are intrinsically noisier.

Same dtype as preprocess_mult.py for pipeline-level compatibility; uses
recip_regimes for classification.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .recip_regimes import (
    NUM_REGIMES,
    REGIME_NAMES,
    TAG_NAMES,
    classify,
)

# Same dtype as preprocess_mult; tag_mask is uint16 to fit 9 tag bits.
TABLE_DTYPE = np.dtype(
    [
        ("a_bits", np.uint8),
        ("b_bits", np.uint8),
        ("result_bits", np.uint8),
        ("regime_id", np.uint8),
        ("tag_mask", np.uint16),
    ]
)

TABLE_SIZE = 256


def build_table() -> np.ndarray:
    """Enumerate all 256 inputs and return the classified table.

    Each entry has b_bits=0x00 (constant padding for binary-shape compat);
    the model treats b as a known no-information channel.
    """
    table = np.zeros(TABLE_SIZE, dtype=TABLE_DTYPE)
    for a in range(256):
        result, regime, tags = classify(a)
        table[a] = (a, 0x00, result, regime, tags)
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
    holdout_frac: float = 0.20,
    min_holdout: int = 1,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split: holdout_frac per regime, floor of min_holdout per regime.

    For recip we use 20% holdout (vs add/mul's 10%) to get ~50 holdout pairs
    rather than ~26, and min_holdout=1 (vs 10) because OVERFLOW has only 2
    entries and we need at least 1 of those in training.

    Raises if any regime would be left with zero training samples.
    """
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    holdout_indices: list[int] = []

    for r in range(NUM_REGIMES):
        regime_indices = np.where(table["regime_id"] == r)[0]
        n = len(regime_indices)
        if n == 0:
            continue
        n_holdout = max(int(round(n * holdout_frac)), min_holdout)
        n_holdout = min(n_holdout, n)
        n_train = n - n_holdout
        if n_train < 1:
            # For 2-entry regimes with min_holdout=1 we get 1 train + 1 holdout.
            # If we still get 0 training, force at least 1 to training.
            if n_holdout > 1:
                n_holdout -= 1
                n_train = 1
            else:
                raise ValueError(
                    f"Regime '{REGIME_NAMES[r]}' has only {n} entries; "
                    f"cannot leave >=1 training sample."
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
            f"  {REGIME_NAMES[r]:20s}  total={n_total:4d}  "
            f"train={n_train:4d}  holdout={n_holdout:4d}  ({pct:5.2f}%)"
        )

    print("\nTag distribution (secondary, overlapping):")
    for bit, name in TAG_NAMES.items():
        n = int(((table["tag_mask"] & bit) != 0).sum())
        pct = n / TABLE_SIZE * 100
        print(f"  {name:20s}  {n:4d}  ({pct:5.2f}%)")

    print(f"\nTotals:  table={TABLE_SIZE}  "
          f"train={len(train_idx)}  holdout={len(holdout_idx)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and (optionally) cache the reciprocal regime table."
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--min-holdout", type=int, default=1)
    args = parser.parse_args()

    print("Building 256-entry classified reciprocal table...")
    table = build_table()
    train_idx, holdout_idx = split_train_holdout(
        table, holdout_frac=args.holdout_frac,
        min_holdout=args.min_holdout, seed=args.seed,
    )

    _print_summary(table, train_idx, holdout_idx)

    if args.save is not None:
        save_table(args.save, table, train_idx, holdout_idx)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
