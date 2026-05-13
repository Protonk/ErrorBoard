"""Sampling and eval batching over the classified pair table.

Per task_spec.md §5:
    - StratifiedSampler: equal-mass-per-regime, with-replacement training sampler.
    - EvalBatcher: ordered, exhaustive iteration (used for per-regime holdout/training eval).
    - per_regime_eval_loaders: factory dict {regime_id -> EvalBatcher} over an index pool.
    - natural_distribution_weights: weights for re-aggregating per-regime losses
      back to the natural occurrence rate ("deployment failure rate" metric).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .regimes import NUM_REGIMES, REGIME_NAMES
from .tokenizer import SEQ_LEN as _BIT_SEQ_LEN, encode_batch as _bit_encode_batch


class StratifiedSampler:
    """Equal-mass-per-regime, with-replacement sampler over a training index pool.

    Each call to sample_batch draws batch_size samples: regime is drawn uniformly
    over 1/NUM_REGIMES, then a pair is drawn uniformly within that regime's pool.

    `encode_fn` lets a caller swap the tokenization (bit-level by default; SEM or
    other tokenizers can be plugged in). The function must accept an (N, 3)
    uint8/int array of (a_bits, b_bits, c_bits) and return an (N, SEQ_LEN) int64
    tensor.
    """

    def __init__(self, table: np.ndarray, train_indices: np.ndarray, seed: int = 0,
                 encode_fn=None):
        self.table = table
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.encode_fn = encode_fn if encode_fn is not None else _bit_encode_batch

        # Per-regime training pools, only for regimes with data. Empty regimes
        # (e.g., underflow-to-zero in FP8 E4M3 -- structurally unreachable) are
        # skipped; sampling is uniform over the active regimes.
        self._pools: dict[int, np.ndarray] = {}
        for r in range(NUM_REGIMES):
            mask = table["regime_id"][self.train_indices] == r
            pool = self.train_indices[mask]
            if len(pool) > 0:
                self._pools[r] = pool

        self.active_regimes: np.ndarray = np.array(sorted(self._pools.keys()), dtype=np.int64)
        if len(self.active_regimes) == 0:
            raise ValueError("StratifiedSampler: no training samples in any regime")

    def sample_batch(self, batch_size: int) -> dict:
        """Draw a stratified batch (uniform over the *active* regimes).

        Returns dict with:
            input      : (batch_size, SEQ_LEN - 1) int64 -- sequence[:-1]
            target     : (batch_size, SEQ_LEN - 1) int64 -- sequence[1:]
            regime_id  : (batch_size,) uint8
            tag_mask   : (batch_size,) uint8
            table_idx  : (batch_size,) int64 -- index into `table` for each sample
        """
        choice = self.rng.integers(0, len(self.active_regimes), size=batch_size)
        regime_ids = self.active_regimes[choice]
        table_idx = np.empty(batch_size, dtype=np.int64)
        for i, r in enumerate(regime_ids):
            pool = self._pools[int(r)]
            table_idx[i] = pool[self.rng.integers(0, len(pool))]

        rows = self.table[table_idx]
        triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
        seqs = self.encode_fn(triples)
        return {
            "input": seqs[:, :-1],
            "target": seqs[:, 1:],
            "regime_id": rows["regime_id"],
            "tag_mask": rows["tag_mask"],
            "table_idx": table_idx,
        }


class EvalBatcher:
    """Ordered, exhaustive iterator over a fixed index pool. No randomness."""

    def __init__(self, table: np.ndarray, indices: np.ndarray, batch_size: int,
                 encode_fn=None):
        self.table = table
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.encode_fn = encode_fn if encode_fn is not None else _bit_encode_batch

    def __iter__(self) -> Iterator[dict]:
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            rows = self.table[batch_idx]
            triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
            seqs = self.encode_fn(triples)
            yield {
                "input": seqs[:, :-1],
                "target": seqs[:, 1:],
                "regime_id": rows["regime_id"],
                "tag_mask": rows["tag_mask"],
                "table_idx": batch_idx,
            }

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def per_regime_eval_loaders(
    table: np.ndarray, indices: np.ndarray, batch_size: int = 256, encode_fn=None,
) -> dict[int, EvalBatcher]:
    """Build a per-regime dict of EvalBatchers over the given index pool."""
    out: dict[int, EvalBatcher] = {}
    indices = np.asarray(indices, dtype=np.int64)
    for r in range(NUM_REGIMES):
        mask = table["regime_id"][indices] == r
        out[r] = EvalBatcher(table, indices[mask], batch_size, encode_fn=encode_fn)
    return out


def natural_distribution_weights(table: np.ndarray) -> np.ndarray:
    """Per-regime natural occurrence rate over the full 65,536-pair grid.

    Used to re-aggregate per-regime losses back to the deployment-distribution scalar
    (per task_spec.md §5's natural-distribution eval stream).
    """
    n_total = len(table)
    weights = np.zeros(NUM_REGIMES, dtype=np.float64)
    for r in range(NUM_REGIMES):
        weights[r] = (table["regime_id"] == r).sum() / n_total
    return weights


def _spot_checks() -> None:
    # Late import to avoid module-import-time cost in callers.
    from .preprocess import build_table, split_train_holdout
    from .tokenizer import BOS_ID, EQ_ID, PLUS_ID, POS_EQ, POS_PLUS

    table = build_table()
    train_idx, holdout_idx = split_train_holdout(table, seed=0)

    # Sampler structural sanity
    sampler = StratifiedSampler(table, train_idx, seed=42)
    batch = sampler.sample_batch(8000)
    assert batch["input"].shape == (8000, _BIT_SEQ_LEN - 1)
    assert batch["target"].shape == (8000, _BIT_SEQ_LEN - 1)
    assert batch["regime_id"].shape == (8000,)
    assert batch["input"][0, 0] == BOS_ID
    assert batch["input"][0, POS_PLUS] == PLUS_ID
    assert batch["input"][0, POS_EQ] == EQ_ID

    # Empirical regime distribution should be approximately uniform across *active* regimes.
    regime_counts = np.bincount(batch["regime_id"], minlength=NUM_REGIMES)
    n_active = len(sampler.active_regimes)
    expected = 8000 / n_active
    # 3-sigma binomial bound ~ sqrt(8000 * p * (1-p)). For p=1/7, sigma ~ 31; allow 200.
    for r in sampler.active_regimes:
        deviation = abs(int(regime_counts[r]) - expected)
        assert deviation < 200, (
            f"regime {REGIME_NAMES[r]}: count={regime_counts[r]}, expected~{expected}"
        )
    # Inactive regimes should never appear.
    for r in range(NUM_REGIMES):
        if r not in sampler.active_regimes:
            assert regime_counts[r] == 0, f"inactive regime {REGIME_NAMES[r]} got samples"

    # EvalBatcher: full coverage of the index pool exactly once.
    eval_loaders = per_regime_eval_loaders(table, holdout_idx, batch_size=64)
    seen = 0
    for r, loader in eval_loaders.items():
        for b in loader:
            seen += len(b["regime_id"])
            # All rows in this batch must be from regime r.
            assert (b["regime_id"] == r).all()
    assert seen == len(holdout_idx), f"saw {seen}, expected {len(holdout_idx)} holdout pairs"

    # Natural-distribution weights sum to 1.
    weights = natural_distribution_weights(table)
    assert abs(weights.sum() - 1.0) < 1e-12

    print("all dataset spot checks passed")


if __name__ == "__main__":
    _spot_checks()
