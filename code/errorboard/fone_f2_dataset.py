"""Dataset + samplers for the F2 (binary FoNE) arm.

Mirror of `fone_dataset.py` but using F2 encode/target functions
(`fone_f2_tokenizer`).
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .fone_f2_tokenizer import encode_batch, encode_targets, SEQ_LEN
from .regimes import NUM_REGIMES, REGIME_NAMES


class StratifiedSamplerFoneF2:
    def __init__(self, table: np.ndarray, train_indices: np.ndarray, seed: int = 0):
        self.table = table
        self.train_indices = np.asarray(train_indices, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self._pools: dict[int, np.ndarray] = {}
        for r in range(NUM_REGIMES):
            mask = table["regime_id"][self.train_indices] == r
            pool = self.train_indices[mask]
            if len(pool) > 0:
                self._pools[r] = pool
        self.active_regimes: np.ndarray = np.array(
            sorted(self._pools.keys()), dtype=np.int64
        )
        if len(self.active_regimes) == 0:
            raise ValueError("StratifiedSamplerFoneF2: no training samples")

    def sample_batch(self, batch_size: int) -> dict:
        choice = self.rng.integers(0, len(self.active_regimes), size=batch_size)
        regime_ids = self.active_regimes[choice]
        table_idx = np.empty(batch_size, dtype=np.int64)
        for i, r in enumerate(regime_ids):
            pool = self._pools[int(r)]
            table_idx[i] = pool[self.rng.integers(0, len(pool))]
        rows = self.table[table_idx]
        triples = np.stack(
            [rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1
        )
        tokens, num_values = encode_batch(triples)
        sign_target, digit_target, is_nan = encode_targets(triples)
        return {
            "input_ids": tokens[:, :-1],
            "target_tokens": tokens[:, 1:],
            "num_values": num_values[:, :-1],
            "sign_target": sign_target,
            "digit_target": digit_target,
            "is_nan": is_nan,
            "regime_id": rows["regime_id"],
            "tag_mask": rows["tag_mask"],
            "table_idx": table_idx,
        }


class EvalBatcherFoneF2:
    def __init__(self, table: np.ndarray, indices: np.ndarray, batch_size: int):
        self.table = table
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[dict]:
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            rows = self.table[batch_idx]
            triples = np.stack(
                [rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1
            )
            tokens, num_values = encode_batch(triples)
            sign_target, digit_target, is_nan = encode_targets(triples)
            yield {
                "input_ids": tokens[:, :-1],
                "target_tokens": tokens[:, 1:],
                "num_values": num_values[:, :-1],
                "sign_target": sign_target,
                "digit_target": digit_target,
                "is_nan": is_nan,
                "regime_id": rows["regime_id"],
                "tag_mask": rows["tag_mask"],
                "table_idx": batch_idx,
            }

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def per_regime_eval_loaders_fone_f2(
    table: np.ndarray, indices: np.ndarray, batch_size: int = 256,
) -> dict[int, EvalBatcherFoneF2]:
    out: dict[int, EvalBatcherFoneF2] = {}
    indices = np.asarray(indices, dtype=np.int64)
    for r in range(NUM_REGIMES):
        mask = table["regime_id"][indices] == r
        out[r] = EvalBatcherFoneF2(table, indices[mask], batch_size)
    return out


def _spot_checks() -> None:
    from .preprocess import build_table, split_train_holdout

    table = build_table()
    train_idx, holdout_idx = split_train_holdout(table, seed=0)

    sampler = StratifiedSamplerFoneF2(table, train_idx, seed=42)
    batch = sampler.sample_batch(128)
    assert batch["input_ids"].shape == (128, SEQ_LEN - 1)
    assert batch["digit_target"].shape == (128, 18)

    eval_loaders = per_regime_eval_loaders_fone_f2(table, holdout_idx, batch_size=64)
    seen = 0
    for r, loader in eval_loaders.items():
        for b in loader:
            seen += len(b["regime_id"])
    assert seen == len(holdout_idx)
    print("all F2 dataset spot checks passed")


if __name__ == "__main__":
    _spot_checks()
