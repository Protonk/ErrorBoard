"""Three-arm failure-consensus comparison.

Compares L4-E048 with three input representations:
  - learned-PE bit-level (`sweep-L4-E048-s{0..19}`)
  - RoPE        bit-level (`rope-L4-E048-s{0..19}`)
  - learned-PE SEM       (`sem-L4-E048-s{0..19}`)

For each arm: per-pair failure count across seeds, lottery distribution
(p̂ binning), per-regime fail rates, and structural-core/lottery/easy strata.
Cross-arm Jaccards on the lottery zones tell us whether the same pairs are
hard under each representation (format-driven) or different (representation-
driven).

Usage:
    python -m errorboard.failure_consensus_sem
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .failure_consensus import binomial_expected
from .hooked_bridge import load_hooked
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES
from . import tokenizer as _bit_tokenizer
from . import sem_tokenizer as _sem_tokenizer


@torch.no_grad()
def correctness_one(checkpoint_path: Path, tokenizer) -> np.ndarray:
    """Per-pair correctness on holdout, using the given tokenizer module.

    Works for both bit-level (vocab 12, seq 28) and SEM (vocab 32, seq 13)
    checkpoints — the checkpoint's saved gpt_config carries vocab/block info.
    """
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    hooked, gpt = load_hooked(checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seqs = tokenizer.encode_batch(triples.astype(np.uint8))
    inputs = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    targets = torch.from_numpy(seqs[:, 1:].astype(np.int64))
    pos_c_start = tokenizer.POS_C_START
    pos_c_end = tokenizer.POS_C_END
    n = inputs.shape[0]
    bsz = 512
    correct = np.zeros(n, dtype=bool)
    for i in range(0, n, bsz):
        idx = inputs[i:i+bsz].to(next(hooked.parameters()).device)
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)
        tgt = targets[i:i+bsz].to(idx.device)
        pred_slice = preds[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        tgt_slice = tgt[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
    return correct


def _load_arm(prefix: str, n_seeds: int, tokenizer, repo: Path,
              ckpt: str = "checkpoint_020000.pt") -> dict[int, np.ndarray]:
    correctness = {}
    for s in range(n_seeds):
        rel = f"runs/{prefix}-s{s}/{ckpt}"
        full = repo / rel
        if not full.exists():
            print(f"  {prefix} s{s}: SKIP (not found at {full})", flush=True)
            continue
        print(f"  {prefix} s{s}: {rel}", flush=True)
        correctness[s] = correctness_one(full, tokenizer)
    return correctness


def _arm_stats(correctness: dict[int, np.ndarray]) -> dict:
    seeds_loaded = sorted(correctness.keys())
    fail_matrix = np.stack([~correctness[s] for s in seeds_loaded], axis=0)
    fail_count = fail_matrix.sum(axis=0)
    n_seeds = len(seeds_loaded)
    n_pairs = fail_matrix.shape[1]
    return dict(
        seeds=seeds_loaded,
        fail_matrix=fail_matrix,
        fail_count=fail_count,
        n_seeds=n_seeds,
        n_pairs=n_pairs,
        p_mean=float(fail_matrix.mean()),
        hist=[int((fail_count == k).sum()) for k in range(n_seeds + 1)],
        var_obs=float(fail_count.var()),
    )


def _arm_section(name: str, st: dict) -> list[str]:
    lines = [f"## {name} — single-arm stats", ""]
    p = st["p_mean"]
    iid_var = st["n_seeds"] * p * (1 - p)
    var_ratio = st["var_obs"] / iid_var if iid_var > 0 else float("nan")
    core = (st["fail_count"] == st["n_seeds"]).mean() * 100
    lot = ((st["fail_count"] >= 1) & (st["fail_count"] <= st["n_seeds"] - 1)).mean() * 100
    easy = (st["fail_count"] == 0).mean() * 100
    lines.append(f"n_seeds={st['n_seeds']}  n_holdout={st['n_pairs']}  "
                 f"mean_fail={p*100:.2f}%  var_ratio={var_ratio:.2f}  "
                 f"core={core:.1f}%  lottery={lot:.1f}%  easy={easy:.1f}%")
    lines.append("")
    return lines


def _comparison_table(arms: list[tuple[str, dict]]) -> list[str]:
    lines = ["## Cross-arm headline table", ""]
    lines.append("| arm | mean fail % | var ratio | core % | lottery % | easy % | "
                 "p̂ ≥ 0.8 pairs | p̂ = 1.0 pairs |")
    lines.append("|-----|------:|------:|------:|------:|------:|------:|------:|")
    for name, st in arms:
        p = st["p_mean"]
        iid_var = st["n_seeds"] * p * (1 - p)
        var_ratio = st["var_obs"] / iid_var if iid_var > 0 else float("nan")
        n_seeds = st["n_seeds"]
        core = int((st["fail_count"] == n_seeds).sum())
        lot = int(((st["fail_count"] >= 1) & (st["fail_count"] <= n_seeds - 1)).sum())
        easy = int((st["fail_count"] == 0).sum())
        n_high = int((st["fail_count"] >= 0.8 * n_seeds).sum())
        n_full = int((st["fail_count"] == n_seeds).sum())
        lines.append(f"| {name} | {p*100:.2f}% | {var_ratio:.2f} | "
                     f"{core/st['n_pairs']*100:.2f}% | {lot/st['n_pairs']*100:.1f}% | "
                     f"{easy/st['n_pairs']*100:.1f}% | {n_high} | {n_full} |")
    lines.append("")
    return lines


def _phat_distribution(arms: list[tuple[str, dict]]) -> list[str]:
    lines = ["## Per-pair p̂ distribution (20 seeds)", ""]
    bins = [(0, 0), (1, 1), (2, 5), (6, 10), (11, 15), (16, 19), (20, 20)]
    header = "| bin | " + " | ".join(name for name, _ in arms) + " |"
    sep = "|-----" + "|------" * len(arms) + "|"
    lines.append(header)
    lines.append(sep)
    for lo, hi in bins:
        label = f"{lo}/20" if lo == hi else f"{lo}-{hi}/20"
        row = f"| {label} |"
        for _, st in arms:
            cnt = int(((st["fail_count"] >= lo) & (st["fail_count"] <= hi)).sum())
            row += f" {cnt} |"
        lines.append(row)
    lines.append("")
    return lines


def _regime_table(arms: list[tuple[str, dict]], regime_ids: np.ndarray) -> list[str]:
    lines = ["## Per-regime mean fail rate", ""]
    header = "| regime | n |"
    for name, _ in arms:
        header += f" {name} |"
    lines.append(header)
    sep = "|-----|------|" + "------|" * len(arms)
    lines.append(sep)
    for ri, rname in enumerate(REGIME_NAMES):
        m = (regime_ids == ri)
        n_tot = int(m.sum())
        if n_tot == 0:
            continue
        row = f"| {rname} | {n_tot} |"
        for _, st in arms:
            rate = float(st["fail_matrix"][:, m].mean()) * 100
            row += f" {rate:.2f}% |"
        lines.append(row)
    lines.append("")
    return lines


def _jaccard_table(arms: list[tuple[str, dict]]) -> list[str]:
    lines = ["## Pairwise lottery-zone Jaccard", "",
             "Lottery zone = pairs failed by 1..n_seeds-1 of the seeds."]
    lines.append("")
    masks = {}
    for name, st in arms:
        n_seeds = st["n_seeds"]
        masks[name] = (st["fail_count"] >= 1) & (st["fail_count"] <= n_seeds - 1)
    names = [name for name, _ in arms]
    header = "| | " + " | ".join(names) + " |"
    sep = "|---|" + "---|" * len(names)
    lines.append(header)
    lines.append(sep)
    for a in names:
        row = f"| {a} |"
        for b in names:
            if a == b:
                row += " 1.000 |"
                continue
            inter = int((masks[a] & masks[b]).sum())
            union = int((masks[a] | masks[b]).sum())
            jacc = inter / union if union > 0 else float("nan")
            row += f" {jacc:.3f} |"
        lines.append(row)
    lines.append("")
    return lines


def _difficult_pair_overlap(arms: list[tuple[str, dict]]) -> list[str]:
    """Where the heavy-tail pairs (p̂ >= 0.8) of each arm live in the others."""
    lines = ["## Heavy-tail (p̂ ≥ 0.8) overlap across arms", ""]
    masks = {}
    for name, st in arms:
        n_seeds = st["n_seeds"]
        masks[name] = st["fail_count"] >= 0.8 * n_seeds
    names = [name for name, _ in arms]
    for a in names:
        n_a = int(masks[a].sum())
        if n_a == 0:
            lines.append(f"- **{a}** heavy tail empty.")
            continue
        lines.append(f"- **{a}** heavy tail = {n_a} pairs; of those:")
        for b in names:
            if b == a:
                continue
            inter = int((masks[a] & masks[b]).sum())
            lines.append(f"  - {inter} ({inter/n_a*100:.1f}%) are also heavy-tail in **{b}**")
    lines.append("")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--out", default="notes/sem_arm_comparison.md")
    args = p.parse_args()
    repo = _repo_root()

    print("Loading learned-PE bit arm...")
    corr_learned = _load_arm("sweep-L4-E048", args.n_seeds, _bit_tokenizer, repo)
    print("Loading RoPE bit arm...")
    corr_rope = _load_arm("rope-L4-E048", args.n_seeds, _bit_tokenizer, repo)
    print("Loading SEM arm...")
    corr_sem = _load_arm("sem-L4-E048", args.n_seeds, _sem_tokenizer, repo)

    if not corr_learned or not corr_sem:
        print("Missing seeds in one or more arms. Exiting.")
        return

    arms: list[tuple[str, dict]] = [
        ("learned-PE bit", _arm_stats(corr_learned)),
        ("RoPE bit", _arm_stats(corr_rope)),
        ("learned-PE SEM", _arm_stats(corr_sem)),
    ]

    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    regime_ids = rows["regime_id"]

    sections = [
        "# Three-arm comparison: bit vs RoPE-bit vs SEM",
        "",
        f"All L4-E048, iter 20k, {args.n_seeds} seeds each. Same holdout split (seed=0, n={len(holdout_idx)}).",
        "",
    ]
    for name, st in arms:
        sections += _arm_section(name, st)
    sections += _comparison_table(arms)
    sections += _phat_distribution(arms)
    sections += _regime_table(arms, regime_ids)
    sections += _jaccard_table(arms)
    sections += _difficult_pair_overlap(arms)

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
