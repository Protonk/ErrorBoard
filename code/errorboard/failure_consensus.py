"""Failure-count-per-pair across seeds.

If failures are i.i.d. lottery with mean rate p≈0.11, the per-pair failure count
across 5 seeds should follow Binomial(5, 0.11): mostly 0s and 1s, vanishingly
few 4s or 5s. If there's a structural core (pairs that ~always fail) and the
rest is lottery, we'd see a bimodal-ish histogram with mass at both endpoints.

The "shuffled" pairs the user is asking about are the middle of the histogram —
pairs that some seeds fail and some don't. These are the lottery component. The
"always failed" tail is the structural core.

This script:
  1. Counts per-pair failures across the 5 L4-E044 seeds at iter 20k.
  2. Compares the histogram to the i.i.d. prediction under various p values.
  3. Stratifies the 5/5 ("structural core") and 1-4/5 ("lottery zone") sets by
     m_c, regime, and bit-error type — to see whether the structural core has
     a distinctive shape vs the lottery zone.
  4. Projects what we'd resolve with more seeds.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .hooked_bridge import load_hooked
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .tokenizer import POS_C_START, POS_C_END, encode_batch
from .regimes import REGIME_NAMES


def _seed_checkpoints(n_seeds: int) -> dict[int, str]:
    return {s: f"runs/sweep-L4-E044-s{s}/checkpoint_020000.pt" for s in range(n_seeds)}


@torch.no_grad()
def correctness_one(checkpoint_path: Path) -> np.ndarray:
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    hooked, gpt = load_hooked(checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seqs = encode_batch(triples.astype(np.uint8))
    inputs = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    targets = torch.from_numpy(seqs[:, 1:].astype(np.int64))
    n = inputs.shape[0]
    bsz = 512
    correct = np.zeros(n, dtype=bool)
    for i in range(0, n, bsz):
        idx = inputs[i:i+bsz].to(next(hooked.parameters()).device)
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)
        tgt = targets[i:i+bsz].to(idx.device)
        pred_slice = preds[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()
        tgt_slice = tgt[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
    return correct


def binomial_expected(N: int, p: float, n_pairs: int) -> list[float]:
    """Expected count of pairs at each failure-count level k=0..N under Binomial(N, p)."""
    return [
        n_pairs * math.comb(N, k) * (p**k) * ((1-p)**(N-k))
        for k in range(N + 1)
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of L4-E044 seeds (0..n_seeds-1) to load.")
    args = p.parse_args()
    repo = _repo_root()

    checkpoints = _seed_checkpoints(args.n_seeds)
    print(f"Loading correctness for L4-E044 seeds 0..{args.n_seeds-1} at iter 20k...")
    correctness = {}
    for seed, path in checkpoints.items():
        full = repo / path
        if not full.exists():
            print(f"  seed {seed}: SKIP (not found at {full})")
            continue
        print(f"  seed {seed}: {path}")
        correctness[seed] = correctness_one(full)

    if not correctness:
        print("No seeds found. Exiting.")
        return
    seeds_loaded = sorted(correctness.keys())
    n_pairs = len(correctness[seeds_loaded[0]])
    n_seeds = len(correctness)
    fail_matrix = np.stack([~correctness[s] for s in seeds_loaded], axis=0)  # (n_seeds, n_pairs)
    fail_count = fail_matrix.sum(axis=0)  # per pair: how many of 5 seeds failed it

    # 1. Histogram.
    hist = [int((fail_count == k).sum()) for k in range(n_seeds + 1)]
    p_mean = float(fail_matrix.mean())  # mean failure rate across all (seed, pair)
    iid_expected = binomial_expected(n_seeds, p_mean, n_pairs)

    sections = [
        "# Failure consensus across L4-E044 seeds",
        "",
        f"Computed per-pair failure count across {n_seeds} seeds (L4-E044, iter 20k).",
        f"Holdout n = {n_pairs}. Mean per-(seed,pair) failure rate = {p_mean*100:.2f}%.",
        "",
        "## Failure-count histogram",
        "",
        f"For each k ∈ 0..{n_seeds}, count of holdout pairs failed by exactly k of {n_seeds} seeds:",
        "",
        "| k (seeds failing) | observed | expected (i.i.d. binomial) | ratio obs/exp |",
        "|------------------:|---------:|--------------------------:|--------------:|",
    ]
    for k in range(n_seeds + 1):
        ratio = hist[k] / iid_expected[k] if iid_expected[k] > 0.01 else float("inf")
        ratio_str = f"{ratio:>6.2f}" if math.isfinite(ratio) else "    ∞"
        sections.append(
            f"| {k} | {hist[k]:>8} | {iid_expected[k]:>26.1f} | {ratio_str:>13} |"
        )

    # 2. Variance check: under i.i.d., Var(count) = N*p*(1-p).
    iid_var = n_seeds * p_mean * (1 - p_mean)
    obs_var = float(fail_count.var())
    sections.append("")
    sections.append(
        f"**Variance of per-pair fail count**: observed {obs_var:.3f}, "
        f"i.i.d. predicted {iid_var:.3f} (ratio {obs_var/iid_var:.2f}). "
        f"Higher than predicted ⟹ underlying per-pair probability is non-uniform "
        f"(some pairs much harder than others)."
    )

    # 3. Stratify the "core" (5/5) vs "lottery zone" (1-4/5) vs "easy" (0/5).
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    result_bits = rows["result_bits"].astype(int)
    result_exp = (result_bits >> 3) & 0xF
    result_m = result_bits & 0x7
    is_normal = (result_exp >= 1) & ~((result_bits == 0x7F) | (result_bits == 0xFF))
    regime_ids = rows["regime_id"]

    core_mask = (fail_count == n_seeds)
    easy_mask = (fail_count == 0)
    lottery_mask = (fail_count >= 1) & (fail_count <= n_seeds - 1)

    sections.append("\n## Three strata\n")
    sections.append("| stratum | n_pairs | fraction |")
    sections.append("|---------|--------:|---------:|")
    sections.append(f"| structural core ({n_seeds}/{n_seeds} fail) | {int(core_mask.sum())} | {core_mask.mean()*100:.1f}% |")
    sections.append(f"| lottery zone (1..{n_seeds-1}/{n_seeds} fail) | {int(lottery_mask.sum())} | {lottery_mask.mean()*100:.1f}% |")
    sections.append(f"| structural easy (0/{n_seeds} fail) | {int(easy_mask.sum())} | {easy_mask.mean()*100:.1f}% |")

    # 4. Per-stratum regime breakdown.
    sections.append("\n## Where each stratum lives (regime)\n")
    sections.append("| regime | n_total | n_core | n_lottery | n_easy | core% | lottery% |")
    sections.append("|--------|--------:|-------:|----------:|-------:|------:|---------:|")
    for ri, name in enumerate(REGIME_NAMES):
        reg_mask = (regime_ids == ri)
        n_total = int(reg_mask.sum())
        if n_total == 0:
            continue
        n_core = int((reg_mask & core_mask).sum())
        n_lottery = int((reg_mask & lottery_mask).sum())
        n_easy = int((reg_mask & easy_mask).sum())
        core_pct = n_core / n_total * 100
        lottery_pct = n_lottery / n_total * 100
        sections.append(
            f"| {name:<18} | {n_total:>7} | {n_core:>6} | {n_lottery:>9} | {n_easy:>6} | "
            f"{core_pct:>4.1f}% | {lottery_pct:>7.1f}% |"
        )

    # 5. m_c stratification (normal-result only) — where are the strata in mantissa-space?
    sections.append("\n## m_c distribution within strata (normal-result only)\n")
    sections.append("| m_c | n_total | core | lottery | easy | core% | lottery% |")
    sections.append("|-----|--------:|-----:|--------:|-----:|------:|---------:|")
    for m_val in range(8):
        m_mask = is_normal & (result_m == m_val)
        n_total = int(m_mask.sum())
        if n_total == 0:
            continue
        n_core = int((m_mask & core_mask).sum())
        n_lottery = int((m_mask & lottery_mask).sum())
        n_easy = int((m_mask & easy_mask).sum())
        sections.append(
            f"| {m_val}/8 | {n_total:>7} | {n_core:>4} | {n_lottery:>7} | {n_easy:>4} | "
            f"{n_core/n_total*100:>4.1f}% | {n_lottery/n_total*100:>7.1f}% |"
        )

    # 6. Projection: with more seeds, what would we resolve?
    sections.append("\n## Projection to more seeds\n")
    sections.append("If the per-pair failure probabilities are stable across seeds, with N seeds")
    sections.append("we resolve per-pair probability to ±√(p(1-p)/N) (1σ).")
    sections.append("For the lottery zone (where p ≈ 0.5), this is:")
    sections.append("")
    sections.append("| N seeds | ±1σ per-pair p estimate | distinguishable bins (≈ 3σ apart) |")
    sections.append("|--------:|------------------------:|----------------------------------:|")
    for N in [5, 10, 20, 50, 100]:
        sigma = math.sqrt(0.5 * 0.5 / N)
        n_bins = int(1 / (3 * sigma)) if sigma > 0 else 0
        sections.append(f"| {N} | ±{sigma:.3f} | ~{n_bins} |")
    sections.append("")
    sections.append(
        "5 seeds give ±0.22 per-pair p resolution — we can distinguish ~1 bin (0 vs ~1).\n"
        "20 seeds would resolve ~3 bins (e.g., 'low', 'mid', 'high' failure-probability classes).\n"
        "100 seeds would resolve ~5-6 bins, enough to characterize the lottery distribution\n"
        "as a continuous shape rather than a binary core-vs-lottery split."
    )

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / "notes/failure_consensus_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
