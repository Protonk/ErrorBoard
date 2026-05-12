"""Probe 2: where do V4's residual errors live?

V4 (L4-E128, iter 20000) is the saturated reference: 99.7% default accuracy.
After saturation, the few errors remaining could be:
  - Fit-driven: clustered in low-density strata (the model didn't see enough)
  - Structure-driven: clustered in algorithmically-hard cases regardless of
                      training mass

Stratify V4's residual errors by orthogonal axes and compare to training mass:
  - Regime (existing classification — gives per-pair training density)
  - Δexp = |unbiased_exp(a) − unbiased_exp(b)|
  - Magnitude band (max of operand binades)
  - Sign relationship (same / opposite)
  - Subnormal involvement
  - Rounding-tie tag

For each stratum, compute (a) error rate and (b) training mass (estimated from
density × stratum size). If errors cluster in low-mass strata: fit-driven.
If errors cluster in high-complexity strata at any mass level: structure-driven.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .hooked_bridge import load_hooked
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES, TAG_TIE, is_tie, unbiased_exp
from .tokenizer import (
    POS_C_START, POS_C_END, SEQ_LEN, encode_batch,
)


CHECKPOINTS = {
    "V4": "runs/sweep-L4-E128-s0/checkpoint_020000.pt",
    "V5": "runs/sweep-L1-E128-s0/checkpoint_020000.pt",
    "V3": "runs/sweep-L4-E044-s0/checkpoint_020000.pt",
}


@torch.no_grad()
def run_model_on_holdout(repo: Path, checkpoint_path: str) -> dict:
    """Forward a model on full holdout, return per-pair correctness info."""
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]

    hooked, gpt = load_hooked(repo / checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seqs = encode_batch(triples.astype(np.uint8))           # (N, SEQ_LEN=28)
    inputs = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    targets = torch.from_numpy(seqs[:, 1:].astype(np.int64))

    # Forward in batches to avoid memory blowup.
    n = inputs.shape[0]
    batch = 512
    correct_per_bit = np.zeros((n, POS_C_END - POS_C_START), dtype=bool)
    for i in range(0, n, batch):
        idx = inputs[i:i+batch].to(next(hooked.parameters()).device)
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)                       # (B, 27)
        tgt = targets[i:i+batch].to(idx.device)
        # Result-predictor positions in the 27-input: 18..25 predict 19..26.
        pred_slice = preds[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()
        tgt_slice = tgt[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()
        correct_per_bit[i:i+batch] = (pred_slice == tgt_slice)

    all_correct = correct_per_bit.all(axis=1)
    return {
        "rows": rows,
        "correct_per_bit": correct_per_bit,
        "all_correct": all_correct,
        "n_correct": int(all_correct.sum()),
        "n_total": n,
    }


def per_stratum_table(rows, errors, axis_labels: list[str], axis_values, name: str,
                      train_density: dict[str, float] | None = None) -> str:
    """Build a markdown table: for each stratum, report (n, n_errors, err_rate,
    train_density)."""
    lines = [f"\n### Stratified by {name}\n"]
    lines.append("| stratum | n holdout | n errors | error rate | train density / pair |")
    lines.append("|---------|----------:|---------:|-----------:|---------------------:|")
    for label, idx in zip(axis_labels, axis_values):
        n_h = int(idx.sum())
        if n_h == 0:
            lines.append(f"| {label:<10} |        0 |       0 |     —       | — |")
            continue
        n_e = int(errors[idx].sum())  # errors is np.array of booleans (True = error)
        rate = n_e / n_h
        dens_str = "—"
        if train_density is not None and label in train_density:
            dens_str = f"{train_density[label]:,.0f}"
        lines.append(f"| {label:<10} | {n_h:>9} | {n_e:>8} | {rate*100:>9.3f}% | {dens_str:>18} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vertex", default="V4", choices=list(CHECKPOINTS.keys()),
                   help="Which pentagon vertex to analyze")
    args = p.parse_args()

    repo = _repo_root()
    checkpoint_path = CHECKPOINTS[args.vertex]
    vertex_label = args.vertex

    # Compute training densities per regime (same calc as Probe 1).
    table = build_table()
    train_idx, _ = split_train_holdout(table, seed=0)
    train_sizes = {r: int((table["regime_id"][train_idx] == i).sum())
                   for i, r in enumerate(REGIME_NAMES)}
    n_active = sum(1 for r in REGIME_NAMES if train_sizes[r] > 0)
    mass_per_regime = 128 * 20000 / n_active
    train_density = {r: mass_per_regime / train_sizes[r] if train_sizes[r] > 0 else 0
                     for r in REGIME_NAMES}

    print(f"Running {vertex_label} ({checkpoint_path}) on full holdout...")
    out = run_model_on_holdout(repo, checkpoint_path)
    rows = out["rows"]
    errors = ~out["all_correct"]    # True = sample got at least one bit wrong
    n = out["n_total"]
    n_err = int(errors.sum())
    print(f"  holdout n = {n}; V4 errors (any bit wrong) = {n_err}; "
          f"overall accuracy = {(n - n_err)/n*100:.2f}%")

    # Stratification axes.
    a_bits = rows["a_bits"].astype(int)
    b_bits = rows["b_bits"].astype(int)
    result_bits = rows["result_bits"].astype(int)
    regime_ids = rows["regime_id"]
    tag_masks = rows["tag_mask"]

    # Compute Δexp (use unbiased_exp which collapses subnormals to -6).
    a_uexp = np.array([unbiased_exp(int(b)) for b in a_bits])
    b_uexp = np.array([unbiased_exp(int(b)) for b in b_bits])
    dexp = np.abs(a_uexp - b_uexp)

    # Magnitude band: max of operand binades, but treat NaN-bearing pairs separately.
    max_uexp = np.maximum(a_uexp, b_uexp)
    # NaN check: 0x7F or 0xFF.
    is_nan = ((a_bits == 0x7F) | (a_bits == 0xFF)
              | (b_bits == 0x7F) | (b_bits == 0xFF))

    # Sign bits (top bit of FP8).
    a_sign = (a_bits >> 7) & 1
    b_sign = (b_bits >> 7) & 1
    same_sign = a_sign == b_sign

    # Subnormal involvement: an operand is subnormal if biased exp == 0 and any mantissa
    # bit is set; biased exp == 0 with zero mantissa is +0/−0.
    def is_subn(bits):
        bits = np.asarray(bits, dtype=int)
        biased = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        return (biased == 0) & (mantissa > 0)
    a_subn = is_subn(a_bits)
    b_subn = is_subn(b_bits)
    r_subn = is_subn(result_bits)
    any_subn = a_subn | b_subn | r_subn

    # Build sections.
    sections = [
        f"# Probe 2 — {vertex_label} residual error stratification",
        "",
        f"{vertex_label} = `{checkpoint_path}`. Holdout n = {n}.",
        f"{vertex_label} errors (≥1 result bit wrong): {n_err} ({n_err/n*100:.2f}%).",
        "",
        "Per-stratum counts and error rates. Compare error rates against train density",
        "to test whether residual errors are fit-driven (cluster in low-density strata)",
        "or structure-driven (cluster regardless of density).",
        "",
    ]

    # By regime.
    regime_idx = [regime_ids == i for i in range(len(REGIME_NAMES))]
    sections.append(per_stratum_table(rows, errors, list(REGIME_NAMES), regime_idx,
                                      "regime", train_density))

    # By Δexp band.
    dexp_buckets = [
        ("0",   dexp == 0),
        ("1",   dexp == 1),
        ("2",   dexp == 2),
        ("3",   dexp == 3),
        ("4-7", (dexp >= 4) & (dexp <= 7)),
        ("8+",  dexp >= 8),
    ]
    sections.append(per_stratum_table(
        rows, errors, [b[0] for b in dexp_buckets], [b[1] for b in dexp_buckets],
        "Δexp (excludes NaN-bearing cases via unbiased_exp; subnormal → −6)"))

    # By magnitude band (excluding NaN cases).
    finite = ~is_nan
    mag_buckets = [
        ("subnormal", finite & (max_uexp == -6)),
        ("very small (e≤-3)", finite & (max_uexp >= -5) & (max_uexp <= -3)),
        ("small (e=-2..0)",   finite & (max_uexp >= -2) & (max_uexp <= 0)),
        ("medium (e=1..3)",   finite & (max_uexp >= 1) & (max_uexp <= 3)),
        ("large (e=4..6)",    finite & (max_uexp >= 4) & (max_uexp <= 6)),
        ("largest (e=7-8)",   finite & (max_uexp >= 7)),
    ]
    sections.append(per_stratum_table(
        rows, errors, [b[0] for b in mag_buckets], [b[1] for b in mag_buckets],
        "max operand magnitude band"))

    # By sign relationship.
    sign_buckets = [
        ("same-sign",     finite & same_sign),
        ("opposite-sign", finite & ~same_sign),
    ]
    sections.append(per_stratum_table(
        rows, errors, [b[0] for b in sign_buckets], [b[1] for b in sign_buckets],
        "sign relationship (finite only)"))

    # By subnormal involvement.
    subn_buckets = [
        ("none (neither subnormal)",            finite & ~a_subn & ~b_subn & ~r_subn),
        ("operand-only (a or b is subnormal)",  finite & (a_subn | b_subn) & ~r_subn),
        ("result-only subnormal",               finite & ~a_subn & ~b_subn & r_subn),
        ("both ops and result subnormal",       finite & a_subn & b_subn & r_subn),
        ("any subnormal involvement",           finite & any_subn),
    ]
    sections.append(per_stratum_table(
        rows, errors, [b[0] for b in subn_buckets], [b[1] for b in subn_buckets],
        "subnormal involvement"))

    # By rounding-tie tag.
    tag_is_tie = (tag_masks & TAG_TIE).astype(bool)
    tie_buckets = [
        ("is a tie",     finite & tag_is_tie),
        ("not a tie",    finite & ~tag_is_tie),
    ]
    sections.append(per_stratum_table(
        rows, errors, [b[0] for b in tie_buckets], [b[1] for b in tie_buckets],
        "rounding-tie tag"))

    # Interpretation guide.
    sections.append("\n## Reading the tables\n")
    sections.append(
        "- **Fit hypothesis**: error rate should correlate inversely with train density.\n"
        "  Rare strata (low density per pair) should have higher error rates.\n"
        "- **Structure hypothesis**: error rate should track algorithmic complexity\n"
        "  irrespective of density. The same-sign vs opposite-sign axis is the cleanest\n"
        "  test, as it cuts across regimes orthogonally to training density.\n"
    )

    output = "\n".join(sections)
    print("\n" + output)

    out_path = repo / f"notes/fit_errors_{vertex_label.lower()}_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
