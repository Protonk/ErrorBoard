"""Targeted probe: when predicting result mantissa bits (c:m2, c:m1, c:m0),
where does the model's attention go?

Motivation: the pentagon Thm-5 probe found that operand-a/b residual streams at
the m0 position diverge sharply for capable models (cos sim 0.99 for all bits
except m0, which drops to -0.03 in V5). The first writeup offered a mechanistic
reading ("result m0 ← operand m1 because rounding"). On further inspection that
explanation doesn't hold for normal-normal doubling, where result mantissa =
operand mantissa exactly. So the reading was wrong; this probe replaces
speculation with measurement.

This probe asks the data: for each of c:m2, c:m1, c:m0 predictor positions, what
fraction of attention lands on each operand bit position? Run on:
  - cancellation regime (matched to the prior pentagon pass)
  - doubling holdout (the regime that drove the Thm-5 finding)

Cross-vertex output prints to stdout and saves to notes/m0_probe_findings.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .hooked_bridge import load_hooked
from .inspection import (
    attention_at_result_positions,
    mantissa_attention_breakdown,
    _doubling_holdout,
    _sample_regime,
)
from .pentagon import VERTICES, _repo_root


def run_one(vpath: Path, regime: str, n_samples: int) -> dict:
    hooked, gpt = load_hooked(vpath, device="cpu")
    if regime == "doubling":
        inputs, _ = _doubling_holdout(n_samples)
    else:
        inputs, _ = _sample_regime(regime, n_samples)
    att = attention_at_result_positions(hooked, inputs)
    return mantissa_attention_breakdown(att)


def cross_vertex_table(results: dict, bucket_idx: int, predictor_idx: int,
                       label: str) -> str:
    """Build a markdown table showing one (bucket, predictor) cell across vertices."""
    bucket_names = ["a:m2", "a:m1", "a:m0", "b:m2", "b:m1", "b:m0", "ctx", "aux"]
    pred_names = ["c:m2", "c:m1", "c:m0"]
    rows = [f"| vertex |  V1  |  V2  |  V3  |  V4  |  V5  |",
            "|--------|------|------|------|------|------|"]
    row_cells = [f"{label}"]
    for v in ["V1", "V2", "V3", "V4", "V5"]:
        val = results[v]["per_predictor"][predictor_idx, bucket_idx]
        row_cells.append(f"{val:.3f}")
    rows.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(rows)


def format_predictor_block(results: dict, predictor_idx: int) -> str:
    """For a single predictor (c:m2 / c:m1 / c:m0), show attention to every
    bucket across all 5 vertices in one table."""
    bucket_names = ["a:m2", "a:m1", "a:m0", "b:m2", "b:m1", "b:m0", "ctx_other", "aux"]
    pred_names = ["c:m2", "c:m1", "c:m0"]
    lines = [f"\n#### Predicting {pred_names[predictor_idx]}\n"]
    lines.append("| target  |   V1  |   V2  |   V3  |   V4  |   V5  |")
    lines.append("|---------|------:|------:|------:|------:|------:|")
    for bi, bn in enumerate(bucket_names):
        cells = " | ".join(
            f"{results[v]['per_predictor'][predictor_idx, bi]:.3f}"
            for v in ["V1", "V2", "V3", "V4", "V5"]
        )
        lines.append(f"| {bn:<7} | {cells} |")
    return "\n".join(lines)


def format_operand_distribution(results: dict, predictor_idx: int) -> str:
    """For a single predictor, normalized distribution over 6 operand mantissa
    positions (a:m2, a:m1, a:m0, b:m2, b:m1, b:m0). Sums to 1 within each vertex."""
    bucket_names = ["a:m2", "a:m1", "a:m0", "b:m2", "b:m1", "b:m0"]
    pred_names = ["c:m2", "c:m1", "c:m0"]
    lines = [f"\n#### Predicting {pred_names[predictor_idx]} — normalized over operand mantissa positions\n"]
    lines.append("| target  |   V1  |   V2  |   V3  |   V4  |   V5  |")
    lines.append("|---------|------:|------:|------:|------:|------:|")
    for bi, bn in enumerate(bucket_names):
        cells = []
        for v in ["V1", "V2", "V3", "V4", "V5"]:
            total = results[v]["op_bit_dist"][predictor_idx].sum()
            val = results[v]["op_bit_dist"][predictor_idx, bi] / total if total > 0 else 0
            cells.append(f"{val:.3f}")
        lines.append(f"| {bn:<7} | {' | '.join(cells)} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=64)
    args = p.parse_args()

    repo = _repo_root()

    print("Running m0 probe on all 5 vertices...")
    results_cancel = {}
    results_doubling = {}
    for vname, vpath, vdesc in VERTICES:
        full_path = repo / vpath
        print(f"  {vname} on cancellation...")
        results_cancel[vname] = run_one(full_path, "cancellation", args.n_samples)
        print(f"  {vname} on doubling...")
        results_doubling[vname] = run_one(full_path, "doubling", args.n_samples)

    sections = [
        "# m0-attention targeted probe",
        "",
        "Probe: for each result-mantissa-bit predictor (c:m2, c:m1, c:m0), what",
        "fraction of attention lands on each operand bit position? Averaged",
        "across batch, heads, and layers.",
        "",
        f"Run on {args.n_samples} samples per regime (doubling holdout truncated",
        "to 31 samples — see Thm-5 finding's caveat).",
        "",
        "## Cancellation regime",
    ]
    for pi in range(3):
        sections.append(format_predictor_block(results_cancel, pi))
        sections.append(format_operand_distribution(results_cancel, pi))
    sections.append("\n## Doubling regime")
    for pi in range(3):
        sections.append(format_predictor_block(results_doubling, pi))
        sections.append(format_operand_distribution(results_doubling, pi))

    output = "\n".join(sections)
    print("\n" + output)

    out_path = repo / "notes/m0_probe_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
