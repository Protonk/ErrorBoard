"""Δexp-conditioned slice of the m0 attention probe.

Hypotheses to distinguish:
  H1 (rounding-bit propagation):  c:m0 → a:m1 attention peaks at Δexp=1
                                  (where operand m1 lands in result m0 slot
                                  via the 1-bit alignment shift).
  H2 (subnormal renormalization): pattern depends on subnormal-result status,
                                  not Δexp directly.
  H3 (LSB-is-hardest, generic):   no Δexp dependence.

This probe restricts to *normal-normal* finite operand pairs (excluding
subnormals, zeros, NaN — those go in H2's regime) and stratifies by Δexp =
|unbiased_exp(a) - unbiased_exp(b)|. For each (vertex, Δexp bucket), runs the
existing mantissa_attention_breakdown and pulls the c:m0 row.

Output cells of interest:
  - c:m0 → a:m0 attention per (Δexp, vertex)
  - c:m0 → a:m1 attention per (Δexp, vertex)
  - ratio a:m1 / a:m0  (rises with capability under H1 specifically at Δexp=1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .hooked_bridge import load_hooked
from .inspection import (
    attention_at_result_positions,
    mantissa_attention_breakdown,
    _make_input_target,
)
from .pentagon import VERTICES, _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import unbiased_exp


DEXP_BUCKETS = [0, 1, 2, 3, "4-7", "8+"]


def _bucket_for_dexp(d: int) -> str:
    if d <= 3:
        return str(d)
    elif d <= 7:
        return "4-7"
    else:
        return "8+"


def _sample_by_dexp(n_per_bucket: int = 64, seed: int = 0) -> dict[str, np.ndarray]:
    """Group holdout rows by Δexp bucket. Returns {bucket_label: (input, target) tuple}.

    Restricts to normal-normal finite pairs (both operands have biased exp != 0
    and aren't NaN). Doesn't filter by sign — the rounding hypothesis applies to
    same-sign addition primarily, but we keep both to maximize sample count and
    let the signal vs noise speak.
    """
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=seed)
    rows = table[holdout_idx]

    # Compute Δexp per row, filter to normal-normal finite.
    a_bits = rows["a_bits"].astype(int)
    b_bits = rows["b_bits"].astype(int)
    a_exp_biased = (a_bits >> 3) & 0xF
    b_exp_biased = (b_bits >> 3) & 0xF
    # Normal: biased exp in [1, 14]. Exp 0 = subnormal/zero. Exp 15 with mantissa=111 = NaN.
    # Exp 15 with other mantissa = a normal (OCP E4M3 — but actually 0x7F and 0xFF are NaN,
    # everything else with exp 15 is normal).
    normal_a = (a_exp_biased >= 1) & ~((a_bits == 0x7F) | (a_bits == 0xFF))
    normal_b = (b_exp_biased >= 1) & ~((b_bits == 0x7F) | (b_bits == 0xFF))
    keep = normal_a & normal_b
    rows = rows[keep]
    a_bits, b_bits = a_bits[keep], b_bits[keep]

    # Vectorize unbiased_exp.
    def to_unbiased(eb):
        return np.where(eb == 0, -6, eb - 7)
    a_uexp = to_unbiased((a_bits >> 3) & 0xF)
    b_uexp = to_unbiased((b_bits >> 3) & 0xF)
    dexp = np.abs(a_uexp - b_uexp)

    rng = np.random.default_rng(seed)
    out: dict[str, np.ndarray] = {}
    for bucket in DEXP_BUCKETS:
        if bucket == "4-7":
            mask = (dexp >= 4) & (dexp <= 7)
        elif bucket == "8+":
            mask = dexp >= 8
        else:
            mask = dexp == int(bucket)
        bucket_rows = rows[mask]
        if len(bucket_rows) == 0:
            out[bucket] = (None, None, 0)
            continue
        # Sample (with replacement if needed, but with this many holdout points
        # we never need to).
        n = min(n_per_bucket, len(bucket_rows))
        idx = rng.choice(len(bucket_rows), size=n, replace=False)
        sel = bucket_rows[idx]
        inputs, targets = _make_input_target(sel)
        out[bucket] = (inputs, targets, n)
    return out


def run_one(vpath: Path, buckets: dict) -> dict[str, dict]:
    """For a single vertex, run the m0-attention breakdown on each Δexp bucket.
    Returns {bucket_label: breakdown_dict}."""
    hooked, gpt = load_hooked(vpath, device="cpu")
    out = {}
    for bucket, payload in buckets.items():
        inputs, _, n = payload
        if inputs is None or n == 0:
            out[bucket] = None
            continue
        att = attention_at_result_positions(hooked, inputs)
        out[bucket] = mantissa_attention_breakdown(att)
    return out


def format_dexp_table(results: dict, bucket_idx_in_predictor: int, label: str,
                      predictor_idx: int = 2) -> str:
    """Build a markdown table: rows = Δexp bucket, cols = vertex, cells = attention
    weight from c:m0 (predictor_idx=2 in [c:m2, c:m1, c:m0]) to the chosen bucket.

    bucket_idx_in_predictor is the index into the 8-bucket breakdown
    [a:m2, a:m1, a:m0, b:m2, b:m1, b:m0, ctx_other, aux].
    """
    bucket_names = ["a:m2", "a:m1", "a:m0", "b:m2", "b:m1", "b:m0", "ctx_other", "aux"]
    lines = [f"\n#### {label}: c:m0 → {bucket_names[bucket_idx_in_predictor]} (attention weight)\n"]
    lines.append("| Δexp | n |  V1  |  V2  |  V3  |  V4  |  V5  |")
    lines.append("|-----:|--:|-----:|-----:|-----:|-----:|-----:|")
    for bucket in DEXP_BUCKETS:
        # Sample count from any vertex (same buckets across vertices).
        any_v = next(iter(results.values()))
        n_str = "—"  # filled below if present
        row_cells = []
        for v in ["V1", "V2", "V3", "V4", "V5"]:
            bd = results[v].get(bucket)
            if bd is None:
                row_cells.append("  —  ")
            else:
                row_cells.append(f"{bd['per_predictor'][predictor_idx, bucket_idx_in_predictor]:.3f}")
        lines.append(f"| {bucket:>4} | — | " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def format_ratio_table(results: dict) -> str:
    """Build a markdown table: c:m0 attention ratio (a:m1 + b:m1) / (a:m0 + b:m0)
    per Δexp bucket per vertex. Ratio > 1 ⟹ model prefers m1 over m0 for
    predicting c:m0 — the rounding-bit-hypothesis signature."""
    lines = ["\n#### c:m0 attention ratio (a:m1 + b:m1) / (a:m0 + b:m0)\n"]
    lines.append("Higher ratio ⟹ model attends more to operand m1 than m0 when predicting c:m0.")
    lines.append("Under the rounding-bit hypothesis (H1), this ratio should peak at Δexp=1 and grow with vertex capability.\n")
    lines.append("| Δexp |  V1  |  V2  |  V3  |  V4  |  V5  |")
    lines.append("|-----:|-----:|-----:|-----:|-----:|-----:|")
    for bucket in DEXP_BUCKETS:
        row_cells = []
        for v in ["V1", "V2", "V3", "V4", "V5"]:
            bd = results[v].get(bucket)
            if bd is None:
                row_cells.append("  —  ")
            else:
                m1 = bd['per_predictor'][2, 1] + bd['per_predictor'][2, 4]  # a:m1 + b:m1
                m0 = bd['per_predictor'][2, 2] + bd['per_predictor'][2, 5]  # a:m0 + b:m0
                ratio = m1 / m0 if m0 > 0 else float('nan')
                row_cells.append(f"{ratio:.2f}")
        lines.append(f"| {bucket:>4} | " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-bucket", type=int, default=64)
    args = p.parse_args()

    repo = _repo_root()
    print(f"Sampling holdout, normal-normal pairs only, by Δexp bucket "
          f"(n_per_bucket={args.n_per_bucket})...")
    buckets = _sample_by_dexp(args.n_per_bucket)
    for b in DEXP_BUCKETS:
        n = buckets[b][2]
        print(f"  Δexp={b}: {n} samples")

    print("\nRunning m0-attention probe per (vertex, Δexp bucket)...")
    results = {}
    for vname, vpath, vdesc in VERTICES:
        full_path = repo / vpath
        print(f"  {vname}: {vpath}")
        results[vname] = run_one(full_path, buckets)

    sections = [
        "# m0 attention by Δexp",
        "",
        "Probe: c:m0 predictor attention to each operand mantissa bit, stratified",
        "by Δexp = |unbiased_exp(a) − unbiased_exp(b)|. Restricted to normal-normal",
        "finite operand pairs (subnormals, zero, NaN excluded). Sample sizes:",
        "",
        "| Δexp bucket | n_samples |",
        "|-------------|----------:|",
    ]
    for b in DEXP_BUCKETS:
        n = buckets[b][2]
        sections.append(f"| {b} | {n} |")
    sections.append("")
    sections.append(format_dexp_table(results, bucket_idx_in_predictor=1, label="a:m1 attention",
                                      predictor_idx=2))
    sections.append(format_dexp_table(results, bucket_idx_in_predictor=2, label="a:m0 attention",
                                      predictor_idx=2))
    sections.append(format_dexp_table(results, bucket_idx_in_predictor=4, label="b:m1 attention",
                                      predictor_idx=2))
    sections.append(format_dexp_table(results, bucket_idx_in_predictor=5, label="b:m0 attention",
                                      predictor_idx=2))
    sections.append(format_ratio_table(results))
    sections.append("")
    sections.append("## Interpretation guide\n")
    sections.append("- If the ratio table peaks at Δexp=1 and grows with vertex capability,")
    sections.append("  that's the rounding-bit (H1) signature.")
    sections.append("- If the ratio is roughly flat across Δexp but varies by vertex,")
    sections.append("  that supports the generic 'LSB is hardest' reading (H3).")
    sections.append("- If the ratio is flat across both Δexp and vertex, the m0 anomaly from")
    sections.append("  the pentagon writeup is not explained by operand-m1 attention shifts,")
    sections.append("  and we'd need a different probe direction.")

    output = "\n".join(sections)
    print("\n" + output)

    out_path = repo / "notes/m0_dexp_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
