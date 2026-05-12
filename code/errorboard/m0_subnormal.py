"""Subnormal-result slice of the m0 attention probe.

Tests H2 (subnormal renormalization) as a contributor to the m0 cos-sim
anomaly from the pentagon writeup.

H2 says: when a + b produces a subnormal result, the renormalization step
(leading-zero detection + leftward mantissa shift) routes information from
operand mantissa bits other than m0 into the result's m0 slot. Hence c:m0
attention should be more m1-shifted for subnormal-result cases.

Strata:
  - normal-result   : finite operands, finite result, result is normal
  - subnormal-result: finite operands, finite result, result is subnormal
                      (this is the SUBNORMAL_RESULT regime — id 3)

We compare the m1/m0 attention ratio across strata. Predictions:
  H2 dominant      : subnormal-result ratio >> normal-result ratio at V4
  H2 contributory  : subnormal-result ratio modestly higher (~+0.10-0.20) at V4
  H2 irrelevant    : ratios match within noise (~0.05)

Per the writeup prediction, expectation is "H2 contributory."
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
from .regimes import SUBNORMAL_RESULT


def _sample_strata(n_per_stratum: int = 128, seed: int = 0) -> dict[str, tuple]:
    """Sample two strata from holdout:
      - subnormal-result: regime id == SUBNORMAL_RESULT
      - normal-result:    result is a finite, non-zero, non-NaN, non-subnormal
                          value (i.e., a true normal); operands both finite

    Returns {label: (inputs, targets, n)}.
    """
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=seed)
    rows = table[holdout_idx]
    rng = np.random.default_rng(seed)

    # Subnormal-result stratum.
    sub_mask = rows["regime_id"] == SUBNORMAL_RESULT
    sub_rows = rows[sub_mask]

    # Normal-result stratum. Exclude:
    #   - regime 0 (special-values, NaN involvement)
    #   - regime 1 (overflow — saturated result)
    #   - regime 2 (underflow-to-zero — structurally empty in OCP E4M3)
    #   - regime 3 (subnormal-result)
    # Additionally require: result bits represent a normal value (biased exp >= 1
    # and result is not zero).
    norm_mask = rows["regime_id"] > SUBNORMAL_RESULT  # 4..7: canc, tie, ldex, default
    norm_rows = rows[norm_mask]
    rb = norm_rows["result_bits"].astype(int)
    result_exp = (rb >> 3) & 0xF
    result_is_normal = (result_exp >= 1) & ~((rb == 0x7F) | (rb == 0xFF))
    norm_rows = norm_rows[result_is_normal]

    def pick(rows_in, label):
        if len(rows_in) == 0:
            return None
        n = min(n_per_stratum, len(rows_in))
        idx = rng.choice(len(rows_in), size=n, replace=False)
        sel = rows_in[idx]
        inputs, targets = _make_input_target(sel)
        return (inputs, targets, n)

    return {
        "subnormal-result": pick(sub_rows, "subnormal-result"),
        "normal-result": pick(norm_rows, "normal-result"),
    }


def run_one(vpath: Path, strata: dict) -> dict[str, dict]:
    hooked, gpt = load_hooked(vpath, device="cpu")
    out = {}
    for label, payload in strata.items():
        if payload is None:
            out[label] = None
            continue
        inputs, _, n = payload
        att = attention_at_result_positions(hooked, inputs)
        out[label] = mantissa_attention_breakdown(att)
    return out


def format_stratum_table(results: dict, stratum: str) -> str:
    lines = [f"\n### Stratum: {stratum}\n"]
    bucket_names = ["a:m2", "a:m1", "a:m0", "b:m2", "b:m1", "b:m0", "ctx_other", "aux"]
    pred_names = ["c:m2", "c:m1", "c:m0"]
    for pi, pn in enumerate(pred_names):
        lines.append(f"\n#### Predicting {pn}")
        lines.append("")
        lines.append("| target  |   V1  |   V2  |   V3  |   V4  |   V5  |")
        lines.append("|---------|------:|------:|------:|------:|------:|")
        for bi, bn in enumerate(bucket_names):
            cells = []
            for v in ["V1", "V2", "V3", "V4", "V5"]:
                bd = results[v].get(stratum)
                if bd is None:
                    cells.append("  —  ")
                else:
                    cells.append(f"{bd['per_predictor'][pi, bi]:.3f}")
            lines.append(f"| {bn:<7} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_ratio_comparison(results: dict) -> str:
    lines = [
        "\n### c:m0 attention ratio (a:m1 + b:m1) / (a:m0 + b:m0) — the H2 test\n",
        "Higher ratio at subnormal-result than at normal-result ⟹ H2 active.\n",
        "| stratum | V1 | V2 | V3 | V4 | V5 |",
        "|---------|---:|---:|---:|---:|---:|",
    ]
    for stratum in ["normal-result", "subnormal-result"]:
        cells = []
        for v in ["V1", "V2", "V3", "V4", "V5"]:
            bd = results[v].get(stratum)
            if bd is None:
                cells.append(" — ")
                continue
            m1 = bd['per_predictor'][2, 1] + bd['per_predictor'][2, 4]
            m0 = bd['per_predictor'][2, 2] + bd['per_predictor'][2, 5]
            ratio = m1 / m0 if m0 > 0 else float('nan')
            cells.append(f"{ratio:.2f}")
        lines.append(f"| {stratum} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("Difference (subnormal − normal):")
    lines.append("")
    lines.append("| stratum | V1 | V2 | V3 | V4 | V5 |")
    lines.append("|---------|---:|---:|---:|---:|---:|")
    cells = []
    for v in ["V1", "V2", "V3", "V4", "V5"]:
        bd_n = results[v].get("normal-result")
        bd_s = results[v].get("subnormal-result")
        if bd_n is None or bd_s is None:
            cells.append("  —  ")
            continue
        m1_n = bd_n['per_predictor'][2, 1] + bd_n['per_predictor'][2, 4]
        m0_n = bd_n['per_predictor'][2, 2] + bd_n['per_predictor'][2, 5]
        m1_s = bd_s['per_predictor'][2, 1] + bd_s['per_predictor'][2, 4]
        m0_s = bd_s['per_predictor'][2, 2] + bd_s['per_predictor'][2, 5]
        r_n = m1_n / m0_n if m0_n > 0 else float('nan')
        r_s = m1_s / m0_s if m0_s > 0 else float('nan')
        cells.append(f"{r_s - r_n:+.2f}")
    lines.append("| Δ ratio | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-stratum", type=int, default=128)
    args = p.parse_args()

    repo = _repo_root()
    print(f"Sampling holdout strata (n_per_stratum={args.n_per_stratum})...")
    strata = _sample_strata(args.n_per_stratum)
    for label, payload in strata.items():
        n = payload[2] if payload else 0
        print(f"  {label}: {n} samples")

    print("\nRunning m0-attention probe per (vertex, stratum)...")
    results = {}
    for vname, vpath, vdesc in VERTICES:
        full_path = repo / vpath
        print(f"  {vname}: {vpath}")
        results[vname] = run_one(full_path, strata)

    sections = [
        "# m0 attention by result-is-subnormal",
        "",
        "Tests H2 (subnormal renormalization) as a contributor to the m0 anomaly.",
        "",
        "## Sample counts",
        "",
        "| stratum | n_samples |",
        "|---------|----------:|",
    ]
    for label, payload in strata.items():
        n = payload[2] if payload else 0
        sections.append(f"| {label} | {n} |")
    sections.append("")
    sections.append("## H2 test — c:m0 m1/m0 attention ratio across strata")
    sections.append(format_ratio_comparison(results))
    sections.append("")
    sections.append("## Full attention breakdowns per stratum")
    sections.append(format_stratum_table(results, "normal-result"))
    sections.append(format_stratum_table(results, "subnormal-result"))

    output = "\n".join(sections)
    print("\n" + output)

    out_path = repo / "notes/m0_subnormal_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
