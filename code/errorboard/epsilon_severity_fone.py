"""FoNE-side severity probe.

Reconstructs each FoNE checkpoint's predicted value from (sign-class, 6-digit)
output, then computes value-level severity (ULP error, log-damage) and the
anti-ε correlation across the 8 result-mantissa bins.

Tests whether anti-ε severity — robust across bit-level / RoPE / SEM —
survives the input-representation flip (FoNE is addition-native, multiplication-
taxed, the dual of FP).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .epsilon_severity import (
    epsilon,
    format_block,
    pearson,
    stratify,
    ulp_at_value,
)
from .fone_model import fone_predict_on_holdout
from .fone_tokenizer import (
    SIGN_NAN_ID,
    SIGN_NEG_ID,
    digits_to_magnitude,
)
from .oracle import decode as fp8_decode
from .pentagon import _repo_root


def _fone_severity_from_predictions(out: dict) -> dict:
    """Compute per-pair (ulp_err, log_damage) from a FoNE prediction bundle.

    Uses the FoNE-decoded real value (digits → magnitude, sign-class → sign)
    and compares to the true FP8 value's decoded real.
    """
    rows = out["rows"]
    sign_pred = out["sign_pred"]
    digit_pred = out["digit_pred"]
    correct = out["correct"]
    is_nan_target = out["is_nan"]

    n = len(rows)
    ulp_err = np.full(n, np.nan)
    log_damage = np.full(n, np.nan)

    for i in range(n):
        true_bits = int(rows[i]["result_bits"])
        true_val, true_kind = fp8_decode(true_bits)
        if correct[i]:
            ulp_err[i] = 0.0
            log_damage[i] = 0.0
            continue
        if true_kind == "nan" or is_nan_target[i]:
            continue  # don't score NaN predictions
        # Reconstruct predicted real from sign-class + digits.
        sign_id = int(sign_pred[i])
        if sign_id == SIGN_NAN_ID:
            log_damage[i] = float("inf")
            continue
        mag = digits_to_magnitude(list(digit_pred[i]))
        pred_val = -mag if sign_id == SIGN_NEG_ID else mag

        diff = abs(pred_val - true_val)
        ulp = ulp_at_value(true_val, true_kind)
        if ulp > 0:
            ulp_err[i] = diff / ulp
        if true_val != 0 and pred_val != 0 and (pred_val > 0) == (true_val > 0):
            log_damage[i] = abs(math.log2(abs(pred_val)) - math.log2(abs(true_val)))
        elif true_val == 0 and pred_val == 0:
            log_damage[i] = 0.0
        else:
            log_damage[i] = float("inf")
    return {"ulp_err": ulp_err, "log_damage": log_damage}


DEFAULT_RUNS = [
    ("learned-PE FoNE s0  (L4-E048)", "runs/fone-L4-E048-s0/checkpoint_020000.pt"),
    ("learned-PE FoNE s8  (L4-E048)", "runs/fone-L4-E048-s8/checkpoint_020000.pt"),
    ("learned-PE FoNE s14 (L4-E048)", "runs/fone-L4-E048-s14/checkpoint_020000.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/fone_severity_findings.md")
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# FoNE severity probe",
        "",
        "Predicted value reconstructed from (sign-class, 6-digit) output;",
        "ULP and log-damage are value-level metrics identical to the bit-level /",
        "SEM versions. Tests anti-ε survival under the FoNE representation.",
        "",
        "**ε(m) reference:**",
        "",
        "| m_c | 0/8 | 1/8 | 2/8 | 3/8 | 4/8 | 5/8 | 6/8 | 7/8 |",
        "|-----|----:|----:|----:|----:|----:|----:|----:|----:|",
        "| ε   | 0.000 | 0.045 | 0.072 | 0.084 | 0.085 | 0.075 | 0.057 | 0.032 |",
    ]
    summary_rows = []
    for vname, vpath in DEFAULT_RUNS:
        print(f"Running {vname}: {vpath}", flush=True)
        out = fone_predict_on_holdout(repo / vpath, device="cpu")
        sev = _fone_severity_from_predictions(out)
        stats = stratify(out["rows"], sev, out["correct"])
        sections.append(format_block(vname, stats))
        eps = [stats[m]["epsilon"] for m in range(8) if stats[m]["n_err"] > 0]
        mean_ulps = [stats[m]["mean_ulp_err"] for m in range(8) if stats[m]["n_err"] > 0]
        mean_logs = [stats[m]["mean_log_damage"] for m in range(8) if stats[m]["n_err"] > 0]
        summary_rows.append((vname, pearson(eps, mean_ulps), pearson(eps, mean_logs)))

    sections.append("\n## Cross-checkpoint summary\n")
    sections.append("| checkpoint | Pearson(ε, mean ULP) | Pearson(ε, mean |log Δ|) |")
    sections.append("|------------|---------------------:|-------------------------:|")
    for vname, r_ulp, r_log in summary_rows:
        sections.append(f"| {vname} | {r_ulp:+.3f} | {r_log:+.3f} |")

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
