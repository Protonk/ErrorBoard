"""FoNE-side digit-decomposition probe.

For each error, identify which digit position(s) went wrong. The 6 digits
correspond to places 10^{-3} (thousandths) through 10^{2} (hundreds).

Stratify by result mantissa bin m_c (still extracted from the true FP8 bit
pattern). Tests:
  - Does FoNE concentrate errors at any privileged digit position?
  - Is the "most significant wrong digit" structured (catastrophic place-error)
    or scattered (uniform rounding)?
  - Do endpoint m_c bins show a different digit-error profile than the
    smooth interior?

This is the FoNE analog of the bit/SEM bit-decomposition probe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .fone_model import fone_predict_on_holdout
from .fone_tokenizer import (
    DIGIT_PLACE_EXPONENTS,
    N_DIGITS,
    SIGN_NAN_ID,
)
from .pentagon import _repo_root


def _classify_error(sign_pred: int, sign_target: int,
                    digit_pred: np.ndarray, digit_target: np.ndarray,
                    is_nan: bool) -> tuple[str, list[int]]:
    """Return (category, list-of-wrong-digit-positions).

    Categories:
        sign_only        — sign wrong, all digits match (rare for FoNE)
        sign+digits      — both wrong
        digit_only       — sign right, ≥1 digit wrong
        correct          — nothing wrong
    """
    if is_nan:
        sign_ok = (sign_pred == sign_target)
        return ("correct" if sign_ok else "sign_only"), []
    sign_ok = (sign_pred == sign_target)
    wrong = [i for i in range(N_DIGITS) if digit_pred[i] != digit_target[i]]
    if not wrong and sign_ok:
        return "correct", []
    if not wrong and not sign_ok:
        return "sign_only", []
    if wrong and sign_ok:
        return "digit_only", wrong
    return "sign+digits", wrong


def _per_m_c_digit_table(rows, sign_pred, digit_pred, sign_target, digit_target,
                          is_nan, correct):
    """For each m_c bin, count errors by (which-digit-positions-wrong) and
    by category.
    """
    result_bits = rows["result_bits"].astype(int)
    result_exp = (result_bits >> 3) & 0xF
    result_m = result_bits & 0x7
    is_normal = (result_exp >= 1) & ~((result_bits == 0x7F) | (result_bits == 0xFF))

    # Per m_c: tallies of n_err, n_correct, sign_only, digit_only_*, sign+digits,
    # and per-position digit-wrong counts.
    stats = {}
    for m_val in range(8):
        stats[m_val] = {
            "n_total": 0, "n_err": 0,
            "sign_only": 0, "digit_only": 0, "sign+digits": 0,
            "wrong_at": [0] * N_DIGITS,         # which positions had a wrong digit
            "n_wrong_distribution": [0] * (N_DIGITS + 1),  # how many digits wrong per error
            "most_sig_wrong": [0] * N_DIGITS,   # most-significant wrong digit position
        }
    for i in range(len(rows)):
        if not is_normal[i]:
            continue
        m_val = int(result_m[i])
        s = stats[m_val]
        s["n_total"] += 1
        if correct[i]:
            continue
        s["n_err"] += 1
        cat, wrong = _classify_error(
            int(sign_pred[i]), int(sign_target[i]),
            digit_pred[i], digit_target[i], bool(is_nan[i]),
        )
        if cat in ("sign_only", "digit_only", "sign+digits"):
            s[cat] += 1
        for w in wrong:
            s["wrong_at"][w] += 1
        s["n_wrong_distribution"][len(wrong)] += 1
        if wrong:
            most_sig = max(wrong)  # highest index = highest place value
            s["most_sig_wrong"][most_sig] += 1
    return stats


def _format_per_m_c(stats: dict, vname: str) -> list[str]:
    lines = [f"\n### {vname}", ""]
    lines.append("**Error categorization by m_c (normal-result only):**")
    lines.append("")
    lines.append("| m_c | n | n_err | sign_only | digit_only | sign+digits |")
    lines.append("|-----|--:|------:|----------:|-----------:|------------:|")
    for m_val in range(8):
        s = stats[m_val]
        lines.append(
            f"| {m_val}/8 | {s['n_total']:>5} | {s['n_err']:>5} | "
            f"{s['sign_only']:>9} | {s['digit_only']:>10} | {s['sign+digits']:>11} |"
        )

    # Place-value labels for digit positions.
    place_labels = [f"10^{e}" for e in DIGIT_PLACE_EXPONENTS]

    lines.append("")
    lines.append("**Wrong-digit-position frequency per m_c bin (% of n_err with that digit wrong):**")
    lines.append("")
    header = "| m_c |" + "".join(f" {p} |" for p in place_labels)
    lines.append(header)
    lines.append("|-----|" + "------|" * len(place_labels))
    for m_val in range(8):
        s = stats[m_val]
        if s["n_err"] == 0:
            lines.append(f"| {m_val}/8 |" + " 0% |" * N_DIGITS)
            continue
        row = f"| {m_val}/8 |"
        for w in s["wrong_at"]:
            pct = w / s["n_err"] * 100
            row += f" {pct:>4.1f}% |"
        lines.append(row)

    lines.append("")
    lines.append("**Most-significant wrong digit per m_c bin (% of digit-errors):**")
    lines.append("")
    header = "| m_c |" + "".join(f" {p} |" for p in place_labels)
    lines.append(header)
    lines.append("|-----|" + "------|" * len(place_labels))
    for m_val in range(8):
        s = stats[m_val]
        total = sum(s["most_sig_wrong"])
        if total == 0:
            lines.append(f"| {m_val}/8 |" + " — |" * N_DIGITS)
            continue
        row = f"| {m_val}/8 |"
        for w in s["most_sig_wrong"]:
            pct = w / total * 100
            row += f" {pct:>4.1f}% |"
        lines.append(row)

    return lines


DEFAULT_RUNS = [
    ("learned-PE FoNE s0  (L4-E048)", "runs/fone-L4-E048-s0/checkpoint_020000.pt"),
    ("learned-PE FoNE s8  (L4-E048)", "runs/fone-L4-E048-s8/checkpoint_020000.pt"),
    ("learned-PE FoNE s14 (L4-E048)", "runs/fone-L4-E048-s14/checkpoint_020000.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/fone_digit_decomp_findings.md")
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# FoNE digit-decomposition probe",
        "",
        "Categorical analog of the bit/SEM bit-decomposition probe. For each",
        "error, identify which digit position(s) went wrong; the 6 digits cover",
        "places 10^-3 through 10^2.",
        "",
        "**Hypotheses:**",
        "- If FoNE inherits FP's binade-boundary cost: errors concentrate at",
        "  m=0/8 and m=7/8 bins, with high-place digits (tens, hundreds) wrong.",
        "- If FoNE makes per-digit rounding mistakes uniformly: errors spread",
        "  across m_c bins, low-place digits (thousandths, hundredths) most",
        "  often wrong.",
        "- The severity probe already shows uniform-across-bins n_err, so the",
        "  second hypothesis is favored heading in.",
    ]
    for vname, vpath in DEFAULT_RUNS:
        print(f"Running {vname}: {vpath}", flush=True)
        out = fone_predict_on_holdout(repo / vpath, device="cpu")
        stats = _per_m_c_digit_table(
            out["rows"], out["sign_pred"], out["digit_pred"],
            out["sign_target"], out["digit_target"],
            out["is_nan"], out["correct"],
        )
        sections.extend(_format_per_m_c(stats, vname))

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
