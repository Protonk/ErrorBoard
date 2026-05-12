"""ε-bit-decomposition experiment.

Severity analysis showed catastrophic errors at endpoint m_c bins (m=0/8, 7/8)
with max ULP errors in the thousands. Hypothesis: these are exponent-coordination
failures — the model gets the mantissa "near right" but flips a sign or exponent
bit at the binade transition. Smooth-interior errors (m=3/8 to 5/8) should be
mantissa-only — single-bit RNE mis-rounding without losing the magnitude.

For each error pair, decompose which output bits went wrong:
  - sign bit (bit 7)
  - exponent bits (bits 3-6)
  - mantissa bits (bits 0-2)

Classify each error as:
  - sign_only:  only sign differs
  - exp_only:   only exponent bit(s) differ
  - mant_only:  only mantissa bit(s) differ
  - exp+mant:   exp and mantissa differ (no sign)
  - sign+exp:   sign and exp differ (no mantissa)
  - sign+mant:  sign and mantissa differ (no exp)
  - all:        all three differ

Stratify by result-mantissa bin m_c. The dual-of-ε prediction: endpoint bins
are exp-heavy, smooth-interior bins are mant-only-heavy.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .hooked_bridge import load_hooked
from .pentagon import VERTICES, _repo_root
from .preprocess import build_table, split_train_holdout
from .tokenizer import POS_C_START, POS_C_END, encode_batch


@torch.no_grad()
def predicted_bits(checkpoint_path: Path) -> dict:
    """Forward on holdout; return predicted FP8 bit patterns."""
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
    pred_bits = np.zeros(n, dtype=np.uint8)
    correct = np.zeros(n, dtype=bool)
    for i in range(0, n, bsz):
        idx = inputs[i:i+bsz].to(next(hooked.parameters()).device)
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)
        tgt = targets[i:i+bsz].to(idx.device)
        pred_slice = preds[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()
        tgt_slice = tgt[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
        for j in range(8):
            tok = pred_slice[:, j]
            bit = (tok & 1).astype(np.uint8)
            pred_bits[i:i+bsz] |= bit << (7 - j)
    return {"rows": rows, "pred_bits": pred_bits, "correct": correct}


# Bit positions in FP8 E4M3:
#   bit 7: sign
#   bits 6,5,4,3: exponent (e3, e2, e1, e0)
#   bits 2,1,0: mantissa (m2, m1, m0)
SIGN_MASK = 0x80
EXP_MASK = 0x78
MANT_MASK = 0x07


def classify_error(pred: int, true: int) -> str:
    xor = pred ^ true
    sign_diff = bool(xor & SIGN_MASK)
    exp_diff = bool(xor & EXP_MASK)
    mant_diff = bool(xor & MANT_MASK)
    if not (sign_diff or exp_diff or mant_diff):
        return "correct"
    flags = (int(sign_diff) << 2) | (int(exp_diff) << 1) | int(mant_diff)
    return {
        0b001: "mant_only",
        0b010: "exp_only",
        0b100: "sign_only",
        0b011: "exp+mant",
        0b101: "sign+mant",
        0b110: "sign+exp",
        0b111: "all",
    }[flags]


CATEGORIES = ["mant_only", "exp_only", "sign_only", "exp+mant", "sign+mant", "sign+exp", "all"]


def stratify_errors(rows, pred_bits, correct) -> dict:
    """Stratify error types by result mantissa bin (normal-result only)."""
    result_bits = rows["result_bits"].astype(int)
    result_exp = (result_bits >> 3) & 0xF
    result_m = result_bits & 0x7
    is_normal = (result_exp >= 1) & ~((result_bits == 0x7F) | (result_bits == 0xFF))
    out = {m: {cat: 0 for cat in CATEGORIES + ["total_err", "total_n"]} for m in range(8)}
    for i in range(len(rows)):
        if not is_normal[i]:
            continue
        m_val = result_m[i]
        out[m_val]["total_n"] += 1
        if correct[i]:
            continue
        cat = classify_error(int(pred_bits[i]), int(result_bits[i]))
        if cat != "correct":
            out[m_val][cat] += 1
            out[m_val]["total_err"] += 1
    return out


def format_block(vname: str, stats: dict) -> str:
    lines = [f"\n### {vname}\n"]
    # Header: m_c, n, n_err, then per-category counts and percentages
    lines.append("| m_c | n | n_err | mant_only | exp_only | sign_only | exp+mant | sign+mant | sign+exp | all |")
    lines.append("|-----|--:|------:|----------:|---------:|----------:|---------:|----------:|---------:|----:|")
    for m_val in range(8):
        s = stats[m_val]
        n = s["total_n"]
        n_err = s["total_err"]
        if n_err == 0:
            row = f"| {m_val}/8 | {n} | {n_err} | " + " | ".join("0" for _ in CATEGORIES) + " |"
        else:
            cells = []
            for cat in CATEGORIES:
                c = s[cat]
                pct = c / n_err * 100 if n_err > 0 else 0
                cells.append(f"{c} ({pct:.0f}%)")
            row = f"| {m_val}/8 | {n} | {n_err} | " + " | ".join(cells) + " |"
        lines.append(row)

    # Summary: for each m_c, what fraction of errors involve mantissa only?
    lines.append("")
    lines.append("**Aggregate views:**")
    lines.append("")
    lines.append("| m_c | mant-only %      | any-exp-error %  | any-sign-error % |")
    lines.append("|-----|-----------------:|-----------------:|-----------------:|")
    for m_val in range(8):
        s = stats[m_val]
        n_err = s["total_err"]
        if n_err == 0:
            lines.append(f"| {m_val}/8 |   —  |   —  |   —  |")
            continue
        mant_only = s["mant_only"]
        any_exp = s["exp_only"] + s["exp+mant"] + s["sign+exp"] + s["all"]
        any_sign = s["sign_only"] + s["sign+mant"] + s["sign+exp"] + s["all"]
        lines.append(
            f"| {m_val}/8 | {mant_only}/{n_err} = {mant_only/n_err*100:>5.1f}% | "
            f"{any_exp}/{n_err} = {any_exp/n_err*100:>5.1f}% | "
            f"{any_sign}/{n_err} = {any_sign/n_err*100:>5.1f}% |"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# ε-bit-decomposition experiment",
        "",
        "For each error pair, decompose which output bits went wrong:",
        "sign (bit 7), exponent (bits 3-6), mantissa (bits 0-2).",
        "",
        "Tests the dual-of-ε reading: endpoint bins (m=0/8, m=7/8) should be",
        "exp-heavy; smooth-interior bins (m=3/8 to 5/8) should be mant_only-heavy.",
    ]
    for vname, vpath, vdesc in VERTICES:
        print(f"Running {vname}...")
        out = predicted_bits(repo / vpath)
        stats = stratify_errors(out["rows"], out["pred_bits"], out["correct"])
        sections.append(format_block(f"{vname}: {vdesc}", stats))

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / "notes/epsilon_bit_decomp_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
