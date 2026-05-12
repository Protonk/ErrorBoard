"""ε-severity experiment.

The earlier ε-trace experiment measured *frequency* of error per result-mantissa
bin and found it doesn't trace ε(m). But the practical manifestation of ε in
arithmetic is "given an error occurred, how damaging is it?" — error severity,
not frequency. This experiment re-runs the same stratification but measures
damage when wrong, not rate of being wrong.

Damage measures per error:
  - ULP-distance: |pred_real − true_real| / ulp(true)
    1 ULP = single-step RNE miss; multi-ULP = more substantive failure.
  - Log-damage:   |log₂|pred_real| − log₂|true_real||
    Dimensionless. The natural ε-comparable scale.

Stratify by result mantissa m_c (8 bins). The hypothesis: if ε surfaces as
damage, log-damage per bin should track ε(m_c) — peak in the middle (m=3/8 to
5/8) where ε is largest, low at endpoints.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .hooked_bridge import load_hooked
from .oracle import decode
from .pentagon import VERTICES, _repo_root
from .preprocess import build_table, split_train_holdout
from .tokenizer import POS_C_START, POS_C_END, encode_batch


def epsilon(m: float) -> float:
    return math.log2(1 + m) - m


def ulp_at_value(value: float, kind: str) -> float:
    """ULP magnitude at a given FP8 value. For normals, 2^(E - p) with p=3.
    For subnormals, 2^(e_min − p) = 2^-9 (constant across all subnormals in E4M3).
    Returns NaN for NaN / 0 inputs."""
    if kind == "nan":
        return float("nan")
    if value == 0:
        return 2.0 ** -9  # smallest representable
    abs_val = abs(value)
    # Binade of |value|: floor(log2(|value|))
    E = int(math.floor(math.log2(abs_val)))
    return 2.0 ** (E - 3)


@torch.no_grad()
def severity_per_pair(checkpoint_path: Path) -> dict:
    """Forward on holdout; return per-pair severity measures."""
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
        pred_slice = preds[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()  # (B, 8) tokens
        tgt_slice = tgt[:, POS_C_START - 1:POS_C_END - 1].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
        # Decode pred tokens back into FP8 bit patterns. Mapping: SIGN_0/SIGN_1, EXP_0/EXP_1,
        # MANT_0/MANT_1. Token id 6 = SIGN_0, 7 = SIGN_1, 8 = EXP_0, 9 = EXP_1, 10 = MANT_0,
        # 11 = MANT_1. The first bit (sign), bits 1-4 (exp), bits 5-7 (mantissa).
        for j in range(8):
            tok = pred_slice[:, j]
            bit = (tok & 1).astype(np.uint8)  # SIGN_1, EXP_1, MANT_1 all have LSB=1
            pred_bits[i:i+bsz] |= bit << (7 - j)

    return {"rows": rows, "pred_bits": pred_bits, "correct": correct, "n": n}


def compute_severity(rows, pred_bits, correct) -> dict:
    """For each pair, compute ULP and log severity. Returns arrays aligned with rows."""
    n = len(rows)
    true_bits = rows["result_bits"].astype(np.uint8)
    ulp_err = np.full(n, np.nan)
    log_damage = np.full(n, np.nan)
    for i in range(n):
        if correct[i]:
            ulp_err[i] = 0.0
            log_damage[i] = 0.0
            continue
        true_val, true_kind = decode(int(true_bits[i]))
        pred_val, pred_kind = decode(int(pred_bits[i]))
        if true_kind == "nan" or pred_kind == "nan":
            continue  # leave NaN
        diff = abs(pred_val - true_val)
        ulp = ulp_at_value(true_val, true_kind)
        if ulp > 0:
            ulp_err[i] = diff / ulp
        if true_val != 0 and pred_val != 0 and (pred_val > 0) == (true_val > 0):
            log_damage[i] = abs(math.log2(abs(pred_val)) - math.log2(abs(true_val)))
        elif true_val == 0 and pred_val == 0:
            log_damage[i] = 0.0
        else:
            log_damage[i] = float("inf")  # sign flip or zero involvement
    return {"ulp_err": ulp_err, "log_damage": log_damage}


def stratify(rows, sev, correct):
    """Bin by result mantissa for normal-result pairs. For each bin, compute:
    n_total, n_err, mean_ulp_err_given_err, mean_log_damage_given_err."""
    result_bits = rows["result_bits"].astype(int)
    result_exp = (result_bits >> 3) & 0xF
    result_m = result_bits & 0x7
    is_normal = (result_exp >= 1) & ~((result_bits == 0x7F) | (result_bits == 0xFF))
    out = {}
    for m_val in range(8):
        mask = is_normal & (result_m == m_val)
        n = int(mask.sum())
        err_mask = mask & ~correct
        n_err = int(err_mask.sum())
        if n_err > 0:
            ulps = sev["ulp_err"][err_mask]
            logs = sev["log_damage"][err_mask]
            finite_ulps = ulps[np.isfinite(ulps)]
            finite_logs = logs[np.isfinite(logs)]
            mean_ulp = float(finite_ulps.mean()) if len(finite_ulps) else float("nan")
            max_ulp = float(finite_ulps.max()) if len(finite_ulps) else float("nan")
            mean_log = float(finite_logs.mean()) if len(finite_logs) else float("nan")
            max_log = float(finite_logs.max()) if len(finite_logs) else float("nan")
        else:
            mean_ulp = max_ulp = mean_log = max_log = float("nan")
        out[m_val] = {
            "m_fraction": m_val / 8.0,
            "epsilon": epsilon(m_val / 8.0),
            "n": n,
            "n_err": n_err,
            "mean_ulp_err": mean_ulp,
            "max_ulp_err": max_ulp,
            "mean_log_damage": mean_log,
            "max_log_damage": max_log,
        }
    return out


def pearson(x, y):
    x = np.asarray([v for v in x if not math.isnan(v)])
    y = np.asarray([v for v in y if not math.isnan(v)])
    if len(x) < 3 or len(y) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def format_block(vname: str, stats: dict) -> str:
    lines = [f"\n### {vname}\n"]
    lines.append("| m_c | ε(m)  | n     | n_err | mean ULP err | max ULP err | mean |log Δ| | max |log Δ| |")
    lines.append("|-----|------:|------:|------:|-------------:|------------:|--------------:|-------------:|")
    for m_val in range(8):
        s = stats[m_val]
        mean_ulp_str = f"{s['mean_ulp_err']:>10.3f}" if s['n_err'] > 0 and not math.isnan(s['mean_ulp_err']) else "  —  "
        max_ulp_str = f"{s['max_ulp_err']:>10.2f}" if s['n_err'] > 0 and not math.isnan(s['max_ulp_err']) else "  —  "
        mean_log_str = f"{s['mean_log_damage']:>10.4f}" if s['n_err'] > 0 and not math.isnan(s['mean_log_damage']) else "  —  "
        max_log_str = f"{s['max_log_damage']:>10.4f}" if s['n_err'] > 0 and not math.isnan(s['max_log_damage']) else "  —  "
        lines.append(f"| {m_val}/8 | {s['epsilon']:>5.4f} | {s['n']:>5} | {s['n_err']:>5} | {mean_ulp_str:>12} | {max_ulp_str:>11} | {mean_log_str:>13} | {max_log_str:>12} |")
    # Pearsons across bins (where n_err > 0).
    eps = [stats[m]['epsilon'] for m in range(8) if stats[m]['n_err'] > 0]
    mean_ulps = [stats[m]['mean_ulp_err'] for m in range(8) if stats[m]['n_err'] > 0]
    mean_logs = [stats[m]['mean_log_damage'] for m in range(8) if stats[m]['n_err'] > 0]
    rho_ulp = pearson(eps, mean_ulps)
    rho_log = pearson(eps, mean_logs)
    lines.append("")
    lines.append(f"Pearson(ε, mean ULP err) = {rho_ulp:+.3f}")
    lines.append(f"Pearson(ε, mean |log damage|) = {rho_log:+.3f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# ε-severity experiment",
        "",
        "Earlier ε-trace measured *frequency* of error per m_c bin; this re-runs",
        "with severity as the dependent variable. Per pair where the model errs,",
        "we compute ULP-distance and absolute log-damage.",
        "",
        "Stratify by result mantissa m_c (normal-result only). Hypothesis: damage",
        "per bin tracks ε(m_c).",
        "",
        "**ε(m) reference**:",
        "",
        "| m_c | 0/8 | 1/8 | 2/8 | 3/8 | 4/8 | 5/8 | 6/8 | 7/8 |",
        "|-----|----:|----:|----:|----:|----:|----:|----:|----:|",
        "| ε   | 0.000 | 0.045 | 0.072 | 0.084 | 0.085 | 0.075 | 0.057 | 0.032 |",
    ]

    summary_rows = []
    for vname, vpath, vdesc in VERTICES:
        print(f"Running {vname}: {vpath}")
        out = severity_per_pair(repo / vpath)
        sev = compute_severity(out["rows"], out["pred_bits"], out["correct"])
        stats = stratify(out["rows"], sev, out["correct"])
        sections.append(format_block(f"{vname}: {vdesc}", stats))

        eps = [stats[m]['epsilon'] for m in range(8) if stats[m]['n_err'] > 0]
        mean_ulps = [stats[m]['mean_ulp_err'] for m in range(8) if stats[m]['n_err'] > 0]
        mean_logs = [stats[m]['mean_log_damage'] for m in range(8) if stats[m]['n_err'] > 0]
        summary_rows.append((vname, pearson(eps, mean_ulps), pearson(eps, mean_logs)))

    sections.append("\n## Cross-vertex summary\n")
    sections.append("| vertex | Pearson(ε, mean ULP) | Pearson(ε, mean |log Δ|) |")
    sections.append("|--------|---------------------:|-------------------------:|")
    for vname, r_ulp, r_log in summary_rows:
        sections.append(f"| {vname} | {r_ulp:+.3f} | {r_log:+.3f} |")

    sections.append("\n## Interpretation\n")
    sections.append(
        "If damage at result-mantissa m_c tracks ε(m_c), the model's errors are\n"
        "concentrated *in real magnitude* where the formal residual peaks. If\n"
        "instead damage is constant (always 1 ULP), the model is making\n"
        "rounding-class mistakes regardless of m_c. If max ULP error is mostly 1,\n"
        "ε's shape might only surface in a different stratification.\n"
    )

    output = "\n".join(sections)
    print("\n" + output)

    out_path = repo / "notes/epsilon_severity_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
