"""ε-trace experiment.

Tests whether the trained model's residual error rate, as a function of the
result mantissa value m_c, traces the shape of the formal ε(m) = log₂(1+m) − m
function.

ε(m) at the 8 representable FP8 mantissa values:
    m_c    0     1/8    2/8    3/8    4/8    5/8    6/8    7/8
    ε(m)   0    0.045  0.072  0.084  0.085  0.075  0.057  0.032

ε peaks near the middle (m=3/8 to 4/8). The hypothesis: V4's error rate as a
function of m_c traces this shape — peaked in the middle, low at the endpoints.
If true: empirical evidence the model's residual is shadowing ε at the resolution
our lossy image permits.

Restricted to normal-result cases (subnormal mantissas live in a different
coordinate and the leading-1 interpretation doesn't apply).
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


def epsilon(m: float) -> float:
    """ε(m) = log₂(1+m) − m for m ∈ [0, 1)."""
    return math.log2(1 + m) - m


@torch.no_grad()
def errors_on_holdout(checkpoint_path: Path) -> dict:
    """Run model on full holdout, return per-pair correctness."""
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
    return {"rows": rows, "correct": correct, "n": n}


def stratify_by_result_mantissa(rows, correct) -> dict[int, dict]:
    """Bin pairs by result mantissa value (0..7). Restrict to normal-result cases."""
    result_bits = rows["result_bits"].astype(int)
    # Biased exponent of result.
    result_exp = (result_bits >> 3) & 0xF
    # Result mantissa bits (0..7).
    result_m = result_bits & 0x7
    # Normal result: biased exp >= 1, not NaN.
    is_normal = (result_exp >= 1) & ~((result_bits == 0x7F) | (result_bits == 0xFF))
    out = {}
    for m_val in range(8):
        mask = is_normal & (result_m == m_val)
        n = int(mask.sum())
        n_err = int((~correct[mask]).sum())
        out[m_val] = {
            "m_fraction": m_val / 8.0,
            "epsilon": epsilon(m_val / 8.0),
            "n": n,
            "n_err": n_err,
            "err_rate": n_err / n if n > 0 else float("nan"),
        }
    return out


def pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation. Returns NaN if either is constant."""
    x = np.asarray(x); y = np.asarray(y)
    if x.std() == 0 or y.std() == 0 or len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def format_vertex_block(vname: str, stats: dict[int, dict]) -> str:
    lines = [f"\n### {vname}\n"]
    lines.append("| m_c   | m (fraction) | ε(m)   | n     | n_err | error rate |")
    lines.append("|-------|-------------:|-------:|------:|------:|-----------:|")
    eps = []
    rates = []
    for m_val in range(8):
        s = stats[m_val]
        rate_str = f"{s['err_rate']*100:.2f}%" if s['n'] > 0 else "  —  "
        lines.append(
            f"| {m_val}/8   | {s['m_fraction']:>12.3f} | {s['epsilon']:>6.4f} | "
            f"{s['n']:>5} | {s['n_err']:>5} | {rate_str:>10} |"
        )
        if s['n'] > 0:
            eps.append(s['epsilon'])
            rates.append(s['err_rate'])
    rho = pearson(eps, rates)
    lines.append("")
    lines.append(f"**Pearson(ε, error_rate) = {rho:+.3f}**")
    if not math.isnan(rho):
        # Identify which m_c has highest error rate.
        max_m = max(stats.keys(), key=lambda k: stats[k]['err_rate'] if stats[k]['n'] > 0 else -1)
        lines.append(f"Peak m_c: {max_m}/8 (error rate {stats[max_m]['err_rate']*100:.2f}%); "
                     f"ε-peak is at m_c=3/8 or 4/8.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# ε-trace experiment",
        "",
        "Tests whether each vertex's error rate, stratified by *result* mantissa",
        "value m_c, traces the formal ε(m) = log₂(1+m) − m function.",
        "",
        "Restricted to normal-result cases (biased exponent ≥ 1, not NaN).",
        "",
        "**ε(m) values across the 8 mantissa bins**:",
        "",
        "| m_c | 0/8 | 1/8 | 2/8 | 3/8 | 4/8 | 5/8 | 6/8 | 7/8 |",
        "|-----|----:|----:|----:|----:|----:|----:|----:|----:|",
        "| ε   | 0.000 | 0.045 | 0.072 | 0.084 | 0.085 | 0.075 | 0.057 | 0.032 |",
        "",
        "Peak at m_c=3/8 or 4/8. If model errors trace ε, error rate should peak there.",
    ]

    per_vertex_corr = {}
    for vname, vpath, vdesc in VERTICES:
        print(f"Running {vname}: {vpath} ({vdesc})")
        full_path = repo / vpath
        out = errors_on_holdout(full_path)
        stats = stratify_by_result_mantissa(out["rows"], out["correct"])
        sections.append(format_vertex_block(f"{vname}: {vdesc}", stats))

        eps = [stats[m]['epsilon'] for m in range(8) if stats[m]['n'] > 0]
        rates = [stats[m]['err_rate'] for m in range(8) if stats[m]['n'] > 0]
        per_vertex_corr[vname] = pearson(eps, rates)

    sections.append("\n## Cross-vertex summary\n")
    sections.append("| vertex | Pearson(ε, err_rate) |")
    sections.append("|--------|---------------------:|")
    for vname in ["V1", "V2", "V3", "V4", "V5"]:
        rho = per_vertex_corr.get(vname, float("nan"))
        sections.append(f"| {vname} | {rho:+.3f} |")

    sections.append("\n## Interpretation guide\n")
    sections.append(
        "- ρ → +1.0: error rate traces ε shape (peak at m=3/8..4/8, low at endpoints)\n"
        "- ρ → 0:    error rate flat or unrelated to ε\n"
        "- ρ → −1.0: error rate anti-correlated (peak at endpoints, low in middle)\n"
        "\n"
        "A cross-vertex pattern where ρ increases from V1 to V4 would suggest that\n"
        "ε's shape becomes visible only at saturation — earlier models are too noisy\n"
        "for ε to surface. V5 (depth-capped) vs V4 (saturated) tests the architectural\n"
        "axis: does depth-1 fail to trace ε even at convergence?"
    )

    output = "\n".join(sections)
    print("\n" + output)

    out_path = repo / "notes/epsilon_trace_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
