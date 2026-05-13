"""Reciprocal-side severity probe, stratified by input magnitude.

For each pair where the model errs, compute:
  - ULP error
  - log-damage |log₂|pred| − log₂|true||
  - relative error vs dayval's format floor (0.1875)

Stratification axes:
  1. Input mantissa m_a (analog of mult's m_a · m_b but unary)
  2. Input binade (exponent)
  3. Distance from dayval's worst input 0x75

Reports per-arm: mean / max relative error, % of pairs at or near the
dayval floor, and the relative-error-vs-input-magnitude correlation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .epsilon_severity import ulp_at_value
from .failure_consensus_recip import (
    _bit_correctness_recip, _fone_predict_on_recip_holdout,
)
from .hooked_bridge import load_hooked
from .oracle import decode as fp8_decode
from .pentagon import _repo_root
from .preprocess_recip import build_table, split_train_holdout
from .recip_regimes import (
    DAYVAL_EPS_FLOOR, DAYVAL_WORST_INPUT, REGIME_NAMES,
    EXACT_RESULT, SPECIAL_VALUES,
)
from . import tokenizer as _bit_tokenizer
from . import sem_tokenizer as _sem_tokenizer


def _bit_predict_real_values_recip(checkpoint_path: Path, tokenizer) -> dict:
    """Forward bit/SEM model on recip holdout; return predicted FP8 bit patterns."""
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    hooked, _gpt = load_hooked(checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seqs = tokenizer.encode_batch(triples.astype(np.uint8))
    inputs = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    targets = torch.from_numpy(seqs[:, 1:].astype(np.int64))
    pos_c_start = tokenizer.POS_C_START
    pos_c_end = tokenizer.POS_C_END
    n = inputs.shape[0]
    bsz = 256
    correct = np.zeros(n, dtype=bool)
    pred_bits = np.zeros(n, dtype=np.uint8)

    for i in range(0, n, bsz):
        idx = inputs[i:i+bsz]
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)
        tgt = targets[i:i+bsz]
        pred_slice = preds[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        tgt_slice = tgt[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
        for j, ps in enumerate(pred_slice):
            try:
                pred_bits[i + j] = tokenizer.decode_fp8_tokens(ps.tolist())
            except Exception:
                pred_bits[i + j] = 0
    return {"rows": rows, "correct": correct, "pred_bits": pred_bits}


def _compute_severity_bit_sem(pred_bits, rows, correct) -> dict:
    """Severity from predicted bit patterns (bit/SEM-shape arms)."""
    n = len(rows)
    ulp_err = np.full(n, np.nan)
    log_damage = np.full(n, np.nan)
    rel_err = np.full(n, np.nan)
    for i in range(n):
        true_bits = int(rows[i]["result_bits"])
        true_val, true_kind = fp8_decode(true_bits)
        if correct[i]:
            ulp_err[i] = 0.0
            log_damage[i] = 0.0
            rel_err[i] = 0.0
            continue
        if true_kind == "nan":
            continue
        pred_val, pred_kind = fp8_decode(int(pred_bits[i]))
        if pred_kind == "nan":
            log_damage[i] = float("inf")
            continue
        diff = abs(pred_val - true_val)
        ulp = ulp_at_value(true_val, true_kind)
        if ulp > 0:
            ulp_err[i] = diff / ulp
        if true_val != 0:
            rel_err[i] = diff / abs(true_val)
        if true_val != 0 and pred_val != 0 and (pred_val > 0) == (true_val > 0):
            log_damage[i] = abs(math.log2(abs(pred_val)) - math.log2(abs(true_val)))
        elif true_val == 0 and pred_val == 0:
            log_damage[i] = 0.0
        else:
            log_damage[i] = float("inf")
    return {"ulp_err": ulp_err, "log_damage": log_damage, "rel_err": rel_err}


def _compute_severity_fone(out: dict, arm: str) -> dict:
    """Severity from FoNE sign+digits predictions."""
    rows = out["rows"]
    sign_pred = out["sign_pred"]
    digit_pred = out["digit_pred"]
    correct = out["correct"]
    is_nan_target = out["is_nan"]

    if arm == "f1":
        from .fone_tokenizer import (
            SIGN_NAN_ID, SIGN_NEG_ID, digits_to_magnitude,
        )
    else:
        from .fone_f2_tokenizer import (
            SIGN_NAN_ID, SIGN_NEG_ID, digits_to_magnitude,
        )

    n = len(rows)
    ulp_err = np.full(n, np.nan)
    log_damage = np.full(n, np.nan)
    rel_err = np.full(n, np.nan)

    for i in range(n):
        true_bits = int(rows[i]["result_bits"])
        true_val, true_kind = fp8_decode(true_bits)
        if correct[i]:
            ulp_err[i] = 0.0
            log_damage[i] = 0.0
            rel_err[i] = 0.0
            continue
        if true_kind == "nan" or is_nan_target[i]:
            continue
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
        if true_val != 0:
            rel_err[i] = diff / abs(true_val)
        if true_val != 0 and pred_val != 0 and (pred_val > 0) == (true_val > 0):
            log_damage[i] = abs(math.log2(abs(pred_val)) - math.log2(abs(true_val)))
        elif true_val == 0 and pred_val == 0:
            log_damage[i] = 0.0
        else:
            log_damage[i] = float("inf")
    return {"ulp_err": ulp_err, "log_damage": log_damage, "rel_err": rel_err}


def _pearson(xs, ys) -> float:
    pairs = [(x, y) for x, y in zip(xs, ys)
             if not (math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y))]
    if len(pairs) < 3:
        return float("nan")
    xs2, ys2 = zip(*pairs)
    x = np.asarray(xs2); y = np.asarray(ys2)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _input_mant(a_bits: int) -> float:
    """Return m/8 for the input (FP8 mantissa as fraction)."""
    return (a_bits & 0x7) / 8.0


def _per_seed_metrics(arm_label: str, prefix: str, seeds: list[int],
                      repo: Path, loader: str) -> list[dict]:
    rows_out = []
    for s in seeds:
        path = repo / f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        if not path.exists():
            print(f"  {prefix} s{s}: SKIP", flush=True)
            continue
        print(f"  {prefix} s{s}: loading...", flush=True)
        if loader == "bit":
            pred = _bit_predict_real_values_recip(path, _bit_tokenizer)
            sev = _compute_severity_bit_sem(pred["pred_bits"], pred["rows"],
                                             pred["correct"])
        elif loader == "sem":
            pred = _bit_predict_real_values_recip(path, _sem_tokenizer)
            sev = _compute_severity_bit_sem(pred["pred_bits"], pred["rows"],
                                             pred["correct"])
        elif loader == "f1":
            pred = _fone_predict_on_recip_holdout(path, "f1")
            sev = _compute_severity_fone(pred, "f1")
        else:
            pred = _fone_predict_on_recip_holdout(path, "f2")
            sev = _compute_severity_fone(pred, "f2")

        rows = pred["rows"]
        rel_err = sev["rel_err"]
        log_damage = sev["log_damage"]

        # Headline stats
        finite_rel = rel_err[np.isfinite(rel_err)]
        mean_rel = float(finite_rel.mean()) if len(finite_rel) > 0 else float("nan")
        max_rel = float(finite_rel.max()) if len(finite_rel) > 0 else float("nan")

        # Floor-relative: how many pairs exceed dayval's 0.1875 floor?
        finite_rel_for_floor = rel_err[np.isfinite(rel_err)]
        n_above_floor = int((finite_rel_for_floor > DAYVAL_EPS_FLOOR + 1e-9).sum())
        n_at_or_below_floor = int((finite_rel_for_floor <= DAYVAL_EPS_FLOOR + 1e-9).sum())

        # Correlation with input mantissa: does severity track m_a?
        m_a = np.array([_input_mant(int(rows[i]["a_bits"])) for i in range(len(rows))])
        rho_logdam_ma = _pearson(m_a.tolist(), log_damage.tolist())
        rho_rel_ma = _pearson(m_a.tolist(), rel_err.tolist())

        # Is the worst input near 0x75?
        finite_idx = np.where(np.isfinite(rel_err))[0]
        if len(finite_idx) > 0:
            worst_local = finite_idx[rel_err[finite_idx].argmax()]
            worst_input_bits = int(rows[worst_local]["a_bits"])
        else:
            worst_input_bits = -1

        n_err = int((~pred["correct"]).sum())
        rows_out.append({
            "seed": s,
            "mean_rel": mean_rel,
            "max_rel": max_rel,
            "n_above_floor": n_above_floor,
            "n_at_or_below_floor": n_at_or_below_floor,
            "rho_logdam_ma": rho_logdam_ma,
            "rho_rel_ma": rho_rel_ma,
            "worst_input_bits": worst_input_bits,
            "n_err": n_err,
        })
    return rows_out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--out", default="notes/recip_severity_findings.md")
    args = p.parse_args()
    repo = _repo_root()
    seeds = list(range(args.n_seeds))

    print("Probing bit-recip...")
    bit_results = _per_seed_metrics("bit", "bit-recip-L4-E048", seeds, repo, "bit")
    print("Probing SEM-recip...")
    sem_results = _per_seed_metrics("sem", "sem-recip-L4-E048", seeds, repo, "sem")
    print("Probing FoNE F1-recip...")
    f1_results = _per_seed_metrics("f1", "fone-recip-L4-E048", seeds, repo, "f1")
    print("Probing FoNE F2-recip...")
    f2_results = _per_seed_metrics("f2", "fone-f2-recip-L4-E048", seeds, repo, "f2")

    sections = [
        "# Reciprocal severity probe (L4-E048, 5 seeds per arm)",
        "",
        f"dayval format-intrinsic floor for FP8 E4M3 reciprocal: "
        f"ε_floor = {DAYVAL_EPS_FLOOR} (worst input bit pattern 0x{DAYVAL_WORST_INPUT:02x}).",
        "No algorithm can achieve relative error below this on the worst input.",
        "",
        "## Per-seed metrics",
        "",
        "| arm | seed | mean rel-err | max rel-err | n above floor | n ≤ floor | "
        "ρ(log Δ, m_a) | worst input |",
        "|-----|-----:|-----:|-----:|------:|------:|------:|------:|",
    ]
    for name, results in [("bit", bit_results), ("SEM", sem_results),
                          ("FoNE F1", f1_results), ("FoNE F2", f2_results)]:
        for r in results:
            worst_str = f"0x{r['worst_input_bits']:02x}" if r["worst_input_bits"] >= 0 else "—"
            rho = f"{r['rho_logdam_ma']:+.2f}" if not math.isnan(r["rho_logdam_ma"]) else "  nan"
            sections.append(
                f"| {name} | {r['seed']} | {r['mean_rel']:.4f} | "
                f"{r['max_rel']:.4f} | {r['n_above_floor']} | "
                f"{r['n_at_or_below_floor']} | {rho} | {worst_str} |"
            )
    sections.append("")

    sections.append("## Mean per arm\n")
    sections.append("| arm | mean rel-err | mean max rel-err | mean ρ(log Δ, m_a) | "
                    "% worst-input at 0x75 |")
    sections.append("|-----|------:|------:|------:|------:|")

    def _mean_ignoring_nan(xs):
        finite = [v for v in xs if not (math.isnan(v) or math.isinf(v))]
        return float(np.mean(finite)) if finite else float("nan")

    for name, results in [("bit", bit_results), ("SEM", sem_results),
                          ("FoNE F1", f1_results), ("FoNE F2", f2_results)]:
        mean_rel = _mean_ignoring_nan([r["mean_rel"] for r in results])
        mean_max = _mean_ignoring_nan([r["max_rel"] for r in results])
        mean_rho = _mean_ignoring_nan([r["rho_logdam_ma"] for r in results])
        worst_at_target = sum(
            1 for r in results
            if abs((r["worst_input_bits"] & 0x7F) - DAYVAL_WORST_INPUT) <= 2
        )
        pct = worst_at_target / len(results) * 100 if results else 0
        sections.append(f"| {name} | {mean_rel:.4f} | {mean_max:.4f} | "
                        f"{mean_rho:+.3f} | {pct:.0f}% |")

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
