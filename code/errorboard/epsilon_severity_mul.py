"""Multiplication-side severity probe stratified by ε_mult = m_a · m_b.

For each pair where the model errs, compute:
  - ULP error
  - log-damage |log₂|pred| − log₂|true||

Stratify by ε_mult = (m_a / 8) · (m_b / 8), the bilinear cost the affine
pseudo-log misses for multiplication (see notes/epsilon_under_multiplication.md).

Reports Pearson(ε_mult, mean |log Δ|) two ways:
  - full population (includes EXACT_RESULT pairs at severity=0, censors signal)
  - rounding-required (excludes EXACT_RESULT and SPECIAL_VALUES)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from .epsilon_severity import ulp_at_value
from .failure_consensus_mul import (
    _bit_correctness_mul,
    _fone_predict_on_mult_holdout,
)
from .hooked_bridge import load_hooked
from .oracle import decode as fp8_decode
from .pentagon import _repo_root
from .preprocess_mult import build_table, split_train_holdout
from .mult_regimes import (
    EXACT_RESULT,
    SPECIAL_VALUES,
    REGIME_NAMES,
)
from . import tokenizer as _bit_tokenizer
from . import sem_tokenizer as _sem_tokenizer
from . import fone_tokenizer as _fone_tokenizer
from . import fone_f2_tokenizer as _fone_f2_tokenizer


# ε_mult = (m_a/8) · (m_b/8) on the discrete FP8 mantissa grid
# Possible values: {0, 1/64, 2/64, ..., 49/64}; 8x8 -> 64 raw products,
# but after dedup we get a smaller set. We bin them for the correlation.

def epsilon_mult(a_bits: int, b_bits: int) -> float:
    """Return m_a · m_b as a real in [0, 49/64]. NaN inputs -> NaN."""
    a_val, a_kind = fp8_decode(a_bits)
    b_val, b_kind = fp8_decode(b_bits)
    if a_kind == "nan" or b_kind == "nan":
        return float("nan")
    a_mant = a_bits & 0x7
    b_mant = b_bits & 0x7
    return (a_mant / 8.0) * (b_mant / 8.0)


def _bit_predict_real_values(checkpoint_path: Path, tokenizer) -> dict:
    """Forward bit/SEM model on the mult holdout; return predicted FP8 real values."""
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
    bsz = 512
    correct = np.zeros(n, dtype=bool)
    pred_bits = np.zeros(n, dtype=np.uint8)

    if tokenizer is _bit_tokenizer:
        bits_per_token = "bit"
    else:
        bits_per_token = "sem"

    for i in range(0, n, bsz):
        idx = inputs[i:i+bsz]
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)
        tgt = targets[i:i+bsz]
        pred_slice = preds[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        tgt_slice = tgt[:, pos_c_start - 1:pos_c_end - 1].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
        # Decode predicted bits.
        for j, ps in enumerate(pred_slice):
            try:
                if bits_per_token == "bit":
                    pred_bits[i + j] = tokenizer.decode_fp8_tokens(ps.tolist())
                else:  # sem
                    pred_bits[i + j] = tokenizer.decode_fp8_tokens(ps.tolist())
            except Exception:
                pred_bits[i + j] = 0  # fallback for invalid sequences
    return {"rows": rows, "correct": correct, "pred_bits": pred_bits}


def _compute_severity_bit_sem(pred_bits, rows, correct) -> dict:
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
        if true_val != 0 and pred_val != 0 and (pred_val > 0) == (true_val > 0):
            log_damage[i] = abs(math.log2(abs(pred_val)) - math.log2(abs(true_val)))
        elif true_val == 0 and pred_val == 0:
            log_damage[i] = 0.0
        else:
            log_damage[i] = float("inf")
    return {"ulp_err": ulp_err, "log_damage": log_damage}


def _compute_severity_fone(out: dict, arm: str) -> dict:
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

    for i in range(n):
        true_bits = int(rows[i]["result_bits"])
        true_val, true_kind = fp8_decode(true_bits)
        if correct[i]:
            ulp_err[i] = 0.0
            log_damage[i] = 0.0
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
        if true_val != 0 and pred_val != 0 and (pred_val > 0) == (true_val > 0):
            log_damage[i] = abs(math.log2(abs(pred_val)) - math.log2(abs(true_val)))
        elif true_val == 0 and pred_val == 0:
            log_damage[i] = 0.0
        else:
            log_damage[i] = float("inf")
    return {"ulp_err": ulp_err, "log_damage": log_damage}


def _stratify_by_epsilon_mult(rows, sev: dict, correct: np.ndarray) -> list[dict]:
    """Bin pairs by ε_mult value and compute mean severity per bin.

    Bins: {0, (0, 1/64], (1/64, 4/64], (4/64, 9/64], (9/64, 16/64], (16/64, 25/64],
           (25/64, 36/64], (36/64, 49/64]}.
    """
    edges = [0.0, 1/64, 4/64, 9/64, 16/64, 25/64, 36/64, 49/64 + 1e-9]
    labels = ["= 0", "(0, 1/64]", "(1/64, 4/64]", "(4/64, 9/64]",
              "(9/64, 16/64]", "(16/64, 25/64]", "(25/64, 36/64]", "(36/64, 49/64]"]
    out = []
    n = len(rows)
    eps_vals = np.array([epsilon_mult(int(rows[i]["a_bits"]),
                                       int(rows[i]["b_bits"]))
                          for i in range(n)])
    for k, label in enumerate(labels):
        if k == 0:
            mask = eps_vals == 0
        else:
            mask = (eps_vals > edges[k]) & (eps_vals <= edges[k + 1])
        # Restrict to non-NaN
        valid = mask & ~np.isnan(eps_vals)
        n_total = int(valid.sum())
        if n_total == 0:
            out.append({"label": label, "n_total": 0, "n_err": 0,
                        "mean_ulp_err": float("nan"),
                        "mean_log_damage": float("nan"),
                        "epsilon_repr": float("nan")})
            continue
        err_mask = valid & ~correct
        n_err = int(err_mask.sum())
        if n_err == 0:
            out.append({"label": label, "n_total": n_total, "n_err": 0,
                        "mean_ulp_err": 0.0,
                        "mean_log_damage": 0.0,
                        "epsilon_repr": float(eps_vals[valid].mean())})
            continue
        ulps = sev["ulp_err"][err_mask]
        logs = sev["log_damage"][err_mask]
        finite_ulps = ulps[np.isfinite(ulps)]
        finite_logs = logs[np.isfinite(logs)]
        out.append({
            "label": label,
            "n_total": n_total,
            "n_err": n_err,
            "mean_ulp_err": float(finite_ulps.mean()) if len(finite_ulps) else float("nan"),
            "mean_log_damage": float(finite_logs.mean()) if len(finite_logs) else float("nan"),
            "epsilon_repr": float(eps_vals[valid].mean()),
        })
    return out


def _pearson(xs, ys) -> float:
    xs = [v for v in xs if not (math.isnan(v) or math.isinf(v))]
    ys = [v for v in ys if not (math.isnan(v) or math.isinf(v))]
    n = min(len(xs), len(ys))
    if n < 3:
        return float("nan")
    x = np.asarray(xs[:n]); y = np.asarray(ys[:n])
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _per_seed_metrics(arm_name: str, prefix: str, seeds: list[int],
                      repo: Path, loader: str) -> list[dict]:
    """For each seed: ρ_full, ρ_rounding, total n_err."""
    rows_out = []
    for s in seeds:
        path = repo / f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        if not path.exists():
            print(f"  {prefix} s{s}: SKIP", flush=True)
            continue
        print(f"  {prefix} s{s}: loading...", flush=True)
        if loader == "bit":
            pred = _bit_predict_real_values(path, _bit_tokenizer)
            sev = _compute_severity_bit_sem(pred["pred_bits"], pred["rows"],
                                             pred["correct"])
        elif loader == "sem":
            pred = _bit_predict_real_values(path, _sem_tokenizer)
            sev = _compute_severity_bit_sem(pred["pred_bits"], pred["rows"],
                                             pred["correct"])
        elif loader == "f1":
            pred = _fone_predict_on_mult_holdout(path, "f1")
            sev = _compute_severity_fone(pred, "f1")
        else:  # f2
            pred = _fone_predict_on_mult_holdout(path, "f2")
            sev = _compute_severity_fone(pred, "f2")

        stats = _stratify_by_epsilon_mult(pred["rows"], sev, pred["correct"])
        eps_vals = [s_["epsilon_repr"] for s_ in stats if s_["n_err"] > 0]
        logs = [s_["mean_log_damage"] for s_ in stats if s_["n_err"] > 0]
        rho_full = _pearson(eps_vals, logs)

        # Rounding-only: exclude EXACT_RESULT and SPECIAL_VALUES regime pairs
        regime_ids = pred["rows"]["regime_id"]
        exact_mask = (regime_ids != EXACT_RESULT) & (regime_ids != SPECIAL_VALUES)
        # Re-stratify on the filtered subset
        rows_filt = pred["rows"][exact_mask]
        sev_filt = {
            "ulp_err": sev["ulp_err"][exact_mask],
            "log_damage": sev["log_damage"][exact_mask],
        }
        correct_filt = pred["correct"][exact_mask]
        stats_filt = _stratify_by_epsilon_mult(rows_filt, sev_filt, correct_filt)
        eps_vals_f = [s_["epsilon_repr"] for s_ in stats_filt if s_["n_err"] > 0]
        logs_f = [s_["mean_log_damage"] for s_ in stats_filt if s_["n_err"] > 0]
        rho_round = _pearson(eps_vals_f, logs_f)

        n_err = int((~pred["correct"]).sum())
        rows_out.append({
            "seed": s, "rho_full": rho_full, "rho_round": rho_round,
            "n_err": n_err,
        })
    return rows_out


SEEDS_DEFAULT = list(range(5))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--out", default="notes/mul_severity_findings.md")
    args = p.parse_args()
    repo = _repo_root()
    seeds = list(range(args.n_seeds))

    print("\nProbing bit-mul...")
    bit_results = _per_seed_metrics("bit", "bit-mul-L4-E048", seeds, repo, "bit")
    print("\nProbing SEM-mul...")
    sem_results = _per_seed_metrics("sem", "sem-mul-L4-E048", seeds, repo, "sem")
    print("\nProbing FoNE F1-mul...")
    f1_results = _per_seed_metrics("f1", "fone-mul-L4-E048", seeds, repo, "f1")
    print("\nProbing FoNE F2-mul...")
    f2_results = _per_seed_metrics("f2", "fone-f2-mul-L4-E048", seeds, repo, "f2")

    sections = [
        "# Multiplication severity: ε_mult = m_a · m_b stratification",
        "",
        "Per-seed Pearson(ε_mult, mean |log Δ|) at L4-E048, 5 seeds per arm.",
        "Severity binned by m_a · m_b in 8 bins. Reports two correlations:",
        "  - full: includes EXACT_RESULT pairs (which have ε_mult = 0 and contribute",
        "    severity 0 for FP-native arms — censors the signal toward 0)",
        "  - rounding-only: excludes EXACT_RESULT and SPECIAL_VALUES regimes",
        "",
        "Predictions from `notes/epsilon_under_multiplication.md`:",
        "  - FP-native (bit, SEM): +0.4 to +0.8 rounding-only ρ",
        "  - FoNE F1: -0.1 to +0.2",
        "  - FoNE F2: 0.0 to +0.3",
        "",
        "## Per-seed ρ table",
        "",
        "| arm | seed | ρ (full) | ρ (rounding-only) | n_err total |",
        "|-----|-----:|--------:|------------------:|------------:|",
    ]
    for name, results in [("bit", bit_results), ("SEM", sem_results),
                          ("FoNE F1", f1_results), ("FoNE F2", f2_results)]:
        for r in results:
            full = f"{r['rho_full']:+.3f}" if not math.isnan(r["rho_full"]) else "  nan"
            rnd = f"{r['rho_round']:+.3f}" if not math.isnan(r["rho_round"]) else "  nan"
            sections.append(f"| {name} | {r['seed']} | {full} | {rnd} | {r['n_err']} |")
    sections.append("")

    # Mean ρ per arm.
    def _mean_ignoring_nan(xs):
        finite = [v for v in xs if not math.isnan(v)]
        return float(np.mean(finite)) if finite else float("nan")

    sections.append("## Mean ρ per arm\n")
    sections.append("| arm | mean ρ (full) | mean ρ (rounding-only) |")
    sections.append("|-----|------:|------:|")
    for name, results in [("bit", bit_results), ("SEM", sem_results),
                          ("FoNE F1", f1_results), ("FoNE F2", f2_results)]:
        mfull = _mean_ignoring_nan([r["rho_full"] for r in results])
        mrnd = _mean_ignoring_nan([r["rho_round"] for r in results])
        sections.append(f"| {name} | {mfull:+.3f} | {mrnd:+.3f} |")

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
