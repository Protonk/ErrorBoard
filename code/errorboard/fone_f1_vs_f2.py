"""F1 (decimal FoNE) vs F2 (binary FoNE) head-to-head comparison.

Loads F1 and F2 checkpoints at matched sizes, runs severity + lottery +
per-regime accuracy probes on both, and reports a side-by-side table.

The test: does aligning FoNE's period basis with FP8's binade structure
(T_i = 2^i) recover anti-ε severity at smaller capacity than the canonical
base-10 variant?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .epsilon_severity import format_block, pearson, stratify
from .epsilon_severity_fone import _fone_severity_from_predictions
from .fone_f2_model import fone_f2_predict_on_holdout
from .fone_f2_tokenizer import (
    SIGN_NAN_ID as F2_SIGN_NAN_ID,
    SIGN_NEG_ID as F2_SIGN_NEG_ID,
    digits_to_magnitude as f2_digits_to_magnitude,
)
from .fone_model import fone_predict_on_holdout
from .oracle import decode as fp8_decode
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES


def _f2_severity_from_predictions(out: dict) -> dict:
    """Same shape as F1's severity computation, using F2's digits_to_magnitude."""
    import math
    from .epsilon_severity import ulp_at_value

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
            continue
        sign_id = int(sign_pred[i])
        if sign_id == F2_SIGN_NAN_ID:
            log_damage[i] = float("inf")
            continue
        mag = f2_digits_to_magnitude(list(digit_pred[i]))
        pred_val = -mag if sign_id == F2_SIGN_NEG_ID else mag

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


def _per_seed_metrics(prefix: str, seeds: list[int], repo: Path,
                       arm: str) -> list[dict]:
    """Returns list of dicts {seed, rho_log, rho_ulp, n_err, default_acc}."""
    out = []
    for s in seeds:
        path = repo / f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        if not path.exists():
            print(f"  {prefix} s{s}: SKIP (not found)", flush=True)
            continue
        print(f"  {prefix} s{s}: loading...", flush=True)
        if arm == "f1":
            pred = fone_predict_on_holdout(path, device="cpu")
            sev = _fone_severity_from_predictions(pred)
        else:
            pred = fone_f2_predict_on_holdout(path, device="cpu")
            sev = _f2_severity_from_predictions(pred)
        stats = stratify(pred["rows"], sev, pred["correct"])
        eps = [stats[m]["epsilon"] for m in range(8) if stats[m]["n_err"] > 0]
        mean_ulps = [stats[m]["mean_ulp_err"] for m in range(8) if stats[m]["n_err"] > 0]
        mean_logs = [stats[m]["mean_log_damage"] for m in range(8) if stats[m]["n_err"] > 0]
        n_err = sum(stats[m]["n_err"] for m in range(8))

        # Per-regime accuracy.
        regime_ids = pred["rows"]["regime_id"]
        default_idx = REGIME_NAMES.index("default")
        default_mask = (regime_ids == default_idx)
        default_acc = float(pred["correct"][default_mask].mean()) if default_mask.sum() else float("nan")

        out.append({
            "seed": s,
            "rho_log": pearson(eps, mean_logs),
            "rho_ulp": pearson(eps, mean_ulps),
            "n_err": n_err,
            "default_acc": default_acc,
        })
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/fone_f1_vs_f2_findings.md")
    p.add_argument("--seeds", type=int, default=5)
    args = p.parse_args()
    repo = _repo_root()
    seeds = list(range(args.seeds))

    sections = [
        "# F1 (decimal FoNE) vs F2 (binary FoNE) — matched comparison",
        "",
        "L=4, matched seeds 0..4, learned PE, 20k iters. F1 periods are",
        "`T_i = 10^i for i ∈ [-2, 3]` (6 decimal digits, FONE_DIM=12). F2",
        "periods are `T_i = 2^i for i ∈ [-8, 9]` (18 binary digits, FONE_DIM=36).",
        "",
        "**Test:** Does aligning FoNE's period basis with FP8's binade",
        "structure recover anti-ε severity at smaller capacity?",
        "",
    ]

    sizes_to_compare = [(48, "f1"), (48, "f2"), (128, "f1"), (128, "f2")]
    results = {}
    for size, arm in sizes_to_compare:
        prefix = f"fone-L4-E{size:03d}" if arm == "f1" else f"fone-f2-L4-E{size:03d}"
        print(f"\nProbing {prefix} ({arm}, E={size})...")
        rows = _per_seed_metrics(prefix, seeds, repo, arm)
        results[(size, arm)] = rows

    # Per-seed table.
    sections.append("## Per-seed Pearson(ε, mean |log Δ|)\n")
    sections.append("| size | arm | s0 | s1 | s2 | s3 | s4 | mean ρ |")
    sections.append("|-----:|-----|----:|----:|----:|----:|----:|------:|")
    for size, arm in sizes_to_compare:
        rows = results[(size, arm)]
        by_seed = {r["seed"]: r["rho_log"] for r in rows}
        cells = []
        rhos = []
        for s in seeds:
            if s in by_seed:
                cells.append(f"{by_seed[s]:+.2f}")
                rhos.append(by_seed[s])
            else:
                cells.append("—")
        mean_r = np.mean(rhos) if rhos else float("nan")
        sections.append(f"| {size} | {arm} | " + " | ".join(cells) + f" | {mean_r:+.2f} |")
    sections.append("")

    # n_err totals and default-acc.
    sections.append("\n## Total errors + default-regime accuracy\n")
    sections.append("| size | arm | total n_err | mean default acc |")
    sections.append("|-----:|-----|------------:|----------------:|")
    for size, arm in sizes_to_compare:
        rows = results[(size, arm)]
        if not rows:
            continue
        total_err = sum(r["n_err"] for r in rows)
        mean_def = np.mean([r["default_acc"] for r in rows])
        sections.append(f"| {size} | {arm} | {total_err} | {mean_def*100:.2f}% |")
    sections.append("")

    # Sign distribution.
    sections.append("\n## Anti-ε sign distribution\n")
    sections.append("| size | arm | strongly negative (ρ < −0.5) | flat (\\|ρ\\| ≤ 0.5) | strongly positive (ρ > +0.5) |")
    sections.append("|-----:|-----|---:|---:|---:|")
    for size, arm in sizes_to_compare:
        rows = results[(size, arm)]
        rhos = [r["rho_log"] for r in rows]
        n_neg = sum(1 for r in rhos if r < -0.5)
        n_mid = sum(1 for r in rhos if -0.5 <= r <= 0.5)
        n_pos = sum(1 for r in rhos if r > 0.5)
        sections.append(f"| {size} | {arm} | {n_neg} | {n_mid} | {n_pos} |")
    sections.append("")

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
