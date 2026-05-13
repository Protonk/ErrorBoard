"""FoNE capacity → anti-ε transition probe.

Sweeps {L4-E048, L4-E064, L4-E096, L4-E128} × {seeds 0..4} and reports:
  - per-seed Pearson(ε, mean |log Δ|)
  - per-arm lottery stats (mean fail rate, var ratio, structural easy/core,
    p̂ histogram)
  - per-arm per-regime accuracy

Output: `notes/fone_transition_findings.md`.

Usage:
    python -m errorboard.fone_transition
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .epsilon_severity import format_block, pearson, stratify
from .epsilon_severity_fone import _fone_severity_from_predictions
from .failure_consensus_fone import _load_fone_arm, _arm_stats
from .fone_model import fone_predict_on_holdout
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .regimes import REGIME_NAMES


SIZES = [48, 64, 96, 128]
SEEDS = [0, 1, 2, 3, 4]


def _per_seed_pearson(prefix: str, seeds: list[int], repo: Path) -> list[tuple[int, float, float, int]]:
    """Returns list of (seed, ρ_ULP, ρ_log, n_err_total)."""
    rows = []
    for s in seeds:
        path = repo / f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        if not path.exists():
            print(f"  {prefix} s{s}: SKIP (not found)", flush=True)
            continue
        print(f"  {prefix} s{s}: loading...", flush=True)
        out = fone_predict_on_holdout(path, device="cpu")
        sev = _fone_severity_from_predictions(out)
        stats = stratify(out["rows"], sev, out["correct"])
        eps = [stats[m]["epsilon"] for m in range(8) if stats[m]["n_err"] > 0]
        mean_ulps = [stats[m]["mean_ulp_err"] for m in range(8) if stats[m]["n_err"] > 0]
        mean_logs = [stats[m]["mean_log_damage"] for m in range(8) if stats[m]["n_err"] > 0]
        n_err = sum(stats[m]["n_err"] for m in range(8))
        rho_ulp = pearson(eps, mean_ulps)
        rho_log = pearson(eps, mean_logs)
        rows.append((s, rho_ulp, rho_log, n_err))
    return rows


def _per_regime_acc(prefix: str, seeds: list[int], repo: Path,
                     regime_ids: np.ndarray) -> dict[str, list[float]]:
    """For each regime, per-seed accuracy across the listed seeds."""
    out: dict[str, list[float]] = {name: [] for name in REGIME_NAMES}
    for s in seeds:
        path = repo / f"runs/{prefix}-s{s}/checkpoint_020000.pt"
        if not path.exists():
            continue
        pred = fone_predict_on_holdout(path, device="cpu")
        correct = pred["correct"]
        for ri, name in enumerate(REGIME_NAMES):
            mask = (regime_ids == ri)
            if mask.sum() == 0:
                continue
            acc = correct[mask].mean()
            out[name].append(float(acc))
    return out


def _params_for_embd(n_embd: int, n_head: int = 4, d_mlp_ratio: int = 4,
                     vocab: int = 10, block: int = 9, n_layer: int = 4) -> int:
    """Rough parameter count for an L=n_layer model."""
    d_mlp = d_mlp_ratio * n_embd
    # embeddings: wte + wpe
    p = vocab * n_embd + block * n_embd
    # per block: c_attn (3*E,E), c_proj (E,E), c_fc (d_mlp,E), c_proj (E,d_mlp), 2x RMSNorm (E)
    per_block = 3 * n_embd * n_embd + n_embd * n_embd + n_embd * d_mlp + d_mlp * n_embd + 2 * n_embd
    p += n_layer * per_block
    # final norm + lm_head
    p += n_embd
    p += n_embd * vocab
    return p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/fone_transition_findings.md")
    args = p.parse_args()
    repo = _repo_root()

    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    regime_ids = rows["regime_id"]

    sections = [
        "# FoNE capacity → anti-ε transition",
        "",
        "Sweeps L=4 with `n_embd ∈ {48, 64, 96, 128}` at 5 matched seeds each.",
        "Tests whether the L4-E048 anti-ε sign-flip resolves as a sharp threshold,",
        "a gradual shift in seed distribution, or persists at intermediate scales.",
        "",
    ]

    # Per-seed Pearson table.
    sections.append("## Per-seed anti-ε Pearson (ε, mean |log Δ|)\n")
    sections.append("| n_embd | params | s0 | s1 | s2 | s3 | s4 |")
    sections.append("|-------:|-------:|---:|---:|---:|---:|---:|")
    all_rhos: dict[int, list[tuple[int, float]]] = {}
    n_err_table: dict[int, list[tuple[int, int]]] = {}
    for n_embd in SIZES:
        prefix = f"fone-L4-E{n_embd:03d}"
        print(f"\nProbing {prefix}...")
        per_seed = _per_seed_pearson(prefix, SEEDS, repo)
        params = _params_for_embd(n_embd)
        all_rhos[n_embd] = [(s, r_log) for s, _, r_log, _ in per_seed]
        n_err_table[n_embd] = [(s, n_err) for s, _, _, n_err in per_seed]
        rho_by_seed = {s: r_log for s, _, r_log, _ in per_seed}
        cells = []
        for s in SEEDS:
            if s in rho_by_seed:
                cells.append(f"{rho_by_seed[s]:+.2f}")
            else:
                cells.append("—")
        sections.append(f"| {n_embd} | {params:,} | " + " | ".join(cells) + " |")
    sections.append("")

    sections.append("\n## Per-seed n_err (normal-result, summed across m_c)\n")
    sections.append("| n_embd | s0 | s1 | s2 | s3 | s4 | total |")
    sections.append("|-------:|---:|---:|---:|---:|---:|------:|")
    for n_embd in SIZES:
        by_seed = {s: n for s, n in n_err_table.get(n_embd, [])}
        cells = []
        total = 0
        for s in SEEDS:
            if s in by_seed:
                cells.append(f"{by_seed[s]}")
                total += by_seed[s]
            else:
                cells.append("—")
        sections.append(f"| {n_embd} | " + " | ".join(cells) + f" | {total} |")
    sections.append("")

    # Distribution summary.
    sections.append("\n## Anti-ε sign distribution across seeds\n")
    sections.append("| n_embd | n seeds with ρ < −0.5 | n with ρ ∈ [−0.5, +0.5] | n with ρ > +0.5 |")
    sections.append("|-------:|----------------------:|------------------------:|----------------:|")
    for n_embd in SIZES:
        rhos = [r for _, r in all_rhos.get(n_embd, [])]
        n_neg = sum(1 for r in rhos if r < -0.5)
        n_mid = sum(1 for r in rhos if -0.5 <= r <= 0.5)
        n_pos = sum(1 for r in rhos if r > 0.5)
        sections.append(f"| {n_embd} | {n_neg} | {n_mid} | {n_pos} |")
    sections.append("")

    # Lottery stats per arm.
    sections.append("\n## Lottery shape across capacity\n")
    sections.append("| n_embd | mean fail % | var ratio | structural easy % | "
                    "lottery % | structural core (pairs) |")
    sections.append("|-------:|------------:|----------:|------------------:|"
                    "----------:|-----------------------:|")
    for n_embd in SIZES:
        prefix = f"fone-L4-E{n_embd:03d}"
        corr = _load_fone_arm(prefix, len(SEEDS), repo)
        st = _arm_stats(corr)
        p_mean = st["p_mean"]
        iid = st["n_seeds"] * p_mean * (1 - p_mean)
        vr = st["var_obs"] / iid if iid > 0 else float("nan")
        easy = (st["fail_count"] == 0).mean() * 100
        lot = ((st["fail_count"] >= 1) &
               (st["fail_count"] <= st["n_seeds"] - 1)).mean() * 100
        core = int((st["fail_count"] == st["n_seeds"]).sum())
        sections.append(f"| {n_embd} | {p_mean*100:.2f}% | {vr:.2f} | "
                        f"{easy:.1f}% | {lot:.1f}% | {core} |")
    sections.append("")

    # Per-regime accuracy across arms.
    sections.append("\n## Per-regime mean accuracy (5 seeds each)\n")
    header = "| regime |" + "".join(f" E={n} |" for n in SIZES)
    sections.append(header)
    sections.append("|--------|" + "------|" * len(SIZES))
    per_regime_by_size = {}
    for n_embd in SIZES:
        prefix = f"fone-L4-E{n_embd:03d}"
        per_regime_by_size[n_embd] = _per_regime_acc(prefix, SEEDS, repo, regime_ids)
    for name in REGIME_NAMES:
        row = f"| {name} |"
        for n_embd in SIZES:
            vals = per_regime_by_size[n_embd].get(name, [])
            if not vals:
                row += " — |"
                continue
            row += f" {np.mean(vals)*100:.2f}% |"
        sections.append(row)
    sections.append("")

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
