"""ε-severity under matched-arch PE comparison.

Same probe as epsilon_severity.py but on configurable checkpoints. Checks
whether the anti-ε correlation (Pearson(ε, mean |log Δ|) ≈ -0.80 at V3)
survives under RoPE on matched L4-E048 architecture.

Usage:
    python -m errorboard.epsilon_severity_pe
"""

from __future__ import annotations

import argparse

from .epsilon_severity import (
    compute_severity,
    format_block,
    pearson,
    severity_per_pair,
    stratify,
)
from .pentagon import _repo_root


DEFAULT_RUNS = [
    ("learned-PE s0 (L4-E048)", "runs/sweep-L4-E048-s0/checkpoint_020000.pt"),
    ("RoPE        s0 (L4-E048)", "runs/rope-L4-E048-s0/checkpoint_020000.pt"),
    ("learned-PE s14 (L4-E048)", "runs/sweep-L4-E048-s14/checkpoint_020000.pt"),
    ("RoPE        s8 (L4-E048)", "runs/rope-L4-E048-s8/checkpoint_020000.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/epsilon_severity_pe_findings.md")
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# ε-severity: PE arm comparison",
        "",
        "Same probe as `epsilon_severity.py` but on matched-architecture (L4-E048)",
        "learned-PE and RoPE checkpoints.",
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
        out = severity_per_pair(repo / vpath)
        sev = compute_severity(out["rows"], out["pred_bits"], out["correct"])
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
