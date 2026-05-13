"""SEM-side severity probe — value-level ULP and log-damage.

Analog of `epsilon_severity.py` for SEM checkpoints. Severity is computed in
value space (decoded FP8 reals), so the metric is identical to bit-level —
only the prediction-decoding step changes (3 tokens → 8 bits → FP8 value).

Tests whether the anti-ε severity correlation (Pearson(ε, mean |log Δ|)
ranging −0.68 to −0.87 across bit-level checkpoints) survives, flattens,
or inverts under SEM.

Usage:
    python -m errorboard.epsilon_severity_sem
"""

from __future__ import annotations

import argparse

from .epsilon_severity import compute_severity, format_block, pearson, stratify
from .epsilon_field_decomp_sem import predicted_bits_sem
from .pentagon import _repo_root


DEFAULT_RUNS = [
    ("learned-PE SEM s0  (L4-E048)", "runs/sem-L4-E048-s0/checkpoint_020000.pt"),
    ("learned-PE SEM s8  (L4-E048)", "runs/sem-L4-E048-s8/checkpoint_020000.pt"),
    ("learned-PE SEM s14 (L4-E048)", "runs/sem-L4-E048-s14/checkpoint_020000.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/sem_severity_findings.md")
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# SEM severity probe",
        "",
        "Analog of `epsilon_severity.py` for SEM checkpoints. Severity (ULP / log-damage)",
        "is value-level so the metric is identical; only the prediction-decoding step changes.",
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
        out = predicted_bits_sem(repo / vpath)
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
