"""ε-bit-decomposition under matched-arch PE comparison.

Runs the same bit-decomposition probe as epsilon_bit_decomp.py but on a
configurable list of checkpoints — used to compare a matched learned-PE
checkpoint and a RoPE checkpoint at the same L4-E048 architecture.

The dual-of-ε reading predicts that the smooth-interior-mant-only vs
endpoint-exp-coupled split is format-driven (FP mantissa structure), not
PE-driven. If that's right, the per-m_c shape should look similar across
both arms.

Usage:
    python -m errorboard.epsilon_bit_decomp_pe
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .epsilon_bit_decomp import predicted_bits, stratify_errors, format_block
from .pentagon import _repo_root


# Sentinel set: one seed per arm, matched architecture (L4-E048).
DEFAULT_RUNS = [
    ("learned-PE s0 (L4-E048)", "runs/sweep-L4-E048-s0/checkpoint_020000.pt"),
    ("RoPE        s0 (L4-E048)", "runs/rope-L4-E048-s0/checkpoint_020000.pt"),
    ("learned-PE s14 (L4-E048)", "runs/sweep-L4-E048-s14/checkpoint_020000.pt"),
    ("RoPE        s8 (L4-E048)", "runs/rope-L4-E048-s8/checkpoint_020000.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/epsilon_bit_decomp_pe_findings.md")
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# ε-bit-decomposition: PE arm comparison",
        "",
        "Same probe as `epsilon_bit_decomp.py` but on matched-architecture",
        "checkpoints (L4-E048) with learned-PE vs RoPE.",
        "",
        "**Format-driven prediction:** smooth-interior mant-only / endpoint-exp-coupled",
        "split survives across both PE arms.",
        "**PE-driven prediction:** the split changes shape under RoPE (eg endpoints",
        "become mant-only-dominated, or smooth-interior gains exp errors).",
    ]
    for vname, vpath in DEFAULT_RUNS:
        print(f"Running {vname}...", flush=True)
        out = predicted_bits(repo / vpath)
        stats = stratify_errors(out["rows"], out["pred_bits"], out["correct"])
        sections.append(format_block(vname, stats))

    output = "\n".join(sections)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")
    print("\n" + output)


if __name__ == "__main__":
    main()
