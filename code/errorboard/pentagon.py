"""Run the three first-pass probes on all 5 pentagon checkpoints and emit
cross-vertex comparison tables.

Per-vertex full dumps (attention patterns, DLA, commutativity, Thm 5 residuals)
are saved to runs/pentagon/V{N}_full.txt for browsing. Raw numpy arrays are
saved to runs/pentagon/V{N}.npz for follow-up analysis. The cross-vertex tables
print to stdout and also save to notes/pentagon_findings.md.

Usage:
    python -m errorboard.pentagon [--regime cancellation] [--n-samples 64]
"""

from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

from .hooked_bridge import load_hooked
from .inspection import (
    attention_at_result_positions,
    commutativity_check,
    direct_layer_attribution,
    operand_residual_similarity,
    print_attention_summary,
    print_attribution_summary,
    print_equality_summary,
    _doubling_holdout,
    _sample_regime,
)


VERTICES = [
    ("V1", "runs/sweep-L4-E044-s0/checkpoint_005000.pt", "early borderline (iter 5k)"),
    ("V2", "runs/sweep-L4-E044-s0/checkpoint_010000.pt", "mid borderline (iter 10k)"),
    ("V3", "runs/sweep-L4-E044-s0/checkpoint_020000.pt", "late borderline (iter 20k)"),
    ("V4", "runs/sweep-L4-E128-s0/checkpoint_020000.pt", "saturated (L4-E128, iter 20k)"),
    ("V5", "runs/sweep-L1-E128-s0/checkpoint_020000.pt", "depth-capped (L1-E128, iter 20k)"),
]

BIT_NAMES = ["sign", "e3", "e2", "e1", "e0", "m2", "m1", "m0"]


def run_one(vpath: Path, regime: str, n_samples: int):
    hooked, gpt = load_hooked(vpath, device="cpu")
    inputs, targets = _sample_regime(regime, n_samples)
    dbl_in, _ = _doubling_holdout(n_samples)
    att = attention_at_result_positions(hooked, inputs)
    attr = direct_layer_attribution(hooked, inputs, targets)
    comm = commutativity_check(hooked, inputs, targets)
    sim = operand_residual_similarity(hooked, dbl_in)
    return {
        "params": sum(p.numel() for p in gpt.parameters()),
        "L": gpt.config.n_layer,
        "E": gpt.config.n_embd,
        "att": att,
        "attr": attr,
        "comm": comm,
        "sim": sim,
    }


def capture_per_vertex_text(result, label: str) -> str:
    """Re-run the inspection print_* helpers, capture to string."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"=== {label} ===")
        print(f"  params: {result['params']:,}  L={result['L']}  E={result['E']}")
        print_attention_summary(result["att"], top_k=4)
        print_attribution_summary(result["attr"])
        print_equality_summary(result["comm"], result["sim"])
    return buf.getvalue()


def cross_vertex_commutativity(results: dict) -> str:
    lines = ["## Per-bit commutativity  f(a, b) == f(b, a)\n"]
    lines.append("| bit  | " + " | ".join(v for v in results) + " |")
    lines.append("|------|" + "|".join("------" for _ in results) + "|")
    for bi, bn in enumerate(BIT_NAMES):
        row = "  ".join(
            f"{results[v]['comm']['per_position_agreement'][bi]:.3f}"
            for v in results
        )
        lines.append(f"| {bn:<4} | " + " | ".join(
            f"{results[v]['comm']['per_position_agreement'][bi]:.3f}"
            for v in results
        ) + " |")
    lines.append("")
    lines.append("Full-result agreement (all 8 bits):")
    lines.append("| metric    | " + " | ".join(v for v in results) + " |")
    lines.append("|-----------|" + "|".join("------" for _ in results) + "|")
    lines.append("| full-comm | " + " | ".join(
        f"{results[v]['comm']['full_result_agreement']:.3f}" for v in results
    ) + " |")
    return "\n".join(lines)


def cross_vertex_thm5(results: dict) -> str:
    lines = ["## Park Thm 5 — operand-a/b residual cosine sim at final layer (doubling inputs)\n"]
    lines.append("| bit  | " + " | ".join(v for v in results) + " |")
    lines.append("|------|" + "|".join("------" for _ in results) + "|")
    for bi, bn in enumerate(BIT_NAMES):
        row_vals = [results[v]["sim"][-1, bi] for v in results]
        lines.append(f"| {bn:<4} | " + " | ".join(f"{x:+.2f}" for x in row_vals) + " |")
    return "\n".join(lines)


def cross_vertex_dla(results: dict) -> str:
    """Per-block contribution to logit at correct class, mean over batch, summed
    across the 8 result bits. Cells show 'block i mean abs contribution'."""
    lines = ["## Direct logit attribution by block (mean over batch, sum over 8 result bits)\n"]
    max_blocks = max(results[v]["L"] for v in results)
    header_stages = ["embed"] + [f"blk{i}" for i in range(max_blocks)]
    lines.append("| vertex | " + " | ".join(header_stages) + " |   final |")
    lines.append("|--------|" + "|".join("------" for _ in header_stages) + "|---------|")
    for v in results:
        contrib = results[v]["attr"]["contributions"]  # (n_stages, B, 8)
        # Sum over result bits, mean over batch
        per_stage = contrib.mean(axis=1).sum(axis=-1)   # (n_stages,)
        final = results[v]["attr"]["final_logits"].mean(axis=0).sum()
        row_cells = []
        for si in range(len(header_stages)):
            if si < len(per_stage):
                row_cells.append(f"{per_stage[si]:+.1f}")
            else:
                row_cells.append("   —  ")
        lines.append(f"| {v}     | " + " | ".join(row_cells) + f" | {final:+.1f}  |")
    return "\n".join(lines)


def cross_vertex_attention_sharpness(results: dict) -> str:
    """Per-vertex attention sharpness summary: at the final layer, average over
    heads and result-predict positions, what's the max attention weight?
    Sharpness ≈ 1 means very peaked; sharpness ≈ 1/T means uniform."""
    lines = ["## Final-layer attention sharpness (max weight at result-predictor positions)\n"]
    lines.append("| vertex | n_layer | mean_max_attn |  uniform_floor |")
    lines.append("|--------|--------:|--------------:|---------------:|")
    for v in results:
        att = results[v]["att"]  # (n_layer, n_head, 8, T)
        final_layer = att[-1]    # (n_head, 8, T)
        mean_max = final_layer.max(axis=-1).mean()
        floor = 1.0 / att.shape[-1]
        lines.append(f"| {v}     | {results[v]['L']:>7} | {mean_max:>13.3f} | {floor:>14.3f} |")
    return "\n".join(lines)


def _repo_root() -> Path:
    """Resolve repo root from this file's location: code/errorboard/pentagon.py."""
    return Path(__file__).resolve().parent.parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="cancellation")
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--out-dir", default=None,
                   help="Output directory (default: <repo>/runs/pentagon)")
    args = p.parse_args()

    repo = _repo_root()
    out_dir = Path(args.out_dir) if args.out_dir else repo / "runs" / "pentagon"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for vname, vpath, vdesc in VERTICES:
        full_path = repo / vpath
        print(f"\nRunning {vname}: {full_path} ({vdesc})")
        results[vname] = run_one(full_path, args.regime, args.n_samples)

        # Save per-vertex full dump.
        txt = capture_per_vertex_text(results[vname], f"{vname}: {vdesc}")
        (out_dir / f"{vname}_full.txt").write_text(txt)

        # Save raw numpy arrays for later analysis.
        np.savez_compressed(
            out_dir / f"{vname}.npz",
            attention=results[vname]["att"],
            contributions=results[vname]["attr"]["contributions"],
            cumulative=results[vname]["attr"]["cumulative"],
            final_logits=results[vname]["attr"]["final_logits"],
            similarity=results[vname]["sim"],
            comm_per_pos=results[vname]["comm"]["per_position_agreement"],
            comm_full=results[vname]["comm"]["full_result_agreement"],
            params=results[vname]["params"],
            L=results[vname]["L"],
            E=results[vname]["E"],
        )

    # Cross-vertex tables.
    sections = [
        f"# Pentagon inspection findings — regime '{args.regime}', n={args.n_samples}",
        "",
        "## Vertices",
        "",
        "| label | path | description | L | E | params |",
        "|-------|------|-------------|--:|--:|-------:|",
    ]
    for vname, vpath, vdesc in VERTICES:
        r = results[vname]
        sections.append(
            f"| {vname} | `{vpath}` | {vdesc} | {r['L']} | {r['E']} | {r['params']:,} |"
        )
    sections.extend([
        "",
        cross_vertex_commutativity(results),
        "",
        cross_vertex_thm5(results),
        "",
        cross_vertex_dla(results),
        "",
        cross_vertex_attention_sharpness(results),
        "",
        f"\nPer-vertex full dumps: `{out_dir}/V*_full.txt`",
        f"Raw numpy arrays:      `{out_dir}/V*.npz`",
    ])
    output = "\n".join(sections)
    print("\n" + output)

    findings_path = repo / "notes/pentagon_findings.md"
    findings_path.write_text(output)
    print(f"\nWrote {findings_path}")


if __name__ == "__main__":
    main()
