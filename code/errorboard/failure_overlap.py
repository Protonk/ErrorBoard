"""Cross-vertex and cross-seed failure-pair overlap.

Question: when different vertices (or different seeds of the same architecture)
fail, do they fail on the same input pairs?

Two sub-experiments:

A. Cross-vertex overlap (V1-V5):
   - V1, V2, V3 are the L4-E044 trajectory snapshots (same model, different iters)
   - V3 vs V4 differs by width
   - V3 vs V5 differs by depth
   - V4 vs V5 differs by depth at same width

B. Cross-seed overlap (L4-E044 seeds 0-4 at iter 20000):
   - Same architecture, same training schedule, different initialization
   - High overlap ⟹ architecture/training trajectory dominates failure location
   - Low overlap ⟹ lottery within architectural constraint

Jaccard(A, B) = |A ∩ B| / |A ∪ B| where A, B are failure-pair sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .hooked_bridge import load_hooked
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from .tokenizer import POS_C_START, POS_C_END, encode_batch


# Vertex checkpoints (includes some redundancy with VERTICES — we want explicit seed support).
CHECKPOINTS = {
    "V1":         "runs/sweep-L4-E044-s0/checkpoint_005000.pt",
    "V2":         "runs/sweep-L4-E044-s0/checkpoint_010000.pt",
    "V3":         "runs/sweep-L4-E044-s0/checkpoint_020000.pt",  # = L4-E044-s0 final
    "V4":         "runs/sweep-L4-E128-s0/checkpoint_020000.pt",
    "V5":         "runs/sweep-L1-E128-s0/checkpoint_020000.pt",
    "L4E044-s1":  "runs/sweep-L4-E044-s1/checkpoint_020000.pt",
    "L4E044-s2":  "runs/sweep-L4-E044-s2/checkpoint_020000.pt",
    "L4E044-s3":  "runs/sweep-L4-E044-s3/checkpoint_020000.pt",
    "L4E044-s4":  "runs/sweep-L4-E044-s4/checkpoint_020000.pt",
}


@torch.no_grad()
def correctness_on_holdout(checkpoint_path: Path) -> np.ndarray:
    """Forward on full holdout; return boolean array (True = all 8 result bits correct)."""
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
    return correct


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard similarity between two failure-pair boolean arrays."""
    fails_a = ~a
    fails_b = ~b
    inter = int((fails_a & fails_b).sum())
    union = int((fails_a | fails_b).sum())
    return inter / union if union > 0 else float("nan")


def format_overlap_table(labels: list[str], correct: dict[str, np.ndarray]) -> str:
    """Build a Jaccard overlap matrix as a markdown table."""
    lines = ["| | " + " | ".join(labels) + " |"]
    lines.append("|---|" + "|".join(["---:" for _ in labels]) + "|")
    for li in labels:
        row = [li]
        for lj in labels:
            row.append(f"{jaccard(correct[li], correct[lj]):.3f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def n_failures(correct: np.ndarray) -> int:
    return int((~correct).sum())


def main() -> None:
    p = argparse.ArgumentParser()
    args = p.parse_args()
    repo = _repo_root()

    print("Running forward on each checkpoint...")
    correct = {}
    for label, path in CHECKPOINTS.items():
        print(f"  {label}: {path}")
        correct[label] = correctness_on_holdout(repo / path)

    sections = [
        "# Failure-pair overlap (cross-vertex + cross-seed)",
        "",
        "Jaccard similarity of failure-pair sets across checkpoints. 1.0 = identical",
        "failure sets; 0.0 = disjoint failure sets.",
        "",
        "## Failure counts",
        "",
        "| checkpoint | n_failures |",
        "|------------|-----------:|",
    ]
    for label in CHECKPOINTS:
        sections.append(f"| {label} | {n_failures(correct[label])} |")

    # A. Cross-vertex matrix (pentagon).
    sections.append("\n## Cross-vertex overlap (Jaccard)\n")
    vertex_labels = ["V1", "V2", "V3", "V4", "V5"]
    sections.append(format_overlap_table(vertex_labels, correct))

    # B. Cross-seed matrix (L4-E044 at iter 20000).
    sections.append("\n## Cross-seed overlap, L4-E044 iter 20000 (Jaccard)\n")
    seed_labels = ["V3", "L4E044-s1", "L4E044-s2", "L4E044-s3", "L4E044-s4"]  # V3 = seed-0
    sections.append(format_overlap_table(seed_labels, correct))

    # C. Combined matrix for cross-architecture-vs-cross-seed contrast.
    sections.append("\n## All-vs-all (for direct comparison)\n")
    all_labels = ["V1", "V2", "V3", "V4", "V5", "L4E044-s1", "L4E044-s2", "L4E044-s3", "L4E044-s4"]
    sections.append(format_overlap_table(all_labels, correct))

    sections.append("\n## Reading\n")
    sections.append(
        "- **V3 vs V1, V2** (same model, training trajectory): high Jaccard reflects\n"
        "  that V1's failure set shrinks toward V3's; V3 ⊂ V1, V2 mostly.\n"
        "- **V3 vs L4E044-s{1..4}** (same architecture, different seeds): tests whether\n"
        "  the failure set is *seed-determined* or *architecture-determined*.\n"
        "  High Jaccard ⟹ architecture wins; low Jaccard ⟹ lottery.\n"
        "- **V3 vs V4** (deeper-wider): V3's failures should mostly be a superset of V4's.\n"
        "- **V3 vs V5** (depth-1 vs depth-4): if low Jaccard, depth produces qualitatively\n"
        "  different failure patterns.\n"
        "- **L4E044 seed pairs**: matrix off-diagonal entries say whether different seeds\n"
        "  converge on the same failures.\n"
    )

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / "notes/failure_overlap_findings.md"
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
