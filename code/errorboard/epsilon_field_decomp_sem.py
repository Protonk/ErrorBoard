"""SEM-side field-decomposition probe — analog of epsilon_bit_decomp.

In bit-level, errors at endpoint m_c bins are exp-coupled (multi-bit
coordination across bits 3-6 vs bit 0). In SEM, "predict exp=8 vs exp=7"
is a single 16-way classification on one token; the binade transition is
no longer a multi-bit coordination problem.

This probe decodes each SEM checkpoint's predicted tokens at the result
positions back into 8-bit FP8 values, then classifies each error by which
field(s) went wrong — sign / exp / mantissa — and stratifies by result
mantissa bin m_c.

Format-driven prediction (from RoPE finding): endpoints (m=0/8, 7/8) remain
exp-coupled in SEM (i.e., the categorical split is FP-format property,
preserved across tokenizations).

Bit-discovery prediction: endpoints become mantissa-only in SEM — the
exp-coupling was specifically the cost of doing multi-bit coordination
on the bit-level substrate.

Usage:
    python -m errorboard.epsilon_field_decomp_sem
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .epsilon_bit_decomp import (
    CATEGORIES,
    EXP_MASK,
    MANT_MASK,
    SIGN_MASK,
    classify_error,
    format_block,
)
from .hooked_bridge import load_hooked
from .pentagon import _repo_root
from .preprocess import build_table, split_train_holdout
from . import sem_tokenizer as _sem


@torch.no_grad()
def predicted_bits_sem(checkpoint_path: Path) -> dict:
    """Forward on holdout (SEM tokens); return predicted FP8 bit patterns."""
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    hooked, gpt = load_hooked(checkpoint_path, device="cpu")
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    seqs = _sem.encode_batch(triples.astype(np.uint8))
    inputs = torch.from_numpy(seqs[:, :-1].astype(np.int64))
    targets = torch.from_numpy(seqs[:, 1:].astype(np.int64))
    pos_start = _sem.POS_C_START - 1   # offsets in target-tensor coordinates
    pos_end = _sem.POS_C_END - 1
    n = inputs.shape[0]
    bsz = 512
    pred_bits = np.zeros(n, dtype=np.uint8)
    correct = np.zeros(n, dtype=bool)
    for i in range(0, n, bsz):
        idx = inputs[i:i+bsz].to(next(hooked.parameters()).device)
        logits = hooked(idx, return_type="logits")
        preds = logits.argmax(dim=-1)
        tgt = targets[i:i+bsz].to(idx.device)
        pred_slice = preds[:, pos_start:pos_end].cpu().numpy()   # (B, 3) token ids
        tgt_slice = tgt[:, pos_start:pos_end].cpu().numpy()
        correct[i:i+bsz] = (pred_slice == tgt_slice).all(axis=1)
        # Decode each row's 3 tokens back to an 8-bit FP8 value. If the model
        # emits a token from the wrong class at a position, we treat that field
        # as the value the token's *class-conditional* offset says — but only
        # if the token is in the legal range for that field; otherwise the field
        # value is set to the maximum of its range so the bit-decomposition
        # still records "field wrong."
        for r in range(pred_slice.shape[0]):
            sign_tok = int(pred_slice[r, 0])
            exp_tok = int(pred_slice[r, 1])
            mant_tok = int(pred_slice[r, 2])
            sign = sign_tok - _sem.SIGN_BASE if _sem.is_sign_token(sign_tok) else 0
            exp = exp_tok - _sem.EXP_BASE if _sem.is_exp_token(exp_tok) else 0xF
            mant = mant_tok - _sem.MANT_BASE if _sem.is_mant_token(mant_tok) else 0x7
            # Clamp into legal ranges defensively.
            sign &= 0x1
            exp &= 0xF
            mant &= 0x7
            pred_bits[i + r] = (sign << 7) | (exp << 3) | mant
    return {"rows": rows, "pred_bits": pred_bits, "correct": correct}


def stratify_errors(rows, pred_bits, correct) -> dict:
    """Same stratification as bit-level: by result m_c, for normal-result pairs."""
    result_bits = rows["result_bits"].astype(int)
    result_exp = (result_bits >> 3) & 0xF
    result_m = result_bits & 0x7
    is_normal = (result_exp >= 1) & ~((result_bits == 0x7F) | (result_bits == 0xFF))
    out = {m: {cat: 0 for cat in CATEGORIES + ["total_err", "total_n"]} for m in range(8)}
    for i in range(len(rows)):
        if not is_normal[i]:
            continue
        m_val = result_m[i]
        out[m_val]["total_n"] += 1
        if correct[i]:
            continue
        cat = classify_error(int(pred_bits[i]), int(result_bits[i]))
        if cat != "correct":
            out[m_val][cat] += 1
            out[m_val]["total_err"] += 1
    return out


DEFAULT_RUNS = [
    ("learned-PE SEM s0  (L4-E048)", "runs/sem-L4-E048-s0/checkpoint_020000.pt"),
    ("learned-PE SEM s8  (L4-E048)", "runs/sem-L4-E048-s8/checkpoint_020000.pt"),
    ("learned-PE SEM s14 (L4-E048)", "runs/sem-L4-E048-s14/checkpoint_020000.pt"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="notes/sem_field_decomp_findings.md")
    args = p.parse_args()
    repo = _repo_root()

    sections = [
        "# SEM field-decomposition probe",
        "",
        "Analog of `epsilon_bit_decomp.py` for SEM tokenization. Decodes each",
        "checkpoint's predicted tokens (sign, exp, mant) into an 8-bit FP8 value,",
        "then classifies each error by which fields differ from the true value.",
        "",
        "**Tests:** does the smooth-interior-mant-only / endpoint-exp-coupled",
        "categorical split survive when field structure is given as a prior?",
        "If yes, the split is format-driven. If endpoints become mant-only in SEM,",
        "the bit-level exp-coupling was a coordination cost of multi-bit decoding.",
    ]
    for vname, vpath in DEFAULT_RUNS:
        print(f"Running {vname}...", flush=True)
        out = predicted_bits_sem(repo / vpath)
        stats = stratify_errors(out["rows"], out["pred_bits"], out["correct"])
        sections.append(format_block(vname, stats))

    output = "\n".join(sections)
    print("\n" + output)
    out_path = repo / args.out
    out_path.write_text(output)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
