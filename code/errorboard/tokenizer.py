"""Bit-level tokenizer for E4M3 add per task_spec.md §1.

12-token vocabulary, fixed 28-token sequences. MSB-first within each FP8 number.

Layout:
    pos 0     : <bos>
    pos 1..8  : operand a  (sign, e3, e2, e1, e0, m2, m1, m0)
    pos 9     : `+`
    pos 10..17: operand b  (same layout)
    pos 18    : `=`
    pos 19..26: result c   (same layout)
    pos 27    : <eos>
"""

from __future__ import annotations

from typing import Final

import numpy as np

# ---- vocabulary ----

BOS_ID: Final[int] = 0
EOS_ID: Final[int] = 1
PAD_ID: Final[int] = 2          # reserved, unused in v1 fixed-length
SCRATCH_ID: Final[int] = 3      # reserved, unused in v1
PLUS_ID: Final[int] = 4
EQ_ID: Final[int] = 5
SIGN_0_ID: Final[int] = 6
SIGN_1_ID: Final[int] = 7
EXP_0_ID: Final[int] = 8
EXP_1_ID: Final[int] = 9
MANT_0_ID: Final[int] = 10
MANT_1_ID: Final[int] = 11

VOCAB_SIZE: Final[int] = 12
SEQ_LEN: Final[int] = 28

# ---- positions (in the 28-token sequence) ----

POS_BOS: Final[int] = 0
POS_A_START: Final[int] = 1
POS_A_END: Final[int] = 9          # exclusive
POS_PLUS: Final[int] = 9
POS_B_START: Final[int] = 10
POS_B_END: Final[int] = 18         # exclusive
POS_EQ: Final[int] = 18
POS_C_START: Final[int] = 19
POS_C_END: Final[int] = 27         # exclusive
POS_EOS: Final[int] = 27

# Target indices in the shifted-target tensor (target[i] = sequence[i + 1]) where the
# model is predicting result-c tokens. Used for result-only eval loss/accuracy.
RESULT_TARGET_POSITIONS: Final[np.ndarray] = np.arange(POS_C_START - 1, POS_C_END - 1)


# ---- single-number encode / decode ----

def encode_fp8_bits(bits: int) -> list[int]:
    """Encode an 8-bit E4M3 pattern as 8 tokens, MSB-first (sign, e3..e0, m2..m0)."""
    if not (0 <= bits <= 0xFF):
        raise ValueError(f"bits must be in [0, 255], got {bits}")
    return [
        SIGN_1_ID if (bits >> 7) & 1 else SIGN_0_ID,
        EXP_1_ID if (bits >> 6) & 1 else EXP_0_ID,
        EXP_1_ID if (bits >> 5) & 1 else EXP_0_ID,
        EXP_1_ID if (bits >> 4) & 1 else EXP_0_ID,
        EXP_1_ID if (bits >> 3) & 1 else EXP_0_ID,
        MANT_1_ID if (bits >> 2) & 1 else MANT_0_ID,
        MANT_1_ID if (bits >> 1) & 1 else MANT_0_ID,
        MANT_1_ID if bits & 1 else MANT_0_ID,
    ]


def decode_fp8_tokens(tokens) -> int:
    """Decode 8 tokens (one number) back to an FP8 bit pattern. Accepts list/tuple/array."""
    if len(tokens) != 8:
        raise ValueError(f"expected 8 tokens, got {len(tokens)}")
    bits = 0
    bits |= (1 if int(tokens[0]) == SIGN_1_ID else 0) << 7
    for i in range(4):
        if int(tokens[1 + i]) == EXP_1_ID:
            bits |= 1 << (6 - i)
    for i in range(3):
        if int(tokens[5 + i]) == MANT_1_ID:
            bits |= 1 << (2 - i)
    return bits


# ---- full-sequence encode / decode ----

def encode_sequence(a_bits: int, b_bits: int, c_bits: int) -> list[int]:
    """Encode (a, b, c) triple as a fixed 28-token sequence."""
    return (
        [BOS_ID]
        + encode_fp8_bits(a_bits)
        + [PLUS_ID]
        + encode_fp8_bits(b_bits)
        + [EQ_ID]
        + encode_fp8_bits(c_bits)
        + [EOS_ID]
    )


def decode_result_from_sequence(seq) -> int:
    """Extract result FP8 bits from a 28-token sequence (positions 19..26)."""
    if len(seq) != SEQ_LEN:
        raise ValueError(f"expected {SEQ_LEN} tokens, got {len(seq)}")
    return decode_fp8_tokens(seq[POS_C_START:POS_C_END])


def encode_batch(triples: np.ndarray) -> np.ndarray:
    """Encode an (N, 3) array of (a, b, c) bit patterns into an (N, SEQ_LEN) int64 tensor."""
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {triples.shape}")
    n = len(triples)
    out = np.empty((n, SEQ_LEN), dtype=np.int64)
    for i in range(n):
        out[i] = encode_sequence(int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2]))
    return out


def _spot_checks() -> None:
    # Single-number round-trip for all 256 patterns
    for bits in range(256):
        toks = encode_fp8_bits(bits)
        assert len(toks) == 8
        assert decode_fp8_tokens(toks) == bits, f"round-trip failed at bits={bits:#04x}"

    # Sequence structure landmarks
    seq = encode_sequence(0x38, 0xB8, 0x00)  # 1.0 + (-1.0) = +0
    assert len(seq) == SEQ_LEN
    assert seq[POS_BOS] == BOS_ID
    assert seq[POS_PLUS] == PLUS_ID
    assert seq[POS_EQ] == EQ_ID
    assert seq[POS_EOS] == EOS_ID
    assert decode_result_from_sequence(seq) == 0x00

    # Batch encoding shape and content
    triples = np.array([[0x38, 0xB8, 0x00], [0x7E, 0x7E, 0x7E]], dtype=np.uint8)
    batch = encode_batch(triples)
    assert batch.shape == (2, SEQ_LEN)
    assert decode_fp8_tokens(batch[1, POS_C_START:POS_C_END].tolist()) == 0x7E

    # Vocab and position invariants
    assert VOCAB_SIZE == 12
    assert POS_C_END - POS_C_START == 8
    assert len(RESULT_TARGET_POSITIONS) == 8

    print("all tokenizer spot checks passed")


if __name__ == "__main__":
    _spot_checks()
