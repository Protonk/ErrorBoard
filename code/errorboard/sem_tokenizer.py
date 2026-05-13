"""3-token (Sign, Exponent, Mantissa) tokenizer for E4M3 add.

Each FP8 number is encoded as exactly 3 tokens:
  - SIGN token (2 possible values)
  - EXP token  (16 possible values, 0..15 = full E4M3 exp field)
  - MANT token (8 possible values, 0..7 = full E4M3 mantissa field)

This is the "field structure handed to the model as a prior" variant tested
in `notes/future_arms.md` Arm 2. Architecture and training schedule stay
identical to the bit-level tokenizer; only the vocab and sequence layout
change.

Layout (13-token sequence):
    pos 0     : <bos>
    pos 1..3  : operand a  (sign, exp, mant)
    pos 4     : `+`
    pos 5..7  : operand b  (sign, exp, mant)
    pos 8     : `=`
    pos 9..11 : result c   (sign, exp, mant)
    pos 12    : <eos>

Vocab layout:
    0  : BOS
    1  : EOS
    2  : PAD
    3  : SCRATCH
    4  : +
    5  : =
    6..7   : SIGN_0, SIGN_1
    8..23  : EXP_0  .. EXP_15
    24..31 : MANT_0 .. MANT_7
    VOCAB_SIZE = 32
"""

from __future__ import annotations

from typing import Final

import numpy as np

# ---- vocabulary ----

BOS_ID: Final[int] = 0
EOS_ID: Final[int] = 1
PAD_ID: Final[int] = 2
SCRATCH_ID: Final[int] = 3
PLUS_ID: Final[int] = 4
EQ_ID: Final[int] = 5
SIGN_BASE: Final[int] = 6     # SIGN_0 = 6, SIGN_1 = 7
EXP_BASE: Final[int] = 8      # EXP_v   = 8 + v   for v in 0..15
MANT_BASE: Final[int] = 24    # MANT_v  = 24 + v  for v in 0..7

VOCAB_SIZE: Final[int] = 32
SEQ_LEN: Final[int] = 13

# ---- positions ----

POS_BOS: Final[int] = 0
POS_A_START: Final[int] = 1
POS_A_END: Final[int] = 4          # exclusive
POS_PLUS: Final[int] = 4
POS_B_START: Final[int] = 5
POS_B_END: Final[int] = 8          # exclusive
POS_EQ: Final[int] = 8
POS_C_START: Final[int] = 9
POS_C_END: Final[int] = 12         # exclusive
POS_EOS: Final[int] = 12

# Sub-position offsets within each 3-token number block (0 = sign, 1 = exp, 2 = mant).
OFFSET_SIGN: Final[int] = 0
OFFSET_EXP: Final[int] = 1
OFFSET_MANT: Final[int] = 2

RESULT_TARGET_POSITIONS: Final[np.ndarray] = np.arange(POS_C_START - 1, POS_C_END - 1)


# ---- single-number encode / decode ----

def encode_fp8_bits(bits: int) -> list[int]:
    """Encode an 8-bit E4M3 pattern as 3 tokens (sign, exp, mantissa)."""
    if not (0 <= bits <= 0xFF):
        raise ValueError(f"bits must be in [0, 255], got {bits}")
    sign = (bits >> 7) & 1
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7
    return [SIGN_BASE + sign, EXP_BASE + exp, MANT_BASE + mant]


def decode_fp8_tokens(tokens) -> int:
    """Decode 3 tokens (one number) back to an FP8 bit pattern."""
    if len(tokens) != 3:
        raise ValueError(f"expected 3 tokens, got {len(tokens)}")
    sign_tok = int(tokens[0])
    exp_tok = int(tokens[1])
    mant_tok = int(tokens[2])
    if not (SIGN_BASE <= sign_tok < SIGN_BASE + 2):
        raise ValueError(f"bad sign token: {sign_tok}")
    if not (EXP_BASE <= exp_tok < EXP_BASE + 16):
        raise ValueError(f"bad exp token: {exp_tok}")
    if not (MANT_BASE <= mant_tok < MANT_BASE + 8):
        raise ValueError(f"bad mant token: {mant_tok}")
    sign = sign_tok - SIGN_BASE
    exp = exp_tok - EXP_BASE
    mant = mant_tok - MANT_BASE
    return (sign << 7) | (exp << 3) | mant


# ---- full-sequence encode / decode ----

def encode_sequence(a_bits: int, b_bits: int, c_bits: int) -> list[int]:
    """Encode (a, b, c) triple as a fixed 13-token sequence."""
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
    """Extract result FP8 bits from a 13-token sequence (positions 9..11)."""
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


# ---- token-class predicates (useful for SEM-aware bit-decomp analog) ----

def is_sign_token(tok: int) -> bool:
    return SIGN_BASE <= tok < SIGN_BASE + 2


def is_exp_token(tok: int) -> bool:
    return EXP_BASE <= tok < EXP_BASE + 16


def is_mant_token(tok: int) -> bool:
    return MANT_BASE <= tok < MANT_BASE + 8


def _spot_checks() -> None:
    # Round-trip for all 256 FP8 patterns.
    for bits in range(256):
        toks = encode_fp8_bits(bits)
        assert len(toks) == 3
        assert decode_fp8_tokens(toks) == bits, f"round-trip failed at bits={bits:#04x}"

    # Sequence structural landmarks.
    seq = encode_sequence(0x38, 0xB8, 0x00)  # 1.0 + (-1.0) = +0
    assert len(seq) == SEQ_LEN
    assert seq[POS_BOS] == BOS_ID
    assert seq[POS_PLUS] == PLUS_ID
    assert seq[POS_EQ] == EQ_ID
    assert seq[POS_EOS] == EOS_ID
    assert decode_result_from_sequence(seq) == 0x00

    # Result-token classes at the right positions.
    seq_b = encode_sequence(0x7E, 0x00, 0x7E)
    assert is_sign_token(seq_b[POS_C_START + OFFSET_SIGN])
    assert is_exp_token(seq_b[POS_C_START + OFFSET_EXP])
    assert is_mant_token(seq_b[POS_C_START + OFFSET_MANT])

    # Batch encoding.
    triples = np.array([[0x38, 0xB8, 0x00], [0x7E, 0x7E, 0x7E]], dtype=np.uint8)
    batch = encode_batch(triples)
    assert batch.shape == (2, SEQ_LEN)
    assert decode_fp8_tokens(batch[1, POS_C_START:POS_C_END].tolist()) == 0x7E

    # Vocab / position invariants.
    assert VOCAB_SIZE == 32
    assert POS_C_END - POS_C_START == 3
    assert len(RESULT_TARGET_POSITIONS) == 3

    # Distinct ranges — no token id overlap between classes.
    for s in range(2):
        for e in range(16):
            for m in range(8):
                tok_s = SIGN_BASE + s
                tok_e = EXP_BASE + e
                tok_m = MANT_BASE + m
                assert tok_s != tok_e and tok_e != tok_m and tok_s != tok_m
                assert is_sign_token(tok_s) and not is_exp_token(tok_s) and not is_mant_token(tok_s)
                assert is_exp_token(tok_e) and not is_sign_token(tok_e) and not is_mant_token(tok_e)
                assert is_mant_token(tok_m) and not is_sign_token(tok_m) and not is_exp_token(tok_m)

    print("all SEM tokenizer spot checks passed")


if __name__ == "__main__":
    _spot_checks()
