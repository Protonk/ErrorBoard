"""F2 (binary-period) FoNE tokenizer for E4M3 add.

Where F1 (the canonical Zhou variant in `fone_tokenizer.py`) uses base-10
period set {10^i for i in [-2,3]} and 6 decimal digits, F2 uses base-2
periods {2^i for i in [-8,9]} and 18 binary digits.

This aligns the FoNE feature space with FP8's actual binade structure:
each binary digit place value 2^p for p in [-9, 8] covers exactly the
representable bit positions of FP8 E4M3 (subnormal step 2^-9 through max
finite 448 < 2^9).

The model should — per the F2 hypothesis from `fone_transition_memo.md` —
converge to FP-shape (anti-ε) error severity at lower capacity than F1,
because it no longer has to internally invert base-10 → base-2 to align
its representation with the format's structure.

Sequence layout is identical to F1: 10-token sequence with [BOS, SIGN_a,
NUM_a, +, SIGN_b, NUM_b, =, SIGN_c, NUM_c, EOS]. Vocab identical.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from .oracle import decode as fp8_decode

# ---- vocabulary (identical to F1) ----

BOS_ID: Final[int] = 0
EOS_ID: Final[int] = 1
PAD_ID: Final[int] = 2
SCRATCH_ID: Final[int] = 3
PLUS_ID: Final[int] = 4
EQ_ID: Final[int] = 5
SIGN_POS_ID: Final[int] = 6
SIGN_NEG_ID: Final[int] = 7
SIGN_NAN_ID: Final[int] = 8
NUM_ID: Final[int] = 9

VOCAB_SIZE: Final[int] = 10
SEQ_LEN: Final[int] = 10

# ---- positions (identical to F1) ----

POS_BOS: Final[int] = 0
POS_A_START: Final[int] = 1
POS_A_END: Final[int] = 3
POS_PLUS: Final[int] = 3
POS_B_START: Final[int] = 4
POS_B_END: Final[int] = 6
POS_EQ: Final[int] = 6
POS_C_START: Final[int] = 7
POS_C_END: Final[int] = 9
POS_EOS: Final[int] = 9

POS_SIGN_C: Final[int] = 7
POS_NUM_C: Final[int] = 8

OFFSET_SIGN: Final[int] = 0
OFFSET_NUM: Final[int] = 1

# ---- F2-specific: base-2 periods + 18 binary digits ----

# Digit at binary place 2^p for p in [-9, 8] — covers exactly the FP8
# representable bit positions (subnormal step 2^-9 through max-finite
# 448 < 2^9).
DIGIT_PLACE_EXPONENTS: Final[list[int]] = list(range(-9, 9))   # length 18
# Each digit is recovered via Lemma 3.3 from period T = 2^(p+1).
PERIOD_EXPONENTS: Final[list[int]] = [p + 1 for p in DIGIT_PLACE_EXPONENTS]
PERIODS: Final[tuple[float, ...]] = tuple(2.0 ** e for e in PERIOD_EXPONENTS)
BASE: Final[int] = 2

N_DIGITS: Final[int] = len(DIGIT_PLACE_EXPONENTS)  # 18
FONE_DIM: Final[int] = 2 * N_DIGITS                # 36


def fp8_to_sign_and_digits(bits: int) -> tuple[int, list[int]]:
    """Decode an FP8 pattern into (sign_class_id, binary_digits).

    Returns:
        sign_class_id ∈ {SIGN_POS_ID, SIGN_NEG_ID, SIGN_NAN_ID}.
        digits: 18 binary digits ordered low-place → high-place
            (bit at 2^-9, 2^-8, ..., 2^7, 2^8).
    """
    value, kind = fp8_decode(bits)
    if kind == "nan":
        return SIGN_NAN_ID, [0] * N_DIGITS
    sign_id = SIGN_NEG_ID if value < 0 else SIGN_POS_ID
    abs_v = abs(value)
    # Convert to integer in units of 2^min_place = 2^-9. Banker's rounding.
    scale = 2.0 ** (-min(DIGIT_PLACE_EXPONENTS))    # 2^9 = 512
    v_int = round(abs_v * scale)                    # in [0, 2^18)
    # FP8 max 448 * 512 = 229376 < 2^18 = 262144, so this fits.
    digits = [(v_int >> i) & 1 for i in range(N_DIGITS)]
    return sign_id, digits


def fp8_to_num_value(bits: int) -> float:
    """Return |x| for FoNE feature computation; NaN → 0.0 (the sign token
    carries the NaN signal)."""
    value, kind = fp8_decode(bits)
    if kind == "nan":
        return 0.0
    return abs(float(value))


def encode_fp8_bits(bits: int) -> tuple[list[int], list[float]]:
    if not (0 <= bits <= 0xFF):
        raise ValueError(f"bits must be in [0, 255], got {bits}")
    sign_id, _ = fp8_to_sign_and_digits(bits)
    return [sign_id, NUM_ID], [0.0, fp8_to_num_value(bits)]


def encode_sequence(a_bits: int, b_bits: int, c_bits: int) -> tuple[list[int], list[float]]:
    sa, na = encode_fp8_bits(a_bits)
    sb, nb = encode_fp8_bits(b_bits)
    sc, nc = encode_fp8_bits(c_bits)
    tokens = [BOS_ID] + sa + [PLUS_ID] + sb + [EQ_ID] + sc + [EOS_ID]
    values = [0.0] + na + [0.0] + nb + [0.0] + nc + [0.0]
    return tokens, values


def encode_batch(triples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError(f"expected (N, 3), got {triples.shape}")
    n = len(triples)
    tokens = np.empty((n, SEQ_LEN), dtype=np.int64)
    values = np.zeros((n, SEQ_LEN), dtype=np.float32)
    for i in range(n):
        t, v = encode_sequence(int(triples[i, 0]), int(triples[i, 1]), int(triples[i, 2]))
        tokens[i] = t
        values[i] = v
    return tokens, values


def encode_targets(triples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(triples)
    sign_target = np.empty(n, dtype=np.int64)
    digit_target = np.zeros((n, N_DIGITS), dtype=np.int64)
    is_nan = np.zeros(n, dtype=bool)
    for i in range(n):
        c_bits = int(triples[i, 2])
        sign_id, digits = fp8_to_sign_and_digits(c_bits)
        sign_target[i] = sign_id
        digit_target[i] = digits
        is_nan[i] = (sign_id == SIGN_NAN_ID)
    return sign_target, digit_target, is_nan


def digits_to_magnitude(digits) -> float:
    """Inverse of digit extraction: assemble base-2 magnitude."""
    return float(sum(d * (2.0 ** e) for d, e in zip(digits, DIGIT_PLACE_EXPONENTS)))


def reconstruct_value(sign_class_id: int, digits) -> tuple[float, str]:
    if sign_class_id == SIGN_NAN_ID:
        return float("nan"), "nan"
    mag = digits_to_magnitude(digits)
    if sign_class_id == SIGN_NEG_ID:
        mag = -mag
    return mag, "real"


def _spot_checks() -> None:
    assert VOCAB_SIZE == 10
    assert SEQ_LEN == 10
    assert N_DIGITS == 18
    assert FONE_DIM == 36
    assert BASE == 2
    assert len(PERIODS) == N_DIGITS
    assert len(DIGIT_PLACE_EXPONENTS) == N_DIGITS

    # Roundtrip: every non-NaN FP8 should decode exactly via 18-bit
    # base-2 fixed-point. Subnormals + normals are powers-of-2 sums, so
    # they should round to themselves with banker's rounding.
    for bits in range(256):
        sign_id, digits = fp8_to_sign_and_digits(bits)
        if sign_id == SIGN_NAN_ID:
            continue
        recon, _ = reconstruct_value(sign_id, digits)
        true_val, _ = fp8_decode(bits)
        # Exact match expected (FP8 values are dyadic rationals with at
        # most 9 fractional bits — within our 18-bit precision).
        assert abs(recon - true_val) < 1e-12, (
            f"bits={bits:#04x}: true={true_val} recon={recon} digits={digits}"
        )

    # Distinctness: every non-NaN value has a unique (sign, 18-bit) key.
    seen = {}
    for bits in range(256):
        sign_id, digits = fp8_to_sign_and_digits(bits)
        if sign_id == SIGN_NAN_ID:
            continue
        key = (sign_id, tuple(digits))
        if key in seen:
            other = seen[key]
            other_val, _ = fp8_decode(other)
            this_val, _ = fp8_decode(bits)
            if abs(this_val) < 1e-12 and abs(other_val) < 1e-12:
                continue  # ±0 collision is fine
            raise AssertionError(
                f"collision: bits={bits:#04x} ({this_val}) and "
                f"{other:#04x} ({other_val})"
            )
        seen[key] = bits

    # Sequence landmarks.
    tokens, values = encode_sequence(0x38, 0xB8, 0x00)
    assert tokens[POS_SIGN_C] == SIGN_POS_ID
    assert tokens[POS_NUM_C] == NUM_ID
    assert values[POS_NUM_C] == 0.0

    # Target for a known value: 1.0 should decode to bit_9 = 1 (place 2^0),
    # all others 0.
    triples = np.array([[0x00, 0x38, 0x38]], dtype=np.uint8)   # 0 + 1 = 1
    _, digit_tgt, _ = encode_targets(triples)
    expected = [0] * N_DIGITS
    expected[9] = 1   # bit at 2^0
    assert list(digit_tgt[0]) == expected, list(digit_tgt[0])

    # NaN handling.
    nan_triple = np.array([[0x7F, 0x00, 0x7F]], dtype=np.uint8)
    sign_tgt, digit_tgt, is_nan = encode_targets(nan_triple)
    assert sign_tgt[0] == SIGN_NAN_ID
    assert is_nan[0]

    print("all F2 tokenizer spot checks passed")


if __name__ == "__main__":
    _spot_checks()
