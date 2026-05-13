"""FoNE-style tokenizer for E4M3 add.

Each FP8 number is encoded as exactly 2 tokens:
  - SIGN token: one of {SIGN_POS, SIGN_NEG, SIGN_NAN}
  - NUM   token: a single [NUM] sentinel; the actual numeric value rides
                 alongside in a parallel `num_values` array, and the model
                 sums FoNE Fourier features onto the NUM-token embedding.

Layout (10-token sequence):
    pos 0    : <bos>
    pos 1..2 : operand a  (SIGN_a, NUM_a)
    pos 3    : `+`
    pos 4..5 : operand b  (SIGN_b, NUM_b)
    pos 6    : `=`
    pos 7..8 : result c   (SIGN_c, NUM_c)
    pos 9    : <eos>

Vocab:
    0  : BOS
    1  : EOS
    2  : PAD
    3  : SCRATCH
    4  : +
    5  : =
    6  : SIGN_POS  (non-negative finite)
    7  : SIGN_NEG  (negative finite)
    8  : SIGN_NAN  (either of FP8's two NaN patterns)
    9  : NUM       (sentinel; FoNE feature carries the value)

Per-NUM digit decoding: each NUM token at the result position is decoded
into 6 base-10 digits via FoNE's cosine-similarity head (m=3 integer
places, n=3 fractional places — covers FP8's [0, 448] range with enough
fractional precision to distinguish adjacent subnormals).
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

from .oracle import decode as fp8_decode

# ---- vocabulary ----

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

# ---- positions ----

POS_BOS: Final[int] = 0
POS_A_START: Final[int] = 1
POS_A_END: Final[int] = 3          # exclusive
POS_PLUS: Final[int] = 3
POS_B_START: Final[int] = 4
POS_B_END: Final[int] = 6          # exclusive
POS_EQ: Final[int] = 6
POS_C_START: Final[int] = 7
POS_C_END: Final[int] = 9          # exclusive (SIGN_c at 7, NUM_c at 8)
POS_EOS: Final[int] = 9

POS_SIGN_C: Final[int] = 7
POS_NUM_C: Final[int] = 8

# Sub-offsets within each number block (0 = sign, 1 = num).
OFFSET_SIGN: Final[int] = 0
OFFSET_NUM: Final[int] = 1

# ---- FoNE digit layout ----
# m=3 integer + n=3 fractional digits, total 6 base-10 digits per number.
# Digit place values, ordered to match the FoNE feature ordering — which is
# small-period (high-precision) → large-period (high-magnitude). So digits
# are listed thousandths → tenths → ones → tens → hundreds:
N_INT_DIGITS: Final[int] = 3
N_FRAC_DIGITS: Final[int] = 3
N_DIGITS: Final[int] = N_INT_DIGITS + N_FRAC_DIGITS

# Periods T_i = 10^i for i ∈ [-n+1, m] following Zhou Def 3.2.
# For (m=3, n=3): i ∈ [-2, 3] → periods [10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3].
# Each period contributes one (cos, sin) pair to the FoNE feature vector.
PERIOD_EXPONENTS: Final[list[int]] = list(range(-(N_FRAC_DIGITS - 1), N_INT_DIGITS + 1))
PERIODS: Final[tuple[float, ...]] = tuple(10.0 ** e for e in PERIOD_EXPONENTS)

# Digit place values aligned with PERIOD_EXPONENTS. Per Zhou's Lemma 3.4, the
# `i`-th FoNE component recovers x mod T_i, which yields the digit at place 10^(i-1).
DIGIT_PLACE_EXPONENTS: Final[list[int]] = [e - 1 for e in PERIOD_EXPONENTS]
# So for periods [0.01, 0.1, 1, 10, 100, 1000], digit places are
# [0.001, 0.01, 0.1, 1, 10, 100] — thousandths through hundreds.

FONE_DIM: Final[int] = 2 * N_DIGITS  # 12


# ---- value → (sign-class, digits) extraction ----

def fp8_to_sign_and_digits(bits: int) -> tuple[int, list[int]]:
    """Decode a single FP8 pattern into (sign_class_id, digits).

    Returns:
        sign_class_id ∈ {SIGN_POS_ID, SIGN_NEG_ID, SIGN_NAN_ID}.
        digits: 6 base-10 digits ordered low-place → high-place
            (thousandths, hundredths, tenths, ones, tens, hundreds).
            For NaN inputs, digits are all zeros (don't-care during loss).
    """
    value, kind = fp8_decode(bits)
    if kind == "nan":
        return SIGN_NAN_ID, [0] * N_DIGITS
    sign_id = SIGN_NEG_ID if value < 0 else SIGN_POS_ID
    abs_v = abs(value)
    # Convert to integer in units of 10^min_exp = 10^-3, with banker's rounding.
    scale = 10.0 ** (-min(DIGIT_PLACE_EXPONENTS))
    v_int = round(abs_v * scale)  # in [0, 10^6)
    digits = []
    remainder = v_int
    # Pull digits low-to-high to match the FoNE feature ordering.
    for _ in range(N_DIGITS):
        digits.append(int(remainder % 10))
        remainder //= 10
    return sign_id, digits


def fp8_to_num_value(bits: int) -> float:
    """Return the magnitude (absolute value) for FoNE feature computation.

    NaN inputs return 0.0 — the sign token already encodes NaN, so the NUM
    embedding is benign. Sign of the value is *not* fed to FoNE here; the
    sign token carries it.
    """
    value, kind = fp8_decode(bits)
    if kind == "nan":
        return 0.0
    return abs(float(value))


# ---- single-number / sequence encode ----

def encode_fp8_bits(bits: int) -> tuple[list[int], list[float]]:
    """Encode an 8-bit E4M3 pattern as 2 tokens.

    Returns (token_ids, num_values):
        token_ids = [SIGN_*, NUM_ID]
        num_values = [0.0, |x|]   — only the NUM position carries a value
    """
    if not (0 <= bits <= 0xFF):
        raise ValueError(f"bits must be in [0, 255], got {bits}")
    sign_id, _ = fp8_to_sign_and_digits(bits)
    num_value = fp8_to_num_value(bits)
    return [sign_id, NUM_ID], [0.0, num_value]


def encode_sequence(a_bits: int, b_bits: int, c_bits: int) -> tuple[list[int], list[float]]:
    """Encode (a, b, c) triple as a (token_ids, num_values) pair of length SEQ_LEN."""
    sa, na = encode_fp8_bits(a_bits)
    sb, nb = encode_fp8_bits(b_bits)
    sc, nc = encode_fp8_bits(c_bits)
    tokens = [BOS_ID] + sa + [PLUS_ID] + sb + [EQ_ID] + sc + [EOS_ID]
    values = [0.0] + na + [0.0] + nb + [0.0] + nc + [0.0]
    return tokens, values


def encode_batch(triples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Encode an (N, 3) array of bit triples.

    Returns:
        token_ids   : (N, SEQ_LEN) int64
        num_values  : (N, SEQ_LEN) float32 — magnitudes at NUM positions, 0 elsewhere
    """
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
    """Compute per-pair training targets for the result-c token.

    Returns:
        sign_target  : (N,) int64 — sign-class id (SIGN_POS/NEG/NAN)
        digit_target : (N, N_DIGITS) int64 — 6 base-10 digits low→high
        is_nan       : (N,) bool — mask for samples where digit-loss is don't-care
    """
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


# ---- single-number reconstruction (for eval) ----

def digits_to_magnitude(digits) -> float:
    """Inverse of the digit extraction: assemble base-10 magnitude."""
    place_values = [10.0 ** e for e in DIGIT_PLACE_EXPONENTS]
    return float(sum(d * pv for d, pv in zip(digits, place_values)))


def reconstruct_value(sign_class_id: int, digits) -> tuple[float, str]:
    """Reconstruct (value, kind) from predicted sign + digits.

    kind ∈ {"normal_or_subnormal", "nan"} — for FoNE we don't distinguish
    these subkinds, since per-digit decoding ignores binade structure.
    """
    if sign_class_id == SIGN_NAN_ID:
        return float("nan"), "nan"
    mag = digits_to_magnitude(digits)
    if sign_class_id == SIGN_NEG_ID:
        mag = -mag
    return mag, "real"


# ---- self-tests ----

def _spot_checks() -> None:
    # Vocab and layout invariants.
    assert VOCAB_SIZE == 10
    assert SEQ_LEN == 10
    assert POS_C_END - POS_C_START == 2
    assert N_DIGITS == 6
    assert FONE_DIM == 12
    assert len(PERIODS) == N_DIGITS
    assert len(DIGIT_PLACE_EXPONENTS) == N_DIGITS

    # Digit / sign roundtrip for all 256 FP8 patterns.
    for bits in range(256):
        sign_id, digits = fp8_to_sign_and_digits(bits)
        if sign_id == SIGN_NAN_ID:
            continue
        # Reconstruct value and check it matches the FP8 decode to within
        # the rounding tolerance (3 decimal places).
        recon_val, _ = reconstruct_value(sign_id, digits)
        true_val, _ = fp8_decode(bits)
        # Tolerance: half the smallest digit place value (= 0.0005).
        assert abs(recon_val - true_val) <= 0.0005 + 1e-9, (
            f"bits={bits:#04x}: true={true_val} recon={recon_val} "
            f"sign={sign_id} digits={digits}"
        )

    # Distinctness: every non-NaN FP8 value should have a unique
    # (sign_id, digit-tuple) when we round to 3 decimal places.
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
            # ±0 will collide; that's OK.
            if abs(this_val) < 1e-9 and abs(other_val) < 1e-9:
                continue
            raise AssertionError(
                f"collision: bits={bits:#04x} ({this_val}) and "
                f"{other:#04x} ({other_val}) share key {key}"
            )
        seen[key] = bits

    # Sequence structural landmarks.
    tokens, values = encode_sequence(0x38, 0xB8, 0x00)  # 1.0 + (-1.0) = +0
    assert len(tokens) == SEQ_LEN
    assert tokens[POS_BOS] == BOS_ID
    assert tokens[POS_PLUS] == PLUS_ID
    assert tokens[POS_EQ] == EQ_ID
    assert tokens[POS_EOS] == EOS_ID
    assert tokens[POS_SIGN_C] == SIGN_POS_ID  # +0
    assert tokens[POS_NUM_C] == NUM_ID
    assert values[POS_NUM_C] == 0.0          # |+0| = 0

    # Batch encoding shape and content.
    triples = np.array([[0x38, 0xB8, 0x00], [0x7E, 0x7E, 0x7E]], dtype=np.uint8)
    tok_b, val_b = encode_batch(triples)
    assert tok_b.shape == (2, SEQ_LEN)
    assert val_b.shape == (2, SEQ_LEN)
    # 0x7E = +448 (max finite). Targets should put SIGN_POS and digits for 448.
    sign_tgt, digit_tgt, is_nan = encode_targets(triples)
    assert sign_tgt.shape == (2,)
    assert digit_tgt.shape == (2, N_DIGITS)
    assert is_nan.shape == (2,)
    assert not is_nan.any()
    assert sign_tgt[1] == SIGN_POS_ID
    # Digits for 448.000 (low→high): 0, 0, 0, 8, 4, 4.
    assert list(digit_tgt[1]) == [0, 0, 0, 8, 4, 4], list(digit_tgt[1])

    # NaN handling.
    nan_triple = np.array([[0x7F, 0x00, 0x7F]], dtype=np.uint8)
    sign_tgt, digit_tgt, is_nan = encode_targets(nan_triple)
    assert sign_tgt[0] == SIGN_NAN_ID
    assert is_nan[0]

    print("all FoNE tokenizer spot checks passed")


if __name__ == "__main__":
    _spot_checks()
