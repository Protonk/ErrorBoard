"""Regime classification for FP8 E4M3 multiplication.

Parallel to `regimes.py` (which classifies addition pairs). Multiplication's
structural cases differ from addition's:

  Dropped from addition:
    - CANCELLATION (no cancellation in multiplication)
    - LARGE_DEXP   (Δexp doesn't shape mult the way it shapes add — mult
                    sums exponents cleanly; the analogous "near-overflow"
                    and "near-underflow" axes are absorbed by OVERFLOW
                    and UNDERFLOW_TO_ZERO and tagged via _LARGE/SMALL_EXP_SUM)

  Kept / repurposed:
    - SPECIAL_VALUES: NaN input OR exactly one operand is zero (zero
                      product is its own thing semantically; sign is XOR
                      and result is ±0)
    - OVERFLOW: |true_product| > 448 (saturates to ±448 per oracle)
    - UNDERFLOW_TO_ZERO: 0 < |true_product| < 2^-10 (rounds to ±0)
    - SUBNORMAL_RESULT: result is a nonzero subnormal
    - ROUNDING_TIE: true_product is the exact midpoint of adjacent FP8 values
    - DEFAULT: everything else (rounding-required, normal-result)

  New for multiplication:
    - EXACT_RESULT: true_product is exactly representable in finite FP8
                    (no rounding needed). Captures power-of-2 multiplications
                    and other clean products like 1.25 × 8 = 10 = exact FP8.

Tags (bitmask):
    - SAME_SIGN, OPPOSITE_SIGN — sign-wise pairing of operands
    - POWER_OF_TWO_INPUT       — at least one operand is a power of 2
                                   (mantissa zero, non-special). Mult by a
                                   power of 2 is just an exponent shift —
                                   trivially exact unless overflow.
    - SAME_BINADE              — operands share the same biased exponent
    - LARGE_EXP_SUM            — unbiased(a) + unbiased(b) >= 5 (near overflow side)
    - SMALL_EXP_SUM            — unbiased(a) + unbiased(b) <= -5 (near underflow side)
    - SUBNORMAL_INPUT          — either operand is subnormal
    - RESULT_EXACT             — result needs no rounding (mirrors EXACT_RESULT regime)
    - TIE                      — mirrors ROUNDING_TIE regime
"""

from __future__ import annotations

import bisect
import math
from typing import Final

from .oracle import NAN_BITS, decode, encode, mul

# ---- primary regime IDs (priority-ordered partition; first match wins) ----

SPECIAL_VALUES: Final[int] = 0
OVERFLOW: Final[int] = 1
UNDERFLOW_TO_ZERO: Final[int] = 2
SUBNORMAL_RESULT: Final[int] = 3
ROUNDING_TIE: Final[int] = 4
EXACT_RESULT: Final[int] = 5
DEFAULT: Final[int] = 6

REGIME_NAMES: Final[tuple[str, ...]] = (
    "special-values",
    "overflow",
    "underflow-to-zero",
    "subnormal-result",
    "rounding-tie",
    "exact-result",
    "default",
)
NUM_REGIMES: Final[int] = len(REGIME_NAMES)

# ---- secondary tag bit positions ----

TAG_SAME_SIGN: Final[int] = 1 << 0
TAG_OPPOSITE_SIGN: Final[int] = 1 << 1
TAG_POWER_OF_TWO_INPUT: Final[int] = 1 << 2
TAG_SAME_BINADE: Final[int] = 1 << 3
TAG_LARGE_EXP_SUM: Final[int] = 1 << 4
TAG_SMALL_EXP_SUM: Final[int] = 1 << 5
TAG_SUBNORMAL_INPUT: Final[int] = 1 << 6
TAG_RESULT_EXACT: Final[int] = 1 << 7
TAG_TIE: Final[int] = 1 << 8

TAG_NAMES: Final[dict[int, str]] = {
    TAG_SAME_SIGN: "same-sign",
    TAG_OPPOSITE_SIGN: "opposite-sign",
    TAG_POWER_OF_TWO_INPUT: "power-of-two-input",
    TAG_SAME_BINADE: "same-binade",
    TAG_LARGE_EXP_SUM: "large-exp-sum",
    TAG_SMALL_EXP_SUM: "small-exp-sum",
    TAG_SUBNORMAL_INPUT: "subnormal-input",
    TAG_RESULT_EXACT: "result-exact",
    TAG_TIE: "tie",
}


# ---- helpers ----

def _build_fp8_grid() -> list[float]:
    """Sorted unique finite FP8 values (NaN excluded; +0 and -0 collapse)."""
    vals = set()
    for b in range(256):
        v, kind = decode(b)
        if kind != "nan":
            vals.add(v)
    return sorted(vals)


_FP8_GRID = _build_fp8_grid()
_FP8_GRID_SET = set(_FP8_GRID)


def unbiased_exp(bits: int) -> int:
    """Unbiased exponent of an E4M3 bit pattern; subnormals collapse to -6."""
    exp = (bits >> 3) & 0xF
    if exp == 0:
        return -6
    return exp - 7


def is_subnormal_bits(bits: int) -> bool:
    exp = (bits >> 3) & 0xF
    mant = bits & 0x7
    return exp == 0 and mant != 0


def is_zero_bits(bits: int) -> bool:
    return (bits & 0x7F) == 0x00


def is_nan_bits(bits: int) -> bool:
    return (bits & 0x7F) == 0x7F


def is_power_of_two_bits(bits: int) -> bool:
    """True if the FP8 value is a power of 2 (mantissa zero, normal or
    smallest subnormal). Excludes zero and NaN."""
    mant = bits & 0x7
    if mant != 0:
        return False
    exp = (bits >> 3) & 0xF
    if exp == 0:
        return False  # zero
    if exp == 0xF:
        # exp=15 mant=0 is 1.000 * 2^8 = 256; valid power of 2.
        return True
    return True


def is_tie_product(true_product: float) -> bool:
    """True if true_product is the exact midpoint of two adjacent FP8 values."""
    if math.isnan(true_product) or math.isinf(true_product):
        return False
    if abs(true_product) > 448.0:
        return False  # overflow takes priority
    idx = bisect.bisect_left(_FP8_GRID, true_product)
    if idx == 0 or idx == len(_FP8_GRID):
        return False
    if _FP8_GRID[idx] == true_product:
        return False
    lower = _FP8_GRID[idx - 1]
    upper = _FP8_GRID[idx]
    return true_product == (lower + upper) / 2.0


def is_result_exact_value(true_product: float) -> bool:
    if math.isnan(true_product) or math.isinf(true_product):
        return False
    return true_product in _FP8_GRID_SET


# ---- classification ----

def classify(a_bits: int, b_bits: int) -> tuple[int, int, int]:
    """Classify (a_bits, b_bits) for multiplication.

    Returns (result_bits, primary_regime_id, tag_mask). Primary regime is
    first-match in priority order.
    """
    a_val, a_kind = decode(a_bits)
    b_val, b_kind = decode(b_bits)

    a_is_nan = a_kind == "nan"
    b_is_nan = b_kind == "nan"
    a_zero = a_kind == "zero"
    b_zero = b_kind == "zero"

    if a_is_nan or b_is_nan:
        true_product = math.nan
        result_bits = NAN_BITS
    else:
        true_product = a_val * b_val
        result_bits = encode(true_product)

    # Primary regime (priority order)
    if a_is_nan or b_is_nan or a_zero or b_zero:
        regime = SPECIAL_VALUES
    elif abs(true_product) > 448.0:
        regime = OVERFLOW
    elif 0.0 < abs(true_product) < 2.0 ** -10:
        regime = UNDERFLOW_TO_ZERO
    elif is_subnormal_bits(result_bits):
        regime = SUBNORMAL_RESULT
    elif is_tie_product(true_product):
        regime = ROUNDING_TIE
    elif is_result_exact_value(true_product):
        regime = EXACT_RESULT
    else:
        regime = DEFAULT

    tag_mask = _compute_tags(a_bits, b_bits, a_val, b_val, a_kind, b_kind, true_product)
    return result_bits, regime, tag_mask


def _compute_tags(
    a_bits: int,
    b_bits: int,
    a_val: float,
    b_val: float,
    a_kind: str,
    b_kind: str,
    true_product: float,
) -> int:
    tags = 0

    # Sign pairing — match XOR-of-signs logic. For zero operands, take the
    # explicit sign bit (Python's copysign handles signed zeros).
    a_sign = (a_bits >> 7) & 1
    b_sign = (b_bits >> 7) & 1
    if a_sign == b_sign:
        tags |= TAG_SAME_SIGN
    else:
        tags |= TAG_OPPOSITE_SIGN

    if is_power_of_two_bits(a_bits) or is_power_of_two_bits(b_bits):
        tags |= TAG_POWER_OF_TWO_INPUT

    a_biased = (a_bits >> 3) & 0xF
    b_biased = (b_bits >> 3) & 0xF
    if a_biased == b_biased and a_kind != "nan" and b_kind != "nan":
        tags |= TAG_SAME_BINADE

    # Sum-of-exponents tags (only meaningful for non-NaN, non-zero inputs).
    if a_kind not in ("nan", "zero") and b_kind not in ("nan", "zero"):
        exp_sum = unbiased_exp(a_bits) + unbiased_exp(b_bits)
        if exp_sum >= 5:
            tags |= TAG_LARGE_EXP_SUM
        if exp_sum <= -5:
            tags |= TAG_SMALL_EXP_SUM

    if a_kind == "subnormal" or b_kind == "subnormal":
        tags |= TAG_SUBNORMAL_INPUT

    if is_result_exact_value(true_product):
        tags |= TAG_RESULT_EXACT

    if is_tie_product(true_product):
        tags |= TAG_TIE

    return tags


# ---- self-tests ----

def _spot_checks() -> None:
    # special-values: NaN
    _, r, _ = classify(0x7F, 0x38)
    assert r == SPECIAL_VALUES, f"NaN * 1.0 -> {REGIME_NAMES[r]}"
    _, r, _ = classify(0x38, 0xFF)
    assert r == SPECIAL_VALUES, f"1.0 * NaN(-) -> {REGIME_NAMES[r]}"

    # special-values: zero input (regardless of other operand)
    _, r, _ = classify(0x00, 0x38)
    assert r == SPECIAL_VALUES, f"+0 * 1.0 -> {REGIME_NAMES[r]}"
    _, r, _ = classify(0x38, 0x80)
    assert r == SPECIAL_VALUES, f"1.0 * -0 -> {REGIME_NAMES[r]}"
    _, r, _ = classify(0x00, 0x00)
    assert r == SPECIAL_VALUES, f"+0 * +0 -> {REGIME_NAMES[r]}"

    # overflow: 448 * 2.0 = 896 > 448
    _, r, _ = classify(0x7E, 0x40)
    assert r == OVERFLOW, f"448 * 2.0 -> {REGIME_NAMES[r]}"

    # underflow-to-zero: 2^-9 * 2^-9 = 2^-18 (rounds to ±0)
    _, r, _ = classify(0x01, 0x01)
    assert r == UNDERFLOW_TO_ZERO, f"2^-9 * 2^-9 -> {REGIME_NAMES[r]}"

    # subnormal-result: 2^-9 * 2.0 = 2^-8 (a subnormal)
    _, r, _ = classify(0x01, 0x40)
    assert r == SUBNORMAL_RESULT, f"2^-9 * 2.0 -> {REGIME_NAMES[r]}"

    # exact-result: 1.0 * 1.0 = 1.0 (exactly representable, no rounding)
    _, r, _ = classify(0x38, 0x38)
    assert r == EXACT_RESULT, f"1.0 * 1.0 -> {REGIME_NAMES[r]}"

    # exact-result: 2.0 * 1.25 = 2.5 (exact, mantissa-wise)
    # 2.5 = 1.25 * 2^1, m=2 e=8 -> 0x42; check FP8 representable.
    bits_25, _, _ = classify(0x40, 0x3A)
    assert bits_25 == 0x42, f"2.0 * 1.25 -> 0x{bits_25:02x}"
    _, r, _ = classify(0x40, 0x3A)
    assert r == EXACT_RESULT, f"2.0 * 1.25 -> {REGIME_NAMES[r]}"

    # rounding-tie: 1.25 * 1.25 = 1.5625, midpoint between 1.5 and 1.625
    _, r, _ = classify(0x3A, 0x3A)
    assert r == ROUNDING_TIE, f"1.25 * 1.25 -> {REGIME_NAMES[r]}"

    # default: 1.125 * 1.125 = 1.265625; not a tie, not exact.
    # 1.125 between 1.0 (0x38) and 1.125 (0x39). m=1.125 needs rounding.
    # 1.125 ULP near 1.25 (0x3A)? Let's compute: 1.265625 between 1.25 (0x3A) and 1.375 (0x3B).
    # Midpoint of 1.25, 1.375 is 1.3125. 1.265625 != 1.3125, so not a tie. Default.
    _, r, _ = classify(0x39, 0x39)
    assert r == DEFAULT, f"1.125 * 1.125 -> {REGIME_NAMES[r]}"

    # tags
    _, _, t = classify(0x38, 0x40)   # 1.0 * 2.0 — both powers of 2
    assert t & TAG_POWER_OF_TWO_INPUT, "POW2 missing on (1.0, 2.0)"
    assert t & TAG_SAME_SIGN, "SAME_SIGN missing on (+, +)"

    _, _, t = classify(0xB8, 0x40)   # -1.0 * 2.0
    assert t & TAG_OPPOSITE_SIGN, "OPPOSITE_SIGN missing on (-, +)"

    _, _, t = classify(0x39, 0x39)   # 1.125 * 1.125, same binade
    assert t & TAG_SAME_BINADE, "SAME_BINADE missing on (0x39, 0x39)"

    _, _, t = classify(0x7E, 0x7E)   # 448 * 448 = exp_sum 16 -> large
    assert t & TAG_LARGE_EXP_SUM, "LARGE_EXP_SUM missing on (448, 448)"

    _, _, t = classify(0x01, 0x01)   # 2^-9 * 2^-9 = exp_sum -12 -> small
    assert t & TAG_SMALL_EXP_SUM, "SMALL_EXP_SUM missing on (2^-9, 2^-9)"
    assert t & TAG_SUBNORMAL_INPUT, "SUBNORMAL_INPUT missing"

    _, _, t = classify(0x38, 0x38)   # exact result
    assert t & TAG_RESULT_EXACT, "RESULT_EXACT missing on 1.0 * 1.0"

    _, _, t = classify(0x3A, 0x3A)   # tie
    assert t & TAG_TIE, "TIE missing on 1.25 * 1.25"

    # Population sanity: classify all 65,536 ordered pairs, see distribution.
    counts = [0] * NUM_REGIMES
    for a in range(256):
        for b in range(256):
            _, r, _ = classify(a, b)
            counts[r] += 1
    total = sum(counts)
    assert total == 65_536
    print("Population per regime (out of 65,536 ordered pairs):")
    for name, count in zip(REGIME_NAMES, counts):
        print(f"  {name:<20s}: {count:>6} ({count/total*100:>5.2f}%)")

    # Sanity: oracle.mul agrees with classify's result_bits (both go through encode()).
    for a in range(256):
        for b in range(256):
            res_bits, _, _ = classify(a, b)
            oracle_res = mul(a, b)
            # NaN bit patterns are equivalent
            if (res_bits & 0x7F) == 0x7F and (oracle_res & 0x7F) == 0x7F:
                continue
            assert res_bits == oracle_res, (
                f"classify vs mul disagreement at ({a:#04x}, {b:#04x}): "
                f"classify={res_bits:#04x} mul={oracle_res:#04x}"
            )

    print("all mult-regime spot checks passed")


if __name__ == "__main__":
    _spot_checks()
