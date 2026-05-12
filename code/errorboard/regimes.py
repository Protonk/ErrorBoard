"""Regime classification per task_spec.md §2 (predicates) and §3 (tags).

Each pair (a_bits, b_bits) maps to:
    - a primary regime in {0..7}, assigned by first-match in priority order
    - a tag_mask byte, with one bit per secondary tag (analysis-only, may overlap)
"""

from __future__ import annotations

import bisect
import math
from typing import Final

from .oracle import NAN_BITS, decode, encode

# ---- primary regime IDs (priority-ordered partition; first match wins) ----

SPECIAL_VALUES: Final[int] = 0
OVERFLOW: Final[int] = 1
UNDERFLOW_TO_ZERO: Final[int] = 2
SUBNORMAL_RESULT: Final[int] = 3
CANCELLATION: Final[int] = 4
ROUNDING_TIE: Final[int] = 5
LARGE_DEXP: Final[int] = 6
DEFAULT: Final[int] = 7

REGIME_NAMES: Final[tuple[str, ...]] = (
    "special-values",
    "overflow",
    "underflow-to-zero",
    "subnormal-result",
    "cancellation",
    "rounding-tie",
    "large-dexp",
    "default",
)
NUM_REGIMES: Final[int] = len(REGIME_NAMES)

# ---- secondary tag bit positions (tag_mask byte) ----

TAG_SAME_SIGN: Final[int] = 1 << 0
TAG_OPPOSITE_SIGN: Final[int] = 1 << 1
TAG_SAME_EXP: Final[int] = 1 << 2
TAG_SMALL_DEXP: Final[int] = 1 << 3
TAG_LARGE_DEXP: Final[int] = 1 << 4
TAG_TIE: Final[int] = 1 << 5
TAG_SUBNORMAL_INPUT: Final[int] = 1 << 6
TAG_RESULT_EXACT: Final[int] = 1 << 7

TAG_NAMES: Final[dict[int, str]] = {
    TAG_SAME_SIGN: "same-sign",
    TAG_OPPOSITE_SIGN: "opposite-sign",
    TAG_SAME_EXP: "same-exp",
    TAG_SMALL_DEXP: "small-dexp",
    TAG_LARGE_DEXP: "large-dexp",
    TAG_TIE: "tie",
    TAG_SUBNORMAL_INPUT: "subnormal-input",
    TAG_RESULT_EXACT: "result-exact",
}


# ---- helpers ----

def _build_fp8_grid() -> list[float]:
    """Sorted unique finite FP8 values (excludes NaN; +0 and -0 collapse to one entry)."""
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
    """True if bit pattern is a (nonzero) subnormal."""
    exp = (bits >> 3) & 0xF
    mantissa = bits & 0x7
    return exp == 0 and mantissa != 0


def is_tie(true_sum: float) -> bool:
    """True if true_sum is the exact midpoint of two adjacent finite FP8 values."""
    if math.isnan(true_sum) or math.isinf(true_sum):
        return False
    idx = bisect.bisect_left(_FP8_GRID, true_sum)
    if idx == 0 or idx == len(_FP8_GRID):
        return False
    if _FP8_GRID[idx] == true_sum:
        return False  # exactly representable, not a tie
    lower = _FP8_GRID[idx - 1]
    upper = _FP8_GRID[idx]
    return true_sum == (lower + upper) / 2.0


def is_result_exact(true_sum: float) -> bool:
    """True if true_sum is exactly representable in finite E4M3."""
    if math.isnan(true_sum) or math.isinf(true_sum):
        return False
    return true_sum in _FP8_GRID_SET


# ---- classification ----

def classify(a_bits: int, b_bits: int) -> tuple[int, int, int]:
    """Classify (a_bits, b_bits) -> (result_bits, primary_regime_id, tag_mask).

    Primary regime is assigned by first-match in priority order per task_spec.md §2.
    Tags are computed regardless of primary regime per task_spec.md §3.
    """
    a_val, a_kind = decode(a_bits)
    b_val, b_kind = decode(b_bits)

    a_is_nan = a_kind == "nan"
    b_is_nan = b_kind == "nan"
    a_zero = a_kind == "zero"
    b_zero = b_kind == "zero"

    if a_is_nan or b_is_nan:
        true_sum = math.nan
        result_bits = NAN_BITS
    else:
        true_sum = a_val + b_val
        result_bits = encode(true_sum)

    # Primary regime (priority order, first match wins)
    if a_is_nan or b_is_nan or (a_zero and b_zero):
        regime = SPECIAL_VALUES
    elif abs(true_sum) > 448.0:
        regime = OVERFLOW
    elif 0.0 < abs(true_sum) < 2.0 ** -10:
        regime = UNDERFLOW_TO_ZERO
    elif is_subnormal_bits(result_bits):
        regime = SUBNORMAL_RESULT
    elif _is_cancellation(a_bits, b_bits, a_val, b_val, true_sum):
        regime = CANCELLATION
    elif is_tie(true_sum):
        regime = ROUNDING_TIE
    elif abs(unbiased_exp(a_bits) - unbiased_exp(b_bits)) >= 4:
        regime = LARGE_DEXP
    else:
        regime = DEFAULT

    tag_mask = _compute_tags(a_bits, b_bits, a_val, b_val, a_kind, b_kind, true_sum)
    return result_bits, regime, tag_mask


def _is_cancellation(
    a_bits: int, b_bits: int, a_val: float, b_val: float, true_sum: float
) -> bool:
    if math.isnan(true_sum):
        return False
    a_sign = (a_bits >> 7) & 1
    b_sign = (b_bits >> 7) & 1
    if a_sign == b_sign:
        return False
    max_abs = max(abs(a_val), abs(b_val))
    if max_abs == 0:
        return False
    return abs(true_sum) <= max_abs / 4.0


def _compute_tags(
    a_bits: int,
    b_bits: int,
    a_val: float,
    b_val: float,
    a_kind: str,
    b_kind: str,
    true_sum: float,
) -> int:
    tags = 0

    if ((a_bits >> 7) & 1) == ((b_bits >> 7) & 1):
        tags |= TAG_SAME_SIGN
    else:
        tags |= TAG_OPPOSITE_SIGN

    dexp = abs(unbiased_exp(a_bits) - unbiased_exp(b_bits))
    if dexp == 0:
        tags |= TAG_SAME_EXP
    elif dexp <= 3:
        tags |= TAG_SMALL_DEXP
    else:
        tags |= TAG_LARGE_DEXP

    if is_tie(true_sum):
        tags |= TAG_TIE

    if a_kind == "subnormal" or b_kind == "subnormal":
        tags |= TAG_SUBNORMAL_INPUT

    if is_result_exact(true_sum):
        tags |= TAG_RESULT_EXACT

    return tags


def _spot_checks() -> None:
    # special-values: NaN input
    _, r, _ = classify(0x7F, 0x00)
    assert r == SPECIAL_VALUES, f"NaN+0 -> {REGIME_NAMES[r]}"

    # special-values: +0 + -0
    _, r, _ = classify(0x00, 0x80)
    assert r == SPECIAL_VALUES, f"+0+-0 -> {REGIME_NAMES[r]}"

    # overflow: 448 + 448
    _, r, _ = classify(0x7E, 0x7E)
    assert r == OVERFLOW, f"448+448 -> {REGIME_NAMES[r]}"

    # cancellation: 1.0 + (-1.0) = 0  (subnormal-result fails because result is +0, not subnormal)
    _, r, _ = classify(0x38, 0xB8)
    assert r == CANCELLATION, f"1.0+(-1.0) -> {REGIME_NAMES[r]}"

    # large-dexp: 1.0 + smallest normal 2^-6 (Δexp = 6 >= 4, no other primary triggers)
    _, r, _ = classify(0x38, 0x08)
    assert r == LARGE_DEXP, f"1.0+2^-6 -> {REGIME_NAMES[r]}"

    # default: 1.0 + 1.0 = 2.0  (clean same-sign small-exp add)
    _, r, _ = classify(0x38, 0x38)
    assert r == DEFAULT, f"1.0+1.0 -> {REGIME_NAMES[r]}"

    # subnormal-result: 2^-9 + 2^-9 = 2^-8 (subnormal)
    _, r, _ = classify(0x01, 0x01)
    assert r == SUBNORMAL_RESULT, f"2^-9+2^-9 -> {REGIME_NAMES[r]}"

    # tags: same-sign on (+0, +0); subnormal-input on (2^-9, *)
    _, _, t = classify(0x00, 0x00)
    # +0+0 -> special-values, but tags still computed
    assert t & TAG_SAME_SIGN, "same-sign tag missing on (+0,+0)"
    _, _, t = classify(0x01, 0x38)  # subnormal + 1.0
    assert t & TAG_SUBNORMAL_INPUT, "subnormal-input tag missing"
    assert t & TAG_LARGE_DEXP, "large-dexp tag missing (Δexp >= 4)"

    print("all regime spot checks passed")


if __name__ == "__main__":
    _spot_checks()
