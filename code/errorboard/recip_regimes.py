"""Regime classification for FP8 E4M3 reciprocal.

Parallel in shape to `regimes.py` and `mult_regimes.py` but unary: classifies
single FP8 inputs rather than ordered pairs. The "table" is 256 entries
(one per input bit pattern) rather than 65,536.

Regimes (priority order, first-match wins):
  0 SPECIAL_VALUES    NaN, ±0
  1 OVERFLOW          1/x > 448 → saturates to ±448 (only the smallest
                      subnormals: |x| < ~0.00223)
  2 SUBNORMAL_RESULT  1/x is a nonzero subnormal in [2^-9, 2^-6) — happens
                      for large-magnitude normal inputs (|x| > 64)
  3 EXACT_RESULT      1/x is exactly representable in finite FP8 (no
                      rounding needed). Power-of-2 inputs always land here;
                      a handful of non-power-of-2 cases too.
  4 ROUNDING_TIE      1/x is exact midpoint of two adjacent FP8 values
  5 DEFAULT           normal-result, rounding required

Tags (bitmask, overlap allowed):
  - SAME_SIGN_POS / NEG       sign of input
  - POWER_OF_TWO_INPUT        x is a power of 2 (mantissa=0, non-special)
  - SUBNORMAL_INPUT
  - LARGE_INPUT               |x| ≥ 16 (1/x small)
  - SMALL_INPUT               |x| ≤ 0.0625 (1/x large)
  - RESULT_EXACT              mirrors EXACT_RESULT
  - TIE                       mirrors ROUNDING_TIE
  - NEAR_WORST                input is within ±2 ULPs of dayval's worst input
                              0x75 (the format-intrinsic ε_floor witness)

dayval's `lowbit` reference (`code/dayval/results/lowbit/fp8e4m3.txt`)
reports the format-intrinsic floor as 0.1875 with worst input 0x75. The
NEAR_WORST tag lets us slice the holdout near that witness specifically.
"""

from __future__ import annotations

import bisect
import math
from typing import Final

from .oracle import NAN_BITS, decode, encode, reciprocal

# ---- primary regime IDs ----

SPECIAL_VALUES: Final[int] = 0
OVERFLOW: Final[int] = 1
SUBNORMAL_RESULT: Final[int] = 2
EXACT_RESULT: Final[int] = 3
ROUNDING_TIE: Final[int] = 4
DEFAULT: Final[int] = 5

REGIME_NAMES: Final[tuple[str, ...]] = (
    "special-values",
    "overflow",
    "subnormal-result",
    "exact-result",
    "rounding-tie",
    "default",
)
NUM_REGIMES: Final[int] = len(REGIME_NAMES)

# ---- secondary tag bit positions ----

TAG_SIGN_POS: Final[int] = 1 << 0
TAG_SIGN_NEG: Final[int] = 1 << 1
TAG_POWER_OF_TWO_INPUT: Final[int] = 1 << 2
TAG_SUBNORMAL_INPUT: Final[int] = 1 << 3
TAG_LARGE_INPUT: Final[int] = 1 << 4
TAG_SMALL_INPUT: Final[int] = 1 << 5
TAG_RESULT_EXACT: Final[int] = 1 << 6
TAG_TIE: Final[int] = 1 << 7
TAG_NEAR_WORST: Final[int] = 1 << 8

TAG_NAMES: Final[dict[int, str]] = {
    TAG_SIGN_POS: "sign-pos",
    TAG_SIGN_NEG: "sign-neg",
    TAG_POWER_OF_TWO_INPUT: "power-of-two-input",
    TAG_SUBNORMAL_INPUT: "subnormal-input",
    TAG_LARGE_INPUT: "large-input",
    TAG_SMALL_INPUT: "small-input",
    TAG_RESULT_EXACT: "result-exact",
    TAG_TIE: "tie",
    TAG_NEAR_WORST: "near-worst",
}

# dayval's format-intrinsic-floor witness for FP8 E4M3 reciprocal.
DAYVAL_WORST_INPUT: Final[int] = 0x75
DAYVAL_EPS_FLOOR: Final[float] = 0.1875


# ---- helpers ----

def _build_fp8_grid() -> list[float]:
    vals = set()
    for b in range(256):
        v, kind = decode(b)
        if kind != "nan":
            vals.add(v)
    return sorted(vals)


_FP8_GRID = _build_fp8_grid()
_FP8_GRID_SET = set(_FP8_GRID)


def unbiased_exp(bits: int) -> int:
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
    """True if the value is a power of 2 (mantissa 0, non-zero, non-NaN)."""
    mant = bits & 0x7
    if mant != 0:
        return False
    exp = (bits >> 3) & 0xF
    if exp == 0:
        return False  # zero
    return True  # exp 1..15 with mant=0; (15, 0) is the +/- 256 binade


def is_tie_value(value: float) -> bool:
    """True if value is the exact midpoint of two adjacent finite FP8 values."""
    if math.isnan(value) or math.isinf(value):
        return False
    if abs(value) > 448.0:
        return False
    idx = bisect.bisect_left(_FP8_GRID, value)
    if idx == 0 or idx == len(_FP8_GRID):
        return False
    if _FP8_GRID[idx] == value:
        return False
    lower = _FP8_GRID[idx - 1]
    upper = _FP8_GRID[idx]
    return value == (lower + upper) / 2.0


def is_result_exact_value(value: float) -> bool:
    if math.isnan(value) or math.isinf(value):
        return False
    return value in _FP8_GRID_SET


# ---- classification ----

def classify(a_bits: int) -> tuple[int, int, int]:
    """Classify reciprocal of a_bits.

    Returns (result_bits, primary_regime_id, tag_mask). Result bits are
    computed via oracle.reciprocal (float64 + RNE + saturation).
    """
    if not (0 <= a_bits <= 0xFF):
        raise ValueError(f"a_bits must be in [0, 255], got {a_bits}")

    result_bits = reciprocal(a_bits)
    a_val, a_kind = decode(a_bits)

    if a_kind == "nan":
        true_recip = float("nan")
    elif a_kind == "zero":
        true_recip = float("inf") if (a_bits & 0x80) == 0 else float("-inf")
    else:
        true_recip = 1.0 / a_val

    # Primary regime (priority order).
    if a_kind == "nan" or a_kind == "zero":
        regime = SPECIAL_VALUES
    elif math.isinf(true_recip) or abs(true_recip) > 448.0:
        regime = OVERFLOW
    elif is_subnormal_bits(result_bits) and result_bits != 0x00 and result_bits != 0x80:
        regime = SUBNORMAL_RESULT
    elif is_tie_value(true_recip):
        regime = ROUNDING_TIE
    elif is_result_exact_value(true_recip):
        regime = EXACT_RESULT
    else:
        regime = DEFAULT

    tag_mask = _compute_tags(a_bits, a_val, a_kind, true_recip)
    return result_bits, regime, tag_mask


def _compute_tags(a_bits: int, a_val: float, a_kind: str, true_recip: float) -> int:
    tags = 0

    if (a_bits >> 7) & 1:
        tags |= TAG_SIGN_NEG
    else:
        tags |= TAG_SIGN_POS

    if is_power_of_two_bits(a_bits):
        tags |= TAG_POWER_OF_TWO_INPUT

    if a_kind == "subnormal":
        tags |= TAG_SUBNORMAL_INPUT

    if a_kind not in ("nan",):
        a_mag = abs(a_val)
        if a_mag >= 16.0:
            tags |= TAG_LARGE_INPUT
        if 0.0 < a_mag <= 0.0625:  # ≤ 1/16
            tags |= TAG_SMALL_INPUT

    if is_result_exact_value(true_recip):
        tags |= TAG_RESULT_EXACT

    if is_tie_value(true_recip):
        tags |= TAG_TIE

    # NEAR_WORST: ±2 ULPs of dayval's worst input 0x75 (= bit pattern around the
    # same binade). Since the worst input is positive, only positive inputs
    # in 0x73..0x77 count. The negative-sign mirror also qualifies.
    abs_bits = a_bits & 0x7F
    if 0x73 <= abs_bits <= 0x77:
        tags |= TAG_NEAR_WORST

    return tags


# ---- self-tests ----

def _spot_checks() -> None:
    import numpy as np

    # NaN
    _, r, _ = classify(0x7F)
    assert r == SPECIAL_VALUES, f"NaN -> {REGIME_NAMES[r]}"
    _, r, _ = classify(0xFF)
    assert r == SPECIAL_VALUES, f"NaN(-) -> {REGIME_NAMES[r]}"

    # Zero
    _, r, _ = classify(0x00)
    assert r == SPECIAL_VALUES, f"+0 -> {REGIME_NAMES[r]}"
    _, r, _ = classify(0x80)
    assert r == SPECIAL_VALUES, f"-0 -> {REGIME_NAMES[r]}"

    # Overflow: 1/(min subnormal) = 1/2^-9 = 512 > 448
    _, r, _ = classify(0x01)
    assert r == OVERFLOW, f"1/2^-9 -> {REGIME_NAMES[r]}"
    _, r, _ = classify(0x81)
    assert r == OVERFLOW, f"1/(-2^-9) -> {REGIME_NAMES[r]}"

    # Subnormal result: 1/448 → smallest positive subnormal
    _, r, _ = classify(0x7E)
    assert r == SUBNORMAL_RESULT, f"1/448 -> {REGIME_NAMES[r]}"

    # Exact result: 1/1 = 1
    _, r, _ = classify(0x38)
    assert r == EXACT_RESULT, f"1/1 -> {REGIME_NAMES[r]}"

    # Exact result: 1/2 = 0.5
    _, r, _ = classify(0x40)
    assert r == EXACT_RESULT, f"1/2 -> {REGIME_NAMES[r]}"

    # Default: 1/3 ≈ 0.333... not exactly representable. 3 in FP8 is 0x42 = 1.5 * 2^1
    # = 3.0, mant=2, exp=8. Wait that's 1.25 * 2 = 2.5 actually. Let me check.
    # bit 0x42 = 0100_0010 → sign=0, exp=1000=8, mant=010=2.
    # value = (1 + 2/8) * 2^(8-7) = 1.25 * 2 = 2.5. So 0x42 is 2.5, not 3.
    # 1/2.5 = 0.4. Is 0.4 representable? 0.4 = 0.011001100..._2, not exact.
    # So 1/2.5 needs rounding → DEFAULT.
    _, r, _ = classify(0x42)
    assert r == DEFAULT, f"1/2.5 -> {REGIME_NAMES[r]}"

    # Tag: dayval's worst input 0x75 should fire NEAR_WORST.
    _, _, t = classify(0x75)
    assert t & TAG_NEAR_WORST, "NEAR_WORST missing on 0x75"

    # Tag: power-of-2 input 0x38 (= 1.0)
    _, _, t = classify(0x38)
    assert t & TAG_POWER_OF_TWO_INPUT, "POW2 missing on 0x38"
    assert t & TAG_RESULT_EXACT, "RESULT_EXACT missing on 0x38"

    # Population census across all 256 inputs.
    counts = [0] * NUM_REGIMES
    tag_counts = {bit: 0 for bit in TAG_NAMES}
    for a in range(256):
        _, r, t = classify(a)
        counts[r] += 1
        for bit in TAG_NAMES:
            if t & bit:
                tag_counts[bit] += 1
    total = sum(counts)
    assert total == 256
    print("Population per regime (out of 256 inputs):")
    for name, count in zip(REGIME_NAMES, counts):
        print(f"  {name:<20s}: {count:>4} ({count/total*100:>5.2f}%)")
    print("\nTag distribution:")
    for bit, name in TAG_NAMES.items():
        n = tag_counts[bit]
        print(f"  {name:<20s}: {n:>4} ({n/total*100:>5.2f}%)")

    # Confirm dayval's floor on input 0x75.
    r_bits, _, _ = classify(0x75)
    r_val, _ = decode(r_bits)
    true_recip = 1.0 / 208.0
    rel_err = abs(r_val - true_recip) / abs(true_recip)
    print(f"\nDayval cross-check: 1/0x75 oracle rel-err = {rel_err:.10f}")
    print(f"dayval ε_floor      = {DAYVAL_EPS_FLOOR}")
    assert abs(rel_err - DAYVAL_EPS_FLOOR) < 1e-10, (
        f"oracle rel-err {rel_err} != dayval floor {DAYVAL_EPS_FLOOR}"
    )

    print("\nall recip-regime spot checks passed")


if __name__ == "__main__":
    _spot_checks()
