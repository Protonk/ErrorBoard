"""Pure-Python E4M3 (OCP / Micikevicius 2022) reference implementation.

Source: Micikevicius et al., "FP8 Formats for Deep Learning," arXiv:2209.05433v2,
Table 1 and §3.1. Same spec as the OCP OFP8 Revision 1.0 (2023-12-01).

Conventions:
    - 1 sign bit, 4 exponent bits, 3 mantissa bits (S.EEEE.MMM).
    - Exponent bias 7.
    - No infinities. S.1111.111 (two bit patterns) are the only NaNs.
    - Max finite = 448 = 1.110 * 2^8 (bit pattern S.1111.110).
    - Min normal = 2^-6. Min subnormal = 2^-9.
    - IEEE-style signed zero: +0 and -0 both representable.
    - Round-to-nearest-even (RNE). Overflow saturates to +-448 (no overflow-to-NaN).

The reference is spec-derived; `torch.float8_e4m3fn` is validated against it elsewhere.
"""

from __future__ import annotations

import math

NAN_BITS = 0x7F  # Canonical NaN we emit (positive); 0xFF is the other valid NaN pattern.
MAX_FINITE_BITS = 0x7E  # +448 = 1.110 * 2^8

EXP_BIAS = 7
MANTISSA_BITS = 3
EXP_MASK = 0xF
MANT_MASK = 0x7


def _is_nan_bits(bits: int) -> bool:
    return (bits & 0x7F) == 0x7F


def _round_half_to_even(x: float) -> int:
    """Round to nearest integer with banker's rounding (round-half-to-even)."""
    floor_x = math.floor(x)
    diff = x - floor_x
    if diff < 0.5:
        return int(floor_x)
    if diff > 0.5:
        return int(floor_x) + 1
    # Exact tie: round to the integer with even LSB.
    return int(floor_x) if (int(floor_x) % 2 == 0) else int(floor_x) + 1


def decode(bits: int) -> tuple[float, str]:
    """Decode an 8-bit E4M3 OCP pattern to (value, kind).

    kind is one of {"normal", "subnormal", "zero", "nan"}.
    For NaN inputs the returned value is float("nan") and the sign bit is dropped
    (we treat the two NaN patterns as equivalent at the value level).
    """
    if not (0 <= bits <= 0xFF):
        raise ValueError(f"bits must be in [0, 255], got {bits}")

    sign = (bits >> 7) & 1
    exp = (bits >> 3) & EXP_MASK
    mant = bits & MANT_MASK
    s = -1.0 if sign else 1.0

    # NaN: S.1111.111 (both signs valid; treated as equivalent NaN at value level)
    if exp == EXP_MASK and mant == MANT_MASK:
        return float("nan"), "nan"

    # Zero: S.0000.000 (signed)
    if exp == 0 and mant == 0:
        return s * 0.0, "zero"

    # Subnormal: S.0000.MMM with M != 0; value = s * (M/8) * 2^-6
    if exp == 0:
        return s * (mant / 8.0) * (2.0 ** -6), "subnormal"

    # Normal: S.EEEE.MMM with E != 0; value = s * (1 + M/8) * 2^(E - 7)
    return s * (1.0 + mant / 8.0) * (2.0 ** (exp - EXP_BIAS)), "normal"


def encode(value: float) -> int:
    """Encode a Python float to an E4M3 OCP bit pattern with RNE and saturation.

    NaN inputs produce NAN_BITS (canonical positive NaN). Infinities saturate to +-448.
    Underflow rounds to +-0 per RNE (the even-mantissa neighbor of the tie is 0).
    Overflow saturates to +-448 (no overflow-to-NaN).
    """
    if math.isnan(value):
        return NAN_BITS

    if value == 0.0:
        # Distinguish +0 and -0 via copysign (Python preserves the sign bit on 0).
        sign = 1 if math.copysign(1.0, value) < 0 else 0
        return sign << 7

    sign = 1 if value < 0 else 0
    abs_value = -value if value < 0 else value

    if math.isinf(abs_value):
        return (sign << 7) | (EXP_MASK << 3) | 0x6  # +-448

    # frexp gives (m, e) with 0.5 <= m < 1 and abs_value == m * 2^e.
    # Shift to 1 <= m < 2 by doubling m and subtracting 1 from e.
    m, e = math.frexp(abs_value)
    m *= 2.0
    e_unbiased = e - 1

    # Subnormal branch: e_unbiased < -6 means smaller than min normal 2^-6.
    if e_unbiased < -6:
        # value / 2^-9 is the would-be subnormal mantissa as a real in [0, 8).
        m_sub_real = abs_value / (2.0 ** -9)
        m_sub_int = _round_half_to_even(m_sub_real)
        if m_sub_int == 0:
            return sign << 7  # underflow to +-0
        if m_sub_int >= 8:
            # Rounded across the subnormal/normal boundary into smallest normal 2^-6.
            return (sign << 7) | (1 << 3) | 0
        return (sign << 7) | (0 << 3) | m_sub_int

    # Normal branch.
    biased_exp = e_unbiased + EXP_BIAS
    frac_real = (m - 1.0) * 8.0  # in [0, 8)
    frac_int = _round_half_to_even(frac_real)

    if frac_int == 8:
        # Mantissa wrapped: 1.111... rounded up. Bump exponent, mantissa = 0.
        frac_int = 0
        biased_exp += 1

    # Overflow handling: max finite is exp=15, mant=6 (=448). exp=15 mant=7 is NaN slot.
    if biased_exp > EXP_MASK or (biased_exp == EXP_MASK and frac_int >= 0x7):
        return (sign << 7) | (EXP_MASK << 3) | 0x6  # saturate to +-448

    return (sign << 7) | (biased_exp << 3) | frac_int


def add(a_bits: int, b_bits: int) -> int:
    """Add two E4M3 numbers (bit patterns); return result bit pattern.

    NaN inputs produce NAN_BITS. The exact sum is computed in float64 (which has 53
    bits of precision -- vastly more than the ~12 bits any E4M3 sum can occupy) and
    then re-encoded under RNE + saturation.
    """
    if _is_nan_bits(a_bits) or _is_nan_bits(b_bits):
        return NAN_BITS

    a_val, _ = decode(a_bits)
    b_val, _ = decode(b_bits)
    true_sum = a_val + b_val
    return encode(true_sum)


def mul(a_bits: int, b_bits: int) -> int:
    """Multiply two E4M3 numbers (bit patterns); return result bit pattern.

    NaN inputs produce NAN_BITS (no NaN-times-zero NaN-injection: OCP E4M3 has no
    inf, so the only NaN source is NaN propagation from the inputs). The exact
    product is computed in float64 -- E4M3's largest finite is 448, so products
    cap at 448^2 = 200,704, well within float64's range -- and re-encoded under
    RNE + saturation.

    Signed zero follows IEEE: Python's `*` preserves the sign bit on zero, and
    encode() reads it via math.copysign.
    """
    if _is_nan_bits(a_bits) or _is_nan_bits(b_bits):
        return NAN_BITS

    a_val, _ = decode(a_bits)
    b_val, _ = decode(b_bits)
    true_product = a_val * b_val
    return encode(true_product)


def reciprocal(a_bits: int) -> int:
    """Compute 1/a for an E4M3 input; return result bit pattern.

    NaN input → NAN_BITS. Zero input (signed) → saturates to ±448 per OCP-fn
    overflow semantics; the result sign matches the input sign (so 1/(+0)=+448,
    1/(-0)=-448). Otherwise: compute 1/a in float64 and re-encode under
    RNE + saturation.

    Smallest positive subnormal 2^-9 gives 1/2^-9 = 512 > 448 → saturates to 448.
    Smallest representable normal 2^-6 gives 1/2^-6 = 64 (representable).
    Largest finite 448 gives 1/448 ≈ 0.002232... → rounds to nearest subnormal.
    """
    if _is_nan_bits(a_bits):
        return NAN_BITS

    a_val, kind = decode(a_bits)
    if kind == "zero":
        sign = (a_bits >> 7) & 1
        return (sign << 7) | (EXP_MASK << 3) | 0x6  # ±448

    true_recip = 1.0 / a_val
    return encode(true_recip)


# ---- spot-check assertions exercised on `python -m errorboard.oracle` ----

def _spot_checks() -> None:
    # Decode: spec landmarks
    assert decode(0x00) == (0.0, "zero")
    assert decode(0x80)[1] == "zero" and math.copysign(1.0, decode(0x80)[0]) < 0
    assert decode(0x7E) == (448.0, "normal")
    assert decode(0xFE) == (-448.0, "normal")
    assert decode(0x7F)[1] == "nan"
    assert decode(0xFF)[1] == "nan"
    assert decode(0x08) == (2.0 ** -6, "normal")          # min positive normal
    assert decode(0x01) == (2.0 ** -9, "subnormal")       # min positive subnormal
    assert decode(0x07) == (0.875 * 2.0 ** -6, "subnormal")  # max positive subnormal

    # Encode: round-trip on representable values
    for bits in range(256):
        if _is_nan_bits(bits):
            continue
        v, _ = decode(bits)
        # +-0 collapse: encode(-0.0) and encode(+0.0) are distinct; encode(decode(bits)) must match bits.
        assert encode(v) == bits, f"round-trip failed: bits={bits:#04x}, value={v}"

    # Encode: NaN
    assert encode(float("nan")) == NAN_BITS

    # Encode: saturation
    assert encode(1e10) == MAX_FINITE_BITS              # +overflow
    assert encode(-1e10) == (0x80 | MAX_FINITE_BITS)     # -overflow
    assert encode(float("inf")) == MAX_FINITE_BITS
    assert encode(float("-inf")) == 0x80 | MAX_FINITE_BITS

    # Encode: underflow direction. 2^-10 ties between 0 and 2^-9; RNE -> 0 (even mantissa).
    assert encode(2.0 ** -10) == 0x00
    # Just-above tie rounds up to min subnormal.
    assert encode(2.0 ** -10 * 1.0001) == 0x01

    # Encode: subnormal-to-normal boundary tie.
    # 7.5/8 * 2^-6 ties between subnormal 0x07 (7/8 * 2^-6) and normal 0x08 (1.0 * 2^-6).
    # Subnormal 0x07 has mantissa LSB 1 (odd); normal 0x08 has mantissa 0 (even). Tie -> 0x08.
    assert encode(7.5 / 8 * 2.0 ** -6) == 0x08

    # Addition: NaN propagation
    assert add(0x00, 0x7F) == NAN_BITS
    assert add(0xFF, 0x08) == NAN_BITS

    # Addition: signed zero per RNE
    assert add(0x00, 0x80) == 0x00      # +0 + -0 = +0
    assert add(0x80, 0x80) == 0x80      # -0 + -0 = -0
    assert add(0x00, 0x00) == 0x00      # +0 + +0 = +0

    # Addition: trivial same-exponent
    assert add(0x08, 0x08) == 0x10      # 2^-6 + 2^-6 = 2^-5  (bits: 1 << 4 = 0x10)
    assert add(0x38, 0x38) == 0x40      # 1.0 + 1.0 = 2.0

    # Addition: total cancellation
    assert add(0x38, 0xB8) == 0x00      # 1.0 + (-1.0) = +0

    # Addition: overflow saturates
    assert add(0x7E, 0x7E) == 0x7E      # 448 + 448 -> saturate to 448
    assert add(0xFE, 0xFE) == 0xFE      # -448 + -448 -> -448

    # Addition: underflow
    assert add(0x01, 0x81) == 0x00      # 2^-9 + (-2^-9) = +0

    # Multiplication: NaN propagation
    assert mul(0x00, 0x7F) == NAN_BITS
    assert mul(0xFF, 0x08) == NAN_BITS
    assert mul(0x7F, 0x7F) == NAN_BITS

    # Multiplication: signed-zero (sign = XOR)
    assert mul(0x00, 0x00) == 0x00                 # +0 * +0 = +0
    assert mul(0x80, 0x80) == 0x00                 # -0 * -0 = +0
    assert mul(0x00, 0x80) == 0x80                 # +0 * -0 = -0
    assert mul(0x38, 0x00) == 0x00                 # 1.0 * +0 = +0
    assert mul(0xB8, 0x00) == 0x80                 # -1.0 * +0 = -0
    assert mul(0xB8, 0x80) == 0x00                 # -1.0 * -0 = +0

    # Multiplication: sign XOR on non-zero operands
    assert mul(0x38, 0x38) == 0x38                 # 1.0 * 1.0 = 1.0
    assert mul(0x38, 0xB8) == 0xB8                 # 1.0 * -1.0 = -1.0
    assert mul(0xB8, 0xB8) == 0x38                 # -1.0 * -1.0 = 1.0

    # Multiplication: power-of-2 (exponent addition, no mantissa rounding)
    assert mul(0x38, 0x40) == 0x40                 # 1.0 * 2.0 = 2.0
    assert mul(0x40, 0x40) == 0x48                 # 2.0 * 2.0 = 4.0
    assert mul(0x08, 0x40) == 0x10                 # 2^-6 * 2.0 = 2^-5

    # Multiplication: exact mantissa product 1.25 * 1.25 = 1.5625 (representable)
    # 1.25 = 1 + 1/4 = mantissa 0x2 at exp 7 -> bit pattern 0x3A
    # 1.5625 = 1 + 9/16: but mantissa has only 3 bits = 1/8 resolution.
    # 1.5 = 1 + 4/8 = 0x3C, 1.625 = 1 + 5/8 = 0x3D. So 1.5625 rounds to 1.5 (RNE: 4 even, 5 odd).
    assert mul(0x3A, 0x3A) == 0x3C                 # 1.25 * 1.25 -> RNE -> 1.5

    # Multiplication: overflow saturates to +-448
    assert mul(0x7E, 0x40) == 0x7E                 # 448 * 2.0 -> saturate to 448
    assert mul(0x7E, 0x7E) == 0x7E                 # 448 * 448 -> saturate
    assert mul(0xFE, 0x40) == 0xFE                 # -448 * 2.0 -> -448

    # Multiplication: underflow rounds to zero (tiny * tiny << min subnormal)
    assert mul(0x01, 0x01) == 0x00                 # 2^-9 * 2^-9 = 2^-18 -> +0
    assert mul(0x01, 0x81) == 0x80                 # 2^-9 * -2^-9 = -2^-18 -> -0

    # Multiplication: subnormal arithmetic
    # 2^-9 * 2.0 = 2^-8, which is a subnormal (2/8 * 2^-6 = 2^-8). Bit 0x02.
    assert mul(0x01, 0x40) == 0x02                 # 2^-9 * 2.0 = 2^-8

    # Reciprocal: NaN propagation
    assert reciprocal(0x7F) == NAN_BITS
    assert reciprocal(0xFF) == NAN_BITS

    # Reciprocal: zero overflows to ±448 (sign preserved)
    assert reciprocal(0x00) == 0x7E                # 1/(+0) = +448
    assert reciprocal(0x80) == 0xFE                # 1/(-0) = -448

    # Reciprocal: identities
    assert reciprocal(0x38) == 0x38                # 1/1 = 1
    assert reciprocal(0xB8) == 0xB8                # 1/(-1) = -1
    assert reciprocal(0x40) == 0x30                # 1/2 = 0.5
    assert reciprocal(0x30) == 0x40                # 1/0.5 = 2

    # Reciprocal: power-of-2 ladder (exponent negation)
    assert reciprocal(0x48) == 0x28                # 1/4 = 0.25
    assert reciprocal(0x28) == 0x48                # 1/0.25 = 4
    assert reciprocal(0x08) == 0x68                # 1/(2^-6) = 64

    # Reciprocal: overflow saturation (smallest normals reach max finite)
    # 1/(min normal 2^-6) = 64 = 1.0 * 2^6 -> bit pattern 0x68 (representable, NOT saturated)
    # 1/(min subnormal 2^-9) = 512 > 448 -> saturates to 448 = 0x7E
    assert reciprocal(0x01) == 0x7E                # 1/(2^-9) overflows
    assert reciprocal(0x81) == 0xFE                # 1/(-2^-9) overflows negatively

    # Reciprocal: subnormal output for very large input
    # 1/448 ≈ 0.002232 → rounds to nearest subnormal/normal.
    # 0.002232 / 2^-9 = 1.143; banker's rounding -> 1 → subnormal m=1 → bit 0x01.
    assert reciprocal(0x7E) == 0x01                # 1/448 → smallest positive subnormal
    assert reciprocal(0xFE) == 0x81                # 1/(-448) → negative smallest subnormal

    # Reciprocal: positive-normal worst input (dayval's 0x75)
    # 0x75 = 1.625 * 2^7 = 208. 1/208 ≈ 0.004807692; smallest representable
    # subnormal step is 2^-9 ≈ 0.001953. 0.004807 / 2^-9 ≈ 2.46 → rounds to 2.
    # So result is 2 * 2^-9 = 0.00390625 → bit 0x02.
    assert reciprocal(0x75) == 0x02                # 1/208 → 2 * 2^-9 = 0.00390625

    # Reciprocal involution: applying twice should give back the original on
    # values where 1/(1/x) round-trips exactly. Powers of 2 always do.
    for bits in [0x38, 0xB8, 0x40, 0x30, 0x48, 0x28, 0x08, 0x68]:
        assert reciprocal(reciprocal(bits)) == bits, (
            f"recip(recip(0x{bits:02x})) != 0x{bits:02x}"
        )

    print("all spot checks passed")


if __name__ == "__main__":
    _spot_checks()
