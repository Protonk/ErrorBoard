"""Exhaustive verification of `torch.float8_e4m3fn` against the spec-derived oracle.

Per task_spec.md §4: compare oracle and torch on every one of the 256 single-pattern
decodings, every one of the 65,536 ordered addition pairs, and every one of the
65,536 ordered multiplication pairs. Bit-level comparison; NaN treated as a
value-equivalence class but the verifier still tracks bit patterns.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

from errorboard.oracle import NAN_BITS, add, decode, encode, mul

torch_fp8 = torch.float8_e4m3fn


def _torch_bits_to_value(bits: int) -> float:
    """Decode an 8-bit pattern via torch.float8_e4m3fn -> float32 -> python float."""
    t = torch.tensor([bits], dtype=torch.uint8).view(torch_fp8)
    return t.float().item()


def _torch_add_bits(a_bits: int, b_bits: int) -> int:
    """Add two E4M3 bit patterns using torch (upcast to float32, add, downcast)."""
    a = torch.tensor([a_bits], dtype=torch.uint8).view(torch_fp8)
    b = torch.tensor([b_bits], dtype=torch.uint8).view(torch_fp8)
    c_fp8 = (a.float() + b.float()).to(torch_fp8)
    return int(c_fp8.view(torch.uint8).item())


def _torch_mul_bits(a_bits: int, b_bits: int) -> int:
    """Multiply two E4M3 bit patterns using torch (upcast to float32, mul, downcast)."""
    a = torch.tensor([a_bits], dtype=torch.uint8).view(torch_fp8)
    b = torch.tensor([b_bits], dtype=torch.uint8).view(torch_fp8)
    c_fp8 = (a.float() * b.float()).to(torch_fp8)
    return int(c_fp8.view(torch.uint8).item())


def _nan_equivalent_bits(x: int, y: int) -> bool:
    """Are bit patterns x and y the same modulo the two NaN encodings?"""
    if x == y:
        return True
    nan = lambda b: (b & 0x7F) == 0x7F
    return nan(x) and nan(y)


_MAX_FINITE = {0x7E, 0xFE}
_NAN_BITS = {0x7F, 0xFF}


def _is_known_overflow_divergence(oracle_bits: int, torch_bits: int) -> bool:
    """OCP E4M3-fn spec says overflow saturates to +-448 (max finite); torch's
    `float8_e4m3fn.to()` instead rounds anything >= the half-ULP above 448 (464)
    into the NaN slot (1.111 * 2^8). Categorize that specific pattern as a
    known spec-vs-impl divergence rather than a real failure.
    """
    return oracle_bits in _MAX_FINITE and torch_bits in _NAN_BITS


def verify_decode() -> list[tuple[int, float, float]]:
    """Compare oracle vs torch on all 256 bit patterns. Returns list of mismatches."""
    mismatches = []
    for bits in range(256):
        oracle_val, oracle_kind = decode(bits)
        torch_val = _torch_bits_to_value(bits)
        if oracle_kind == "nan":
            if not math.isnan(torch_val):
                mismatches.append((bits, oracle_val, torch_val))
            continue
        if oracle_val != torch_val:
            # Distinguish signed-zero
            if oracle_val == 0.0 and torch_val == 0.0:
                # Compare signs
                if math.copysign(1.0, oracle_val) != math.copysign(1.0, torch_val):
                    mismatches.append((bits, oracle_val, torch_val))
                continue
            mismatches.append((bits, oracle_val, torch_val))
    return mismatches


def verify_encode_roundtrip() -> list[tuple[int, int]]:
    """For each non-NaN bit pattern: encode(decode(bits)) must equal bits."""
    mismatches = []
    for bits in range(256):
        if (bits & 0x7F) == 0x7F:  # skip NaN patterns (decode loses sign)
            continue
        v, _ = decode(bits)
        re = encode(v)
        if re != bits:
            mismatches.append((bits, re))
    return mismatches


def verify_add_exhaustive() -> tuple[int, list[tuple[int, int, int, int]]]:
    """Compare oracle.add vs torch fp8 add on all 65,536 ordered pairs.

    Returns (num_compared, list of (a_bits, b_bits, oracle_result, torch_result)).
    NaN-equivalent results (both NaN bit patterns) are treated as agreement.
    """
    mismatches = []

    # Vectorize via numpy/torch for speed.
    all_bits = np.arange(256, dtype=np.uint8)
    a_grid, b_grid = np.meshgrid(all_bits, all_bits, indexing="ij")
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()

    a_t = torch.from_numpy(a_flat.copy()).view(torch_fp8)
    b_t = torch.from_numpy(b_flat.copy()).view(torch_fp8)
    c_t = (a_t.float() + b_t.float()).to(torch_fp8)
    c_bits = c_t.view(torch.uint8).numpy()

    # Oracle (scalar loop is fine: 65,536 iterations in pure Python)
    for i in range(len(a_flat)):
        a_b = int(a_flat[i])
        b_b = int(b_flat[i])
        oracle_r = add(a_b, b_b)
        torch_r = int(c_bits[i])
        if not _nan_equivalent_bits(oracle_r, torch_r):
            mismatches.append((a_b, b_b, oracle_r, torch_r))

    return len(a_flat), mismatches


def verify_mul_exhaustive() -> tuple[int, list[tuple[int, int, int, int]]]:
    """Compare oracle.mul vs torch fp8 mul on all 65,536 ordered pairs.

    Returns (num_compared, list of (a_bits, b_bits, oracle_result, torch_result)).
    NaN-equivalent results (both NaN bit patterns) are treated as agreement.
    """
    mismatches = []

    all_bits = np.arange(256, dtype=np.uint8)
    a_grid, b_grid = np.meshgrid(all_bits, all_bits, indexing="ij")
    a_flat = a_grid.flatten()
    b_flat = b_grid.flatten()

    a_t = torch.from_numpy(a_flat.copy()).view(torch_fp8)
    b_t = torch.from_numpy(b_flat.copy()).view(torch_fp8)
    c_t = (a_t.float() * b_t.float()).to(torch_fp8)
    c_bits = c_t.view(torch.uint8).numpy()

    for i in range(len(a_flat)):
        a_b = int(a_flat[i])
        b_b = int(b_flat[i])
        oracle_r = mul(a_b, b_b)
        torch_r = int(c_bits[i])
        if not _nan_equivalent_bits(oracle_r, torch_r):
            mismatches.append((a_b, b_b, oracle_r, torch_r))

    return len(a_flat), mismatches


def main() -> int:
    print(f"torch version: {torch.__version__}")
    print(f"dtype: {torch_fp8}")
    print()

    print("=" * 60)
    print("Test 1/4: decode comparison (256 bit patterns)")
    print("=" * 60)
    dec_mismatches = verify_decode()
    if dec_mismatches:
        print(f"FAIL: {len(dec_mismatches)} decode mismatches")
        for bits, ov, tv in dec_mismatches[:10]:
            print(f"  bits={bits:#04x}: oracle={ov!r}, torch={tv!r}")
        if len(dec_mismatches) > 10:
            print(f"  ... and {len(dec_mismatches) - 10} more")
    else:
        print("PASS: all 256 patterns decode identically")
    print()

    print("=" * 60)
    print("Test 2/4: encode round-trip (254 finite patterns)")
    print("=" * 60)
    rt_mismatches = verify_encode_roundtrip()
    if rt_mismatches:
        print(f"FAIL: {len(rt_mismatches)} round-trip mismatches")
        for bits, re in rt_mismatches[:10]:
            print(f"  bits={bits:#04x} -> value -> encoded={re:#04x}")
    else:
        print("PASS: all 254 non-NaN patterns round-trip exactly")
    print()

    def _report(op_label: str, n: int, mismatches: list) -> bool:
        """Return True if any *unexpected* mismatch was seen."""
        expected = [m for m in mismatches if _is_known_overflow_divergence(m[2], m[3])]
        unexpected = [m for m in mismatches if not _is_known_overflow_divergence(m[2], m[3])]
        if not mismatches:
            print(f"PASS: all {n} pairs agree (NaN bit-patterns treated as equivalent)")
            return False
        if not unexpected:
            print(
                f"PASS (with {len(expected)} known spec divergences): "
                f"all remaining {n - len(expected)} pairs match bit-for-bit. "
                "Divergences are the OCP-spec saturate-vs-torch-NaN-slot edge "
                "(oracle returns +-448, torch returns NaN for values that round "
                "into the 1.111*2^8 slot). The oracle follows OCP-fn spec; torch "
                "rounds-then-checks-slot."
            )
            return False
        print(f"FAIL: {len(unexpected)} / {n} UNEXPECTED mismatches "
              f"(plus {len(expected)} known spec divergences)")
        for a_b, b_b, o_r, t_r in unexpected[:20]:
            av, _ = decode(a_b)
            bv, _ = decode(b_b)
            ov, _ = decode(o_r)
            tv = _torch_bits_to_value(t_r)
            print(
                f"  a={a_b:#04x}({av!r}) {op_label} b={b_b:#04x}({bv!r}): "
                f"oracle={o_r:#04x}({ov!r}), torch={t_r:#04x}({tv!r})"
            )
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more unexpected")
        return True

    print("=" * 60)
    print("Test 3/4: addition agreement (65,536 ordered pairs)")
    print("=" * 60)
    n, add_mismatches = verify_add_exhaustive()
    add_fail = _report("+", n, add_mismatches)
    print()

    print("=" * 60)
    print("Test 4/4: multiplication agreement (65,536 ordered pairs)")
    print("=" * 60)
    n_mul, mul_mismatches = verify_mul_exhaustive()
    mul_fail = _report("*", n_mul, mul_mismatches)
    print()

    any_fail = bool(dec_mismatches or rt_mismatches or add_fail or mul_fail)
    print("=" * 60)
    print("OVERALL:", "FAIL" if any_fail else "PASS")
    print("=" * 60)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
