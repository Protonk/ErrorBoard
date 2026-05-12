"""Exhaustive verification of `torch.float8_e4m3fn` against the spec-derived oracle.

Per task_spec.md §4: compare oracle and torch on every one of the 256 single-pattern
decodings and every one of the 65,536 ordered addition pairs. Bit-level comparison;
NaN treated as a value-equivalence class but the verifier still tracks bit patterns.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import torch

from errorboard.oracle import NAN_BITS, add, decode, encode

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


def _nan_equivalent_bits(x: int, y: int) -> bool:
    """Are bit patterns x and y the same modulo the two NaN encodings?"""
    if x == y:
        return True
    nan = lambda b: (b & 0x7F) == 0x7F
    return nan(x) and nan(y)


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


def main() -> int:
    print(f"torch version: {torch.__version__}")
    print(f"dtype: {torch_fp8}")
    print()

    print("=" * 60)
    print("Test 1/3: decode comparison (256 bit patterns)")
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
    print("Test 2/3: encode round-trip (254 finite patterns)")
    print("=" * 60)
    rt_mismatches = verify_encode_roundtrip()
    if rt_mismatches:
        print(f"FAIL: {len(rt_mismatches)} round-trip mismatches")
        for bits, re in rt_mismatches[:10]:
            print(f"  bits={bits:#04x} -> value -> encoded={re:#04x}")
    else:
        print("PASS: all 254 non-NaN patterns round-trip exactly")
    print()

    print("=" * 60)
    print("Test 3/3: addition agreement (65,536 ordered pairs)")
    print("=" * 60)
    n, add_mismatches = verify_add_exhaustive()
    if add_mismatches:
        print(f"FAIL: {len(add_mismatches)} / {n} addition mismatches")
        for a_b, b_b, o_r, t_r in add_mismatches[:20]:
            av, _ = decode(a_b)
            bv, _ = decode(b_b)
            ov, _ = decode(o_r)
            tv = _torch_bits_to_value(t_r)
            print(
                f"  a={a_b:#04x}({av!r}) + b={b_b:#04x}({bv!r}): "
                f"oracle={o_r:#04x}({ov!r}), torch={t_r:#04x}({tv!r})"
            )
        if len(add_mismatches) > 20:
            print(f"  ... and {len(add_mismatches) - 20} more")
    else:
        print(f"PASS: all {n} pairs agree (NaN bit-patterns treated as equivalent)")
    print()

    any_fail = bool(dec_mismatches or rt_mismatches or add_mismatches)
    print("=" * 60)
    print("OVERALL:", "FAIL" if any_fail else "PASS")
    print("=" * 60)
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
