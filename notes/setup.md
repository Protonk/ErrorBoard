# FP8 E4M3 — bit layout, special values, range

This note collects the FP8 E4M3 specification as it appears in primary references. Values are not re-derived; everything is quoted or transcribed from the cited source. **Important:** there are multiple incompatible E4M3 conventions in active use; the two principal variants ("OCP/Micikevicius" and "FNUZ") are tabulated side-by-side below.

## Primary reference (OCP / Micikevicius et al. 2022)

Source: Micikevicius, Stosic, Judd, Kamalu, Oberman, Shoeybi, Siu, Wu (NVIDIA); Burgess, Ha, Grisenthwaite (Arm); Mellempudi, Cornea, Heinecke, Dubey (Intel). *"FP8 Formats for Deep Learning."* arXiv:2209.05433v2, 29 Sep 2022. The same spec is normative in the Open Compute Project document *OCP 8-bit Floating Point Specification (OFP8), Revision 1.0*, 2023-12-01 ([opencompute.org](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1)).

### Bit layout

- 1 sign bit · 4 exponent bits · 3 mantissa (trailing significand) bits — 8 bits total.
- Notation used in the source: `S.EEEE.MMM` (subscript 2 indicates binary).

### Table 1 of Micikevicius et al. (verbatim, E4M3 column)

| Field         | E4M3 value (verbatim)                       |
|---------------|---------------------------------------------|
| Exponent bias | `7`                                         |
| Infinities    | `N/A`                                       |
| NaN           | `S.1111.111₂`                               |
| Zeros         | `S.0000.000₂`                               |
| Max normal    | `S.1111.110₂ = 1.75 ∗ 2⁸ = 448`             |
| Min normal    | `S.0001.000₂ = 2⁻⁶`                         |
| Max subnorm   | `S.0000.111₂ = 0.875 ∗ 2⁻⁶`                 |
| Min subnorm   | `S.0000.001₂ = 2⁻⁹`                         |

### Departures from IEEE 754 (Micikevicius §3.1)

Quoted from §3.1 ("Special value representations"):

> "We extend the narrow dynamic range of the E4M3 format by representing fewer special values, adopting their bit patterns for normal values. Infinities are not represented (see Section 2 for overflow handling details) and we retain only one mantissa bit-pattern for NaNs. This modification extends the dynamic range by one extra power of 2, from 17 to 18 binades. We gain the representation of seven more magnitudes (256, 288, 320, 352, 384, 416, 448), corresponding to the biased exponent value 1111₂. The maximum representable magnitude without this modification would be 240."

Concretely, relative to IEEE 754:

- **No ±∞.** The exponent-all-ones pattern is reused for finite values and for NaN.
- **Two NaN encodings only.** `S.1111.111₂` for both signs of S (i.e., `0x7F` and `0xFF`).
- **Zeros and NaN keep IEEE-style symmetry:** the spec explicitly keeps both `+0` and `−0`, and both `+NaN` and `−NaN`. From §3.1: *"For consistency with IEEE 754 conventions we retain positive and negative representations for zero and NaN. While we could gain one additional representable magnitude, 480, by having just one encoding for zero and one for NaN, this would require breaking the symmetry of positive and negative representations inherent in the IEEE 754 formats, complicating or invalidating algorithm implementations that rely on this property."*
- **Exponent bias unchanged from IEEE-style** (bias = 7 for a 4-bit exponent), per §3.2: *"Both E4M3 and E5M2 retain IEEE-like exponent biases: 7 and 15 for E4M3 and E5M2, respectively."*

### Derived range numbers (from the table; not re-derived)

- Max finite: **448** (binary `0.1111.110`, hex `0x7E`).
- Min positive normal: **2⁻⁶ ≈ 1.5625 × 10⁻²**.
- Min positive subnormal: **2⁻⁹ ≈ 1.953 × 10⁻³**.
- Dynamic range: 18 binades (per §3.1, with the special-value reclamation).

## Discrepancy: OCP E4M3 vs. "FNUZ" E4M3

A second, incompatible E4M3 convention is in active hardware and software use. It originates with Graphcore and is used by AMD CDNA3 and is recognized in ONNX as `float8e4m3fnuz` ("finite, NaN, unsigned zero"). The two variants are **not bitwise interchangeable** — the same 8-bit pattern decodes to different real values.

| Property              | OCP E4M3 (Micikevicius / OFP8)    | E4M3FNUZ (Graphcore / AMD CDNA3 / ONNX)  |
|-----------------------|-----------------------------------|------------------------------------------|
| Exponent bias         | 7                                 | 8                                        |
| Max finite value      | 448                               | 240                                      |
| NaN encoding(s)       | `S.1111.111` (two patterns)       | `1.0000.000` only (one pattern)          |
| Signed zero           | Both `+0` and `−0`                | Only one zero (no `−0`)                  |
| Infinities            | Not represented                   | Not represented                          |

Sources for the FNUZ column:

- **ONNX Float8 docs** ([onnx.ai/onnx/technical/float8.html](https://onnx.ai/onnx/technical/float8.html)) — names `float8e4m3fn` (the OCP convention) and `float8e4m3fnuz` as distinct dtypes.
- **AMD ROCm HIP docs** ([rocm.docs.amd.com — low_fp_types](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html)) — "CDNA3: FNUZ E4M3 only. CDNA4 & RDNA4: OCP E4M3 (with FNUZ available on gfx94x)."

**Note for ErrorBoard.** When PyTorch's `torch.float8_e4m3fn` is used the OCP convention applies; `torch.float8_e4m3fnuz` selects the FNUZ convention. The two have different max-finite, different bias, and different NaN encoding. Any experiment that quotes "FP8 E4M3" needs to fix which variant.

## Verified torch quirk: overflow-to-NaN, not saturation

`code/errorboard/verify_oracle.py` was run on 2026-05-11 against a spec-derived Python oracle (`code/errorboard/oracle.py`); 254/254 round-trips and 256/256 decodes agreed bit-exact with `torch.float8_e4m3fn`. **Addition disagreed on 436 / 65,536 pairs.** All 436 mismatches are the same case:

- **OCP/Micikevicius spec** (§3.1 and §2): overflow saturates to the maximum representable magnitude (±448). Quote: *"Values that overflow are then saturated to the maximum representable magnitude."*
- **`torch.float8_e4m3fn` behavior** (PyTorch 2.11.0+cu130): casting `float32 → float8_e4m3fn` produces NaN (`0x7F`) for any value `|x| ≥ 464` (the round-to-even midpoint between 448 and the would-be-NaN slot 480 = `1.111 × 2⁸`).

Distribution: 218 positive-overflow pairs + 218 negative-overflow pairs, all with `|true_sum| ≥ 464`. No other discrepancies; decode and round-trip are bit-exact across all 256 patterns.

**Implication for ErrorBoard.** Per `task_spec.md` §4 ("if any pair disagrees, the reference wins"), training labels for the `overflow` regime are generated by the oracle (saturated to ±448), not by `torch.float8_e4m3fn` casting. If torch fp8 arithmetic is needed at runtime, wrap with explicit `.clamp(-448, 448)` before the `.to(torch.float8_e4m3fn)` cast to match OCP semantics.

## Other references

- The Park, Park, Hwang paper (arXiv:2601.16450, §2.2) cites Micikevicius et al. 2022 for E4M3 / E5M2 and uses p = 3, q = 4 for E4M3, satisfying its Condition 1 (`2 ≤ p ≤ 2^{q-1} − 3`).
- IEEE has not issued a final 8-bit float standard as of this writing (IEEE P3109 working group is active); both E4M3 conventions above are de-facto industry specs, not IEEE-754 binary formats.
