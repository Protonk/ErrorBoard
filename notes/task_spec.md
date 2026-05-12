# Task specification — ErrorBoard "mouse" v1

Frozen concrete spec for the first ErrorBoard experiment. Methodology principles live in `methodology.md`; this file specifies their instantiation. Changes here are experimental design changes — track them in git.

## 1. Tokenization

**Granularity: Option A — bit-level.** Every bit of every FP8 E4M3 number is its own token. Alternatives considered:

- *B (field-level, three tokens per number):* not adopted — forces the model to learn ordering-among-discrete-tokens, which is a different learning problem from bit-arithmetic.
- *C (BitTokens-style, one token per number with bit-structured embedding):* not adopted in v1; may be added later as a parallel comparison arm if the published-work comparator becomes useful.

### Vocabulary (12 tokens, disjoint role ranges)

| ID  | Role                          | Token / value     |
|-----|-------------------------------|-------------------|
| 0   | special                       | `<bos>`           |
| 1   | special                       | `<eos>`           |
| 2   | special (reserved, v1 unused) | `<pad>`           |
| 3   | special (reserved, v1 unused) | `<scratch>`       |
| 4   | operator                      | `+`               |
| 5   | operator                      | `=`               |
| 6   | sign-bit                      | value `0`         |
| 7   | sign-bit                      | value `1`         |
| 8   | exponent-bit                  | value `0`         |
| 9   | exponent-bit                  | value `1`         |
| 10  | mantissa-bit                  | value `0`         |
| 11  | mantissa-bit                  | value `1`         |

The same value-token is used at every position within a role. Position-within-field and operand identity (a / b / c) are conveyed by sequence position and the `+` / `=` separators. This gives a clean role partition in embedding space — `W_E` has six "content" rows that are directly inspectable.

### Sequence format

Fixed length **28 tokens**. MSB-first within each FP8 number (matches standard storage order and matches the FP-add algorithm's exponent-first flow):

```
<bos>  s e3 e2 e1 e0 m2 m1 m0  +  s e3 e2 e1 e0 m2 m1 m0  =  s e3 e2 e1 e0 m2 m1 m0  <eos>
  0    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26   27   (positions 0..27)
       ←──────── operand a ───────→     ←────── operand b ──────→     ←────── result c ─────→
```

Position indices are fixed and known: positions 1–8 carry operand a, positions 10–17 carry operand b, positions 19–26 carry result c. The operator tokens at positions 9 and 18 are structural separators.

### Loss masking

Train with cross-entropy over **all** non-`<bos>` positions (standard nanoGPT next-token prediction).

**Per-regime evaluation uses result-token-only loss/accuracy** (the 8 tokens at positions 19–26). Echoing the inputs is not the interesting signal — the model's performance on the result is the load-bearing measurement.

## 2. Regime predicates (partition with priority order)

Each sample receives exactly one primary regime label by first-match in this order. `a`, `b`, and `true_sum = a + b` are decoded as real numbers from their E4M3 bit patterns and summed exactly (float64 is exact for E4M3 inputs). `exp(x)` = unbiased exponent (subnormals collapsed to exp = −6). `iszero` returns True for both signed zeros.

| # | Regime              | Predicate (first-match wins)                                          |
|---|---------------------|-----------------------------------------------------------------------|
| 1 | special-values      | `isnan(a) or isnan(b) or (iszero(a) and iszero(b))`                   |
| 2 | overflow            | `abs(true_sum) > 448`                                                 |
| 3 | underflow-to-zero   | `0 < abs(true_sum) < 2**-10`  *(structurally empty in FP8 E4M3 — see §5)* |
| 4 | subnormal-result    | `round_to_fp8(true_sum)` has biased exponent `== 0` and is nonzero    |
| 5 | cancellation        | `sign(a) != sign(b) and abs(true_sum) <= max(abs(a), abs(b)) / 4`     |
| 6 | rounding-tie        | `true_sum` is the exact midpoint of two adjacent FP8 representables   |
| 7 | large-Δexp          | `abs(exp(a) - exp(b)) >= 4`                                           |
| 8 | default             | anything not matching 1–7                                             |

**Free parameters with structural justifications** (both trivially adjustable if a sweep shows them mis-calibrated):

- Cancellation severity `/4` (= ≥2 mantissa bits of significance lost; meaningful given E4M3's 3-bit mantissa).
- Large-Δexp threshold `≥4` (= mantissa width; at this point the smaller operand is fully shifted out of the result's 3-bit mantissa).

## 3. Secondary tags (analysis only, not sampling)

Each sample additionally carries a bitmask of zero-or-more secondary tags for post-hoc re-bucketing. Tags do not affect training data composition; they let us slice eval logs after the fact.

- `same-sign`, `opposite-sign`
- `same-exp`, `small-Δexp` (`|Δexp| ∈ {1, 2, 3}`), `large-Δexp` (`|Δexp| ≥ 4`)
- `tie` (would-be rounding tie even when not the primary regime, e.g., a cancellation that's also a tie)
- `subnormal-input` (≥1 operand is subnormal)
- `result-exact` (true_sum is representable in E4M3 with no rounding)

## 4. Oracle (ground-truth labeler)

**Source of truth for `a + b` training labels.**

- Build a pure-Python E4M3 reference (~100 lines) directly from the Micikevicius/OCP spec:
  - `decode(bits: uint8) → (value: float, kind: str)` with kind ∈ {`normal`, `subnormal`, `zero`, `nan`}
  - `encode(value: float) → bits` — round-to-nearest-even, saturate to ±448 (no overflow-to-NaN)
  - `add(a_bits, b_bits) → result_bits` — decode, sum in float64 (exact for E4M3 inputs: operands have ≤11 bits of precision; float64 has 53), re-encode with RNE
- **First check** whether `ml_dtypes` (Google, used by JAX/TF) or another existing library already supplies an E4M3 reference of equivalent fidelity. If so, use it as a second cross-check rather than as the primary — the primary remains spec-derived because we want to know we read the spec correctly.
- Validate `torch.float8_e4m3fn` against the reference on **all 65,536 ordered pairs**. Compare **bit patterns**, not values (NaN-vs-NaN comparisons require bit-level handling). If all 65,536 agree, use torch as the runtime labeler for speed; if any pair disagrees, the reference wins and we document the discrepancy as a torch quirk.

**Edge cases earning explicit test coverage:**

- Both NaN bit patterns (`0x7F`, `0xFF`) propagating correctly
- `(+0) + (-0) = +0` under round-to-even
- `448 + 448` saturates to 448 (verify torch matches OCP, not some overflow-to-NaN convention)
- Subnormal-subnormal addition
- Underflow rounding direction: `2**-10` rounds to 0 under RNE (the even-mantissa neighbor is the zero one)

**NaN equivalence in the loss.** When the oracle output is NaN, any NaN bit pattern from the model is scored as correct. Which specific NaN appears is a representation artifact (depends on the input NaN's sign bit), not arithmetic content. Implemented as a `NaN-class` equivalence in the result-token comparison.

## 5. Sample weighting

The full input space is 65,536 ordered FP8 pairs — enumerable, so we design sampling exactly rather than rely on probability.

### Preprocessing (one-time)

1. Enumerate all 65,536 pairs through `ref.add`.
2. Classify each pair by primary regime + tag mask.
3. Store as a fixed table: `(a_bits, b_bits, result_bits, regime_id, tag_mask)`. Total ≈ 512 KB.

### Train / holdout split

- **10% per regime, stratified.**
- **Floor of 10 pairs per regime in holdout** (matters for tail buckets; rounding-tie may have only ~100 pairs total in the full enumeration).

### Training-time sampling

- Draw a regime uniformly (`1/8` each).
- Draw a pair uniformly within that regime's training portion.
- **With replacement** — step count and pair-exposure are decoupled.

### Three eval streams logged separately at every checkpoint

1. **per-regime holdout loss** (8 curves) — main signal for capability emergence on unseen pairs.
2. **natural-distribution eval** — per-regime holdout losses re-weighted by each regime's natural occurrence rate (its share of the 65,536 pairs); single scalar, the "deployment failure rate."
3. **per-regime training loss** — sanity check that the model learned what it actually saw, separable from generalization.

### No curriculum

Equal mass per regime throughout training. A curriculum would convolve capability-emergence dynamics with the schedule and obscure when each capability turned on.

### Empty regimes (structural)

Enumeration (2026-05-11) found that **underflow-to-zero** has zero pairs in FP8 E4M3. All E4M3 values are integer multiples of `2⁻⁹` (the smallest subnormal spacing), so any pairwise sum is also an integer multiple of `2⁻⁹`. The open interval `(0, 2⁻¹⁰)` is therefore unreachable: any nonzero sum has `|x| ≥ 2⁻⁹ > 2⁻¹⁰`. The predicate remains in the priority list for completeness, but the regime is empty and is skipped during sampling. Equal-mass weighting is therefore over the seven active regimes.

Full enumeration counts (out of 65,536):

| regime              | count  | share   |
|---------------------|-------:|--------:|
| special-values      |  1,024 |  1.56%  |
| overflow            |    836 |  1.28%  |
| underflow-to-zero   |      0 |  0.00%  |
| subnormal-result    |    590 |  0.90%  |
| cancellation        |  1,440 |  2.20%  |
| rounding-tie        |  6,100 |  9.31%  |
| large-Δexp          | 37,540 | 57.28%  |
| default             | 18,006 | 27.47%  |
