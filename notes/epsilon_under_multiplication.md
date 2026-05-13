# Where ε lives under multiplication

**Question (pre-experiment).** Park's ε(m) = log₂(1+m) − m governed our
addition results: severity anti-correlated with ε across all four FP-
native settings (bit, RoPE, SEM, FoNE-at-scale-once-it-saturates). For
multiplication, does ε still live? If so, on what axis?

This is a theoretical sketch before we wire up the multiplication
oracle to the training pipeline. The goal is to know what to *expect* —
in particular, what observable would falsify "ε is universal."

## TL;DR

ε lives, but the axis changes from m_c (single, the result mantissa) to
m_a · m_b (joint, a bilinear product of input mantissas). The bilinear
form is exactly the residual the affine pseudo-log can't capture for
multiplication; the addition arc's m_c-based ε was a degenerate case of
this for the addition operation.

Predicted shape:

- **FP-native arms (bit / SEM / RoPE):** severity *positively* correlates
  with m_a · m_b. Exact-result pairs (where m_a · m_b = 0, e.g. any
  power-of-2 input) drop out and have severity ≈ 0. The DEFAULT and
  ROUNDING_TIE regimes contribute most of the signal. The 21% EXACT_RESULT
  population effectively "censors" the correlation by parking ~a fifth
  of the dataset at severity 0 — but the correlation among the
  rounding-needed pairs should remain measurable.
- **FoNE F1 / F2:** severity should NOT correlate with m_a · m_b. The
  Fourier basis is non-local for multiplication (BitTokens Prop 4.3);
  errors should distribute roughly uniformly across the bilinear range
  rather than concentrating at high m_a · m_b. The bilinear cost has
  nowhere to land in the Fourier representation — every product is
  approximately as hard as every other.

**Sign-flip vs addition.** Addition gave us anti-ε (severity *anti*-
correlated with ε(m_c)). Multiplication should give *pro-ε* (severity
*positively* correlated with ε_mult = m_a · m_b). This is because
mult's "irreducible cost" is the rounding of the bilinear product
itself — it tracks ε rather than tracking the binade-boundary
coordination that anti-correlated with addition's ε.

## Derivation

### What ε measured for addition

ε(m) = log₂(1+m) − m, computed for the *result* mantissa m_c.

It is the residual of the affine pseudo-log approximation of `log₂(1+m_c)`
— i.e., it quantifies how much the format's affine treatment of mantissa
(treating bit pattern as integer = E + m) deviates from the true log.

This residual has a bell shape:
- ε(0) = 0 (start of binade)
- ε peaks near m ≈ 0.44 (where log₂(1+m) − m is maximal)
- ε(1) = 0 (end of binade / start of next)

The endpoints (m_c ≈ 0 and m_c ≈ 1) are where the format's affine reading
is exact. They are *also* where the binade-boundary coordination cost
lives: at m_c → 0, the result has just crossed *up* into a new binade;
at m_c → 1, it's about to cross. The model has to coordinate the
exponent and mantissa decisions cleanly at these transitions, and this
is the source of catastrophic errors.

**Anti-ε severity** for addition: errors concentrate at endpoints (small
ε, large coordination cost). The two costs live in disjoint regions of
the mantissa axis.

### What the affine pseudo-log gets wrong for multiplication

The pseudo-log says: pseudolog(a × b) = pseudolog(a) + pseudolog(b). In
the affine reading, pseudolog(a) = E_a + m_a, so the prediction is:

```
pseudolog(a × b) ≈ (E_a + E_b) + (m_a + m_b)
```

The truth (no rounding) is:

```
log₂(a × b) = (E_a + E_b) + log₂(1 + m_a) + log₂(1 + m_b)
            = (E_a + E_b) + (m_a + ε(m_a)) + (m_b + ε(m_b))
            = pseudolog-prediction + ε(m_a) + ε(m_b)
```

So the per-operand pseudo-log residual is ε(m_a) + ε(m_b), and these
add cleanly under the pseudo-log multiplication identity. **The
multi-operand ε for multiplication is just the sum of per-operand ε's.**

But there's a *separate* error that addition didn't have: the mantissa
product itself doesn't add cleanly. Pre-rounding:

```
(1 + m_a)(1 + m_b) = 1 + m_a + m_b + m_a · m_b
```

The pseudo-log addition predicts the mantissa is `m_a + m_b`. The
truth has a `m_a · m_b` bilinear term. This is what the affine
treatment fundamentally misses, and it's what creates rounding-required
multiplication results.

**Define `ε_mult(m_a, m_b) := m_a · m_b`** as the format's irreducible
bilinear residual.

Properties of `ε_mult`:
- ε_mult(0, m_b) = 0 for any m_b → power-of-2 × anything is exact
- ε_mult(m_a, 0) = 0 → mirror
- Range: [0, 49/64] over the FP8 mantissa grid {0/8, 1/8, ..., 7/8}
- Maximum at m_a = m_b = 7/8: ε_mult_max = 49/64 ≈ 0.766
- Joint zero set (the "exact" subspace) is the union of axes m_a = 0
  and m_b = 0 — exactly the POWER_OF_TWO_INPUT pairs

This last property is the formal reason `EXACT_RESULT` is 21% of the
mult pair table: the joint-zero subspace `{m_a · m_b = 0}` is a
non-trivial fraction of the discrete grid.

### How this changes the stratification axis

Addition's severity was stratified by **m_c** (one number per result,
ranging over 8 discrete bins).

Multiplication's severity should be stratified by **m_a · m_b** (one
number per *pair* of inputs). The natural binning:

| m_a · m_b range | meaning | predicted EXACT_RESULT density |
|---|---|---|
| 0 | one input is power of 2 | 100% (by construction) |
| (0, 1/64] | one input is near power of 2 | high |
| (1/64, 9/64] | both inputs have small mantissa | medium |
| (9/64, 25/64] | both inputs have medium mantissa | low |
| (25/64, 49/64] | both inputs have large mantissa | very low |

The product m_a · m_b on the discrete FP8 grid {0, 1/8, 2/8, ..., 7/8}^2
has 64 possible value combinations, with m_a · m_b ∈ {0, 1/64, 2/64, ...,
49/64}. After dedup and overlap, ~16 distinct values. Severity stratified
across these gives the analog of addition's m_c-binned severity.

### Predicted shape per arm

For FP-native arms (bit / SEM / RoPE), multiplication's irreducible cost
is just rounding the bilinear product (1+m_a)(1+m_b) to 3 mantissa bits:

- When m_a · m_b = 0 (exact-result pairs): no rounding, zero error
- When m_a · m_b is large: more "bit pressure" in the mantissa
  product, higher chance of rounding-tie or rounding-by-a-half-ULP
- Severity should scale roughly with m_a · m_b magnitude

Predicted Pearson(ε_mult, mean |log Δ|) for FP-native arms: **positive**.
Range expected: maybe ρ ∈ [+0.4, +0.8]. Higher than zero, but not
necessarily as strong as addition's anti-ε (which was ρ ≈ −0.9) because:

- Multiplication has no equivalent of addition's binade-coordination
  catastrophe at endpoints. The largest-severity events are still bounded
  by 1/16 ULP (half-step) rounding, not by exponent flips.
- The EXACT_RESULT regime contributes 21% of pairs at severity 0, pulling
  the correlation down toward zero.

The user's phrase "**widely spread ε floor for models minus FoNE**" is
exactly what this predicts: a positive but modest correlation, distributed
across the m_a · m_b axis rather than peaked at endpoints.

### For FoNE arms specifically

FoNE's encoding is additively homomorphic: F(a + b) = F(a) ⊙ F(b)
(Hadamard product in Fourier space). Multiplication is *not* a Hadamard
product on Fourier features — it requires convolution across all
component-pairs (BitTokens Prop 4.3).

This has two consequences:

1. **FoNE multiplication has high errors everywhere.** Not concentrated
   at high m_a · m_b vs low m_a · m_b. The model has to learn the
   bilinear product through its non-native operation, and the cost is
   distributed across the input space.
2. **EXACT_RESULT pairs (m_a · m_b = 0) might NOT be easy for FoNE.**
   Power-of-2 multiplication "should" be trivial — just an exponent
   shift — but FoNE's per-digit decoder doesn't natively know which
   inputs are powers of 2. F2 (binary FoNE) might recognize this through
   its bit-alignment; F1 (decimal FoNE) probably won't.

Predicted Pearson(ε_mult, mean |log Δ|) for FoNE: **near zero**. Severity
spread roughly uniformly across the m_a · m_b axis. F1 should look
clearly flat; F2 might show a weak positive correlation if it discovers
the power-of-2 shortcut.

### How EXACT_RESULT distorts the signal

Twenty-one percent of pairs have m_a · m_b = 0 *and* land in the
EXACT_RESULT regime. For FP-native arms these contribute zero error.
For the Pearson(ε_mult, severity) correlation, this is a giant pile of
(0, 0) points that drag the correlation toward zero from whatever true
positive value it would otherwise have.

To recover a clean signal, the severity probe should report two
correlations:

- **Full-population ρ:** the correlation as observed, including
  EXACT_RESULT.
- **Rounding-required ρ:** the correlation excluding EXACT_RESULT and
  SPECIAL_VALUES. This is the correlation among pairs the model could
  actually be wrong about.

For FP-native arms, full-population ρ might be modest (+0.3-ish)
because of the EXACT_RESULT mass, but rounding-required ρ should be
strong (+0.6 to +0.8).

For FoNE arms, both correlations should be near zero. If FoNE shows
non-zero correlation only because of EXACT_RESULT (i.e., it has
*non-zero* errors on EXACT_RESULT pairs and *zero* elsewhere — which
is hard to even imagine), that would be diagnostic of a very different
failure mode.

### "Dissipation into exactness" — the capacity prediction

The user's intuition: as capacity scales, FP-native arms should
"discover exactness." Concretely: the residual that lives in DEFAULT
and ROUNDING_TIE under multiplication should *shrink with capacity*,
and the EXACT_RESULT regime should remain at zero errors throughout
(it was always easy).

In severity-probe terms: the bilinear-tracking positive ρ should:
- Be measurable at small capacity (the "ε floor" the user mentioned).
- Decrease toward zero at large capacity, not by flipping sign but by
  the residual distribution shrinking.
- For F2 specifically (with bit-aligned Fourier features), the
  EXACT_RESULT pairs might be solved very early, with the residual
  concentrated on DEFAULT/ROUNDING_TIE — a hint of the precision-vs-
  shape recovery distinction we saw for addition.

This is the inverse of addition's transition: there, capacity
*recovered* anti-ε in seeds that learned the binade structure well.
Here, capacity should *dissipate* pro-ε as the model learns to handle
the bilinear product cleanly.

## Concrete observables to compute when training lands

For each arm × scale × seed:

1. **Severity stratified by m_a · m_b** (16 bins, roughly).
2. **Pearson(ε_mult, mean |log Δ|), full population.**
3. **Pearson(ε_mult, mean |log Δ|), rounding-required only.**
4. **Per-regime accuracy** with the new mult_regimes — particular
   attention to EXACT_RESULT (should be ≈ 100% for FP-native arms,
   variable for FoNE).
5. **EXACT_RESULT-vs-DEFAULT ratio of errors.** If FoNE makes MORE
   errors on EXACT_RESULT than on DEFAULT, that's a different failure
   mode than what addition gave us.

If we wanted, an analog of the bit-decomp probe stratified by
m_a · m_b magnitude rather than m_c would show how the *kind* of
mistake (sign / exp / mant flip) interacts with the bilinear cost.

## What would falsify "ε is universal"

The "ε is universal" claim would be falsified if any of these hold:

- FP-native arms show ρ ≈ 0 even after excluding EXACT_RESULT and
  SPECIAL_VALUES. This would mean the bilinear residual isn't structuring
  the model's residual at all.
- FoNE shows strongly positive ρ. This would mean FoNE somehow inherits
  the FP arithmetic structure under multiplication, even though it
  shouldn't via the encoding.
- A consistent sign flip from the prediction (FP-native goes negative, or
  FoNE goes strongly negative).

If any of these show up empirically, the formalism above needs revision.

## Pre-registered predictions

| arm | full ρ (m_a · m_b vs severity) | rounding-only ρ |
|---|---|---|
| bit | +0.25 to +0.50 | +0.50 to +0.75 |
| RoPE | similar to bit | similar to bit |
| SEM | +0.40 to +0.60 (very few errors, signal noisier) | +0.6 to +0.8 |
| FoNE F1 | −0.1 to +0.2 | −0.1 to +0.2 |
| FoNE F2 | 0.0 to +0.3 | 0.0 to +0.3 |

Confidence: weak. The bilinear-cost framing is theoretically motivated
but the experimental literature is sparse and our addition-side
predictions were partially wrong (anti-ε was capacity-modulated, not
universal). Treat these as testable, not certain.

## Files

- Multiplication oracle: `code/errorboard/oracle.py` (`mul()`).
- Regime classifier: `code/errorboard/mult_regimes.py`.
- This memo: `notes/epsilon_under_multiplication.md`.
- Future: probes that stratify by m_a · m_b instead of m_c.
