# Multiplication: four-arm findings (L4-E048, 20 seeds, iter 20k)

**Setup.** Same architecture and training schedule as the addition arc.
Four arms — bit, SEM, FoNE F1, FoNE F2 — each trained for 20 seeds at
L4-E048 (~112K params), iter 20k, learned PE, FP8 multiplication oracle
labels. Holdout split seed=0, n=6554. Probes:

- `failure_consensus_mul.py` → `notes/mul_arm_comparison.md`
- `epsilon_severity_mul.py` → `notes/mul_severity_findings.md`
- Theoretical pre-registration: `notes/epsilon_under_multiplication.md`

## TL;DR

The pre-registered predictions were partially wrong, in a specific and
informative way. The dominant story is **sign-XOR is the multiplication-
specific failure mode**, and only one arm — FoNE F2, with its bit-aligned
Fourier features — learns it at L4-E048 scale.

| arm | mean fail % | sign-token accuracy | what failed |
|---|---:|---:|---|
| bit | 49.26% | 50.5% | sign-XOR; magnitude (7 of 8 bits) at 99.9%+ |
| SEM | 49.08% | 50.4% | sign-XOR |
| FoNE F1 | 63.87% | 52.9% | sign-XOR + magnitude (Fourier non-locality dominates) |
| **FoNE F2** | **8.28%** | **100.0%** | small mantissa-rounding residue, mostly DEFAULT |

The four-arm inversion is sharper than the addition arc's:

- **Addition:** bit / SEM / RoPE thrive (FP's tax is absorbed by the
  format-native encoding); FoNE arms struggle (multiplication non-locality
  in Fourier doesn't apply, but mant-product non-locality does).
- **Multiplication:** **only FoNE F2 thrives.** bit / SEM / F1 all stall
  at ~50% because they cannot compute sign-XOR through their token
  layouts within L4-E048 capacity.

The pre-registered ε_mult = m_a · m_b prediction was correct in form
(format-pinned residual structure) but couldn't be tested cleanly on
bit / SEM because the sign-XOR failure swamps the bilinear signal.
F2's residual *does* show anti-ε severity (ρ = −0.66 rounding-only) —
the same shape as addition's anti-ε, suggesting the binade-coordination
cost is the irreducible residual once an arm is past the sign hurdle.

## Finding 1 — sign-XOR is the multiplication-specific bottleneck

Bit-level per-result-position accuracy on the mult holdout (bit-mul s0):

| result bit position | role | accuracy |
|---|---|---:|
| 0 | sign | **50.5%** |
| 1–4 | exponent | 99.9%+ |
| 5–7 | mantissa | 99.9%+ |

The model has perfectly learned the magnitude of multiplication results
but cannot compute `sign(a) XOR sign(b)` through bit-level tokenization
at this capacity. Per-pair correctness is sign-bound to ~50%.

Cross-arm sign-token accuracy (seed 0):

| arm | sign-token accuracy |
|---|---:|
| bit | 50.49% |
| SEM | 50.37% |
| FoNE F1 | 52.88% |
| **FoNE F2** | **100.00%** |

Three of four arms are stuck near random on the sign decision. F2's bit-
aligned Fourier features make sign trivially accessible (every binary
bit position of the input has a dedicated Fourier component, including
the sign-bit position). The other three arms have a single sign token
without a clean direct signal — and at L4-E048 they don't learn the
XOR.

The variance ratio diagnostic confirms this is a per-seed random
phenomenon:

| arm | var ratio (obs/iid) |
|---|---:|
| bit | 0.84 |
| SEM | 0.79 |
| F1 | 8.96 |
| F2 | 3.32 |

bit and SEM are at ~0.8 — barely below i.i.d. Their failure pattern is
roughly Binomial(20, 0.5): every pair is failed by ~half the seeds
independently. Each seed converges to a *different* random fixed point
on sign-XOR. F1 has a heavy-tailed lottery (var ratio 8.96) with a
1,070-pair structural core — its failure is concentrated AND severe.

## Finding 2 — F2's pre-registered "collapse on mult" prediction was wrong

The pre-registration (`notes/epsilon_under_multiplication.md`) called for:

| arm | predicted full ρ | predicted rounding-only ρ |
|---|---|---|
| bit / SEM | +0.25 to +0.50 | +0.50 to +0.75 |
| FoNE F1 | −0.1 to +0.2 | −0.1 to +0.2 |
| **FoNE F2** | **0.0 to +0.3** | **0.0 to +0.3** |

Observed:

| arm | mean ρ (full) | mean ρ (rounding-only) |
|---|---:|---:|
| bit | +0.090 | +0.026 |
| SEM | nan (all log-damages = ∞ from sign flips) | nan |
| FoNE F1 | +0.237 | −0.048 |
| **FoNE F2** | **−0.516** | **−0.656** |

Three of four arms came in near zero, censored by the sign-XOR failure
mode. F2 came in **strongly negative** — the opposite of the predicted
0..+0.3 range. This is the anti-ε shape from addition.

**The corrected reading:** ε_mult = m_a · m_b is the *formal* affine
pseudo-log residual for multiplication, but the irreducible *measurable*
residual is the binade-coordination cost that emerges once an arm has
solved the operation correctly. F2 — the only arm that solved
multiplication at this scale — shows anti-ε severity in the same shape
as bit/SEM's addition residuals (ρ ≈ −0.7 to −0.9 across all those
prior measurements). The format pins shape; the operation pins which
arms can reach the shape.

## Finding 3 — Per-regime breakdown sharpens the story

Mean fail rate per regime, 20 seeds:

| regime | n | bit | SEM | FoNE F1 | **FoNE F2** |
|---|---:|---:|---:|---:|---:|
| special-values | 203 | 20.94% | 20.86% | 0.00% | 0.00% |
| overflow | 1015 | 50.36% | 50.17% | 27.44% | **0.05%** |
| underflow-to-zero | 216 | 49.40% | 50.69% | 0.00% | 0.00% |
| subnormal-result | 563 | 50.66% | 50.09% | 42.74% | **0.28%** |
| rounding-tie | 391 | 49.91% | 49.81% | 57.63% | **0.50%** |
| exact-result | 1378 | 50.05% | 50.17% | 82.57% | 4.36% |
| **default** | 2788 | 50.14% | 49.76% | 82.62% | **17.16%** |

bit and SEM are uniformly stuck at ~50% — the sign-XOR randomness
shows through every regime equally. F1 is differentially bad on
EXACT_RESULT and DEFAULT (where rounding is needed) and unexpectedly
good on UNDERFLOW_TO_ZERO (always rounds to 0; easy memorization).

F2 essentially saturates every structural regime — even EXACT_RESULT is
at only 4% fail (vs F1's 83%). The lone hard regime is DEFAULT at 17% —
the rounding-required mantissa-product cases. This is where F2's residual
lives, and where the anti-ε severity correlation gets its signal.

## Finding 4 — Lottery overlaps tell a clean story

Pairwise Jaccard on lottery zones (pairs failed by 1..19/20 seeds):

| | bit | SEM | F1 | F2 |
|---|---:|---:|---:|---:|
| bit | 1.000 | **1.000** | 0.785 | 0.539 |
| SEM | 1.000 | 1.000 | 0.785 | 0.539 |
| F1 | 0.785 | 0.785 | 1.000 | 0.425 |
| F2 | 0.539 | 0.539 | 0.425 | 1.000 |

- **bit and SEM have lottery Jaccard 1.000** — identical lotteries.
  This is the sign-XOR failure mode: both arms fail uniformly across the
  full pair set, so every pair is in both lotteries. Same effective
  random process.
- **F1 shares 79% lottery with bit/SEM** — its non-locality cost
  partially overlaps with their sign-XOR failure. F1 is also failing on
  sign + bilinear cost.
- **F2 shares ~54% lottery with bit/SEM**, **43% with F1**. F2's residual
  is in different pairs than the others' lotteries — DEFAULT cases where
  bilinear rounding lives.

## Finding 5 — Inversion vs the addition arc

Comparing same-architecture results:

| metric | addition L4-E048 | multiplication L4-E048 |
|---|---|---|
| best arm | SEM (0.68% fail) | **FoNE F2 (8.28% fail)** |
| worst arm | FoNE F1 (14.59%) | **FoNE F1 (63.87%)** — far worse than addition |
| arms at task | bit, SEM, RoPE all work; FoNE F1 in flat regime | only F2 works |
| sign | trivially correct everywhere | the bottleneck for 3/4 arms |
| anti-ε | survives across all FP-native arms | only F2 shows it |
| format-pinned shape | endpoints (m_c near 0 or 1) | bilinear (m_a · m_b small) — but censored |

The addition arc told us "format pins shape, encoding shapes which arms
find it." Multiplication tells us the same — but with a different shape
and only one arm finding it at this capacity.

**The asymmetry is not symmetric in the sense pre-registered.** We
predicted bit/SEM would thrive on mult (FP-native operation) and FoNE
would struggle (taxed operation). The reality:

- bit and SEM struggle on mult because **sign-XOR through their token
  layouts is hard at this capacity**, independent of operation alignment.
- F2 thrives because its **bit-aligned encoding gives direct access to
  sign bits AND clean mantissa-product representation**.
- F1 struggles for the same Fourier-non-locality reason we predicted,
  but adds sign-XOR-as-second-cost.

The encoding that "wins" multiplication at our scale isn't FP-native;
it's the one that gives the model the most direct circuit-level access
to the operation's structure — and that's the binary-bit-aligned
encoding F2 provides.

## Finding 6 — F2's anti-ε is real and parallels addition

F2's per-seed ρ (rounding-only):

| seed | ρ (rounding-only) | n_err |
|---|---:|---:|
| 0 | −0.480 | 83 |
| 1 | **−0.774** | 1034 |
| 2 | **−0.748** | 187 |
| 3 | **−0.967** | 216 |
| 4 | −0.310 | 320 |

Three of five F2 seeds hit |ρ| > 0.7. The mean (−0.656) is in the same
territory as addition's strongest anti-ε results (SEM s8 at −0.92, bit
at −0.7 to −0.9 across multiple seeds).

This says: **when an arm solves multiplication correctly, the residual
takes the same anti-ε shape as the residual of addition for FP-native
arms.** The anti-ε pattern is an FP-discreteness signature: the binade-
boundary coordination cost the format imposes regardless of operation.

F2's special property — bit-aligned Fourier features — is what lets it
get *past* sign-XOR and start exhibiting the residual shape we expect
from FP discreteness. Without that, you can't see the anti-ε signal
because you're not solving multiplication.

## Implications

1. **Sign-XOR matters more than expected.** A 2-way classification at
   one position seemed like nothing; in fact it's the dominant blocker
   for bit/SEM/F1 multiplication at L4-E048. The signal of "this is hard"
   is per-token-accuracy stratified by token position, not by regime.
2. **F2 wins multiplication at small scale.** The bit-aligned encoding
   that "won" addition at large scale also wins multiplication at small
   scale. The right inductive bias dominates capacity.
3. **The pro-ε bilinear prediction needs revisiting.** It assumed a
   clean baseline where mantissa-product rounding was the only failure
   mode. It isn't. To test the bilinear hypothesis cleanly, we need an
   arm that has solved sign-XOR (F2 is the only one). F2 then shows
   anti-ε, not pro-ε — same shape as addition. The theoretical sketch
   needs updating: the binade-boundary cost is the universal FP residual
   shape, regardless of operation, *once* the operation is computable
   by the model.
4. **Multiplication is harder at small scale than addition.** Addition
   at L4-E048 gives 3/4 arms 90%+ accuracy. Multiplication at L4-E048
   gives 3/4 arms ~50% accuracy. The operation that's "structurally
   easier in FP" is empirically harder for these small models because
   sign-XOR specifically is hard to learn through several attention
   patterns.
5. **L4-E128 follow-up for bit/SEM/F1 is the natural next step.**
   They're capacity-limited on sign-XOR; more capacity should fix it.
   This is the multiplication analog of our addition-arc finding that
   F1 anti-ε recovers with capacity.

## Pre-registration audit

Predictions in `notes/epsilon_under_multiplication.md`:

| prediction | outcome |
|---|---|
| ε lives, on bilinear axis m_a · m_b | partially — F2 confirms; bit/SEM censored by sign |
| sign-flip vs addition (anti-ε → pro-ε) | **wrong** — F2 still shows anti-ε at ρ = −0.66 |
| FP-native arms +0.25 to +0.75 ρ | wrong — bit/SEM/F1 all near zero or undefined |
| F2 0.0 to +0.3 ρ | wrong — F2 at −0.66 |
| EXACT_RESULT distortion | confirmed; rounding-only ρ is the cleaner read |
| "Dissipation into exactness" with capacity | testable in follow-up |

The dominant theoretical error: assuming "mant-product rounding is the
residual" ignores the sign-XOR computation cost that dominates at small
capacity for 3/4 arms. The theoretical framing should be updated:
**ε_mult formally lives on the bilinear axis, but the dominant
*observable* residual depends on which sub-task the arm has actually
learned.** At L4-E048 only F2 has learned past sign-XOR, and its
residual then takes the format-pinned anti-ε shape.

## Open follow-ups

- **L4-E128 bit/SEM/F1 mult pilots.** Test whether sign-XOR is
  capacity-bound. If so, the bilinear-cost prediction becomes testable
  and we can ask whether the binade pattern (anti-ε) wins over the
  bilinear pattern (pro-ε) once arms are at full task.
- **Why F1 is uniquely worst.** F1's 64% fail rate + 1,070 structural
  core is dramatic. Worth a brief investigation: is F1 failing at
  sign-XOR AND mantissa-product AND something else?
- **F2 at scale on mult.** F2 at L4-E048 already does well. Does L4-E128
  saturate to <1%? The addition F2 result was 0.012% errors at L4-E128;
  multiplication might tell us how operation-related the F2 ceiling is.
- **Updated theory.** The ε_mult bilinear-cost framing should be
  retired or revised. The empirical observation is that F2's residual
  on multiplication takes the same shape as addition's residual on
  FP-native arms (anti-ε). That suggests the relevant ε is *format-
  discreteness*-pinned, not operation-pinned. A cleaner formal restatement
  might be: ε(m_c) governs the residual *of the output's discretization*,
  regardless of which operation produced it.

## Files

- Per-arm sweep launchers: `bit_mul_seeds.py`, `sem_mul_seeds.py`,
  `fone_mul_seeds.py`, `fone_f2_mul_seeds.py`.
- Probes: `failure_consensus_mul.py`, `epsilon_severity_mul.py`.
- Raw outputs: `notes/mul_arm_comparison.md`, `notes/mul_severity_findings.md`.
- Pre-registration: `notes/epsilon_under_multiplication.md`.
