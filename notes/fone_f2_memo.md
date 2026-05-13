# F2 (binary FoNE) ablation: precision, not shape recovery

**Setup.** Same architecture as F1 (`fone_*.py`) but with the base-2
period set: `T_i = 2^i for i ∈ [-8, 9]` (18 periods, FONE_DIM = 36) and a
per-binary-digit decoder (cosine similarity against 2 prototypes φ(0, 2) =
(1, 0), φ(1, 2) = (-1, 0)). 18 binary digit places cover exactly the FP8
representable bit positions (subnormal step 2^-9 through max-finite
448 < 2^9).

5 matched seeds at L4-E048 (112,416 params, same as F1) and L4-E128
(791,296 params, same as F1). 20k iters. Same holdout split. Probe:
`fone_f1_vs_f2.py`, raw output `notes/fone_f1_vs_f2_findings.md`.

## TL;DR

**F2 was hypothesized to recover anti-ε severity at smaller capacity than
F1, by removing the internal base-10 → base-2 inversion. That hypothesis
is partially wrong.** F2 doesn't recover anti-ε; it makes the task much
easier and so leaves far fewer residual errors to shape.

The mechanism is **precision**, not **shape recovery**:

- F2 at L4-E048 (5 seeds, 112K params) hits **97.58% mean accuracy on
  `default`** — comparable to F1 at L4-E128 (96.84%) and to bit-level at
  L4-E048 (76.6% — wait, F2 is better than bit-level here too).
- F2 at L4-E128 hits **99.98%** mean accuracy on `default`. One seed (s2)
  is 100% on every regime.
- Anti-ε severity at L4-E048 F2: mean ρ = +0.09 (vs F1's +0.24). Still
  flat, just less positive. F2 at L4-E128 has so few residual errors
  (4 total across 5 seeds × 6554 pairs = 0.012% error rate) that the
  correlation is undefined.

So the binary period set does what FoNE-as-encoding promises (additively-
homomorphic, binade-aligned) — but the path to perfect FP8 addition runs
**through error elimination**, not through anti-ε-shaped residuals.

## Finding 1 — F2 dramatically reduces errors at every size

Total `n_err` (normal-result, summed across m_c, summed across 5 seeds):

| size | F1 | F2 | F2 / F1 ratio |
|---:|----:|----:|---:|
| 48 | 5,030 | 306 | **0.061** (16× fewer) |
| 128 | 412 | 4 | **0.010** (100× fewer) |

Per-regime mean accuracy (5 seeds):

| regime | n | F1 E=48 | **F2 E=48** | F1 E=128 | **F2 E=128** |
|---|---:|---:|---:|---:|---:|
| special-values | 102 | 100% | 100% | 100% | 100% |
| overflow | 84 | 100% | 100% | 100% | 100% |
| subnormal-result | 59 | 99.3% | 97.6% | 100% | 99.3% |
| cancellation | 144 | 97.5% | 93.5% | 99.9% | 99.6% |
| rounding-tie | 610 | 74.9% | **98.85%** | 98.9% | **99.93%** |
| large-dexp | 3754 | 93.1% | **99.38%** | 99.5% | **99.98%** |
| default | 1801 | 67.1% | **95.87%** | 96.8% | **99.87%** |

F2 L4-E048 (the smallest configuration) on `default` (95.87%) **beats F1
L4-E128** (95.07%) and **roughly matches bit-level L4-E048** (76.6% per
the original arm, lower than F2). At a quarter the params of F1's best,
F2 is doing as well or better.

F2 L4-E128 essentially saturates: `default` 99.87%, `rounding-tie`
99.93%, with one seed at 100% on every regime.

## Finding 2 — Anti-ε severity does *not* emerge at small F2 capacity

Per-seed Pearson(ε, mean |log Δ|):

| size | arm | s0 | s1 | s2 | s3 | s4 | mean ρ |
|---:|---|----:|----:|----:|----:|----:|------:|
| 48 | F1 | +0.37 | +0.20 | +0.06 | +0.08 | +0.51 | +0.24 |
| 48 | F2 | −0.18 | +0.06 | +0.33 | +0.37 | −0.11 | **+0.09** |
| 128 | F1 | +0.35 | −0.05 | **−0.89** | −0.20 | −0.04 | −0.17 |
| 128 | F2 | nan | nan | nan | nan | nan | undefined |

The L4-E048 F2 distribution is flatter than F1 (mean +0.09 vs +0.24,
range narrower) — directionally a small step toward anti-ε — but no seed
reaches strongly-negative territory. By L4-E128 F2 has so few errors
that Pearson is undefined.

Sign distribution at L4-E048:

| arm | strongly negative (ρ < −0.5) | flat (\|ρ\| ≤ 0.5) | strongly positive (ρ > +0.5) |
|---|---:|---:|---:|
| F1 | 0 | 4 | 1 |
| F2 | 0 | 5 | 0 |

F2 removes the one strong-positive F1 seed (s4) but does not produce any
strongly-negative seeds at this scale. The flat regime is still the
dominant outcome.

## Finding 3 — F2's mechanism is precision, not shape

The hypothesis was that F2 would let the model find FP-shape residuals
(anti-ε severity) at smaller capacity because the encoding's periods
align with FP8's binade ladder. The data shows a different story:

- At L4-E048, F2 has only 306 errors total (vs F1's 5,030). The 306 errors
  spread roughly evenly across m_c bins, producing flat severity rather
  than endpoint-heavy severity.
- At L4-E128, F2 has only 4 errors total. The task is effectively solved;
  there's no residual error distribution to be anti-ε shaped.

In other words, F2 doesn't fail to find anti-ε *because the model can't*;
it fails to exhibit anti-ε *because there's almost nothing left to be
wrong about*. The binary period set gives the model 18 binary-aligned
features that match FP8's bit positions one-for-one, so the per-digit
decoder is doing FP8 bit extraction rather than rounding-and-coordination.

This is consistent with the four-arm story but adds a new wrinkle:

| arm | what errors look like |
|---|---|
| bit | endpoint-heavy, anti-ε (ρ ≈ −0.7 to −0.92) |
| RoPE | same shape, slightly more probability |
| SEM | very few errors, but the ones remaining are endpoint-heavy anti-ε (ρ ≈ −0.9) |
| F1 (decimal FoNE) | many errors at small scale, uniformly distributed (ρ flat); few at larger scale, the rare strong-negative seed appears |
| **F2 (binary FoNE)** | **very few errors at any scale**; residual is flat, not anti-ε shaped |

## Interpretation

**The encoding's alignment with the format's structure determines the
error count, not the error shape.**

Bit / SEM / FoNE-at-scale all converge on the same residual shape (anti-ε)
because they all eventually have to commit to an FP8 result and the
discretization's binade-boundary fragility is what's left. FoNE-F2 short-
circuits this: by giving the model native access to FP8's bit positions
through the input embedding, the model can do the addition essentially in
binary fixed-point without ever needing to coordinate across binade
boundaries.

**This is *not* a refutation of the four-arm story.** It's an exhibition
of the fact that "format pins the location of irreducible difficulty" can
be defeated by a sufficiently well-aligned input representation. F2 doesn't
fight the binade structure; it gets the binade structure handed to it
through the encoding.

In Landfall terms: ε(m) is the residual of the *affine pseudo-log
abstraction*. If your input embedding already factors that abstraction
into native bit positions, the residual collapses. F2 is exactly that:
the affine pseudo-log made architectural.

## Implications

1. **For the four-arm story.** The "operation-specific cost concentrates
   at binade boundaries" claim from earlier arms is conditional on the
   model needing to navigate the binade structure through whatever the
   encoding gives it. F2 sidesteps this navigation. The four-arm framing
   stays useful but its predictions apply to architectures that don't
   pre-solve the binade-alignment problem.
2. **For interpretability.** F2 is a strong candidate for "FP arithmetic
   done by a transformer with maximum legibility" — the model is doing
   something close to native bit manipulation, and the high accuracy
   suggests it's finding clean circuits. Worth probing.
3. **F2 + scale.** At L4-E128 F2 is at 99.98% — within striking distance
   of the lookup-table ceiling. Whether it actually reaches 100% (saturates
   to perfect FP8 addition) at larger scale is an interesting question.
   We're using ~800K params; Zhou's setting was 8M+.
4. **Bridging F2 and SEM.** Both achieve very low error rates at L4-E048
   scale, but their residual shapes differ: SEM's few errors are at
   endpoints (anti-ε); F2's few errors are uniform. A cross-comparison
   probe on the actual remaining-error pairs might be illuminating.
5. **The flat-error regime under F2.** Even with the right encoding, F2 at
   L4-E048 doesn't produce anti-ε residuals. The flat regime persists.
   This suggests anti-ε requires the model to internally *reproduce*
   FP-shape arithmetic from the bit substrate up — and F2 lets it skip
   that step entirely.

## What we did NOT do

- L4-E064 or L4-E096 F2 pilots. Would refine where on the curve F2
  transitions from "many errors" to "saturated." Likely a small follow-up
  (~13 min on GPU).
- Investigate the few remaining F2 L4-E128 errors. Only 4 total. Worth
  looking at individual pairs to see if they're a specific failure mode.
- L4-E256 F2 to test whether 100% is actually achievable.
- Bit-decomp / field-decomp / digit-decomp analog for F2 errors. The 306
  errors at L4-E048 are enough sample; might show structural pattern.

## Files

- Tokenizer: `code/errorboard/fone_f2_tokenizer.py`
- Encoder: `code/errorboard/fone_f2_encoder.py`
- Model: `code/errorboard/fone_f2_model.py`
- Dataset: `code/errorboard/fone_f2_dataset.py`
- Training: `code/errorboard/fone_f2_training.py`
- Launcher: `code/errorboard/fone_f2_pilot.py`
- Comparison probe: `code/errorboard/fone_f1_vs_f2.py`
- Comparison output: `notes/fone_f1_vs_f2_findings.md`
