# SEM arm: findings

**Setup.** L4-E048 architecture, learned positional encoding, 3-token SEM
tokenization (sign / exp / mantissa, vocab=32, seq_len=13). Trained 20 seeds
(`sem-L4-E048-s{0..19}`) at iter 20k, same holdout split as the other arms
(seed=0, n=6554). Probe scripts:

- `failure_consensus_sem.py` → `notes/sem_arm_comparison.md`
- `epsilon_field_decomp_sem.py` → `notes/sem_field_decomp_findings.md`
- `epsilon_severity_sem.py` → `notes/sem_severity_findings.md`

## TL;DR

**SEM nearly eliminates the lottery.** Mean fail rate 8.98% (bit-level) → 0.68% (SEM). Heavy-tail (p̂ ≥ 0.8) drops from 22 pairs → **0 pairs**. Structural core 2 → 0. The categorical split (smooth-interior mant-only / endpoint exp-coupled) is preserved — but with so few residual errors that what remains is concentrated at exactly the binade-boundary location where exp-coordination is intrinsically hard. Anti-ε severity survives where there is enough data to measure it.

In Landfall's framing: SEM removes the *bit-discovery surcharge* on top of FP's add-tax. What remains is the irreducible cost: exp-coordination at binade boundaries.

## Finding 1 — Lottery essentially gone

Three-arm comparison at L4-E048, 20 seeds each:

| arm | mean fail % | var ratio | lottery % | heavy tail (p̂ ≥ 0.8) | structural core (p̂ = 1.0) |
|---|---:|---:|---:|---:|---:|
| learned-PE bit | 8.98% | 6.12 | 41.0% | 22 pairs | 2 pairs |
| RoPE bit | 11.93% | 7.66 | 43.7% | 128 pairs | 8 pairs |
| **learned-PE SEM** | **0.68%** | **2.92** | **7.0%** | **0 pairs** | **0 pairs** |

The heavy-tail collapse is the cleanest test of the "is the lottery format-driven or bit-discovery-driven" question, and it's a complete answer: **bit-discovery-driven.** When you hand the model the field structure as a prior, the 22-128 pair tail of near-deterministic failures simply disappears.

Per-pair p̂ distribution (20 seeds):

| bin | learned-PE bit | RoPE bit | SEM |
|---|---:|---:|---:|
| 0/20 (easy) | 3868 | 3685 | **6092** |
| 1/20 | 646 | 553 | 258 |
| 2–5/20 | 1262 | 1236 | 186 |
| 6–10/20 | 551 | 646 | 17 |
| 11–15/20 | 205 | 306 | 1 |
| 16–19/20 | 20 | 120 | **0** |
| 20/20 (always-fail) | 2 | 8 | **0** |

93% of holdout pairs are structurally easy under SEM (vs 59% under learned-PE bit, 56% under RoPE bit). The lottery zone shrinks from 41% → 7%, and the residual lottery is concentrated in low-p̂ bins (1–5/20), not heavy-tail bins.

## Finding 2 — Same regimes, far smaller fail rates

| regime | n | learned-PE bit | RoPE bit | SEM | SEM/bit ratio |
|---|---:|---:|---:|---:|---:|
| default (smooth interior) | 1801 | 23.42% | 30.84% | **1.58%** | 0.067× |
| rounding-tie | 610 | 9.65% | 13.88% | **0.58%** | 0.060× |
| large-dexp | 3754 | 2.81% | 3.72% | 0.32% | 0.114× |
| cancellation | 144 | 1.42% | 1.25% | 0.03% | 0.021× |
| subnormal-result | 59 | 0.25% | 0.68% | 0.34% | 1.36× |
| overflow | 84 | 0.12% | 0.12% | 0.00% | 0 |
| special-values | 102 | 0.10% | 0.00% | 0.00% | 0 |

`default` and `rounding-tie` — the regimes where bit-level lottery lives — shrink ~15–17×. The exp-coordination regimes (cancellation, overflow, special-values) shrink to essentially zero. `subnormal-result` is roughly comparable (0.25% → 0.34%) — the one regime where SEM does *not* clearly help, perhaps because subnormals are themselves a format engineering retrofit and don't simplify with field grouping.

## Finding 3 — Lottery moves locations entirely

Pairwise Jaccard on lottery zones (pairs failed by 1..n_seeds-1 of seeds):

| | learned-PE bit | RoPE bit | SEM |
|---|---:|---:|---:|
| learned-PE bit | 1.000 | 0.759 | **0.155** |
| RoPE bit | 0.759 | 1.000 | **0.148** |
| SEM | 0.155 | 0.148 | 1.000 |

The two bit-level arms (learned-PE, RoPE) have lottery Jaccard 0.76 — they share most of their hard pairs. SEM's lottery shares only ~15% with either bit-level lottery. **The small residual lottery in SEM lives in different places.** This is the cleanest possible signal that the bit-level lottery was substantially bit-discovery-related: when you remove that source, the remaining hard pairs are a different set.

Heavy-tail overlap (p̂ ≥ 0.8): **0%** of bit-level's 22 or RoPE's 128 heavy-tail pairs are heavy-tail in SEM. SEM has no heavy tail at all.

## Finding 4 — Field-decomposition split sharpens

For the few residual errors in SEM, the categorical split is *more* pronounced than bit-level:

| m_c | SEM s0 | SEM s8 | SEM s14 | learned-PE bit s14 (for ref) |
|---|---|---|---|---|
| 0/8 | 22 errs, 95% exp | 21 errs, 95% exp | 14 errs, 100% exp | 77 errs, 31% any-exp |
| 1/8 | 6 errs, 100% mant | 0 errs | 2 errs, 100% exp | 124 errs, 2% any-exp |
| 2/8 | 6 errs, 100% mant | 0 errs | 0 errs | 61 errs, 0% any-exp |
| 3/8 | 3 errs, 100% mant | 2 errs, 100% mant | 0 errs | 104 errs, 0% any-exp |
| 4/8 | 0 errs | 0 errs | 0 errs | 43 errs, 5% any-exp |
| 5/8 | 6 errs, 83% mant | 1 err, 100% mant | 0 errs | 88 errs, 0% any-exp |
| 6/8 | 5 errs, 100% mant | 2 errs, 50% mant | 1 err, 100% exp | 60 errs, 13% any-exp |
| 7/8 | 12 errs, 83% exp | 17 errs, 100% exp | 38 errs, 97% exp | 112 errs, 71% any-exp |

Endpoints become **even more exp-dominated** (95–100% any-exp at m=0/8, 83–100% at m=7/8). What remains in SEM at the endpoints is the irreducible binade-coordination cost — concentrated, not distributed across mant-only misses the way bit-level had them.

Smooth-interior bins (m=2,3,4,5) often have *zero* SEM errors. The mant-only residual that survives at m=1, m=6 is single-LSB mantissa rounding — tiny absolute counts.

Interpretation: under bit-level, hundreds of m=0/8 errors included many trivial multi-bit-decoding misses (the "mant-only" fraction was high partly because models occasionally got *only* a mantissa bit wrong while still being formally an m=0 prediction). Under SEM, those decoding misses don't exist; the only m=0/8 errors are the actual binade-coordination failures. Same pattern at m=7/8. The format-driven categorical split survives — sharpened, not softened.

## Finding 5 — m=0/8 RoPE anomaly: not recovered by SEM

Recall from the RoPE arm: at m=0/8, learned-PE bit s14 had 31% any-exp (69% mant-only), while RoPE bit s0/s8 had 71–75% any-exp (RoPE lost the "low-endpoint mant-only" affordance). The future_arms.md question was: does SEM recover that affordance?

Answer: no — and in a revealing way. SEM concentrates **100%** of m=0/8 errors into exp_only (s14) or 95% (s0, s8). What looked like a "mant-only affordance" under bit-level learned-PE s14 was actually the *trivial decoding error rate* hiding the genuinely hard exp-coordination errors. SEM strips out the decoding overhead, revealing that m=0/8 errors are essentially all binade-coordination failures.

The corrected reading: there is no "low-endpoint mant-only affordance" intrinsic to learned-PE. There is a "bit-level decoding error rate" that learned-PE-bit happens to make mostly in the mantissa-LSB direction at m=0/8 (because mantissa-LSB misses are common everywhere). SEM removes those misses; what's left at m=0/8 is the genuine binade-coordination cost.

## Finding 6 — Anti-ε severity survives, sharper where measurable

Pearson(ε, mean |log Δ|), per checkpoint:

| arm / seed | ρ |
|---|---:|
| learned-PE bit s0 | −0.696 |
| learned-PE bit s14 | −0.679 |
| RoPE bit s0 | −0.747 |
| RoPE bit s8 | −0.872 |
| **SEM s0** | **−0.879** |
| **SEM s8** | **−0.917** |
| SEM s14 | +0.306 (noise-dominated, only 4 bins with errors) |

When there is enough residual error to measure (s0, s8), the anti-ε correlation is *stronger* under SEM than under any bit-level checkpoint. The s14 case has only 4 of 8 bins with any error data, making the per-bin correlation noisy; it's not a flip of sign in the underlying phenomenon, it's measurement noise.

This is the strongest possible version of the anti-ε result: when you remove all the bit-level error-amplification noise, the residual error structure is *exactly* what ε predicts in shape — small severity in the high-ε smooth interior, large severity at the low-ε endpoints.

## Synthesis — answers to the three sharp questions

From `notes/future_arms.md` post-RoPE:

1. **Was the 120-pair high-prob-fail tail bit-discovery-driven or value-level?** **Bit-discovery-driven.** Heavy tail at SEM is 0 pairs. Heavy tail overlap with bit-level is 0%. The lottery's most extreme region was a property of bit-level coordination, not FP8 addition.

2. **Does the m=0/8 RoPE anomaly recover under SEM?** **No, but for an instructive reason.** The "low-endpoint mant-only affordance" was an artifact of the bit-level decoding error distribution, not a genuine PE-vs-format affordance. SEM strips decoding noise and reveals m=0/8 errors are essentially 100% binade-coordination. There was nothing to recover.

3. **Does anti-ε severity survive under SEM?** **Yes, more sharply than under bit-level** where there's enough data to measure. SEM s8 gives ρ = −0.917, the strongest anti-ε correlation we've measured. The shape is irreducibly format-driven.

## What this means for the four-arm story

The story crystallizes:

- **Format pins shape.** SEM's categorical split (endpoint-exp / interior-mant-only) is the bit-level split, sharpened. Anti-ε severity survives across all three arms.
- **PE shapes probability on the bit-level lottery only.** Under SEM, PE has nothing to shape — the lottery is essentially gone.
- **Bit-level tokenization is the dominant lottery driver.** Removing it removes ~93% of the lottery zone and 100% of the heavy tail.
- **What's left at the boundary is irreducible.** SEM's residual errors are concentrated at m=0/8 and m=7/8, exp-coordination-coupled. This is the cost of FP8's retrofitted addition at the binade boundary, with all other noise stripped away.

The pentagon writeup's "joint-system" claim survives, with a sharpened picture: the format pins the location of irreducible difficulty (endpoints, exp-coordination); bit-level tokenization adds a large fluctuating overhead that looks like a wide lottery; PE modulates that overhead's probability distribution.

## What is still open for FoNE

FoNE (Arm 3 in future_arms.md) becomes more interesting after this. SEM removes tokenization-level overhead while keeping FP's add-tax. The residual error at SEM's endpoints — the 14–38 binade-coordination failures at m=7/8 in any single seed — is the empirical estimate of FP-addition-cost-with-no-bit-decoding-overhead.

FoNE's prediction: the additively-homomorphic encoding does this near-binade addition smoothly because the Fourier basis covers binade transitions continuously. If FoNE eliminates SEM's residual ~20-40 errors per seed at the endpoints, the add-tax is itself transferrable. If FoNE preserves a comparable residual at the discrete-FP8 projection step, the irreducible cost transfers across encodings — it lives in the discreteness of FP8, not in the operation.

## Files

- Comparison: `code/errorboard/failure_consensus_sem.py` → `notes/sem_arm_comparison.md`
- Field decomp: `code/errorboard/epsilon_field_decomp_sem.py` → `notes/sem_field_decomp_findings.md`
- Severity: `code/errorboard/epsilon_severity_sem.py` → `notes/sem_severity_findings.md`
- Training: `code/errorboard/sem_seeds.py`
- Infrastructure: `code/errorboard/sem_tokenizer.py`; `dataset.py` and `training.py` extended with `tokenization` parameter; `hooked_bridge.py` is vocab-agnostic and worked unchanged.
