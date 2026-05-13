# RoPE arm: findings

**Setup.** L4-E048 architecture (d_head=12, RoPE-compatible). Trained 20 seeds
with learned absolute PE (`sweep-L4-E048-s{0..19}`) and 20 seeds with RoPE
(`rope-L4-E048-s{0..19}`). Both at iter 20k, same holdout split (seed=0,
n=6554). Probes re-run with parallel scripts:

- `failure_consensus_pe.py` → `notes/pe_arm_comparison.md`
- `epsilon_bit_decomp_pe.py` → `notes/epsilon_bit_decomp_pe_findings.md`
- `epsilon_severity_pe.py` → `notes/epsilon_severity_pe_findings.md`

## TL;DR

RoPE is worse, **and worse in exactly the regimes that need per-position
binade shaping** — but the *shape* of the error space (which bins are
mant-only, where severity peaks vs ε) is preserved across PE arms. The PE
modulates the *probability* of failure on already-hard pairs; it does not
relocate the lottery.

In Landfall's framing: PE shapes the *additive* affordance the model gets
from its position encoding. Learned PE gives free per-position bias toward
binade-correct shaping. RoPE gives only rotation-symmetric phase mixing,
which is less useful for the addition operation our format taxes.

## Finding 1 — RoPE pays exactly where addition is hardest

Per-regime mean fail rate, n=20 seeds each:

| regime | learned-PE | RoPE | Δ |
|---|---:|---:|---:|
| default (smooth interior) | 23.42% | **30.84%** | **+7.42pp** |
| rounding-tie | 9.65% | 13.88% | +4.23pp |
| large-dexp | 2.81% | 3.72% | +0.91pp |
| subnormal-result | 0.25% | 0.68% | +0.42pp |
| cancellation | 1.42% | 1.25% | −0.17pp |
| overflow | 0.12% | 0.12% | 0 |
| special-values | 0.10% | 0.00% | −0.10pp |

The damage is concentrated in `default` (the smooth-interior mantissa-rounding
regime) and `rounding-tie` — the two regimes where the model's only job is to
get the LSB right via per-position mantissa shaping. Cancellation, overflow,
and special-values, all of which are about exponent/sign coordination, are
roughly invariant to PE. This is the predicted pattern: addition's tax shows
up at the LSB.

## Finding 2 — RoPE has a heavier high-prob-fail tail, not a bigger lottery

Per-pair p̂ distribution (lottery shape, 20 seeds):

| bin | learned-PE pairs | RoPE pairs | ratio |
|---|---:|---:|---:|
| 0/20 (structural easy) | 3868 | 3685 | 0.95× |
| 1/20 | 646 | 553 | 0.86× |
| 2–5/20 | 1262 | 1236 | 0.98× |
| 6–10/20 | 551 | 646 | 1.17× |
| 11–15/20 | 205 | 306 | 1.49× |
| **16–19/20** | **20** | **120** | **6.0×** |
| 20/20 (structural core) | 2 | 8 | 4× |

Mean fail rate: 8.98% → 11.93% (+2.95pp). Variance ratio (obs/i.i.d.): 6.12 → 7.66.
Lottery zone size only grows from 41.0% → 43.7%, **but the distribution
within the lottery shifts hard toward "almost always fails."** RoPE is not
just rolling more dice; it's making some dice nearly deterministic.

## Finding 3 — Same lottery, different probabilities

Lottery-zone Jaccard between PE arms: **0.759**.

- 89.2% of learned-PE's lottery is also in RoPE's lottery
- 83.6% of RoPE's lottery is also in learned-PE's lottery

The set of *which pairs are hard* is largely PE-independent — it's
format-driven (FP-mantissa structure picks them). PE moves the probability
of failure on those pairs, not their location.

This is the key claim that severs format from PE: a different position
encoding does not generate a different lottery; it shifts the same pairs
up or down the difficulty curve.

## Finding 4 — Bit-decomposition shape survives

The smooth-interior-mant-only / endpoint-exp-coupled categorical split
holds across both PE arms (4 checkpoints sampled: learned s0/s14 + RoPE s0/s8):

| m_c | mant-only % (range across 4 ckpts) |
|---|---|
| 0/8 (low endpoint) | 22%–69%, exp-heavy in all |
| 1/8 | 96%–98% mant-only |
| 2/8 | 93%–100% mant-only |
| 3/8 | 100% mant-only |
| 4/8 | 95%–97% mant-only |
| 5/8 | 100% mant-only |
| 6/8 | 84%–87% mant-only |
| 7/8 (high endpoint) | 27%–55%, exp-coupled in all |

Smooth-interior bins are essentially 100% mant-only in every checkpoint
including RoPE. Endpoint bins are exp-coupled in every checkpoint including
RoPE. The categorical zones are FP-format properties, not PE properties.

One PE-flavored sub-finding at m=0/8: learned-PE s14 keeps ~69% mant-only;
RoPE s0/s8 drop to ~22–28% mant-only. So RoPE specifically loses some of
the low-endpoint binade-coordination affordance. (This is one of the
*coordination* affordances learned-PE seems to give for free.)

## Finding 5 — Anti-ε severity survives

Pearson(ε, mean |log Δ|), across 8 m_c bins:

| checkpoint | ρ |
|---|---:|
| learned-PE s0 | −0.696 |
| RoPE s0 | −0.747 |
| learned-PE s14 | −0.679 |
| RoPE s8 | **−0.872** |

The earlier headline result — model errors are catastrophic at endpoints
(small ε) and gentle in the middle (large ε), the opposite of formal
ε's shape — is preserved under RoPE, with comparable or stronger magnitude.
The "irreducible residual surfaces as severity-when-wrong, not frequency"
claim is robust to PE choice.

## Synthesis: what this means for the four-arm plan

The RoPE arm tests whether learned-PE's affordances are doing the work, or
whether the format itself drives the failure structure. The answer is
**both, separably**:

- **Format drives:** (a) which pairs are hard (Jaccard 0.76 lottery overlap),
  (b) the mant-only / exp-coupled categorical split, (c) the anti-ε severity
  shape. None of these change qualitatively under RoPE.

- **PE shapes:** (a) the per-pair failure *probability* (the lottery
  distribution shifts heavier-tailed under RoPE), (b) the low-endpoint
  m=0/8 mant-only fraction (~69% → ~25% under RoPE), (c) the smooth-interior
  fail rate (23% → 31%).

This is consistent with the Landfall framing: ε is irreducible at any PE.
Learned-PE just absorbs more of it through per-position bias; RoPE has
nowhere to put it except into the residual stream and pays accordingly.

**Implications for the remaining arms (per `future_arms.md`):**

- **Bit-level model (priority 2).** The {S,E,M} / bit-decomposition model
  should preserve the anti-ε severity result if it's truly format-driven.
  If bit-decomposition flips the sign of ρ (errors track ε rather than
  anti-track it), the anti-correlation was an artifact of the byte
  tokenization, not of the format.

- **FoNE arm (priority 3).** FoNE is the proper dual: addition becomes
  cheap, multiplication-like-coordination becomes hard. If our
  hypothesis is right that addition is format-taxed under FP, FoNE
  should *invert* the per-regime ordering — cancellation and overflow
  (which need addition-like coordination of magnitudes) should improve,
  while binade-multiplication-like regimes should regress. The RoPE
  result raises the prior that the per-regime ordering is the right
  observable.

## Files

- Comparison script: `code/errorboard/failure_consensus_pe.py`
- Bit-decomp PE script: `code/errorboard/epsilon_bit_decomp_pe.py`
- Severity PE script: `code/errorboard/epsilon_severity_pe.py`
- Raw outputs: `notes/pe_arm_comparison.md`,
  `notes/epsilon_bit_decomp_pe_findings.md`,
  `notes/epsilon_severity_pe_findings.md`
- Training launcher: `code/errorboard/rope_seeds.py`
- Bridge support: `code/errorboard/hooked_bridge.py` (RoPE-aware
  `make_hooked_config` / `convert_state_dict`)
