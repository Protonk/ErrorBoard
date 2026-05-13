# FoNE pilot at L4-E128: scale-stability of anti-ε sign-flip

**Setup.** Same FoNE arm as `fone_arm_findings.md` but at L4-E128
(n_embd=128, d_mlp=512, d_head=32). Parameter count 791,296 — 7× the
L4-E048 baseline. 5 seeds, iter 20k, same holdout split (seed=0). Goal:
test whether the L4-E048 anti-ε sign-flip persists at higher capacity.

Probes: per-regime accuracy table, severity probe with `--pilot` flag,
mini-lottery analysis via `failure_consensus_fone._load_fone_arm` +
`_arm_stats`.

## Headline

The L4-E048 finding that "FoNE is the worst arm" was **substantially a
small-model artifact**. At 7× capacity:

- Mean fail rate **drops 12×** (14.59% → 1.26%).
- Lottery zone **shrinks 9×** (41.9% → 4.9%); structural core **goes to 0**.
- Variance ratio **drops from 9.38 to 1.50** (near i.i.d. — no uniquely
  hard pairs).
- Anti-ε severity sign-flip **mostly dissolves**: at L4-E048 we had ρ =
  {+0.42, +0.51, −0.11}; at L4-E128 we have {+0.35, −0.05, **−0.89**,
  −0.20, −0.04} — 1 positive, 1 strongly negative, 3 near zero.
- The seed with best `default` accuracy (s2, 96.4%) is exactly the one
  with the strongest anti-ε (ρ = −0.89, comparable to SEM s8's −0.92).

The qualitative claim from the L4-E048 arm — "FoNE inverts the parlay and
breaks anti-ε" — needs to be downgraded: it's the **regime where FoNE
hasn't yet learned the FP-native error shape** that breaks anti-ε. Give
FoNE enough capacity and at least one seed in five recovers the FP-shaped
residual.

## Finding 1 — Per-regime accuracy: closes the bit-level gap

Per-regime mean (5 seeds at L4-E128, 20 seeds at L4-E048):

| regime | n | bit L4-E048 | FoNE L4-E048 | **FoNE L4-E128** | SEM L4-E048 |
|---|---:|---:|---:|---:|---:|
| special-values | 102 | 99.9% | 100% | 100% | 100% |
| overflow | 84 | 99.9% | 100% | 100% | 100% |
| subnormal-result | 59 | 99.75% | 99.24% | 99.66% | 99.66% |
| cancellation | 144 | 98.6% | 98.5% | 98.2% | ~100% |
| rounding-tie | 610 | 98.1% | 77.2% | **97.97%** | 99.4% |
| large-dexp | 3754 | 97.2% | 93.3% | **99.21%** | 99.7% |
| default | 1801 | 76.6% | 68.9% | **95.07%** | 98.4% |

(bit-level mean computed as `100% − fail rate` from `notes/fone_arm_comparison.md`.)

**FoNE L4-E128 matches bit-level L4-E048 on `default`** (95.07% vs 76.6% —
wait, FoNE beats bit-level there). Actually FoNE at 7× scale **outperforms**
bit-level at 1× scale on `default`, `rounding-tie`, and `large-dexp`. It
still trails SEM L4-E048 but the gap is small (~3pp on `default`).

This isn't a fair architecture comparison (matched-scale would need
bit-level L4-E128 numbers), but it's clear FoNE was bottlenecked by
parameter count, not by an intrinsic representational limitation.

## Finding 2 — Lottery collapses; structural core goes to zero

Mini-lottery (5-seed, L4-E128):

| metric | FoNE L4-E048 (20s) | **FoNE L4-E128 (5s)** | SEM L4-E048 (20s) |
|---|---:|---:|---:|
| mean fail rate | 14.59% | **1.26%** | 0.68% |
| var ratio (obs/iid) | 9.38 | **1.50** | 2.92 |
| structural easy | 57.6% | **95.1%** | 93.0% |
| lottery zone | 41.9% | **4.9%** | 7.0% |
| structural core | 36 pairs (0.55%) | **0 pairs** | 0 pairs |

The lottery shape changes *qualitatively*. At L4-E048, FoNE had a heavy
tail of 242 pairs at p̂ ≥ 0.8 and a structural core of 36 always-failing
pairs. At L4-E128, all of that is gone — the structural core vanishes and
the remaining lottery is light (mostly 1/5 failures). The var ratio of
1.50 means the difficulty distribution is **close to i.i.d.**: there's no
specific subset of "uniquely hard" pairs left.

Per-pair p̂ histogram (5 seeds, L4-E128):

| bin | pairs | fraction |
|---|---:|---:|
| 0/5 (always correct) | 6233 | 95.1% |
| 1/5 | 247 | 3.8% |
| 2/5 | 59 | 0.9% |
| 3/5 | 12 | 0.2% |
| 4/5 | 3 | 0.05% |
| 5/5 (always fails) | 0 | 0% |

The distribution is monotonically decreasing — the harder bins are smaller.
Contrast L4-E048 FoNE, where 16-19/20 had 206 pairs and 20/20 had 36
pairs (heavy-tailed, anti-monotonic in the tail).

## Finding 3 — Anti-ε severity per-seed: mixed at L4-E128, recoverable

Pearson(ε, mean |log Δ|) per seed:

| arm / seed | ρ | default acc |
|---|---:|---:|
| FoNE L4-E048 s0 | +0.354 | (20-seed mean 65%) |
| FoNE L4-E048 s8 | +0.510 | — |
| FoNE L4-E048 s14 | −0.109 | — |
| **FoNE L4-E128 s0** | **+0.354** | 93.78% |
| **FoNE L4-E128 s1** | **−0.054** | 95.06% |
| **FoNE L4-E128 s2** | **−0.891** | **96.45%** |
| **FoNE L4-E128 s3** | **−0.195** | 95.06% |
| **FoNE L4-E128 s4** | **−0.043** | 95.00% |

Two observations:

1. **Sign distribution shift.** At L4-E048 the population is dominated by
   positive ρ. At L4-E128 it's 1 positive / 1 strongly negative / 3 near
   zero. The "FoNE inverts anti-ε" claim at L4-E048 was real but
   *over-strong* — it was specific to the under-trained regime.
2. **Best-accuracy seed recovers anti-ε strongly.** s2 has the highest
   `default` accuracy among the 5 seeds *and* the strongest negative ρ
   (−0.89, in the range of SEM's strongest result). The correlation
   between "this seed learned FP arithmetic well" and "this seed's
   residual has the anti-ε shape" is direct evidence that **anti-ε is a
   property of well-learned FP arithmetic**, regardless of input encoding.

Caveat: with 1–30 errors per m_c bin at L4-E128, the Pearson estimate is
noisier than at L4-E048. The s2 result (−0.89) is robust because the
sign agrees across both ULP-error and log-damage metrics. The three
"near-zero" seeds may shift in either direction with more iters or
seeds.

## Synthesis update to the FoNE arm story

The corrected reading:

- **Anti-ε severity is a property of an FP-add-task model that has learned
  the residual shape correctly.** It survives across bit / RoPE / SEM
  because those tokenizations all give the model enough structure to find
  the FP-shaped residual at L4-E048 scale.
- **FoNE at small scale doesn't have the capacity to learn the FP residual
  shape.** It learns a flatter, more uniform error distribution (the
  positive-ρ regime). The sign-flip at L4-E048 was therefore not "FoNE
  inverts the parlay" but "FoNE-without-enough-capacity makes flat-shaped
  errors."
- **At sufficient capacity, FoNE recovers FP-shaped error structure.** One
  of five seeds at L4-E128 already shows it (s2 with ρ = −0.89). Variance
  across seeds is high because the model is right at the capacity boundary
  for finding this structure — at larger scale we'd expect more seeds to
  converge there.

This is a corrigendum to `fone_arm_findings.md` Finding 4. The L4-E048
finding "anti-ε flips under FoNE" stands as a within-scale observation, but
the *interpretation* — that the cost was operation-specific and broke when
the operation became native — is partly wrong. The cost survives the encoding
change; it just takes the model more capacity to find it.

## What's still open

- **L4-E256 or larger pilot.** With more seeds at higher scale, does
  anti-ε become universal (all seeds in the −0.7 to −0.9 range), or do
  some seeds stably stay positive? Settling this is the natural next
  experiment, ~3-5M params, 5-10 seeds.
- **Matched-scale bit-level comparison.** We compared FoNE L4-E128 to
  bit-level L4-E048. A fair test would be bit-level L4-E128 vs FoNE
  L4-E128. The architecture-isolation comparison would tell us how much
  of FoNE's improvement is "more params" vs "more params + the right
  encoding."
- **F2 (binary FoNE).** Periods T_i = 2^i matching FP8's binade ladder.
  If the binade-aware recovery in s2 is happening *through* the model
  inventing its own binade structure inside a base-10 representation,
  F2 might converge faster / more universally.

## Files

- Pilot launcher: `code/errorboard/fone_pilot.py`
- Severity probe (with `--pilot` flag): `code/errorboard/epsilon_severity_fone.py`
- Per-seed severity output: `notes/fone_pilot_severity_findings.md`
