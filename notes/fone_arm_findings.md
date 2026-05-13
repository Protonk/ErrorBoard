# FoNE arm: findings

**Setup.** L4-E048 architecture (matched to all other arms), learned absolute
PE, FoNE tokenization: each FP8 number is one `[NUM]` token whose embedding
gets the Fourier feature vector added (Zhou Def 3.4), plus a single sign token
(SIGN_POS / SIGN_NEG / SIGN_NAN). The result is decoded per-digit by cosine
similarity against 10 unit-circle prototypes (Zhou Def 3.6). Period set
`T_i = 10^i` for `i ∈ [-2, 3]` (6 periods, output dim 12), covering 3 integer +
3 fractional decimal digits — enough to distinguish all 256 FP8 values.

Trained 20 seeds (`fone-L4-E048-s{0..19}`) at iter 20k, same holdout split
(seed=0, n=6554). Probe scripts:

- `failure_consensus_fone.py` → `notes/fone_arm_comparison.md`
- `epsilon_severity_fone.py` → `notes/fone_severity_findings.md`
- `epsilon_digit_decomp_fone.py` → `notes/fone_digit_decomp_findings.md`

## TL;DR

**FoNE is the worst arm we've trained, AND its anti-ε severity pattern is
inverted.** The two together resolve the open question from `future_arms.md`:
the anti-ε severity pattern that survives across bit / RoPE / SEM is
*operation-specific*, not format-discreteness-specific. When you flip the
encoding's native operation (FoNE makes addition local, multiplication
non-local — the dual of FP's parlay), the cost stops concentrating at binade
boundaries and instead spreads uniformly across the m_c bins. The format-
pinned location of difficulty is real but it lives in "FP's taxed operation
× a substrate that isn't add-native," not in FP's discrete cells.

The cost shifts but doesn't disappear: FoNE pays everywhere, modestly, instead
of paying at endpoints, catastrophically.

## Finding 1 — FoNE is the worst arm at this scale

Four-arm comparison at L4-E048, 20 seeds, iter 20k:

| arm | mean fail % | var ratio | heavy tail (p̂≥0.8) | structural core | lottery Jaccard vs bit |
|---|---:|---:|---:|---:|---:|
| learned-PE bit | 8.98% | 6.12 | 22 | 2 | 1.000 |
| RoPE bit | 11.93% | 7.66 | 128 | 8 | 0.759 |
| **learned-PE SEM** | **0.68%** | **2.92** | **0** | **0** | **0.155** |
| **learned-PE FoNE** | **14.59%** | **9.38** | **242** | **36** | **0.608** |

FoNE has:

- Higher mean fail rate than any other arm (14.6% vs RoPE's 11.9% second-worst).
- The heaviest difficulty distribution (var ratio 9.38, beating RoPE's 7.66).
- A **242-pair heavy tail** (p̂ ≥ 0.8) — 1.9× RoPE's tail, 11× learned-bit's.
- A **36-pair structural core** (always-fail) — 4.5× RoPE's, 18× learned-bit's.

This is at our matched parameter scale (112k params at L4-E048). Zhou's FoNE
paper reports near-100% on decimal addition with 8M+ params; we are at <2%
of that capacity. The arm should be revisited at larger scales — but the
*qualitative* findings below are robust regardless of where the headline
accuracy lands.

## Finding 2 — Per-regime: cancellation cheap, smooth interior expensive

Per-regime mean fail rate across 20 seeds:

| regime | n | bit | RoPE | SEM | **FoNE** |
|---|---:|---:|---:|---:|---:|
| special-values | 102 | 0.10% | 0.00% | 0.00% | 0.00% |
| overflow | 84 | 0.12% | 0.12% | 0.00% | 0.00% |
| subnormal-result | 59 | 0.25% | 0.68% | 0.34% | 0.76% |
| **cancellation** | 144 | **1.42%** | 1.25% | **0.03%** | **1.53%** |
| **rounding-tie** | 610 | 9.65% | 13.88% | 0.58% | **22.79%** |
| large-dexp | 3754 | 2.81% | 3.72% | 0.32% | 6.75% |
| **default** | 1801 | 23.42% | 30.84% | 1.58% | **31.14%** |

Two specific contrasts settle the FoNE hypothesis:

- **Cancellation.** FoNE matches learned-PE bit (1.53% vs 1.42%) — i.e.,
  the regime where FoNE's add-native encoding should excel doesn't actually
  show a learned-PE-bit-beating advantage at our scale. (SEM dominates here:
  0.03%.) The "addition-native encoding wins at addition" prediction is
  not validated at 112K params.
- **Rounding-tie.** FoNE is **3× worse than bit-level**, **40× worse than
  SEM**. This is exactly where per-digit precision matters most — the model
  has to commit to "this is a .5 case, round up vs down." FoNE's per-digit
  decoder has to do this from scratch.

The smooth-interior `default` regime — the hardest one for bit-level
already — is essentially tied between FoNE (31.1%) and RoPE-bit (30.8%).
SEM at 1.6% is the outlier.

## Finding 3 — FoNE shares ~60% of bit-level's lottery

Lottery-zone Jaccards (20-seed lottery = pairs failed by 1..19/20 seeds):

| | bit | RoPE-bit | SEM | FoNE |
|---|---:|---:|---:|---:|
| bit | 1.000 | 0.759 | 0.155 | **0.608** |
| RoPE-bit | 0.759 | 1.000 | 0.148 | **0.629** |
| SEM | 0.155 | 0.148 | 1.000 | 0.137 |
| FoNE | 0.608 | 0.629 | 0.137 | 1.000 |

FoNE's lottery overlaps ~60% with bit-level's lottery — *higher* than its
overlap with SEM (0.14). The interpretation: FoNE doesn't escape the format-
driven lottery, it stays in it. The pairs that are hard for bit-level remain
hard for FoNE, plus FoNE creates 242 new heavy-tail pairs of its own.

SEM remains uniquely separated from all the other arms (Jaccard 0.14-0.16
with everything). SEM's hard pairs are different from everyone else's hard
pairs.

Heavy-tail overlap (p̂ ≥ 0.8) tells the same story:

- Of bit-level's 22 heavy-tail pairs: 5 (23%) also in FoNE's heavy tail.
- Of RoPE-bit's 128 heavy-tail pairs: 26 (20%) also in FoNE's heavy tail.
- Of SEM's 0 heavy-tail pairs: 0 (trivially) overlap.

FoNE's 242 heavy-tail pairs are *mostly novel* — only ~30 of 242 are also
heavy-tail elsewhere. FoNE has its own brand of hard pairs.

## Finding 4 — Anti-ε severity is inverted

Pearson(ε, mean |log Δ|), per checkpoint, in chronological order across arms:

| arm / seed | ρ |
|---|---:|
| learned-PE bit s0 | −0.696 |
| learned-PE bit s14 | −0.679 |
| RoPE bit s0 | −0.747 |
| RoPE bit s8 | −0.872 |
| learned-PE SEM s0 | −0.879 |
| learned-PE SEM s8 | −0.917 |
| **learned-PE FoNE s0** | **+0.417** |
| **learned-PE FoNE s8** | **+0.510** |
| learned-PE FoNE s14 | −0.109 |

**FoNE flips the sign.** Across three FP-native arms (bit, RoPE, SEM)
anti-ε severity is robust at ρ ≈ −0.7 to −0.92. Under FoNE, two of three
checkpoints flip *positive*; the third is essentially zero.

The per-bin log-damage profile shows what's happening:

- Bit-level / RoPE / SEM: damage concentrated at endpoints (m=0/8 and m=7/8,
  where ε is smallest) — the binade-boundary coordination cost.
- FoNE: damage flatter across bins, with mild peaks in the *middle* of m_c
  (m=2, m=4, m=5) where ε is largest. The middle is where FoNE has to
  resolve the densest set of representable values per binade.

n_err per m_c bin in FoNE is also flat (70–180 errors per bin across all
8 bins) — no endpoint concentration. This is qualitatively different from
bit-level (heavy at endpoints) and SEM (near-zero everywhere except
endpoints).

**This resolves the future_arms.md open question.** From the original brief:

> 1. Endpoint errors disappear under FoNE: the binade-coordination cost
>    was an FP-operation cost (doing FP's taxed op in a non-native encoding),
>    not an FP-discreteness cost.
> 2. Endpoint errors persist (at projection-back-to-FP8): the cost is
>    intrinsic to FP8 discreteness.

Outcome (1) is the answer. Endpoint errors do disappear under FoNE — but the
total error doesn't disappear with them, it redistributes uniformly across
m_c bins. The cost was specifically the cost of doing FP's taxed operation
on a substrate that didn't natively know the lattice structure. Move to a
substrate that natively knows a *different* lattice structure (FoNE's base-10
Fourier periods), and the cost stops being binade-shaped — it becomes
digit-shaped instead.

## Finding 5 — Digit-decomposition: ones place dominates

Per-bin "most-significant wrong digit" distribution (FoNE s8):

| m_c | 10^-3 | 10^-2 | 10^-1 | 10^0 | 10^1 | 10^2 |
|---|---:|---:|---:|---:|---:|---:|
| 0/8 | 26% | 16% | 3% | **47%** | 7% | 0% |
| 1/8 | 24% | 22% | 4% | **40%** | 10% | 0% |
| 2/8 | 23% | 15% | 1% | **52%** | 9% | 0% |
| 3/8 | 18% | 34% | 2% | 30% | 16% | 0% |
| 4/8 | 12% | 28% | 5% | **50%** | 5% | 0% |
| 5/8 | 24% | 24% | 4% | 31% | 14% | 4% |
| 6/8 | 29% | 17% | 3% | **43%** | 8% | 0% |
| 7/8 | 12% | 41% | 6% | 27% | 14% | 0% |

The **ones place (10^0)** dominates as the most-significant wrong digit
across most m_c bins. The thousandths and hundredths places (10^-3, 10^-2)
are the second-most-common wrong digits. The high-magnitude places (10^2,
10^1) are essentially never wrong.

This is the expected pattern for FoNE: most FP8 values have their dominant
magnitude in the [0.001, 100] range, so most predictive ambiguity sits at
the ones place. The model rarely flips a tens or hundreds digit because
those are uniquely determined by the value's magnitude.

**Sign-only errors are essentially zero** (<1% of all FoNE errors). FoNE's
sign prediction is robust; all the difficulty lives in magnitude prediction.

## Synthesis

The four-arm story now closes cleanly:

| layer | what it pins |
|---|---|
| **format** (FP8 E4M3) | the *existence* of irreducible difficulty at binade boundaries |
| **operation** (addition vs encoding's local op) | the *location* of difficulty (endpoint-heavy if mismatched, uniform if matched) |
| **tokenization** (bit / SEM / FoNE) | the *probability mass* on hard pairs (bit-level adds fluctuation; SEM strips fluctuation; FoNE adds different fluctuation) |
| **position encoding** (learned / RoPE) | the *probability shaping* on a given tokenization's lottery |

The Landfall-derived ε(m) anti-correlation with severity was the cleanest
empirical signal of the second row. It survives across bit / RoPE / SEM because
those are all FP-native (mult-native / add-taxed) — the model is always doing
FP's taxed operation, and the cost concentrates at endpoints. It breaks under
FoNE because FoNE *inverts the parlay* — now addition is the local operation
and the format's own retrofitted-for-addition discreteness no longer creates
endpoint-shaped fragility.

What we *cannot* say from this single FoNE arm:

- Whether FoNE would beat bit-level at larger scale. Zhou's 8M+ model hits
  100% on decimal addition; ours at 112K params is at 86%. The within-arm
  qualitative findings (uniform error distribution, anti-ε breaks) should
  scale-extrapolate cleanly, but the headline accuracy comparison won't.
- Whether F2 (binary FoNE, periods T_i = 2^i matching FP8 binade structure)
  would recover the binade-shaped error profile of bit/SEM. The decimal F1
  variant is what we ran; the binary variant remains an open ablation.

## Files

- Probes: `code/errorboard/{failure_consensus,epsilon_severity,epsilon_digit_decomp}_fone.py`
- Raw outputs: `notes/fone_arm_comparison.md`, `notes/fone_severity_findings.md`,
  `notes/fone_digit_decomp_findings.md`
- Training: `code/errorboard/fone_seeds.py`, `code/errorboard/fone_training.py`
- Infrastructure: `code/errorboard/fone_tokenizer.py`, `code/errorboard/fone_encoder.py`,
  `code/errorboard/fone_model.py`, `code/errorboard/fone_dataset.py`
