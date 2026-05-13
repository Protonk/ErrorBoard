# ErrorBoard

Mechanistic-interpretability study of small transformers learning FP8 E4M3
addition. Trains decoder-only mice (L=1–4, E=32–128) on the full
65,536-pair table and instruments the errors. Probes are structured to
isolate three system layers:

- **format** (FP8 E4M3, its mult-native / add-taxed parlay)
- **tokenization** (bit-level vs SEM 3-token vs eventually FoNE)
- **position encoding** (learned absolute vs RoPE)

The throughline is the irreducible residual function ε(m) = log₂(1+m) − m
from [Landfall](https://adampunk.com/documents/landfall.pdf) (the precursor paper this work tests
in practice). FP's exponent-linear mantissa-discrete encoding is *almost*
a logarithm; the "almost" is ε. Our task is FP's structurally taxed
operation, addition, all the way through — every probe is measuring that
tax.

## Headline findings (addition, closed)

Each arm has a synthesis memo. Read in order:

1. [pentagon_writeup.md](notes/pentagon_writeup.md) — 5-vertex baseline
   across training time × architecture × seed; the binade-coordination
   failure mode at endpoints, the lottery zone, the dual-of-ε severity
   reading.
2. [rope_arm_findings.md](notes/rope_arm_findings.md) — RoPE vs learned-PE
   at matched L4-E048 × 20 seeds. RoPE pays exactly where addition is
   hardest (`default` fail 23.4% → 30.8%); lottery Jaccard 0.76 between
   PE arms (same pairs hard, PE shifts probability not location);
   high-probability tail explodes 22 → 128 pairs.
3. [sem_arm_findings.md](notes/sem_arm_findings.md) — 3-token (sign/exp/
   mantissa) tokenization at matched L4-E048 × 20 seeds. Lottery nearly
   eliminated (mean fail 8.98% → 0.68%, heavy tail 22 → 0 pairs).
   Categorical split preserved and sharpened (95–100% exp_only at
   endpoints). Anti-ε severity ρ = −0.917 — strongest yet.
4. [fone_arm_findings.md](notes/fone_arm_findings.md) — F1 (decimal FoNE,
   `T_i = 10^i`) at matched L4-E048 × 20 seeds. Inverts FP's parlay:
   addition is the encoding's local operation. At small capacity FoNE has
   a wide flat-error regime (anti-ε breaks, ρ ≈ +0.4) — *but this is a
   capacity artifact*, see ↓
5. [fone_pilot_findings.md](notes/fone_pilot_findings.md) and
   [fone_transition_memo.md](notes/fone_transition_memo.md) — FoNE scale
   sweep at L=4, n_embd ∈ {48, 64, 96, 128}. Mean ρ moves monotonically
   +0.24 → +0.17 → +0.04 → −0.17. The flat-error regime is real and
   spans a 4× capacity range; strong-negative anti-ε only starts emerging
   at L4-E128 (one seed at ρ = −0.89).
6. [fone_f2_memo.md](notes/fone_f2_memo.md) — F2 (binary FoNE, `T_i = 2^i`)
   ablation. Aligning the period set with FP8's binade bit positions
   collapses errors by 16× at L4-E048 and 100× at L4-E128 — but F2's
   residual is flat, not anti-ε. F2's mechanism is precision (error
   elimination), not shape recovery.

Single-claim summary for addition: **format pins the shape of irreducible
difficulty; bit-level tokenization adds a large fluctuating overhead on
top; position encoding modulates that overhead's probability distribution
but cannot shift the format-pinned location.** With one caveat from F2:
an encoding that hands the model FP8's binade structure architecturally
(binary periods) bypasses the navigation cost and saturates accuracy
without ever producing anti-ε residuals — the model has nothing to be
anti-ε about.

## What's next: multiplication

Addition is the operation FP is taxed for. Multiplication is FP's native
operation. Re-running the four-arm cross under the multiplication oracle
inverts every prediction:

- **bit / RoPE / SEM** (FP-native arms): should do *better* than they did
  on addition. No binade-coordination cost.
- **F1 / F2 (FoNE)**: should do *worse*. Multiplication in the Fourier
  basis is non-local (BitTokens Prop 4.3): two values' product requires
  convolution across all Fourier components.

The multiplication arc is the next experimental thread. Infrastructure
work is sketched in [`notes/future_arms.md`](notes/future_arms.md).

## Layout

```
code/errorboard/        training, model, probes
  tokenizer.py            bit-level (12 tokens, seq 28)
  sem_tokenizer.py        SEM 3-token (32 tokens, seq 13)
  fone_tokenizer.py       F1 decimal FoNE (vocab 10, seq 10, FONE_DIM=12)
  fone_f2_tokenizer.py    F2 binary FoNE  (vocab 10, seq 10, FONE_DIM=36)
  training.py             main loop; tokenization ∈ {bit, sem}
  fone_training.py        F1 FoNE training loop
  fone_f2_training.py     F2 FoNE training loop
  model.py                decoder-only GPT (pre-norm RMS, no biases, SiLU)
  fone_model.py           F1 FoNE GPT (input feature-add + per-digit head)
  fone_f2_model.py        F2 FoNE GPT (binary digits, 2-prototype head)
  hooked_bridge.py        nanoGPT-style ↔ TransformerLens conversion
  preprocess.py           build the classified 65,536-pair table
  regimes.py              7-regime classifier
  dataset.py              stratified sampler + eval batchers (bit / SEM)
  fone_dataset.py         FoNE F1 sampler
  fone_f2_dataset.py      FoNE F2 sampler
  sweep.py                model-size sweep driver
  rope_seeds.py           RoPE arm 20-seed launcher
  sem_seeds.py            SEM arm 20-seed launcher
  fone_seeds.py           FoNE F1 20-seed launcher
  fone_pilot.py           FoNE F1 5-seed launcher with --n-embd
  fone_f2_pilot.py        FoNE F2 5-seed launcher with --n-embd
  fone_transition.py      F1 capacity-sweep analyzer (E ∈ {48,64,96,128})
  fone_f1_vs_f2.py        F1 vs F2 head-to-head comparison
  pentagon.py             5-vertex inspection harness
  epsilon_*.py            severity + bit-decomp probes (per arm variants)
  failure_*.py            lottery / consensus / overlap probes (per arm variants)
  m0_*.py                 m0-anomaly investigations
  fit_*.py                fit-vs-structure probes
notes/                  memos, findings, planning
runs/                   training checkpoints + metrics (gitignored)
```

## Setup

Python 3.10. CPU is fine for L4-E048-scale (~60s/seed at 20k iters).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt   # nanoGPT + TransformerLens 3.2.1
cd code
python -m errorboard.tokenizer       # round-trip self-tests (256 patterns)
python -m errorboard.sem_tokenizer
python -m errorboard.dataset
```

Smoke-test a SEM training run (CPU, ~13s):

```bash
python -m errorboard.training \
  --run-name sem-smoketest --runs-dir ../runs \
  --n-layer 4 --n-head 4 --n-embd 48 --d-mlp 192 \
  --tokenization sem --max-iters 1000 --device cpu
```

Reproduce an arm (each ~7-27 min on GPU depending on size):

```bash
python -m errorboard.rope_seeds      --runs-dir ../runs       # learned-PE + RoPE bit-level
python -m errorboard.sem_seeds       --runs-dir ../runs       # SEM 3-token
python -m errorboard.fone_seeds      --runs-dir ../runs       # F1 decimal FoNE (20 seeds @ L4-E048)
python -m errorboard.fone_pilot      --runs-dir ../runs --n-embd 128  # F1 capacity pilot
python -m errorboard.fone_f2_pilot   --runs-dir ../runs --n-embd 48   # F2 binary FoNE
```

Run the post-hoc probes:

```bash
python -m errorboard.failure_consensus_fone    # four-arm lottery (bit / RoPE / SEM / F1)
python -m errorboard.fone_transition           # F1 capacity sweep analysis
python -m errorboard.fone_f1_vs_f2             # F1 vs F2 head-to-head
python -m errorboard.epsilon_severity_fone     # F1 severity
python -m errorboard.epsilon_digit_decomp_fone # F1 digit-position decomp
```

Each probe writes a markdown findings file under `notes/`.

## Theoretical context

The throughline is described in [notes/papers.md](notes/papers.md) with
per-paper memos:

- [park_memo.md](notes/park_memo.md) — Park et al., binade structure
- [bittokens_memo.md](notes/bittokens_memo.md) — non-locality of multiplication
- [fone_memo.md](notes/fone_memo.md) — additively-homomorphic Fourier encoding
- [feng_memo.md](notes/feng_memo.md) — bit-decoding interpretability priors

The "Landfall" framing — FP as the mult-native dual of FoNE, ε as the
cost of retrofitting addition onto a multiplicative encoding — is
articulated in `future_arms.md`'s Arm 3 section.

## Status

**Addition: closed.** All four arms trained, probed, and synthesized.
Five-vertex pentagon, RoPE comparison, SEM tokenization, FoNE F1 with
capacity transition, FoNE F2 ablation — see headline findings above.

**Multiplication: open, next.** Re-runs the same four-arm cross under
the FP8 multiplication oracle. Predicted inversion: FP-native arms (bit,
RoPE, SEM) should improve; FoNE arms should regress (multiplication is
non-local in Fourier basis, BitTokens Prop 4.3). Infrastructure-shared
with addition — new oracle, new regime classifier, same model + training
pipeline. Plan: [`notes/future_arms.md`](notes/future_arms.md).

**Future cross-axis** (after multiplication): RoPE × SEM, FoNE × precision,
SEM × RoPE × multiplication, etc.
