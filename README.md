# ErrorBoard

Mechanistic-interpretability study of small transformers learning FP8 E4M3
addition. Trains decoder-only mice (L=1–4, E=32–128) on the full
65,536-pair table and instruments the errors. Probes are structured to
isolate three system layers:

- **format** (FP8 E4M3, its mult-native / add-taxed parlay)
- **tokenization** (bit-level vs SEM 3-token vs eventually FoNE)
- **position encoding** (learned absolute vs RoPE)

The throughline is the irreducible residual function ε(m) = log₂(1+m) − m
from [Landfall](https://adampunk.com) (the precursor paper this work tests
in practice). FP's exponent-linear mantissa-discrete encoding is *almost*
a logarithm; the "almost" is ε. Our task is FP's structurally taxed
operation, addition, all the way through — every probe is measuring that
tax.

## Headline findings

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
   endpoints). Anti-ε severity ρ = −0.917 — strongest yet. Bit-level
   tokenization accounted for ~93% of what the bit-level lottery looked
   like.
4. [future_arms.md](notes/future_arms.md) — what's queued (FoNE,
   cross-axis), with the post-RoPE / post-SEM sharpening incorporated.

Single-claim summary: **format pins the shape of irreducible difficulty;
bit-level tokenization adds a large fluctuating overhead on top; position
encoding modulates that overhead's probability distribution but cannot
shift the format-pinned location.**

## Layout

```
code/errorboard/        training, model, probes
  tokenizer.py            bit-level (12 tokens, seq 28)
  sem_tokenizer.py        SEM 3-token (32 tokens, seq 13)
  training.py             main loop; tokenization ∈ {bit, sem}
  model.py                decoder-only GPT (pre-norm RMS, no biases, SiLU)
  hooked_bridge.py        nanoGPT-style ↔ TransformerLens conversion
  preprocess.py           build the classified 65,536-pair table
  regimes.py              7-regime classifier
  dataset.py              stratified sampler + eval batchers
  sweep.py                model-size sweep driver
  rope_seeds.py           RoPE arm 20-seed launcher
  sem_seeds.py            SEM arm 20-seed launcher
  pentagon.py             5-vertex inspection harness
  epsilon_*.py            severity + bit-decomp probes (+ _pe / _sem variants)
  failure_*.py            lottery / consensus / overlap probes (+ _pe / _sem variants)
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

Reproduce an arm (each ~20 min for 20 seeds at L4-E048):

```bash
python -m errorboard.rope_seeds --runs-dir ../runs   # learned-PE + RoPE arms
python -m errorboard.sem_seeds  --runs-dir ../runs   # SEM arm
```

Run the post-hoc probes:

```bash
python -m errorboard.failure_consensus_sem      # three-arm lottery comparison
python -m errorboard.epsilon_field_decomp_sem   # SEM bit-decomp analog
python -m errorboard.epsilon_severity_sem
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

Open arms (priority order, per `future_arms.md`):

- **FoNE** — addition-native, multiplication-taxed encoding. The only
  arm in the plan that inverts FP's parlay rather than varying access
  to it. Tests whether the format pins shape *or* the operation does.
- **Cross-axis** — RoPE × SEM, FoNE × precision, etc.

Closed: pentagon (bit-level baseline), RoPE arm, SEM arm.
