# Methodology — ErrorBoard "mouse" experiment

Design conventions for the first ErrorBoard experiment: a small decoder-only transformer trained on FP8 E4M3 addition, probed for predicted error structures. Settled choices come first; planned comparison arms come next; tabled variants come last.

**Concrete instantiation:** see `task_spec.md` for tokenization, regime predicates, oracle plan, and sample weighting.

## Sizing principle

Find the smallest `(n_layers, d_model, n_heads)` at which the model **just barely solves the task on the hard regimes** — not on aggregate accuracy. Excess capacity learns redundant, distributed circuits that are hard to read; a model at the capacity edge is forced into clean factorizations because it has nowhere else to put things.

The "hard regimes" are the cases where ε-cost actually lives: opposite-sign cancellation, rounding ties, large-exponent-difference. A model that aces easy cases and fails on these is the wrong frontier to study.

Workflow: sweep down from a reasonable starting box, watching **per-regime** loss curves; pitch the interp work just on the solving side of the frontier.

Starting box (to sweep down from): `d_model=128`, `n_layers=4`, `n_heads=4`, `d_head=32`, `d_mlp=512` (≈1M params, ~minutes per run on the 4090). Likely sweep targets: `n_layers=2`, `d_model=64`. FP add has ~4-5 conceptual stages and can sometimes be parallelized, so >4 layers would be surprising.

## Architecture knobs (settled)

- **No biases anywhere.** Llama-style. One linear object per role — cleaner for weight analysis than `Wx + b`.
- **RMSNorm, pre-norm.** Cleaner residual stream than LayerNorm post-norm. The residual stream is the main inspection target; treat it preciously.
- **SiLU or GELU in the MLP.** Either fine; be consistent within an experiment.
- **No dropout.** Synthetic task with infinite training data — regularization just adds noise to analysis.
- **Untied embeddings.** `W_E` and `W_U` as separate objects. `W_E` rows = "what does this token mean as input"; `W_U` columns = "what direction in residual space promotes this token at output." Want to read them independently.

## Training backbone

- **Training:** unmodified nanoGPT (Karpathy), forked only as needed to swap activation/norm and remove biases per the knobs above.
- **Analysis:** convert trained checkpoints into TransformerLens `HookedTransformer` for observables. The conversion is ~50 lines of weight renaming plus a forward-pass equivalence check; the equivalence check itself is a useful cross-check that the converted object is the same model.

Rationale for keeping the split: nanoGPT keeps total ownership of training; HookedTransformer gives the shared conventions and free hooks for interp without us reinventing them.

## Vocabulary

Build by hand. No BPE. Reserve **disjoint ID ranges per role** so an embedding row's role is obvious from its index alone:

- sign tokens
- exponent tokens
- mantissa tokens
- operators (`+`, `=`)
- special tokens (`<bos>`, `<eos>`, `<pad>`, plus a reserved `<scratch>` slot — see Variants)

## Dataset & regimes

Generate synthetically with **stratified sampling over named regimes:**

- exact-aligned add
- large-exponent-difference (subnormal-ish behavior)
- opposite-sign cancellation (the hard case for rounding error)
- rounding ties
- overflow
- underflow
- special values

**Log per-regime loss during training.** Aggregate loss is the wrong instrument: it averages over regimes that probably emerge at very different steps. When (e.g.) rounding-ties loss drops off a cliff at step 12k, we grab checkpoints just before and just after and ask what changed. The stratification is doing half the interpretability.

## Training rhythm

- **Optimizer:** AdamW, cosine schedule with brief warmup, train to clean convergence.
- **Checkpoints:** log-spaced (steps 100, 300, 1000, 3000, 10000, …). Cheap and essential for capability-emergence analysis.
- **Seeds:** **≥5 per configuration.** Different seeds sometimes learn different circuits for the same task; with fewer than 5 we can't separate seed variance from circuit variance.
- **Determinism:** seed Python, NumPy, PyTorch, CUDA. Fix data ordering. Worth the effort.

## Inspection harness (build day one)

Before any serious interp, the harness must produce, given `(model, input)`:

- residual-stream snapshots at each `(layer, position)`
- all attention patterns
- per-head contributions to the residual stream
- MLP inputs and outputs

`HookedTransformer.run_with_cache` gives this essentially for free post-conversion. This is the lens through which everything will be seen; it's the one piece of infrastructure worth building carefully.

## Comparison arms (planned, not fallbacks)

Per-regime loss and per-head structure get compared **across** arms, not just within one arm.

- **Model size** — the sweep-down axis above.
- **Seed** — ≥5 per cell.
- **Positional embedding** — two arms compared directly:
  - *Default arm:* learned absolute positional embeddings. Argument: sequences are short and fixed-length; learned-absolute is the most inspection-friendly choice, with no rotational geometry inside attention.
  - *Comparison arm:* RoPE. Argument: it's what modern models use; we want to know whether the error-structure story is PE-invariant or PE-specific. If circuits match across arms, that's a strong cross-check; if they diverge, the divergence is itself the finding.

## Variants tabled (revisit; do not bake in)

- **Scratchpad positions.** Appending `N` learned `<scratch>` positions between input and answer would give the model dedicated externalization slots and could increase legibility. But it is an intervention on the very question "where does the model put error?" — adopting it shapes the answer we're trying to measure. If we want to add this, we add it as a *third arm* (no-scratchpad / 4-scratch / 6-scratch), not as a default. The `<scratch>` token is reserved in the vocabulary so flipping this on doesn't require retokenizing.
