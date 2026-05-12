# Kreitner et al. — BitTokens source-extraction memo

**Citation.** Linus Kreitner, Paul Hager, Jonathan Mengedoht, Georgios Kaissis, Daniel Rueckert, Martin J. Menten. *Efficient numeracy in language models through single-token number embeddings.* arXiv:2510.06824v1, 8 Oct 2025. Local: `papers/bittokens-2510.06824.pdf` (main p. 1–9, refs p. 10+). Page numbers below refer to v1.

Reference code: `https://github.com/AnonymousAuthor553/BitTokens` — cloned locally as `code/bittokens-ref/` (gitignored).

This is the "baseball stats" extraction. Prose context lives in `notes/papers.md`. §4 is the **load-bearing section for us** because it predicts FoNE's failure mode on multiplication, which is a candidate contrast arm for ErrorBoard once we scale past addition. A second view of FoNE will come from extracting Zhou et al. 2025 directly.

---

## 1. Scope of the paper

- **Problem framing** (p. 1–2): frontier LLMs need 5k–30k reasoning tokens for one calculation; tokenization splits numbers into many tokens; this limits problem complexity. Authors want **single-token number embeddings** that LLMs can do arithmetic on natively.
- **Three contributions** (p. 3): (a) benchmark of 8 frontier LLMs on 9 numeracy tasks; (b) D1–D9 desiderata + analysis showing existing methods fail several; (c) BitTokens, a new IEEE-754-based single-token encoding.

---

## 2. Notation glossary

| Symbol     | Definition                                                                | Page |
|------------|---------------------------------------------------------------------------|------|
| `[NUM]`    | Learnable single token that "is" a number                                 | 5,7  |
| `F`        | Sinusoidal encoding map `R → T^|Φ|` (FoNE-style)                          | 5    |
| `b`        | Frequency base in sinusoidal encoding (`b > 1`)                           | 5    |
| `Φ`        | Frequency index set `⊆ Z`                                                 | 5    |
| `⊙`        | Element-wise (Hadamard) product on the torus                              | 5    |
| `⊗_φ`      | Hypothetical FP-mult operator in sinusoidal space (Prop 4.3)              | 5–6  |
| `P`        | Per-component encoding precision (number of distinct states a dim holds)  | 6    |
| `S^x_φ`    | Subset of input frequencies `⊗_φ` reads from `F(x)`                       | 6    |
| sMAPE      | `|ŷ − y| / (|y| + |ŷ| + ε)`, `ε = 10⁻¹⁰⁰`                                 | 4    |
| log-sMAPE  | `min(1, −log₁₀(sMAPE + ε) / M)` with `M = 15`; "fraction of correct sig figs" | 4    |

### Desiderata D1–D9 (§3, p. 5) — quote-verbatim, terse

- **D1 Token efficiency.** Every number is represented by a single token.
- **D2 Uniqueness.** Each value has exactly one valid encoding, with a unique inverse mapping.
- **D3 Structured.** The encoding geometry reflects numeric order and distance.
- **D4 Scale invariance.** A wide range of magnitudes can be represented with high precision.
- **D5 Normalization.** Encodings are bounded and information-preserving under LayerNorm/RMSNorm.
- **D6 Numerical stability.** Representations remain accurate when using low-precision data type formats (e.g., fp8).
- **D7 Continuity.** Encodings vary smoothly with the underlying value (compatible with gradient-based optimization).
- **D8 Robustness.** Values can be decoded reliably even under stochastic prediction noise.
- **D9 Arithmetic.** Encodings admit learnable algorithms for core mathematical operations.

---

## 3. §2 frontier-LLM benchmark — trimmed

Skipped except for two facts we may cite:

- 8 models tested (GPT-OSS 120B, Qwen3 235B 2507, DeepSeek v3.1, Kimi K2 0905, GPT 5, Gemini 2.5 Pro, Gemini 2.5 Flash, Llama 4 Maverick) × 9 tasks (Add, Mult, Div, Mean, Std, MinMax, Sorting, Interval, Exp). Reasoning settings: Maximal / Minimal / None.
- Headline: **MinMax/Sorting/Interval are saturated. Mult, Div, Exp, Std all collapse to near-zero log-sMAPE without reasoning;** with reasoning enabled, frontier models burn 5k–30k tokens per single calculation (p. 4, "Results"). This is the "motivation" frame, not a result we'll directly rely on.

Full extraction skipped intentionally.

---

## 4. §4 — comparison to existing single-token methods (the load-bearing section)

§4 analyzes the two prior single-token encodings against the desiderata. **This is the section you flagged as the comparison-method we'd test.**

### 4.a xVal (Golkar et al. 2023)

- **Mechanism.** A learnable `[NUM]` token whose embedding is scaled by the numerical value being encoded. The number is decoded from the final hidden state via a separate number head.
- **Satisfies.** D1 (single token), D2 (unique), D3 (structured), D7 (continuous).
- **Fails.** Must rescale inputs to `[−5, 5]` to satisfy D5 and avoid LayerNorm collapsing the value. This violates **D4** (scale invariance), **D6** (numerical stability), and by extension **D9** (arithmetic).

### 4.b FoNE (Zhou et al. 2025) — the contrast arm

- **Mechanism.** A single `[NUM]` token whose embedding is composed of sin/cos features at base-10 frequencies. Each dim corresponds to a sine or cosine of a per-digit-magnitude frequency. The LM's final hidden state is interpreted directly as a sinusoidal encoding; output digits are predicted by maximum cosine similarity against embeddings of `[0..9]` per magnitude.
- **Satisfies.** D1–D8 (token-efficient, unique, structured, scale-invariant, RMS-normed since sin/cos give constant norm, numerically stable, continuous, robust via per-digit max-similarity).
- **Fails.** **D9 — for multiplication specifically.** Addition works (Lemma 4.2 below); multiplication is provably non-local (Prop 4.3 below).

### 4.c Def 4.1 — sinusoidal encoding (p. 5)

> A sinusoidal encoding `F : R → T^|Φ|` maps real numbers to a |Φ|-dimensional torus. Given frequencies `b^φ` with base `b > 1` and `φ ∈ Φ ⊆ Z`:
> `F(x) := [cos(2π b^φ x), sin(2π b^φ x)]_{φ∈Φ} = [e^{i 2π b^φ x}]_{φ∈Φ}`

### 4.d Lemma 4.2 — additive homomorphism (p. 5)

> `F : (R, +) → (T^|Φ|, ⊙)` is a group homomorphism, where `⊙` is the Hadamard product.

Proof (one line, Euler's formula): `F(x₁ + x₂) = [e^{i2πb^φ(x₁+x₂)}]_φ = [e^{i2πb^φ x₁} · e^{i2πb^φ x₂}]_φ = F(x₁) ⊙ F(x₂)`.

**Implication for ErrorBoard.** Under exact arithmetic, sinusoidal encoding turns addition into pointwise complex multiplication on the torus — trivially a local op, no carry-propagation needed. So FoNE should perform comparably to BitTokens on addition. **Below the multiplication threshold, FoNE is not a falsifiable contrast.** This is why the contrast arm becomes interesting only when we extend to multiplication.

### 4.e Prop 4.3 — non-locality and computational complexity of multiplication (p. 5–6)

**Setup.** `X := {ε, …, U}` is the input set with resolution `ε = b^m`, choose `Φ = {m, …, n}` so that `F` uniquely encodes the full range. Assume each encoding component has finite precision `P` (i.e., can hold `P` distinct states). Suppose there exists an operator `⊗_φ : T^|Φ| × T^|Φ| → T` with `⊗_φ(F(x), F(y)) := F_φ(xy)` for each output frequency `φ ∈ Φ`. Let `S^x_φ, S^y_φ ⊆ Φ` be the subsets of input frequencies `⊗_φ` reads from `F(x), F(y)`.

**Claim 1 — non-locality.** `⊗_φ` must access at least `d = O(log_P(U/ε))` components from each input, with `|S^x_φ|, |S^y_φ| ≥ ⌈log_P |X|⌉`.

**Claim 2 — computational complexity.** `⊗` must perform a computation functionally equivalent to polynomial multiplication.

**Proof sketch (extracted, p. 6):**

- *Non-locality*: counting argument. If `|S^x_φ| < log_P |X|`, projection of `F(x)` onto `S^x_φ` has ≤ `P^|S^x_φ|` states. Pigeonhole gives two distinct `x ≠ x'` with same projection. Pick `y* ∉ Y* := {y : b^φ Δy ∈ Z}` (a strict subset of X, so this is possible). Then `F_φ(xy*) ≠ F_φ(x'y*)` but `⊗_φ` sees identical inputs from `F(x)` and `F(x')`. Contradiction.
- *Complexity*: write `x = Σ_ψ k_ψ b^ψ`, `y = Σ_φ l_φ b^φ`. Then `xy = Σ_τ (k*l)_τ b^τ` where `(k*l)_τ = Σ_{ψ+φ=τ} k_ψ l_φ`. Computing `F_φ(xy)` from `F_ψ(x), F_φ(y)` via the bilinear expansion produces redundant cross-terms; canceling them is at least as complex as performing the convolution + carry propagation explicitly. Under finite precision, postponing the disentanglement "amplifies error as intermediate sums of large wrapped phases cause quantization collapse."

**Corollary (p. 6, last ¶ of §4).** Any successful `⊗` in sinusoidal space must internally execute: (1) non-local decode of sinusoidal inputs into integer coefficient sequences, (2) convolve, (3) carry-propagate, (4) re-encode. This is exactly the work that bit-wise encodings give the network for free.

### 4.f Failed workaround — log-space (p. 6)

Authors tried log-space preprocessing: in `log` space, multiplication becomes Hadamard-linear (since `ln(xy) = ln x + ln y` reduces to Lemma 4.2). But then **addition becomes non-trivial** (requires `exp-sum-log`, same non-locality problem). MoE routing to switch between linear and log spaces was attempted but "unsuccessful." We can cite this if anyone proposes log-space encoding as a fix.

---

## 5. §5 — BitTokens construction (p. 7)

Encoding target: **IEEE-754 float64**.

```
v = (−1)^s × (1 + Σ_{i=1}^{52} b_{52−i} · 2⁻ⁱ) × 2^{E−1023}
```

with `s ∈ {0,1}` sign bit, `E ∈ {0..2047}` 11-bit exponent biased by 1023, `b_j ∈ {0,1}` for `j = 0..51` (52 significand bits). Total: **64 bits → 64 embedding dims**, one bit per dim.

Range: `[2.23×10⁻³⁰⁸, 1.8×10³⁰⁸]` with 15–17 significant decimal digits. Supports `±0, ±∞, NaN`.

### Token construction (p. 7)

- Type-reinterpret float64 → 64-bit binary, then bit-shift to extract each bit.
- **Concat reciprocal's bit representation** to ease division learning.
- Scale bit vector to `[−1, 1]` (so `0 → −1, 1 → +1`), giving unit RMS norm — satisfies D5.
- Zero-pad to model embedding dim.
- Add to a learned `[NUM]` token embedding.

### Decode (p. 7)

- LM outputs `[NUM]`. Last hidden state passes through a **dedicated number head**: linear layer → sigmoid → threshold 0.5.
- Loss: **bit-wise binary cross-entropy**, equal weighting per dim.
- They note BCE doesn't strictly satisfy D7 (continuity in numeric value): e.g., target `y=7=0b0111`, predictions `ŷ₁=8=0b1000` is numerically closer but incurs higher BCE than `ŷ₂=3=0b0011`. They argue equal weighting is more important than numeric continuity for learning low-significance bits.

### Arithmetic properties — D9 satisfied (p. 7)

Three structural reasons why bit-wise embeddings make arithmetic learnable:

1. **Direct bit access** unlocks classical hardware algorithms — Booth (1951), Wallace (2006), Brent–Kung (1982), Kogge–Stone (2009), Brent–Zimmermann (2010) — that operate on bits and are IEEE-754-compatible.
2. **IEEE-754 separates** magnitude (log-space exponent) from significand (linear). Add = align-shift + add; mul = add exponents, multiply significands, XOR signs. Each step has small fan-in.
3. **Bit-wise arithmetic over Z₂** = Boolean gates: `(x·y) mod 2 = x ∧ y`, `(x+y) mod 2 = x ⊕ y`. So the simplest arithmetic primitives become parallelizable per-bit.

---

## 6. §6 — empirical results (p. 7–9)

Setup: nanoGPT-2 trained from scratch. Five tokenizers compared: **Subword, Single-digit, Triple-digit (subword), xVal, FoNE, BitTokens**. xVal rescaled log to `[−5, 5]` with `[−]` prepended for negatives. FoNE uses 17 integer + 32 fraction frequencies with base `b = 10`.

Setting: multi-task on 7 tasks (Addition, Mult, Division, MinMax, Interval, Sorting, Fineweb-perplexity) + solo-task on 3 hard tasks (Mean, Std, Exponentiation).

**Key empirical facts:**

- **xVal underperforms on all tasks** due to limited precision in I/O.
- **FoNE learns addition** but **fails on multiplication and division "as predicted in Section 4."** ← this is the cite-able empirical confirmation of Prop 4.3.
- **Single-digit (multi-token) beats all single-token strategies** for some tasks because it generates output one digit at a time, effectively reasoning across forward passes.
- **BitTokens beats all other single-token methods** on 7 tasks; matches or near-matches single-digit. Authors call comparison and single-step calculations "near-perfect."
- **All methods struggle on Std and Exponentiation** (intrinsically sequential).

---

## 7. Mapping to ErrorBoard

### Tokenization: BitTokens vs Option A (ours) vs FoNE

Three single-token-ish strategies for floats, distinguished by **what counts as a token**:

| Strategy            | Tokens per number | Per-token meaning                  | Used in ErrorBoard? |
|---------------------|-------------------|-------------------------------------|---------------------|
| BitTokens           | 1                 | 64-dim embedding, one bit per dim   | No                  |
| FoNE                | 1                 | Sin/cos features per digit-magnitude| No (candidate arm)  |
| **Option A (ours)** | 8 (per FP8)       | One token per bit, embedded via lookup | **Yes** (`task_spec.md` §1) |
| Subword (BPE)       | varies            | Subword string                      | No                  |

**Our Option A is closest to BitTokens in spirit** (bit-level structure) but differs structurally: BitTokens is one *token* with a bit-structured embedding (geometric prior); we have one token *per bit* (algorithmic prior on attention to learn cross-bit interactions). Both should support addition; the empirical question of which generalizes better to multiplication is open and orthogonal to FoNE-vs-bit.

### FoNE as a candidate contrast arm

Not for the current task (FP8 addition) — at addition, FoNE and BitTokens both work per Lemma 4.2. The **contrast becomes falsifiable only at multiplication.** If/when we extend ErrorBoard to FP8 mult:

- Build a FoNE-style embedding for FP8 (frequencies tuned to FP8's range and resolution: `ε = 2⁻⁹`, `U = 448`, so `log_P(U/ε) ≈ log_P(2.3×10⁵)` components needed for mult). The BitTokens paper used `b = 10` for decimal; for FP8 it's natural to use `b = 2`.
- Predict: FoNE arm matches Option A on add; FoNE arm collapses on mult while Option A does not. Falsifies/confirms Prop 4.3 at the FP8 scale.
- Defer until we're past addition. Recorded here so we don't lose it.

### Direct cites we'll use in the mouse

- **Lemma 4.2** to justify why all three structured-embedding strategies should succeed on FP8 addition (predicts no learnability gap at our current task).
- **Prop 4.3** to motivate why per-bit access matters in principle, even if we can't *test* the prediction on addition alone.

---

## 8. Open questions / flags

- **Prop 4.3 lower-bound interpretation.** "Functionally equivalent to polynomial multiplication" — is this a circuit-depth, parameter-count, or attention-head-count lower bound? Memo flags this; check Zhou 2025 for an independent statement of the same claim.
- **Lemma 4.2 under RNE round-off.** `F(a+b) = F(a)⊙F(b)` is *exact*. With FP8 RNE, `a ⊕ b ≠ a + b` in general (round-off), so the homomorphism becomes approximate. The approximation error depends on the binade of the result. We should verify with our oracle whether FoNE-encoded FP8 addition has bounded reconstruction error per regime — could be a useful side-experiment.
- **D7 (continuity) and BitTokens.** BCE loss on bits is *discontinuous* in numeric value (the `7 → 8` jump example). For our mouse this is fine because addition is exact-at-result-bits anyway, but if we ever do regression-style targets, we need to revisit.
- **Reference code.** `code/bittokens-ref/` is in repo (gitignored). Look there if we need actual FoNE / xVal implementations rather than re-deriving from the paper.
- **BitTokens' float64 vs our FP8.** Their construction is float64 (sign + 11 exp + 52 mantissa). Direct FP8 analog: sign + 4 exp + 3 mantissa = 8 dims. We could test BitTokens-style single-token embeddings on FP8 as a third arm alongside Option A and FoNE. Not currently planned but worth recording.
