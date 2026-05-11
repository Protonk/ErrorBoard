# Reference papers — load-bearing claims for ErrorBoard

Each summary names the central claim, the relevant theorem/proposition numbers, and the scaling/threshold predictions. Theorem statements are quoted verbatim from the source PDFs in `papers/`. Symbols are transcribed in Markdown; refer to the PDFs for exact typesetting.

---

## BitTokens — Kreitner, Hager, Mengedoht, Kaissis, Rueckert, Menten (2025)
*Efficient numeracy in language models through single-token number embeddings.* arXiv:2510.06824v1, 8 Oct 2025. `papers/bittokens-2510.06824.pdf`. Reference code: `code/bittokens-ref/` (clone of https://github.com/AnonymousAuthor553/BitTokens).

**Central claim.** A bit-wise IEEE-754 float64 embedding (`BitTokens`) is the first single-token number encoding that simultaneously satisfies a set of nine desiderata (D1–D9) the authors postulate for numeric tokens, and it enables small language models (nanoGPT-2) to learn arithmetic algorithms from scratch. Two formal results undergird this: sinusoidal (Fourier) encodings are an additive homomorphism (so addition is trivial), but multiplication in sinusoidal space is provably non-local and equivalent to polynomial multiplication.

**Relevant formal statements (Section 4).**

- **Definition 4.1 (sinusoidal encoding).** "*A sinusoidal encoding F : R ↦ T^|Φ| maps real numbers to a |Φ|-dimensional torus, which forms a compact abelian lie group. Given frequencies b^φ with base b > 1 and φ ∈ Φ ⊆ Z, then F with base b and Φ is defined as: F(x) := [cos(2π b^φ x), sin(2π b^φ x)]_{φ∈Φ} = [e^{i 2π b^φ x}]_{φ∈Φ}*"

- **Lemma 4.2 (additive homomorphism).** "*The encoding map F is a group homomorphism from the additive group of real numbers (R, +) to the multiplicative torus (T^|Φ|, ⊙), where ⊙ denotes the element-wise (Hadamard) product.*"

- **Proposition 4.3 (non-locality and computational complexity of multiplication).** "*Let X := {ε, …, U} be the set of input numbers with resolution ε = b^m and choose Φ = {m, …, n} so that F uniquely encodes the entire number range. Assume each encoding component has a finite precision P (i.e., can represent P distinct states). Suppose there exists an operator ⊗_φ : T^|Φ| × T^|Φ| → T such that ⊗_φ(F(x), F(y)) := F_φ(xy) for each output frequency φ ∈ Φ. Let S^x_φ, S^y_φ ⊆ Φ be the subsets of input frequencies that ⊗_φ is required to read from F(x) and F(y), respectively. Then:*
  1. *Non-locality. The operator ⊗_φ must access at least d = O(log_P(U/ε)) components from each input vector with |S^x_φ|, |S^y_φ| ≥ ⌈log_P |X|⌉.*
  2. *Computational complexity. The operator ⊗ must perform a computation functionally equivalent to polynomial multiplication.*"

**Scaling/threshold predictions ErrorBoard cares about.** Bit-wise encoding renders bit-level binary arithmetic algorithms (Booth, Wallace, Brent–Kung, Kogge–Stone, Brent–Zimmermann) "directly accessible" to the network because each bit occupies a separate embedding dimension (paper §5, "Arithmetic properties"). The empirical scaling claim is that BitTokens "outperforms all other methods and achieves near-perfect performance on comparison and single-step calculation tasks" in a multi-task nanoGPT-2 setup; exponentiation, mean, and standard deviation remain hard for small LMs and require solo-task training (paper §6, "Results").

---

## Park, Park, Hwang (2026)
*On the Expressive Power of Floating-Point Transformers.* arXiv:2601.16450v1, 23 Jan 2026. `papers/park-2601.16450.pdf`.

**Central claim.** Floating-point transformers — transformers with floating-point parameters and floating-point ⊕, ⊗ (non-associative) — do **not** inherit the expressivity guarantees of real-arithmetic transformers. They (a) can represent some non-permutation-equivariant functions even without positional encoding (exploiting non-associativity of FP addition), (b) cannot represent all permutation-equivariant functions when the sequence length n is large, but (c) can represent all permutation-equivariant functions when n is bounded relative to the mantissa width 2^p. Their minimal equivariance is the swap-first-two permutation π^n_{(1,2)}.

**Relevant theorem statements (Section 3).** The paper writes p, q for mantissa and exponent widths; F_{p,q} is the finite-FP set; F̄ adds {∞, −∞, NaN}. Condition 1 requires 2 ≤ p ≤ 2^{q-1} − 3 (FP8 E4M3 has p = 3, q = 4 and satisfies this).

- **Theorem 1.** "*Let d_in, d_out, n ∈ N such that n ≥ 2 and Δ_n := {[x_1, …, x_n] ∈ F^{d_in × n} : x_i ≠ x_j ∀ i ≠ j}. Then, for any π^n_{(1,2)}-equivariant f* : Δ_n → F^{d_out × n}, there exists a floating-point transformer f : F^{d_in × n} → F^{d_out × n} such that f = f* on Δ_n.*"

- **Theorem 2.** "*Let d_in, d_out, n, α, β ∈ N such that α ≥ 3 × 2^p, β − α ≥ 6 × 2^p, and n ≥ β. Then, for any (α, β)-similar X, Y ∈ F^{d_in × n} and for any floating-point transformer f : F^{d_in × n} → F^{d_out × n}, f(X) and f(Y) are also (α, β)-similar.*"

- **Theorem 3.** "*Let d_in, d_out, n ∈ N such that n ≤ 6 × 2^p − 2. Then, for any permutation-equivariant function f* : F^{d_in × n} → F^{d_out × n}, there exists a floating-point transformer f : F^{d_in × n} → F^{d_out × n} such that f = f* on F^{d_in × n}.*"

- **Theorem 4.** "*Let d_in, d_out, n ∈ N such that n ≥ 2. Then, for any floating-point transformer f : F^{d_in × n} → F^{d_out × n}, f is π^n_{(1,2)}-equivariant.*"

(Theorem 5 concerns positional encoding; the paper's main thresholds for ErrorBoard are 2–4.)

**Scaling/threshold predictions ErrorBoard cares about.** The boundaries scale with 2^p:

- **Full representability ceiling (Thm 3):** n ≤ 6 · 2^p − 2.
- **Saturation / collapse onset (Thm 2):** kicks in once α ≥ 3 · 2^p and β − α ≥ 6 · 2^p with n ≥ β, so the impossibility class shows up at n ≥ 9 · 2^p.

For FP8 E4M3 (p = 3, so 2^p = 8): full representability up to **n ≤ 46**, impossibility class appears at **n ≥ 72**. The paper explicitly flags that "9 × 2^p can be small, especially for low-precision formats: e.g., 8-bit formats E5M2 (p = 2), E4M3 (p = 3)" (paper §3.2 discussion following Thm 2).

---

## Feng, Yang, Gu, Ai, Luo, Sun, He, Li, Wang (2024/2025)
*How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs.* arXiv:2410.13857v2, 21 Jun 2025. `papers/feng-2410.13857.pdf`.

**Central claim.** Numerical precision is the controlling factor for transformer arithmetic capacity. Constant-precision (e.g., int4, int8, fp8) transformers of constant depth require super-polynomial hidden dimension to solve iterated addition (IterADD) or integer multiplication (MUL); standard / logarithmic-precision transformers solve all three elementary tasks (ADD, IterADD, MUL) with constant or polynomial size. The summary table (Table 1) makes the contrast explicit.

| Task | Standard precision | Low precision |
|---|---|---|
| Integer Addition ADD_p(n) | Constant | O(n²) |
| Iterated Addition IterADD_p(n,k) | Constant | Super-polynomial |
| Integer Multiplication Mul_p(n,l) | O(n²) | Super-polynomial |

**Relevant theorem statements (Sections 4 and 5).** T_c is a tokenizer with token width at most c digits (Eq. 1). "Constant-precision" means each neuron carries c bits independent of input length; "logarithmic-precision" means O(log n) bits per neuron.

- **Theorem 4.1.** "*Fix integers p ≥ 2 and c ∈ N*. Consider the tokenizer T_c defined in Eq. (1) for processing the input and output sequences. There exist constant-precision Transformers with constant depth (independent of n) and hidden dimension d = O(n²) that can solve the ADD_p(n) task.*"

- **Theorem 4.2.** "*Fix integers p ≥ 2 and c, L ∈ N*. Consider the tokenizer T_c defined in Eq. (1) for processing the input and output sequences. For any polynomial f, there exist problem scales n and k such that no constant-precision autoregressive Transformer with L layers and hidden dimension d < f(n,k) can correctly solve the IterADD_p(n,k) task.*"

- **Theorem 4.3.** "*Fix integers p ≥ 2 and c, L ∈ N*. Consider the tokenizer T_c defined in Eq. (1) for processing the input and output sequences. For any polynomial f, there exist problem scales n and l such that no constant-precision autoregressive Transformer with L layers and hidden dimension d < f(n,l) can correctly solve the MUL_p(n,l) task.*"

- **Theorem 5.1.** "*Fix integers p ≥ 2 and c ∈ N*. Consider the tokenizer T_c defined in Eq. (1) for processing the input and output sequences. There exists a logarithmic-precision Transformer with constant depth and hidden dimension (independent of n) that can generate the correct output for any input on the ADD_p(n) task.*"

- **Theorem 5.2.** "*Fix integers p ≥ 2 and c ∈ N*. Consider the tokenizer T_c defined in Eq. (1) for processing the input and output sequences. For any integers n and k, there exists a logarithmic-precision Transformer with constant depth and hidden dimension d (independent of n and k) that can generate the correct output for any input on the IterADD_p(n,k) task.*"

- **Theorem 5.3.** "*Fix integers p ≥ 2 and c ∈ N*. Consider the tokenizer T_c defined in Eq. (1) for processing the input and output sequences. For any integers n and l ≤ 2n, there exists a logarithmic-precision Transformer with constant depth (independent of n and k) and hidden dimensions O(n²) that can generate the correct output for any input on the MUL_p(n,l) task.*"

**Scaling/threshold predictions ErrorBoard cares about.** The impossibility proofs reduce IterADD and MUL to Majority, a problem outside AC^0 (per Razborov 1987, Smolensky 1987); constant-precision transformers fall in AC^0 by Li et al. 2024, hence the super-polynomial lower bound. Empirically (paper §6.2), `bfloat16` accuracy on base-2 IterADD with three numbers collapses between digit lengths 7–10, while `float32` stays near-perfect. For base-10 MUL, `bfloat16` accuracy collapses by length-4 multiplicands at 3 layers (Figure 3). LLaMA-3.1-8B Instruct quantized to `int4` shows a comparable collapse versus `bfloat16` on the same tasks (Figure 4).
