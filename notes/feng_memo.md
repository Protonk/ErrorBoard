# Feng et al. — source-extraction memo

**Citation.** Guhao Feng, Kai Yang, Yuntian Gu, Xinyue Ai, Shengjie Luo, Jiacheng Sun, Di He, Zhenguo Li, Liwei Wang. *How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs.* arXiv:2410.13857v2, 21 Jun 2025. Local: `papers/feng-2410.13857.pdf` (10 pages, then references and appendices not extracted).

This is the lightest of the four extraction memos. Feng's results are theoretically clean but **pragmatically less load-bearing for the mouse** than Park (which targets FP arithmetic directly) or BitTokens/Zhou (which give us implementation paths). I keep the parts that scale-set our expectations (ADD at our precision should work; IterADD/MUL won't) and trim the proof-mechanism deep-dive.

---

## 1. Scope and central claim

- **Numerical precision is the controlling factor for transformer arithmetic capacity.** (p. 1, abstract)
- Constant-precision (e.g., int4, int8, fp8) transformers of constant depth require **super-polynomial** hidden dimension to solve IterADD or MUL.
- Standard / logarithmic-precision transformers solve all three elementary tasks (ADD, IterADD, MUL) with constant or polynomial size.
- Empirically: bf16 collapses at digit length 7–10 on base-2 IterADD; bf16 fails on base-10 MUL at length-4 multiplicands with 3 layers; int4-quantized LLaMA-3.1-8B Instruct shows a comparable collapse vs bf16.

### Summary table (Table 1, p. 2)

| Task                          | Standard precision | Low precision    |
|-------------------------------|--------------------|------------------|
| Integer Addition `ADD_p(n)`   | Constant           | `O(n²)`          |
| Iterated Addition `IterADD_p(n,k)` | Constant       | **Super-polynomial** |
| Integer Multiplication `MUL_p(n,l)` | `O(n²)`        | **Super-polynomial** |

---

## 2. Setup (§3, p. 3)

### Integer representations

- All integers are non-negative, base-`p` for fixed `p ≥ 2`. An `n`-digit integer is `(x_{n−1} … x_0)_p`.
- **Three tasks:**
  - `ADD_p(n)` = `a + b`, with `a, b` each ≤ `n` digits in base `p`.
  - `IterADD_p(n, k)` = `a_1 + a_2 + … + a_k`, each ≤ `n` digits.
  - `MUL_p(n, l)` = `a × b` truncated to `l` digits (`l ≤ 2n`). Standard multiplication is the `l = n_1 + n_2` case (Remark 3.1).

### Tokenizer T_c (Eq. 1, p. 3)

A **multi-digit tokenizer**: partitions the integer into tokens, each containing at most `c` contiguous digits. For `x = (x_{n−1} … x_0)_p` and `k = ⌈n/c⌉`:

```
T_c(x) = [t_{k−1}, …, t_0]
```

where `t_i = [x_{ic}, x_{ic+1}, …, x_{ic+c−1}]` for `i < k−1`, and `t_{k−1} = [x_{(k−1)c}, …, x_{n−1}]`.

**Key contrast with prior work:** Feng et al. 2023 and Yang et al. 2024 treated each integer as a *single* token. Feng's setup tokenizes into chunks ≤ `c` digits, matching real LLMs. The results are stated for any fixed `c ≥ 1`.

### Constant- vs log- vs standard-precision (p. 4–5)

- **Constant-precision Transformer:** internal neurons hold `c` bits each, `c` independent of sequence length. Matches `int4`, `int8`, `float8` deployments. Formal def in App. B.2.
- **Logarithmic-precision Transformer:** neurons can represent `O(log n)` bits, `n` = max sequence length. Paper argues `32 bits ↔ 100,000 context length` makes this the natural model of practical LLMs. Formal def in App. B.
- "Standard precision" in the paper is the log-precision model. Real `float32` falls here.

---

## 3. Six theorems (statement-only)

All theorems fix `p ≥ 2`, `c, L ∈ N*`, tokenizer `T_c` from Eq. (1).

**Theorem 4.1 — ADD at constant precision (p. 4).** There exist **constant-precision** Transformers with constant depth (independent of `n`) and hidden dimension `d = O(n²)` that solve `ADD_p(n)`.

**Theorem 4.2 — IterADD impossibility (p. 4).** For any polynomial `f`, there exist `n, k` such that no constant-precision autoregressive Transformer with `L` layers and hidden dim `d < f(n, k)` can solve `IterADD_p(n, k)`.

**Theorem 4.3 — MUL impossibility (p. 4–5).** Same form: for any polynomial `f`, exist `n, l` such that no constant-precision Transformer with `d < f(n, l)` can solve `MUL_p(n, l)`.

**Theorem 5.1 — ADD at log precision (p. 5).** Logarithmic-precision Transformer with constant depth and **hidden dim independent of `n`** solves `ADD_p(n)`.

**Theorem 5.2 — IterADD at log precision (p. 5).** Same: constant depth, hidden dim independent of `n, k`.

**Theorem 5.3 — MUL at log precision (p. 5).** Constant depth, hidden dim `O(n²)`.

---

## 4. Why the lower bounds — circuit-complexity sketch (p. 5)

- Constant-precision Transformers with polynomial size and bounded depth ⊆ **AC⁰** (Li et al. 2024).
- IterADD and MUL reduce to **Majority**, which is outside AC⁰ (Razborov 1987, Smolensky 1987).
- ⇒ super-polynomial hidden dim required at constant precision.

This is the proof mechanism for Thms 4.2 and 4.3. I record it here in one line because the AC⁰ framing is useful context but not load-bearing for our task design.

---

## 5. Empirical results (§6, p. 6–8)

### Setup

- Custom Transformer trained from scratch (not LLaMA). GeLU activations.
- Batch 512, 100k steps, 51.2M training samples. Cross-entropy loss on answer tokens. Exact-match accuracy on 50k held-out test samples.
- Architectures: 3-layer and 5-layer.
- Weight precisions tested: `float32` (the "standard" arm) vs `bfloat16` (the "low" arm).
- Bases: `p ∈ {2, 10}`.

### Results — extracted facts (no plot reproduction)

**ADD (p. 6).** Both `bfloat16` and `float32` hold above 94% accuracy out to **digit length 32** in both base-2 and base-10. Predicted-to-be-easy, observed-to-be-easy.

**IterADD with 3 numbers (Figure 2, p. 7).**

| Base | Digit length where bf16 collapses | float32 status         |
|------|-----------------------------------|------------------------|
| 2    | **7–10** (drops from ~100% to near 0%) | Stays near-perfect through 11 |
| 10   | **3–4** (3-layer drops to ~0% at length 4; 5-layer holds ~50%) | Stays near-perfect       |

**MUL (Figure 2, p. 7; Figure 3).**

| Base | Digit length where bf16 collapses | float32 status                                       |
|------|-----------------------------------|------------------------------------------------------|
| 2    | **~9** (3-layer drops sharply; 5-layer holds longer) | Degrades by length 13 (3-layer ~30%, 5-layer ~60%) |
| 10   | **3–4** (collapses to 0% by length 5) | 5-layer still has partial accuracy at length 4 |

So even `float32` degrades on base-10 multiplication at length 5 — Thm 5.3 requires `O(n²)` hidden dim, and they did not scale `d` with `n`.

### LLaMA-3.1-8B Instruct, base-10 tasks (Figure 4, p. 8)

Four conditions:

- `bfloat16` original
- `int4` (AWQ-quantized)
- `bfloat16 + LoRA` fine-tuned
- `int4 + QLoRA` fine-tuned

Findings:

- For all four tasks (ADD, 3-num IterADD, 5-num IterADD, MUL), **int4 underperforms bf16, and QLoRA underperforms LoRA**, with the gap widening as digit length grows.
- For 3-num IterADD, dropping precision costs ~20% accuracy in some cases.
- **Fine-tuning a low-precision model with QLoRA does not necessarily surpass the original bf16 model** — i.e., post-hoc quantization-aware tuning doesn't recover full-precision arithmetic capacity.

---

## 6. Mapping to ErrorBoard

### Where Feng's results bind on us

Our task is **FP8 E4M3 addition with two 8-bit operands**. Feng's framework asks "what hidden dim does a constant-precision Transformer need to learn ADD?" Mapping each piece:

| Feng's parameter             | ErrorBoard value                                         |
|------------------------------|----------------------------------------------------------|
| Base `p`                     | 2 (we represent FP8 bit-by-bit)                          |
| Digit length `n`             | 8 (each FP8 operand is 8 bits)                           |
| Tokenizer chunk size `c`     | 1 (Option A: one token per bit)                          |
| Task                         | Effectively `ADD_2(8)` plus FP8 rounding semantics       |
| Model precision (Feng's "neuron precision") | bfloat16/float32 weights in our nanoGPT-style backbone |

By Thm 4.1, `d = O(n²) = O(64)` suffices at constant precision. **Our sweep results corroborate this:** the smallest cell that reliably crosses 99% on default is `L2-E128` at 400k params with `d = 128`. The full borderline ladder starts at `E ≈ 32–48` (Round 2 multi-seed). So we're sitting comfortably above Thm 4.1's `O(n²)` envelope.

By Thm 5.1, **at standard precision (our actual training regime, bf16/fp32), hidden dim is independent of `n`** — i.e., scaling down `E` should still keep ADD learnable. Round 2 confirms: even `L1-E016` (~4k params) reaches 74% default accuracy. Not perfect, but the task isn't a hard wall.

### Where Feng's results predict failure if we scale

- **IterADD extension.** If we ever do `a + b + c + …` (multiple FP8 operands), Thms 4.2/5.2 predict an empirical wall once `n` grows: constant-precision arithmetic doesn't scale.
- **MUL extension.** `MUL_2(8)` is at the edge of where bf16 collapsed in Feng's experiments (their `n ≤ 14` regime). We'd expect FP8 multiplication of two FP8 numbers to be solvable at modest hidden dim but already harder than addition.

### One reframing worth pinning

Feng's "constant precision" is about **model precision** (the neurons themselves are quantized). Our setup trains in **bfloat16 / float32 weights** but the *task data type* is FP8. We're not in Feng's constant-precision regime — we're in his standard-precision regime, but the model has to learn to simulate FP8 rounding semantics from data.

This is **why Park is more directly applicable than Feng** for our setup: Park asks about expressivity *given* FP arithmetic in the model; Feng asks about precision constraints *on the model's internal computations*. Park binds even at our scale; Feng mostly tells us we shouldn't have to worry until we add IterADD / MUL.

---

## 7. Why this is a thin memo — trimmed sections

For reference, what I deliberately did not extract:

- **The detailed proof constructions** (Apps. D, E). The AC⁰ ↔ Majority reduction is summarized in §4 as a one-line mechanism; the actual gadget constructions are not load-bearing for ErrorBoard.
- **The full plots from Figures 2/3/4.** I extracted collapse-point digit-lengths and qualitative trends; reading exact accuracy values off the plots adds nothing.
- **Their LLaMA fine-tuning hyperparameters** (Table 6, App. F.4). Not aligned with our setup.
- **References to Razborov 1987, Smolensky 1987, Li et al. 2024.** Named for cross-reference; not summarized.

---

## 8. Open questions / flags

- **Feng's bf16 collapses at digit-7 for IterADD base-2; FP8 should collapse earlier.** We don't have a direct Feng-style experiment for FP8 in their paper. If we add IterADD, the FP8 wall should sit at lower `n` than bf16's wall. Worth a side-table once we get there.
- **Feng's tokenizer `T_c` is integer-only.** Their setup has no notion of sign, decimal point, or special values. Our task spec already handles these (sign bit, subnormals, NaN). When we cite "Thm 4.1 says ADD is OK at constant precision," that's strictly speaking only for the integer-bit-string portion of our task; the special-value regime isn't covered.
- **The Park/Feng overlap.** Park's `n ≤ 6·2^p − 2` and Feng's `d = O(n²)` are different framings of the same domain (low-precision FP transformers + arithmetic). They are *not* mutually redundant: Park bounds which functions are representable in principle; Feng bounds what hidden dim is needed for specific tasks. For ErrorBoard, Park is the binding constraint at our scale (we have `n = 2` operands, below Park's `n = 72` threshold); Feng's `O(n²)` for ADD is a much looser sufficient condition.
- **The "fine-tune doesn't recover" finding (Figure 4).** Suggests that a model trained at bf16 cannot be cheaply ported down to int4 — implies our future "could we deploy this in fp8 weights?" question is not free. Not relevant to the mouse; flag for future deployment discussions.
