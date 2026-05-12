# Zhou, Fu, Soltanolkotabi, Jia, Sharan — FoNE source-extraction memo

**Citation.** Tianyi Zhou, Deqing Fu, Mahdi Soltanolkotabi, Robin Jia, Vatsal Sharan. *FoNE: Precise Single-Token Number Embeddings via Fourier Features.* arXiv:2502.09741v2, 21 Apr 2026 (ICLR 2026). Local: `papers/zhou-2502.09741.pdf` (10 main pages + appendices). Page numbers below refer to v2.

Reference code: `code/fone-ref/` (clone of `github.com/KevinZhoutianyi/FoNE`), gitignored.

This memo is the second view on FoNE — the first was indirect, via BitTokens §4. **The two papers disagree empirically about whether FoNE can do multiplication.** That disagreement is the most important finding here.

---

## 1. Scope of the paper

- Single-token number embedding via per-digit sinusoidal features.
- Trained from scratch in a 38M-param Llama-style transformer; outperforms fine-tuned Llama-3.2-1B on add/sub/mult (p. 1, abstract).
- **Only method to hit 100% on 6-digit decimal add, 6-digit integer add, 5-digit subtraction, and 3-digit multiplication** in their evaluation suite (p. 7).
- 6-digit decimal addition: 99% accuracy at 6,400 samples — **64× less data than digit-wise / subword** (p. 7).
- Companion: FoNE+Abacus combination strictly improves over Abacus alone (Figure 4(a)); 60-digit add via chunking achieves 97.42% avg (Figure 4(b)).

---

## 2. Notation glossary

| Symbol               | Definition                                                                | Page |
|----------------------|---------------------------------------------------------------------------|------|
| `φ(x, T)`            | Circular embedding: `(cos(2πx/T), sin(2πx/T)) ∈ R²`                       | 3    |
| `T_i`                | `10^i` — period for digit position `i`                                    | 3    |
| `m`                  | Integer digit length (e.g., for "1234.56", `m = 4`)                       | 3    |
| `n`                  | Decimal digit length (e.g., for "1234.56", `n = 2`)                       | 3    |
| `FoNE(x, m, n)`      | `[φ(x, T_{−n+1}); φ(x, T_{−n+2}); …; φ(x, T_m)] ∈ R^{2(m+n)}`             | 3    |
| `[NUM]`              | Special vocab token; all numbers map to it pre-embedding                  | 5    |
| `{φ(0,10),…,φ(9,10)}`| 10 unit-circle prototypes used for per-digit classification               | 5    |
| `W ∈ R^{d×2(m+n)}`   | Optional learnable projection from FoNE space to model embed dim          | 3    |

---

## 3. §3.1 — the encoding (Def 3.1, 3.2)

**Def 3.1 — Circular embedding (p. 3):**

> `φ(x, T) := (cos(2πx/T), sin(2πx/T))`

**Def 3.2 — Fourier Number Embedding (p. 3):**

> `FoNE(x, m, n) := [φ(x, T_{−n+1}); φ(x, T_{−n+2}); …; φ(x, T_m)]`
>
> where `T_i = 10^i` for `i` ranging `−n+1` to `m`. Output dim is `2(m+n)`.

Two ways to align FoNE's `2(m+n)`-dim output with model embed dim `d`:
- **Learnable linear projection** `W ∈ R^{d × 2(m+n)}`, or
- **Zero-padding** to length `d`.

Both reach comparable accuracy (§4.3, Table 2; on decimal/integer addition both give 100%; on multiplication, 99.95% vs 99.91%). So the projection choice is not load-bearing.

### Why multiple periods, not one big `T` (p. 4)

If FoNE used a single large `T = 10^M` (long enough to uniquely encode `x`), then `(cos(2π(x+1)/T), sin(2π(x+1)/T))` would be **arbitrarily close to** `(cos(2πx/T), sin(2πx/T))` — finite-precision floats cannot distinguish adjacent integer values on the unit circle. Solution: stack multiple `T`s spanning many decades, so the smallest-period dimension distinguishes 1-apart values while the largest-period dimension distinguishes magnitude.

This is the key porting consideration for FP8: we need a *spread* of periods, not one fitting FP8's whole range.

### Worked example (Example 3.5, p. 4)

For `x = 4.17`: `FoNE(4.17, 1, 1) = [φ(4.17, 0.1); φ(4.17, 1); φ(4.17, 10)]`. By Lemma 3.3 (below), this recovers `[4.17 mod 0.1, 4.17 mod 1, 4.17 mod 10] = [0.07, 0.17, 4.17]` — one digit each at the tenths, units, and tens positions.

---

## 4. §3.2 — representational guarantee (Lemma 3.3, 3.4)

**Lemma 3.3 (informal, p. 3):** Given `(cos(2πx/T), sin(2πx/T))`, we can recover `x mod T`.

(Formal version is Lemma D.1 in appendix; not extracted.)

**Lemma 3.4 (FoNE preserves numeracy, p. 4):** Given `FoNE(x, m, n)`, we can recover `x mod 10^i` for each `i` in `−n+1` to `m`. Together, these mod-residues determine `x` exactly within the digit range.

Proof structure: apply Lemma 3.3 component-wise across the period list `{10^{−n+1}, …, 10^m}`.

---

## 5. §3.3 — decoding head and loss (Def 3.6, 3.7)

**Def 3.6 — FoNE loss (p. 5):** Let `h ∈ R^d` be the last-layer hidden state, `y_i` the `i`-th digit of the label.

> `L_FoNE(h, y, i) := L_CE(y_i,  ([h[2i], h[2i+1]]_{1×2})  ·  ([φ(0,10); …; φ(9,10)]_{2×10})^⊤)`

Per-digit, the loss treats positions `(h[2i], h[2i+1])` of the hidden state as a 2-vector and dots it against each of the 10 unit-circle prototypes `φ(0,10), …, φ(9,10)`. Then standard cross-entropy against the digit label. Final loss is the average over digit positions `i`.

**Important fine-print:** The comparison set `{φ(0,10), …, φ(9,10)}` is the **same for every digit position `i`** — i.e., every digit is decoded against the same `mod 10` ring, regardless of magnitude. The mapping between `(h[2i], h[2i+1])` and the actual digit-`i` Fourier component is positional, not by period.

**Def 3.7 — prediction (p. 5):**

> `ŷ_i := argmax_{j ∈ {0..9}} ([h[2i], h[2i+1]] · φ(j, 10)^⊤)`

Final number reconstructed by concatenating digit-wise predictions with their place values.

**Why classification, not regression?** (Q1, p. 9): "Regression produces continuous values, making it impossible to integrate number-related tasks with general language modeling. For example, when predicting the year '1997', regression may output '1996.9999', which is acceptable under regression metrics but unusable in sequence generation." Keeping classification loss preserves drop-in compatibility with vocab-prediction LM heads.

---

## 6. §3.4 — integration procedure (p. 5)

7-step recipe:

1. Parse all numbers from the input string → replace each with `[NUM]` token + canonical numeric value.
2. Tokenize the modified sequence normally.
3. Embed token list with standard word-embedding lookup.
4. For each `[NUM]` token, compute its FoNE vector.
5. **Add** the FoNE vector to the `[NUM]` token's word embedding (so `[NUM]` is the same vocab item everywhere, but its post-embedding value carries the number).
6. Feed combined embeddings to transformer.
7. At output: if predicted token is `[NUM]`, decode the hidden state via Def 3.7; otherwise predict next token normally.

---

## 7. §4 — empirical results

### Per-task accuracy (Table 1, p. 7, single-task, 38M params)

| Method     | Decimal Add | Subtraction | 3-digit Mult | Tokens per number |
|------------|------------:|------------:|-------------:|------------------:|
| **FoNE**   | **100%**    | **100%**    | **98.56%**   | **1**             |
| Digit-wise |     99.85%  |     99.71%  |     81.21%   |    7              |
| Subword    |     97.94%  |     91.66%  |      8.05%   |    3              |
| xVAL       |      0.44%  |      3.41%  |      0%      |    1              |

Also from Figure 3 (p. 6) — same trends on 6-digit integer addition, 5-digit subtraction, 3-digit and 4-digit integer multiplication: FoNE error rate drops cleanly with data + model size; xVAL plateaus near 100% error; subword plateaus high; digit-wise tracks FoNE but needs more data.

### Data efficiency (§4.2, p. 7)

- 6-digit decimal add: 99% at 6,400 samples; **64× less data** than digit-wise/subword (which need 409,600).
- 100% accuracy at 51,200 samples.
- 26.62M-param FoNE surpasses fine-tuned Llama-3.2-1B (~1.2B params).
- 8.31M-param FoNE, 1 layer, reaches 97% at 200k examples.

### §4.3 ablations (p. 7–8)

**Period choice (Table 3):** Periods `{2,5,10}` ≈ `{10}` >> `{5}` or `{7}`. Decimal/integer add still hit 100% with `{5}`, but multiplication collapses to 3.67% (vs 99.95% with `{2,5,10}`). The paper picks `{10}` as default for parameter efficiency; the **reference code defaults to `period_base_list=[2, 5]`** (see `fone-ref/number_encoders/FNE.py:13`) which expands via `_get_period_list` to all `b · 10^i` for `b ∈ {2,5}` across the digit range.

**Linear vs zero-pad (Table 2):** Both achieve ≥99.91% on all tasks. Use whichever fits the architecture.

**Sinusoidal-is-necessary (p. 8, "Necessity of Sine and Cosine Encoding"):** Tried *direct* encoding — each digit `0..9` mapped to a separate one-hot embedding dimension. Even with LR search and 100 epochs, max accuracy on 6-digit decimal add was 99.3%. FoNE hit 100% in 6 epochs with same data + model. Reason: LayerNorm makes near-similar one-hot encodings (e.g., for 999 vs 888) collapse — the "small difference between adjacent digits" vanishes after normalization. Sinusoidal encodings preserve the difference because they sit on different points of a unit circle, not different magnitudes of one-hot dims.

### §4.4 long sequences and complementarity (p. 8)

**Chunking:** `float64` represents only 15 sig figs, so FoNE cannot directly encode larger. For `>15`-digit numbers, chunk into 5-digit groups, FoNE each chunk independently, concatenate. 8-layer transformer trained on chunked input achieves 97.42% on **60-digit addition**, single forward pass (Figure 4b).

**FoNE + Abacus:** Replacing per-digit Abacus position embeddings (McLeish 2024a) with FoNE strictly improves over Abacus-only on extrapolation (train on ≤10-digit, test up to 50-digit; Figure 4a, mean diff +16.04% accuracy). So FoNE composes with other arithmetic-friendly inductive biases.

---

## 8. §5 — Q&A rationale (terse)

- **Q1 (regression vs classification):** see §5 above; classification = LM-compatible.
- **Q2 (why FoNE beats xVAL/DICE/SALSA):** FoNE encodes via **ratios between entries** (cosine/sine pairs at fixed periods) — invariant to LayerNorm/RMSNorm (Lemma D.2). xVAL encodes via *magnitude*, which LN destroys. DICE/SALSA use one unit circle, limiting magnitude range.
- **Q3 (why base 10):** pretrained LLMs already learn `mod 10`-style components (Zhou et al. 2024 interpretability work). And text uses base 10. Other bases (5, 16, etc.) work too; see Table 7 in appendix.
- **Q4 (does FoNE harm semantic ability?):** No. Continual-pretraining Llama-3.1-1B with simplified FoNE on 15B MegaMath-Web-Pro tokens: 4-digit add 51%→59%, 4-digit sub 30%→40%, 5-digit add 29%→36%, MMLU unchanged (38.10% vs 38.21%). From-scratch GPT-2-117M on 10B FineWeb: FoNE achieves *lower* perplexity (46.86) than BPE (67.39), SingleDigit (55.92), xVAL (49.16). On number-containing-only subsequences FoNE is also lowest.

---

## 9. ⚠ Divergence with BitTokens §6 — the empirical question

| Claim source                          | Multiplication result for FoNE                                   |
|---------------------------------------|------------------------------------------------------------------|
| **BitTokens (Kreitner) §6, p. 8**     | "FoNE is able to learn addition, it struggles with multiplication and division as predicted in Section 4." (multi-task nanoGPT-2)             |
| **Zhou (this paper) §4, Table 1**     | **FoNE: 98.56% on 3-digit integer mult** (single-task 38M Llama-style; digit-wise baseline only 81.21%) |
| **Zhou §4, Figure 3(c)/(d)**          | 3-digit and 4-digit integer multiplication solved cleanly with sufficient data + model size |

**The two papers cannot both be straightforwardly right.** Hypotheses:

1. **Multi-task degrades FoNE on mult.** BitTokens trained 7 tasks simultaneously with dynamic rebalancing; Zhou trained single-task. If FoNE's per-digit decoding head competes with other tasks for hidden-state real estate, multi-task could disproportionately hurt it on the hardest task (mult).
2. **Period set matters.** Zhou's Table 3 shows multiplication is highly period-sensitive (`{5}` alone → 3.67% on mult, `{2,5,10}` → 99.95%). BitTokens used "17 integer + 32 fraction frequencies with base `b = 10`" (their §6 setup, p. 8) — that's a different period structure from Zhou's default; could be undertrained for mult.
3. **Task scale differs.** "Multiplication" in BitTokens isn't specified to a digit count; their evaluation in §2 includes products like `4314.97 × 4080000` (large + decimal). Zhou's mult tests are 3-digit and 4-digit *integers*. FoNE may scale poorly to large-decimal mult even if it solves small-integer mult.
4. **BitTokens chose hyperparameters that disadvantage FoNE.** Possible; the BitTokens paper does not publish a per-method LR/hparam search.

**Implication for ErrorBoard:** the FoNE-on-mult question is open. If we ever build a FoNE arm at the FP8-mult stage, we run it head-to-head on the same data/training budget rather than citing either paper.

---

## 10. Mapping to ErrorBoard

### Two ways to apply FoNE to FP8 values

**Option F1 — decimal FoNE on the decoded value.**

- Decode FP8 bits → real value via `oracle.decode`. FP8 range: smallest positive subnormal `2⁻⁹ ≈ 0.002`, largest finite `448` (OCP E4M3).
- This spans 5–6 decimal decades.
- Set `m = 3` (max integer digit ≈ 448), `n = 3` (smallest subnormal needs 3 fractional decimal digits).
- FoNE output dim: `2(m+n) = 12`. Period set: `T_i = 10^i` for `i ∈ [−2, 3]` (6 periods).
- Pros: closest to Zhou's published method; ref code drops in.
- Cons: base-10 mismatch with FP8's binary structure; subnormals < 0.001 will lose precision.

**Option F2 — binary FoNE.**

- Periods `T_i = 2^i` for `i` covering FP8's binade range. Smallest binade `2⁻⁹`, largest binade `2⁸ = 256`, so `i ∈ [−9, 8]`. 18 periods.
- Output dim: `2 × 18 = 36`.
- Decoding head: per-binade 2-way classification (the analog of Zhou's per-digit 10-way) — but this is no longer "Zhou's FoNE." We'd be inventing a base-2 variant.
- Pros: matches FP8's actual structure; bit alignment.
- Cons: untested; effectively becomes "BitTokens-with-sinusoidal-embeddings" rather than FoNE.

**Recommendation if/when we add the arm:** start with Option F1 (faithful FoNE on decoded values). If it underperforms, attempt F2 as an ablation.

### What's interesting to test

- At **FP8 addition (current task):** Lemma 4.2 of BitTokens predicts FoNE works fine. Zhou's Table 1 confirms. So at the mouse stage, FoNE arm should match Option A (our bit-level tokenization). Mostly a sanity check.
- At **FP8 mult (future):** the BitTokens-vs-Zhou disagreement makes this *the* interesting comparison. Either FoNE collapses (vindicating BitTokens' Prop 4.3 prediction at FP8 scale) or doesn't (vindicating Zhou and undercutting Prop 4.3's empirical force).

### Negative-number handling (open in the paper)

Zhou's Def 3.1/3.2 don't address negatives. BitTokens §6 reports they used a `[−]` token prefix for negatives when implementing FoNE for comparison. In the **reference code** (`fone-ref/number_encoders/FNE.py`):

- `forward()` masks zero values (`number_scatter != 0`) — zeros get a zero embedding, not `FoNE(0)`.
- No sign handling visible in `_turn_numbers_to_cosxsinx`; values pass through `.to(dtype=torch.float64)` directly. Negative `x` would just give `cos(2π·x/T) = cos(−2π|x|/T) = cos(2π|x|/T)` (sign-loss for cosine, sign-preserve for sine).
- For our FP8 task: sign needs separate channel. Cleanest: prepend a learned sign token, or add a sign-bit dim to the FoNE vector.

---

## 11. Reference code map (`code/fone-ref/`)

| Path                                       | Purpose                                                   |
|--------------------------------------------|-----------------------------------------------------------|
| `number_encoders/FNE.py`                   | FoNE encoder: forward, loss, prediction. ~250 lines.      |
| `number_encoders/XVAL.py`                  | xVAL baseline.                                            |
| `number_encoders/vanilla.py`               | Standard subword/digit-wise embedding baseline.           |
| `train/train.py`, `train/train_pipeline.py`| Training loop.                                            |
| `train/eval.py`                            | Evaluation.                                               |
| `utils/data_utils.py`                      | Dataset construction (`[operand a][operator][operand b]=`).|
| `main.py`                                  | Entry point with arg parsing.                             |
| `scripts/test.sh`                          | Example run command.                                      |

**Useful internals:**

- `FNE.__init__` (`FNE.py:13`): default `period_base_list=[2, 5]`. Periods constructed via `_get_period_list` which generates `base · 10^i` for `base ∈ period_base_list`, `i` spanning the digit range.
- `FNE.fourier_compute_loss` (`FNE.py:136`): implements Def 3.6 by `matmul(slices, precomputed_cos_sin_matrix)` then `F.cross_entropy`. Pre-computed prototype matrix is stored as a buffer.
- `FNE.fourier_compute_prediction` (`FNE.py:182`): implements Def 3.7 via `argmax` per digit; final number assembled from `predicted_digit * 10^place`.

---

## 12. Open questions / flags

- **BitTokens-vs-Zhou disagreement on mult.** Highest-priority open question for any future contrast-arm work. Whichever of us builds the FoNE arm should test FoNE+nanoGPT-2 multi-task (BitTokens-style) and FoNE+single-task (Zhou-style) on the same mult dataset to localize the source.
- **Per-digit decoding head fan-in.** Each digit reads exactly 2 hidden-state dims (`h[2i], h[2i+1]`). At our embedding sizes (E=32 to E=128 in the sweep), this is a vanishingly thin slice of the residual stream. Worth flagging if FoNE arm underperforms at small embed dim.
- **Reference code masks zero.** `mask = (number_scatter != 0)` in `FNE.fourier_embedding`. Zero-input gets a zero embedding, not `FoNE(0)`. For FP8 this is a structural concern — zero is a representable value, and the `±0` subnormal regime needs to be embedded as something, not all-zeros (which collides with padding). Flag for any port.
- **Lemma 3.3 under FP arithmetic.** Zhou's recovery-of-`x mod T` is in real arithmetic. With FP8 in the encoder's input (or with FP-arithmetic-trained weights, à la Park), the recovery becomes approximate. Magnitude of the approximation error not analyzed by either Zhou or BitTokens.
- **Sign handling.** Zhou's paper doesn't address signed numbers. Ref code doesn't either. Needs a design decision before any FP8 port. Either: (a) prepend `[−]` token (BitTokens-style), (b) add a learned sign channel concatenated to FoNE, (c) FoNE on `|x|` plus a separate sign predictor.
- **Q3 (base choice).** Paper claims bases other than 10 work (Table 7 in appendix, not extracted). Would be worth pulling Table 7 if we go binary-FoNE route — it likely contains exactly the data we'd want.
