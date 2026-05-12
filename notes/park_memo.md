# Park, Park, Hwang (2026) — source-extraction memo

**Citation.** Sejun Park, Yeachan Park, Geonho Hwang. *On the Expressive Power of Floating-Point Transformers.* arXiv:2601.16450v1, 23 Jan 2026. Local: `papers/park-2601.16450.pdf` (main paper p. 1–10, then proofs/appendices). Page numbers below refer to v1.

This is the "baseball stats" extraction: facts, symbols, thresholds, and how they map to ErrorBoard. Prose context lives in `notes/papers.md`. Open the PDF only for proof details (§5 and appendices) not extracted here.

---

## 1. Scope of the paper

- Two questions (p. 1, abstract; p. 2, §1.1):
  1. Are FP transformers permutation-equivariant?
  2. Can FP transformers represent all FP permutation-equivariant sequence-to-sequence functions?
- **Both answers are no.** Real-arithmetic results (Yun et al. 2020a) do not port.
- Three positive/negative results: (a) FP transformers can represent some *non*-perm-equiv functions even without PE; (b) cannot represent all perm-equiv functions when `n` large; (c) *can* represent all perm-equiv functions when `n` small.
- The minimal equivariance every FP transformer has is `π^n_{(1,2)}` (swap-first-two).

---

## 2. Notation glossary

| Symbol         | Definition                                                                      | Page |
|----------------|---------------------------------------------------------------------------------|------|
| `p, q`         | Mantissa width, exponent width (positive integers)                              | 3    |
| `F_{p,q}`      | Set of *finite* FP numbers (normals + subnormals + ±0)                          | 3    |
| `F̄_{p,q}`     | `F_{p,q} ∪ {∞, −∞, NaN}` — single NaN encoding                                  | 4    |
| `e_min`        | `−2^{q−1} + 2`                                                                  | 4    |
| `e_max`        | `2^{q−1} − 1`                                                                   | 4    |
| `ω`            | `2^{e_min − p}` (smallest positive subnormal)                                   | 4    |
| `Ω`            | `(2 − 2^{−p}) × 2^{e_max}` (largest positive normal)                            | 4    |
| `[·]_F`        | RNE rounding `R ∪ {±∞, NaN} → F̄`; ties-to-even                                 | 4    |
| `⌈σ⌋`          | Correctly-rounded ReLU/exp on F̄                                                | 4    |
| `⊕,⊖,⊗,⊘`      | FP add/sub/mul/div on F (extended to F̄ in App. A.1); non-associative           | 4    |
| `⊕_{i=1}^n`    | Left-associative chained add: `x₁ ⊕ x₂ ⊕ … ⊕ xₙ`                                | 5    |
| `ρ`            | `⌈ReLU⌋`                                                                        | 5    |
| `σ`            | FP softmax: `σ(x)_i = ⌈exp⌋(x_i − x⋆) ⊘ ⊕_j ⌈exp⌋(x_j − x⋆)`, `x⋆ = max(x)`     | 5    |
| `Δ_n`          | `{[x₁,…,xₙ] ∈ F^{d×n} : xᵢ ≠ xⱼ ∀ i ≠ j}` — distinct-column inputs              | 6    |
| `π^n_{(1,2)}`  | Permutation swapping coordinates 1 and 2; identity on 3..n                      | 6    |
| `(α,β)`-similar | See §5 below — pair X,Y differing only at column α                             | 7    |

**Condition 1** (assumed throughout): `2 ≤ p ≤ 2^{q−1} − 3` (p. 4).

| Format   | p  | q  | Cond 1 holds? |
|----------|---:|---:|:--------------|
| FP8 E4M3 |  3 |  4 | 2 ≤ 3 ≤ 5  ✓  |
| FP8 E5M2 |  2 |  5 | 2 ≤ 2 ≤ 13 ✓  |
| FP16     | 10 |  5 | 2 ≤ 10 ≤ 13 ✓ |
| bfloat16 |  7 |  8 | 2 ≤ 7 ≤ 125 ✓ |
| FP32     | 23 |  8 | 2 ≤ 23 ≤ 125 ✓|

---

## 3. FP model fine-print

Paper's `F̄_{p,q}` (p. 3–4):

- `p + q + 1` bits total. Two of the `2^q` exponent codes are reserved for ±∞ and NaN (leaving `2^q − 2` "real" exponent codes).
- Includes both ±∞ and a single NaN encoding.
- `[·]_F` rounds out-of-range values to ±∞ (`x ≥ Ω + 2^{e_max−p−1}` → ∞).
- ⊕/⊗ on F̄ defined in App. A.1; on F they're standard IEEE.
- Chained adds and matmuls evaluate **left-to-right** (Eq. 5–6, p. 5).
- Matmul: `(M⊗N)_{ij} = ⊕_k (M_{ik} ⊗ N_{kj})` with FP ⊕.

**Transformer architecture (p. 5–6):**

```
FFN(X) = X ⊕ (W₂ ⊗ ρ(W₁⊗X ⊕ b₁1ₙᵀ) ⊕ b₂1ₙᵀ)             (residual FFN)
AT(X)  = X ⊕ ⊕ᵢ Wᵢᴼ ⊗ (Vᵢ ⊗ σ(Kᵢᵀ ⊗ Qᵢ))                (residual multi-head attn)
f(X)   = W_out ⊗ g(W_in ⊗ X ⊕ b_in 1ₙᵀ) ⊕ b_out 1ₙᵀ      (encoder, g = block stack)
```

NaN propagation in softmax: `x⋆ := NaN` if any `xᵢ = NaN` (footnote, p. 5).

### ⚠ Divergence from OCP E4M3 (Micikevicius 2022) — load-bearing

Park's `F̄_{p,q}` with `(p=3, q=4)`:

- `e_max = 7` → `Ω = (2 − 2⁻³) × 2⁷ = 240`
- Has both ±∞ and a single NaN
- `2^q − 2 = 14` real exponent codes

OCP E4M3 (our oracle, `setup.md`, `oracle.py`):

- `e_max = 8` → max = **448** (extra binade; all-ones exponent repurposed for normals)
- **No ±∞.** Saturate-on-overflow.
- Two NaN encodings (`0x7F`, `0xFF`)
- 15 real exponent codes (only one code reserved, for NaN)

**Implications for citing Park's results in ErrorBoard:**

| Result   | Depends on overflow/∞? | Ports to OCP E4M3? |
|----------|------------------------|---------------------|
| Thm 1    | No (round-off in attn) | ✓                   |
| Thm 2    | No (round-off in ⊕)    | ✓ (threshold unchanged) |
| Thm 3    | No (lookup construction over finite F̄) | ✓ (with F instead of F̄) |
| Thm 4    | No (left-associativity only) | ✓                   |
| Thm 5    | No                     | ✓                   |
| Lemma 6  | No                     | ✓                   |
| App. A.1 | Defines ⊕/⊗ on ±∞     | Does NOT port — flag if cited |

The `n ≤ 6·2^p − 2` and `n ≥ 9·2^p` thresholds depend only on `p`, not on overflow behavior. So **the numbers are unchanged for OCP E4M3.**

Already documented in `setup.md` §3: 436 / 65,536 ordered FP8 pairs disagree between torch (NaN-on-overflow) and OCP (saturate). All 436 are the same overflow case.

---

## 4. Main theorems (Section 3) — statement-only quotes

All theorems take `d_in, d_out, n ∈ N`. Each is paraphrased; the verbatim hypothesis/conclusion appears in `notes/papers.md`.

### Theorem 1 — representing non-perm-equiv functions on distinct inputs (p. 6)

For `n ≥ 2`: for any `π^n_{(1,2)}`-equivariant `f* : Δ_n → F̄^{d_out × n}`, there exists an FP transformer `f` with `f = f*` on Δ_n.

**Proof mechanism** (sketch, p. 7): exploits non-associativity. With exact arithmetic, `π(VΣ) = (πV)Σ^π`. With FP, `π(V⊗Σ) ≠ (πV)⊗Σ^π` in general. So the attention layer `Vᵢ ⊗ σ(Kᵢᵀ⊗Qᵢ)` can be made non-perm-equiv. Full construction in §5.1 (uses Lemmas 7-9, attention as triplet-max sort over `(F̄^{d_in})³`).

### Theorem 2 — impossibility class for large `n` (p. 8)

For `α ≥ 3·2^p`, `β − α ≥ 6·2^p`, `n ≥ β`: for any `(α,β)`-similar `X, Y ∈ F̄^{d_in × n}` and any FP transformer `f`, the outputs `f(X), f(Y)` are also `(α,β)`-similar.

**Consequence:** FP transformers cannot represent all perm-equiv FP functions once `n ≥ 9·2^p` (the minimum `β` satisfying the hypothesis).

**Proof mechanism** (sketch, p. 8): Thms 14, 15 in appendix prove `⊕_{i=1}^α a ⊕ ⊕_{i=α+1}^β b = ⊕_{i=1}^{α−1} a ⊕ ⊕_{i=α}^β b` for large `α, β`. I.e., once the running sum has absorbed enough copies, adding one more (or one fewer) leaves it RNE-identical. Then any FP transformer applied to `(α,β)`-similar inputs produces `(α,β)`-similar outputs at every layer. Full proof §5.3.

### Theorem 3 — full representability for small `n` (p. 8)

For `n ≤ 6·2^p − 2`: for any permutation-equivariant `f* : F̄^{d_in × n} → F̄^{d_out × n}`, there exists an FP transformer `f` with `f = f*` on all of `F̄^{d_in × n}`.

**Proof mechanism** (p. 9–10, §5.2): construct `f = W_out ⊗ (ψ ∘ φ(W_in⊗X))`. Lemma 7 says `f*` can be written as `[f̃(x_i, X)]_i` for some `f̃`, since `f*` is determined by knowing each column and the *multiset* of columns. Lemma 8: knowing `max{π(i₁),π(i₂),π(i₃)}` for all `1 ≤ i₁ < i₂ < i₃ ≤ n` determines `π` up to `π^n_{(1,2)}`. The block `φ` collects this triplet-max information into each column; `ψ` then computes the per-column output via Thm 1.

### Theorem 4 — minimal equivariance (p. 8)

For `n ≥ 2`: every FP transformer `f : F̄^{d_in × n} → F̄^{d_out × n}` is `π^n_{(1,2)}`-equivariant.

**Proof** (one-line, p. 8): `⊕_{i=1}^n xᵢ = (⋯((x₁⊕x₂)⊕x₃)⊕⋯⊕xₙ) = ⊕_{i=1}^n x_{π^n_{(1,2)}(i)}`. Left-associative chained add is invariant under swap of the first two operands. All FFN/AT ops are built from such chains.

§4.1 also notes: for any non-trivial `π ≠ π^n_{(1,2)}`, Thm 1 gives an FP transformer that is *not* `π`-equivariant — so `π^n_{(1,2)}` is the unique minimal equivariance.

### Theorem 5 — equality preservation (p. 9)

"FP transformers preserve equality." Formally (Def. 2): if `xᵢ = xⱼ` in the input, then `yᵢ = yⱼ` in the output. Proof: §C.2.

### Lemma 6 — additive PE is not injective (p. 9)

For any `z ∈ F̄ \ {0}`, the map `x ↦ x ⊕ z` is not injective on F̄. Hence additive PE `[x₁+p₁, …, xₙ+pₙ]` is not injective unless all `pᵢ = 0`. **Strictly worse than real transformers**, where additive PE strictly improves expressivity.

§4.2 workaround: concat a position channel. Use `[Xᵀ, (1,…,n)ᵀ]ᵀ` so columns become distinct; then Thm 1 applies and all functions are representable on the new input.

---

## 5. Definition 1 — `(α,β)`-similarity (p. 7)

Let `d, n, α, β ∈ N` with `α < β ≤ n`. `X, Y ∈ F̄^{d×n}` are **(α,β)-similar** if there exist `z₁, z₂ ∈ F̄^d` such that:

```
        col:  1     2    ...  α-1   α    α+1  ...  β    β+1  ...  n
X      =     [z₁   z₁  …  z₁   z₁  z₂  …  z₂   x_{β+1}  …  x_n]
Y      =     [z₁   z₁  …  z₁   z₂  z₂  …  z₂   y_{β+1}  …  y_n]
```

with `x_j = y_j` for all `j ∈ {β+1, …, n}`.

X and Y differ **only at column α** — X has one more `z₁` and one fewer `z₂` than Y, and the trailing `n − β` columns match exactly.

Thm 2 says: starting from such an X, Y pair, the outputs remain `(α,β)`-similar — i.e., the transformer cannot un-confuse them.

---

## 6. Threshold table — paper's `n`-bounds across FP formats

| Format    | p  | 2^p   | Thm 3: `n ≤ 6·2^p − 2`         | Thm 2: `n ≥ 9·2^p`             |
|-----------|---:|------:|--------------------------------:|--------------------------------:|
| FP8 E5M2  |  2 |     4 | 22                              | 36                              |
| **FP8 E4M3** |  **3** | **8** | **46**                       | **72**                          |
| FP16      | 10 |  1024 | 6142                            | 9216                            |
| bfloat16  |  7 |   128 | 766                             | 1152                            |
| FP32      | 23 | 2²³   | ≈ 5.03 × 10⁷                    | ≈ 7.55 × 10⁷                    |

Park comment, p. 8 (just after Thm 2): "9 × 2^p can be small, especially for low-precision formats: e.g., 8-bit formats E5M2 (p = 2), E4M3 (p = 3)."

---

## 7. Mapping to ErrorBoard's setup

Park's `n` = number of FP-vector columns in the input matrix `X ∈ F̄^{d_in × n}`.

In ErrorBoard's binary-addition mouse:

- Input is **two operands** `(a, b)`. Reading our task as Park-style FP input, `n = 2` — far below both thresholds.
- Our actual model sees a 28-token integer sequence (bit-level tokenizer, `SEQ_LEN = 28`, vocab `[0..11]`), not an `F̄^{d×n}` matrix. Park's formal model does not directly apply to the bit-tokenized input.
- The `n ≤ 46` and `n ≥ 72` thresholds **are not binding for binary FP8 addition at our scale.** They become live when/if we add multi-operand `IterADD` (Feng's regime).

**What we still cite Park for, in this project:**

- **Thm 4** (`π^n_{(1,2)}`-equivariance is automatic): trivially satisfied — `a ⊕ b = b ⊕ a` for FP8 RNE addition, and our oracle confirms it.
- **Thm 5** (equality preservation): non-trivial probe target. In our task, the `a = b` (doubling) regime feeds identical operand columns. Thm 5 says any FP transformer must produce identical column representations at those two positions throughout the network. We can verify this in HookedTransformer caches as a sanity check on the learned representation.
- **Lemma 6** (additive PE not injective on F̄): warns against any future variant that injects FP values directly into the residual stream with additive PE. We currently embed integer token IDs, so it doesn't bite — but record this if we ever change tokenization.
- **Thms 2, 3**: cite as the long-`n` story when discussing why our scale-up plan (if it includes IterADD) needs to stay under `n = 72`.

---

## 8. Open questions / flags

- **(α,β)-similarity over F vs F̄**: Def. 1 is stated over `F̄^{d×n}` (includes ±∞, NaN). Does the impossibility (Thm 2) survive restricting inputs to `F` (no ±∞), as in OCP E4M3? Proof appeals to round-off in chained ⊕ on finite values, so likely yes — but unverified in the appendix. Flag if we ever cite the precise statement.
- **App. A.1** has the ⊕/⊗ extension rules to F̄ (NaN propagation, ∞ arithmetic). Not extracted here; consult if we need to argue about NaN-input behavior in the model.
- **§5.1–5.3** are the proofs of Thms 1–3 with Lemmas 7–9 (sort-via-attention construction). Not extracted; consult if we want to *imitate* their construction in an analysis (e.g., "is the model doing something like triplet-max?").
- **Thm 5 as probe target**: at the `a = b` doubling subset of our task, every layer should produce identical activations at the `a` and `b` operand positions. If our trained transformer violates this, it would either mean (a) we've broken Park's framing somewhere (PE adds a position channel that breaks column-wise equality), or (b) implementation isn't a "Park-style FP transformer" (which is true — we use BF16/FP32 training, not FP8 weights). Worth checking once the HookedTransformer bridge lands.
