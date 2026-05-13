# Future arms — ordered priority list

**Status as of 2026-05-13:** the four addition arms are closed (RoPE, SEM, FoNE F1, FoNE F2). The next experimental thread is **multiplication** — re-running the four-arm cross under FP8 multiply rather than add. See "Arm 5: Multiplication" below for the scoping. The earlier arm sections (RoPE, SEM, FoNE F1, FoNE F2) are preserved as historical context — they were the planning documents for arms now completed, and the findings memos linked in `README.md` are the current authoritative writeup.

The arms are listed in priority order based on what we've learned from the bit-level + learned-PE work in `pentagon_writeup.md`. Each arm tests a specific question that emerged from the current findings; the question is what makes it worth doing now (rather than as a generic ablation).

---

## Why these arms are now valuable

The pentagon writeup characterizes the joint system (architecture + training + format + regime balance) and identifies several findings that depend, in unclear proportion, on choices we made about the model–data interface:

- **Bit-decomposition**: errors at endpoint m_c bins are categorically different from smooth-interior errors (mant-only vs exp-coupled). This pattern emerges from a model that has to *discover* field structure (sign/exp/mantissa) from bit-level tokens. Does it survive when the field structure is given as a prior?
- **44% lottery zone**: a wide band of holdout pairs at non-trivial per-pair failure probability. Is this an artifact of the model having to learn field decomposition, or is it intrinsic to FP8 addition?
- **L=4 minimum depth**: enough composition depth for the multi-stage algorithm. Would a model with field-level priors need less depth?
- **Operand-asymmetry at V5**: depth-1 attention biased toward operand b. Is this an absolute-position artifact (which RoPE will tell us) or also a function of how operands are presented?
- **Severity dual-of-ε**: residual error severity anti-correlates with ε(m), concentrated at binade boundaries. Does this pattern depend on the model emitting bits, or is it format-intrinsic at the value level?

The four arms below test these questions at different layers of the system.

---

## Arm 2: {Sign, Exponent, Mantissa} 3-token tokenization

**What it changes vs current bit-level:** each FP8 number is tokenized as 3 tokens (sign, exponent value 0..15, mantissa value 0..7) instead of 8 bit-tokens. Sequence length per number drops from 8 to 3; vocab grows from 12 to ~30 (BOS, EOS, +, =, 2 sign tokens, 16 exp tokens, 8 mantissa tokens). Architecture, training schedule, learned-PE all stay identical.

### Why this is sharper after RoPE

The RoPE arm established a clean split: **format drives shape (lottery Jaccard 0.76 between PE arms, mant-only/exp-coupled split preserved, anti-ε severity preserved); PE shifts probability (mean fail rate 9.0% → 11.9%, high-prob-fail tail 20 pairs → 120 pairs).** But the test couldn't address a hidden third variable: the model has to *learn the field decomposition* from bit-level tokens. SEM is the clean next cut because it removes that learning task while keeping the FP8 mult-native/add-taxed parlay intact. Two of the three system layers (format + position encoding) hold fixed; only the input-representation granularity changes.

### Three sharpened questions

1. **The 120-pair high-prob-fail tail.** Under RoPE we found 120 pairs with p̂ ∈ [0.8, 0.95] failure probability — pairs that fail almost every seed. Are these hard because *FP8 addition is hard there*, or because *bit-level field discovery + FP8 addition is hard there*? Specific prediction: if SEM's high-prob-fail tail collapses (eg from 120 → fewer than 30 in that bin), the heavy tail was substantially seeded by field-discovery variance. If the tail survives at comparable size, the lottery is value-level FP8 addition difficulty.

2. **The m=0/8 RoPE anomaly.** Bit-decomp PE found one specific PE-flavored regression: at the low endpoint (m=0/8), learned-PE keeps ~69% mant-only errors but RoPE drops to ~25%. That is RoPE losing the "binade-coordination at the bottom" affordance. Under SEM, exp/mantissa coordination at m=0/8 becomes a 2-token decision (which exp token × which mantissa token), not 5-bit coordination. If SEM recovers the mant-only fraction at m=0/8 under either PE, the bit-level coordination cost was the issue. If SEM still loses the mant-only fraction at m=0/8 under RoPE specifically, then there's something position-dependent about binade-boundary handling that survives field-grouping — strongest "PE matters intrinsically" signal we could find.

3. **Anti-ε severity** survived RoPE strongly (ρ = −0.87 at RoPE s8). If it survives SEM too, severity-anti-ε is value-level (ε is doing real work in the output space, independent of bit decoding). If SEM flattens or inverts it, severity was specifically the cost of bit-level rounding-boundary decisions and ε's claim weakens by one rung.

### Probe portability

- **Bit-decomp** doesn't port literally (no bits). The natural analog: at endpoint m_c bins (m=0/8 and m=7/8), do errors flip the *exp token* (analog of exp-coupled) or only the *mantissa token* (analog of mant-only)? If endpoints stay exp-token-heavy under SEM, the categorical split is format-driven. If endpoints become mantissa-only under SEM, the split was bit-level-coordination-driven. This is testable cleanly because m_c is still a single mantissa token's value 0..7.
- **Failure consensus** ports directly (correctness is still a per-pair indicator).
- **Severity** ports directly (ULP and log-damage are value-level metrics).
- **Lottery Jaccard** ports directly — and importantly, computing Jaccard(SEM, bit-level) at the same holdout split tells us whether SEM moved the lottery or just shrank it.
- **Regime stratification** ports directly.

### What it tests (carried over from pre-RoPE writeup)

- **Does the binade-coordination failure mode require multi-bit field discovery?** Pentagon found exp-coupled errors at endpoints. Under SEM, "exp=8 vs exp=7" is a unitary decision, not a 4-bit coordination. If SEM eliminates exp-coupling at endpoints, the binade-coordination failure mode was bit-tokenization-driven.
- **Does the lottery zone shrink?** If field decomposition was a substantial source of seed variability, giving it as a prior should compress the per-pair failure probability distribution.
- **Does the model need less depth?** L=4 was the minimum at bit-level. If SEM saturates at L=2 or L=3, some of that depth was field discovery, not addition.

### Cost

~4-6 hours infrastructure (new tokenizer ~50 lines; output head per-field-classification; encoder embedding tweaks). Then same per-cell training cost as bit-level (~75s). Probe scripts mostly run unchanged; bit-decomp port is the one piece of new analysis code.

### What would update us in either direction

- If SEM dramatically reduces the lottery zone, collapses the high-prob-fail tail, or eliminates exp-coupling at endpoints: the bit-decomp finding from pentagon is *partly* an artifact of bit-level tokenization. The corrected claim becomes "format + bit-tokenization pins shape" rather than "format pins shape." This is a meaningful weakening but not a refutation — RoPE already showed format pins which pairs are hard; SEM would tell us bit-tokenization is upstream of why.
- If SEM produces similar failure shape, lottery distribution, and bit-decomp analog: the failure structure is format-intrinsic regardless of tokenization granularity. The pentagon's claims survive externalization through two independent variations (PE and tokenization).

### Caveats

Could go either way. SEM might also *introduce* new failure modes — a 16-way exp classification is a different problem than predicting 4 bit-tokens; an 8-way mantissa is different than predicting 3 mantissa bits. Intra-field coordination might become harder, not easier. The cleanest interpretation requires running the same holdout split and comparing per-pair correctness, not just per-pair accuracy.

---

## Arm 3: FoNE (Fourier Number Embedding)

### Background framing — this arm tests a different question than the others

The other arms (bit-level, RoPE, {S, E, M}) vary how the model *accesses* FP8's structure but keep the format's own design intact. FoNE goes deeper: it changes *which arithmetic operation is the encoding's local operation*. Recovering the context:

- **FP wasn't designed for arithmetic in general; it was designed for multiplicative scaling.** The exponent field is a power-of-2 ladder; the mantissa is the residual within a rung. Multiplication is structurally easy (exp_a + exp_b, mantissa_a × mantissa_b — stay on the lattice). Addition is the operation the format had to be *retrofitted* to handle: align by Δexp (denormalize the smaller operand), add mantissas, renormalize (re-snap to a rung), round (deal with the leftover).
- **ε is the cost of that retrofitting.** It exists because the affine pseudo-log abstraction (treat bit pattern as integer = E + m) is *almost* but not quite log. Almost-but-not-quite because the format chose to treat the exponent linearly — making multiplication clean and leaving addition with a residual.
- **Two-player race intuition:** two values both doubling stay locked forever — multiplication doesn't separate them. One additive ε perturbation puts them out of phase. Subsequent doublings preserve the log-space offset but scale the linear gap. Getting them back in sync requires precise additive corrections at exactly the right scale. The format's discrete rung structure means most such corrections land between representable values.
- **Subnormals are addition-specific engineering near zero.** Near-zero addition is *especially* taxed because the multiplicative-ladder rungs spread out (in log space) as you approach zero. Subnormals flatten that region into a special low-precision layer to keep addition tractable there. Multiplication doesn't need this — multiplication of subnormals is just multiplication.

Our addition-only ErrorBoard task therefore measures the model's performance on **FP's structurally taxed operation** throughout. Every binade-coordination failure, every endpoint exp-error we found in the pentagon work is happening at the cost-of-retrofitting boundary. The bit-level model's binade-fragility is the model paying for FP's add-tax through a substrate (attention + MLP) that doesn't natively know the lattice structure.

### What FoNE inverts

FoNE picks a basis where **addition is the local operation**: Lemma 4.2 says F(a+b) = F(a) ⊙ F(b) exactly in real arithmetic. The Hadamard product on the Fourier torus is pointwise. The lattice in FoNE is the *Fourier basis itself* — additively closed by construction. Binade boundaries that bit-level crashes at are smooth in FoNE because Fourier features at adjacent periods cover the transition continuously.

The price: BitTokens Prop 4.3 — multiplication in FoNE is non-local (requires accessing Ω(log_P |X|) components per operand). Two values that should multiply to a definite result need explicit convolution-and-carry that no finite local circuit can produce. **FoNE is add-native, mult-taxed — the dual of FP's mult-native, add-taxed.**

Subnormals get absorbed: at base-2 with periods T_i = 2^i covering FP8's range (i ∈ [−9, 8], ~18 components), each binade including the subnormal binades has its own dedicated Fourier component. The engineered work that subnormals were doing — keeping addition tractable near zero — is *built into the encoding* rather than special-cased in the format.

### What this means for our experiment

The other arms test variations of "how does the model navigate the format's mult-native/add-taxed structure?" FoNE tests something categorically different: **is our measured failure shape a property of FP's taxed-operation × substrate interaction, or a property of FP8's discrete representation independent of which operation is native?**

Two outcomes:

- **Binade-fragility disappears in FoNE addition** (smoother error continuum, residuals concentrate at the projection-back-to-discrete-FP8 step rather than at binade boundaries): the bit-level binade-fragility we measured was specifically the cost of doing FP's taxed operation in a representation that respects the mult-native ladder. The shape was about the operation, not the format's discreteness.
- **Binade-fragility persists in FoNE addition** (same anti-ε severity shape, same endpoint-vs-interior split): the binade-coordination cost is intrinsic to FP8's discrete structure regardless of encoding. Even when the encoding is add-native, the *output projection* still has to commit to a discrete FP8 value, and that commitment has rounding-boundary fragility of its own.

This is a much higher-stakes test than "does field-grouping help." It's the test that distinguishes "format-disfavored operation × substrate" from "format-discrete-structure, full stop." Neither bit-level nor RoPE nor {S, E, M} can answer this because they all preserve the mult-native/add-taxed pairing.

### Predicted observable shifts (held lightly)

- The 76 high-probability lottery pairs we found at L4-E044 (20-seed upper-tail) might be specifically the near-tie additive cases where post-addition real-valued result sits near a representable-value midpoint. FoNE's smooth encoding should resolve these *or* push the fragility into the discrete projection step.
- V5's operand-asymmetry should disappear: FoNE has no operand-position structure (each operand is one token).
- The bit-decomp's mant-only vs exp-coupled split should look different because FoNE never emits "exp bits" or "mantissa bits" — it emits a number-token whose decoded value rounds to a representable.

### What it changes vs current bit-level

- Sequence length: 28 → ~5 (BOS, NUM_a, +, NUM_b, =, NUM_c, EOS or similar)
- Vocab: 12 → ~5 (BOS, EOS, +, =, NUM)
- Input embedding: each NUM gets a learned `[NUM]` embedding *plus* its FoNE Fourier-feature stack
- Output prediction: instead of 8 bit-tokens, predict a single NUM token; decode the hidden state at that position via per-digit (or per-binade) cosine-similarity head against prototypes
- Loss: cross-entropy per digit decoded from the Fourier embedding (per Zhou §3.3, Def 3.6)

### What probes need new infrastructure

- **Bit-decomp doesn't apply.** No bits emitted. The natural analog: stratify by which "digit" (or binade) of the result is wrong. Doesn't quite map but a value-space analog exists.
- **Severity probes work in value space.** |pred_real − true_real| and |log₂|pred| − log₂|true|| are well-defined.
- **Failure consensus, regime stratification, attention patterns** all work but at value-level rather than bit-level.

### Cost

~1-2 days infrastructure (FoNE encoder + decoding head + token plumbing — reference impl exists at `code/fone-ref/`). New probe set (~2-3 hours). Per-cell training is comparable.

### What would update us in either direction

- FoNE produces same anti-ε severity shape: the shape is format-intrinsic at the discrete-FP8 level. Strongest version of Finding 6.
- FoNE produces different shape (smoother across binades but failing on projection-step): bit-level's binade-fragility was the cost of FP's taxed operation × bit-substrate, not the cost of FP8's discretization in general.
- FoNE handles subnormals with no special difficulty: confirms subnormals are engineering-overhead for addition specifically, and that overhead can be absorbed into the right encoding.

### Where this stands relative to the other arms

This is the only arm in our planned set that **inverts FP's parlay** rather than just varying how the model accesses FP's native parlay. It's the test of: "did we measure the cost of doing FP's taxed operation, or the cost of FP8's discrete representation?" That question doesn't survive {S, E, M} (which keeps the add-tax) or RoPE (which is purely architectural).

For an agent recovering this context: the key insight is that **our addition-only ErrorBoard task is throughout measuring the format's structurally disfavored operation.** FoNE is the only arm where that's no longer true. Anything we measure in FoNE that *differs* from the bit-level measurements is specifically the contribution of "doing FP's taxed operation" to the residual shape.

---

## Arm 4: cross-axis experiments

After arms 2 and 3 are in place, the natural follow-ups span two or more axes simultaneously:

- **RoPE × {S, E, M}**: do field-level tokenization findings change with relative-position encoding?
- **FoNE × {model precision}**: train FoNE at bf16 vs fp32 weights and see if the format's own ε-shape interacts with the Fourier embedding's smoothness.
- **{S, E, M} × {scratchpad}**: if we tabled the scratchpad arm originally (per methodology.md), the field-level tokenization makes scratchpad steps more naturally interpretable — they could expose intermediate field-level state.

These are not yet scoped in detail. Recorded so we don't lose them.

---

## Priority order

1. ~~RoPE~~ — closed, [`rope_arm_findings.md`](rope_arm_findings.md).
2. ~~{S, E, M} 3-token~~ — closed, [`sem_arm_findings.md`](sem_arm_findings.md).
3. ~~FoNE F1 (decimal)~~ — closed, [`fone_arm_findings.md`](fone_arm_findings.md),
   [`fone_pilot_findings.md`](fone_pilot_findings.md),
   [`fone_transition_memo.md`](fone_transition_memo.md).
4. ~~FoNE F2 (binary)~~ — closed (ablation), [`fone_f2_memo.md`](fone_f2_memo.md).
5. **Multiplication** — next experimental thread, see new section below.
6. Cross-axis experiments — after multiplication.

**Status of the addition arc:** all four input-representation arms tested
at L4-E048 baseline, three of four (RoPE, SEM, FoNE F1) probed with scale
followups. The single-claim summary in the README captures what we
learned. The FoNE F2 ablation closed the loop on "does encoding alignment
with format structure recover anti-ε" — answer: no, it recovers accuracy
instead (precision, not shape recovery).

---

## Arm 5: Multiplication

The addition arc treated FP8's structurally taxed operation. Multiplication
is FP's native operation, and the four-arm cross under multiplication is
predicted to invert nearly every effect:

- **bit / RoPE / SEM** (FP-native): multiplication should be *easier*
  than addition. No binade-coordination; no cancellation; the operation
  decomposes cleanly into `exp_a + exp_b` and `mantissa_a × mantissa_b`,
  with renormalization. Predicted: lottery substantially smaller than the
  addition lottery for the same architecture and seeds.
- **FoNE F1 / F2** (add-native via Fourier embedding): multiplication
  should be *harder*. By BitTokens Prop 4.3, multiplication in the
  Fourier basis is non-local — the product of two values requires
  convolution across all Fourier components. The Hadamard product that
  makes FoNE additively homomorphic does not extend to multiplication.

The interpretive payoff: addition told us "operation alignment matters
when the model has to navigate the binade structure through its encoding."
Multiplication tests the dual claim: "operation misalignment matters
when the model has to construct a non-local operation through a local-
operation-friendly encoding." If FoNE F2 — which essentially solved
addition — collapses on multiplication while bit/SEM thrive, the
asymmetry is symmetric: each encoding has a native operation and a
taxed operation, and the four-arm story is just one half of a two-by-two.

### What changes infrastructurally

- **Oracle (new)**: `oracle_mult.py` implementing FP8 E4M3 multiplication
  per OCP spec. Operation decomposes as: sign XOR; exponent sum; mantissa
  product; renormalize; round-to-nearest-even; handle subnormals,
  overflow, NaN.
- **Regime classifier (new)**: addition's 7 regimes don't all apply.
  Multiplication regimes I'd expect: special-values (NaN, ±0), overflow,
  underflow-to-zero/subnormal-result, exact-product (a×b is exactly
  representable), rounding-tie (the half-even cases), default.
  Cancellation doesn't exist for multiplication; large-Δexp is folded
  into overflow/underflow.
- **Pair table generation**: same shape — 256×256 = 65,536 pairs — but
  with the new oracle's result-bits. The split / sampler / eval loaders
  are encoding-agnostic and don't need to change.
- **Tokenizers and training**: completely shared. The bit / SEM /
  FoNE F1 / FoNE F2 tokenizers all encode (a, b, c) triples without
  caring whether c = a + b or c = a × b. We pass the new pair table
  through the existing pipeline.
- **Probes**: most port directly. Severity (ULP, log-damage) is
  operation-agnostic. Failure consensus, regime stratification, lottery
  analysis all unchanged. Field/digit/bit-decomp ports per arm. Anti-ε
  framing needs reconsideration — ε is the residual of an *affine
  pseudo-log against the true log*, and multiplication interacts with
  log differently than addition does (multiplication is the linear
  operation in log space, addition is the messy one). Specifically, the
  Park anti-ε result needs to be re-derived for the multiplication
  oracle before we can claim it as a target.

### Estimated infrastructure work

- `oracle_mult.py` (~150 LOC, mirroring `oracle.py`): half a day
- New regime classifier (~50 LOC): half a day
- Verify against `torch.float8_e4m3fn`: half a day
- Re-derive anti-ε for multiplication (~theoretical, ~1 day)
- Re-run four arms × matched seeds: ~1 hour GPU
- Probe and synthesis: ~half a day

Total: ~3 days infrastructure before we can hand-off the same probe set
to the new operation.

### What we'd predict in advance (held lightly)

| arm | addition mean fail % | mult mean fail % (predicted) | direction |
|---|---:|---:|---|
| bit | 8.98% | <5% | better |
| RoPE | 11.93% | <8% | better |
| SEM | 0.68% | <0.5% | better |
| F1 (decimal FoNE) | 14.59% | >30% | worse |
| F2 (binary FoNE) | <2% | >15% | much worse |

If F2 (which essentially saturates addition) collapses on multiplication,
we have the cleanest possible symmetry: same architecture, same training
schedule, flip the operation, flip the encoding's native-vs-taxed status.

### Open scoping questions

- Whether to repeat the full scale sweep (L4-E048 → L4-E128) for each
  arm, or just match-test at L4-E048. The scale curve under addition
  taught us a lot; multiplication might or might not need it.
- Whether to extend to L4-E256+ for FoNE F2 specifically, to see if its
  multiplication collapse is recoverable with capacity (parallel to how
  F1's anti-ε recovered with capacity for addition).
- Whether to include a {addition, multiplication} multi-task arm at
  some point. Probably premature — clean single-task results first.

---

## Arm 6: Cross-axis experiments (post-multiplication)

Once both operations have four-arm coverage, the natural cross-axis
follow-ups span operation × representation:

- **RoPE × SEM × multiplication**: do the addition findings transfer?
- **F2 × {addition, multiplication, both}**: how does the encoding's
  native operation drop accuracy under the dual? Is there a multi-task
  benefit or interference?
- **F1 vs F2 head-to-head under multiplication**: the F2 vs F1
  asymmetry on addition is precision-driven. Under multiplication F2
  should still have its bit-position alignment but lose its operational
  advantage. Interesting test of "what does F2 still get right when its
  native operation isn't there?"

---

## What this doc is and isn't

This is a planning document. The five arms above the multiplication line
are closed; the multiplication arm is the next experimental commitment.
The cross-axis section is queued and not yet scoped in detail.
