# F1 (decimal FoNE) vs F2 (binary FoNE) — matched comparison

L=4, matched seeds 0..4, learned PE, 20k iters. F1 periods are
`T_i = 10^i for i ∈ [-2, 3]` (6 decimal digits, FONE_DIM=12). F2
periods are `T_i = 2^i for i ∈ [-8, 9]` (18 binary digits, FONE_DIM=36).

**Test:** Does aligning FoNE's period basis with FP8's binade
structure recover anti-ε severity at smaller capacity?

## Per-seed Pearson(ε, mean |log Δ|)

| size | arm | s0 | s1 | s2 | s3 | s4 | mean ρ |
|-----:|-----|----:|----:|----:|----:|----:|------:|
| 48 | f1 | +0.37 | +0.20 | +0.06 | +0.08 | +0.51 | +0.24 |
| 48 | f2 | -0.18 | +0.06 | +0.33 | +0.37 | -0.11 | +0.09 |
| 128 | f1 | +0.35 | -0.05 | -0.89 | -0.20 | -0.04 | -0.17 |
| 128 | f2 | +nan | +nan | +nan | +nan | +nan | +nan |


## Total errors + default-regime accuracy

| size | arm | total n_err | mean default acc |
|-----:|-----|------------:|----------------:|
| 48 | f1 | 5030 | 67.08% |
| 48 | f2 | 306 | 97.58% |
| 128 | f1 | 412 | 96.84% |
| 128 | f2 | 4 | 99.98% |


## Anti-ε sign distribution

| size | arm | strongly negative (ρ < −0.5) | flat (\|ρ\| ≤ 0.5) | strongly positive (ρ > +0.5) |
|-----:|-----|---:|---:|---:|
| 48 | f1 | 0 | 4 | 1 |
| 48 | f2 | 0 | 5 | 0 |
| 128 | f1 | 1 | 4 | 0 |
| 128 | f2 | 0 | 0 | 0 |
