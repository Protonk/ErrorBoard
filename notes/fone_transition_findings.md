# FoNE capacity → anti-ε transition

Sweeps L=4 with `n_embd ∈ {48, 64, 96, 128}` at 5 matched seeds each.
Tests whether the L4-E048 anti-ε sign-flip resolves as a sharp threshold,
a gradual shift in seed distribution, or persists at intermediate scales.

## Per-seed anti-ε Pearson (ε, mean |log Δ|)

| n_embd | params | s0 | s1 | s2 | s3 | s4 |
|-------:|-------:|---:|---:|---:|---:|---:|
| 48 | 112,416 | +0.37 | +0.20 | +0.06 | +0.08 | +0.51 |
| 64 | 199,040 | +0.00 | +0.20 | +0.21 | +0.37 | +0.08 |
| 96 | 446,016 | +0.19 | +0.27 | -0.30 | +0.43 | -0.38 |
| 128 | 791,296 | +0.35 | -0.05 | -0.89 | -0.20 | -0.04 |


## Per-seed n_err (normal-result, summed across m_c)

| n_embd | s0 | s1 | s2 | s3 | s4 | total |
|-------:|---:|---:|---:|---:|---:|------:|
| 48 | 1057 | 1073 | 863 | 1095 | 942 | 5030 |
| 64 | 726 | 400 | 432 | 507 | 501 | 2566 |
| 96 | 298 | 215 | 160 | 146 | 170 | 989 |
| 128 | 156 | 74 | 60 | 67 | 55 | 412 |


## Anti-ε sign distribution across seeds

| n_embd | n seeds with ρ < −0.5 | n with ρ ∈ [−0.5, +0.5] | n with ρ > +0.5 |
|-------:|----------------------:|------------------------:|----------------:|
| 48 | 0 | 4 | 1 |
| 64 | 0 | 5 | 0 |
| 96 | 0 | 5 | 0 |
| 128 | 1 | 4 | 0 |


## Lottery shape across capacity

| n_embd | mean fail % | var ratio | structural easy % | lottery % | structural core (pairs) |
|-------:|------------:|----------:|------------------:|----------:|-----------------------:|
| 48 | 15.39% | 2.78 | 67.8% | 29.3% | 189 |
| 64 | 7.84% | 2.36 | 79.3% | 19.8% | 60 |
| 96 | 3.03% | 1.74 | 89.3% | 10.6% | 5 |
| 128 | 1.26% | 1.50 | 95.1% | 4.9% | 0 |


## Per-regime mean accuracy (5 seeds each)

| regime | E=48 | E=64 | E=96 | E=128 |
|--------|------|------|------|------|
| special-values | 100.00% | 100.00% | 100.00% | 100.00% |
| overflow | 100.00% | 100.00% | 100.00% | 100.00% |
| underflow-to-zero | — | — | — | — |
| subnormal-result | 99.32% | 100.00% | 100.00% | 100.00% |
| cancellation | 97.50% | 99.31% | 98.47% | 99.86% |
| rounding-tie | 74.92% | 89.67% | 96.30% | 98.89% |
| large-dexp | 93.11% | 96.56% | 98.92% | 99.50% |
| default | 67.08% | 82.21% | 92.62% | 96.84% |
