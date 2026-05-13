# Four-arm multiplication comparison (L4-E048, 20 seeds, iter 20k)

Multiplication pair table; holdout split seed=0, n=6554.

## Cross-arm headline table

| arm | mean fail % | var ratio | core % | lottery % | easy % | p̂ ≥ 0.8 | p̂ = 1.0 |
|-----|------:|------:|------:|------:|------:|------:|------:|
| bit (mul) | 49.26% | 0.84 | 0.00% | 98.2% | 1.8% | 6 | 0 |
| SEM (mul) | 49.08% | 0.79 | 0.00% | 98.2% | 1.8% | 5 | 0 |
| FoNE F1 (mul) | 63.87% | 8.96 | 16.33% | 77.1% | 6.5% | 3071 | 1070 |
| FoNE F2 (mul) | 8.28% | 3.32 | 0.00% | 52.9% | 47.1% | 1 | 0 |

## Per-regime mean fail rate

| regime | n | bit (mul) | SEM (mul) | FoNE F1 (mul) | FoNE F2 (mul) |
|--------|------|------|------|------|------|
| special-values | 203 | 20.94% | 20.86% | 0.00% | 0.00% |
| overflow | 1015 | 50.36% | 50.17% | 27.44% | 0.05% |
| underflow-to-zero | 216 | 49.40% | 50.69% | 0.00% | 0.00% |
| subnormal-result | 563 | 50.66% | 50.09% | 42.74% | 0.28% |
| rounding-tie | 391 | 49.91% | 49.81% | 57.63% | 0.50% |
| exact-result | 1378 | 50.05% | 50.17% | 82.57% | 4.36% |
| default | 2788 | 50.14% | 49.76% | 82.62% | 17.16% |

## Pairwise lottery-zone Jaccard

Lottery zone = pairs failed by 1..n_seeds-1 of the seeds.

| | bit (mul) | SEM (mul) | FoNE F1 (mul) | FoNE F2 (mul) |
|---|---|---|---|---|
| bit (mul) | 1.000 | 1.000 | 0.785 | 0.539 |
| SEM (mul) | 1.000 | 1.000 | 0.785 | 0.539 |
| FoNE F1 (mul) | 0.785 | 0.785 | 1.000 | 0.425 |
| FoNE F2 (mul) | 0.539 | 0.539 | 0.425 | 1.000 |
