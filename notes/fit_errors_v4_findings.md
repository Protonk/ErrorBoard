# Probe 2 — V4 residual error stratification

V4 = L4-E128 seed 0, iter 20000. Holdout n = 6554.
V4 errors (≥1 result bit wrong): 63 (0.96%).

Per-stratum counts and error rates. Compare error rates against train density
to test whether residual errors are fit-driven (cluster in low-density strata)
or structure-driven (cluster regardless of density).


### Stratified by regime

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| special-values |       102 |        0 |     0.000% |                397 |
| overflow   |        84 |        0 |     0.000% |                486 |
| underflow-to-zero |        0 |       0 |     —       | — |
| subnormal-result |        59 |        3 |     5.085% |                689 |
| cancellation |       144 |        0 |     0.000% |                282 |
| rounding-tie |       610 |        4 |     0.656% |                 67 |
| large-dexp |      3754 |       11 |     0.293% |                 11 |
| default    |      1801 |       45 |     2.499% |                 23 |

### Stratified by Δexp (excludes NaN-bearing cases via unbiased_exp; subnormal → −6)

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| 0          |       448 |        9 |     2.009% |                  — |
| 1          |       780 |       16 |     2.051% |                  — |
| 2          |       717 |       16 |     2.232% |                  — |
| 3          |       674 |       11 |     1.632% |                  — |
| 4-7        |      2153 |       10 |     0.464% |                  — |
| 8+         |      1782 |        1 |     0.056% |                  — |

### Stratified by max operand magnitude band

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| subnormal  |       110 |        8 |     7.273% |                  — |
| very small (e≤-3) |       567 |       26 |     4.586% |                  — |
| small (e=-2..0) |       989 |        8 |     0.809% |                  — |
| medium (e=1..3) |      1425 |       13 |     0.912% |                  — |
| large (e=4..6) |      1897 |        6 |     0.316% |                  — |
| largest (e=7-8) |      1465 |        2 |     0.137% |                  — |

### Stratified by sign relationship (finite only)

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| same-sign  |      3263 |       39 |     1.195% |                  — |
| opposite-sign |      3190 |       24 |     0.752% |                  — |

### Stratified by subnormal involvement

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| none (neither subnormal) |      5716 |       37 |     0.647% |                  — |
| operand-only (a or b is subnormal) |       678 |       23 |     3.392% |                  — |
| result-only subnormal |        20 |        0 |     0.000% |                  — |
| both ops and result subnormal |        16 |        1 |     6.250% |                  — |
| any subnormal involvement |       737 |       26 |     3.528% |                  — |

### Stratified by rounding-tie tag

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| is a tie   |       610 |        4 |     0.656% |                  — |
| not a tie  |      5843 |       59 |     1.010% |                  — |

## Reading the tables

- **Fit hypothesis**: error rate should correlate inversely with train density.
  Rare strata (low density per pair) should have higher error rates.
- **Structure hypothesis**: error rate should track algorithmic complexity
  irrespective of density. The same-sign vs opposite-sign axis is the cleanest
  test, as it cuts across regimes orthogonally to training density.
