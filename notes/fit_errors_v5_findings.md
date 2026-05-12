# Probe 2 — V5 residual error stratification

V5 = `runs/sweep-L1-E128-s0/checkpoint_020000.pt`. Holdout n = 6554.
V5 errors (≥1 result bit wrong): 1021 (15.58%).

Per-stratum counts and error rates. Compare error rates against train density
to test whether residual errors are fit-driven (cluster in low-density strata)
or structure-driven (cluster regardless of density).


### Stratified by regime

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| special-values |       102 |        1 |     0.980% |                397 |
| overflow   |        84 |        0 |     0.000% |                486 |
| underflow-to-zero |        0 |       0 |     —       | — |
| subnormal-result |        59 |        8 |    13.559% |                689 |
| cancellation |       144 |       11 |     7.639% |                282 |
| rounding-tie |       610 |       87 |    14.262% |                 67 |
| large-dexp |      3754 |      138 |     3.676% |                 11 |
| default    |      1801 |      776 |    43.087% |                 23 |

### Stratified by Δexp (excludes NaN-bearing cases via unbiased_exp; subnormal → −6)

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| 0          |       448 |       99 |    22.098% |                  — |
| 1          |       780 |      347 |    44.487% |                  — |
| 2          |       717 |      273 |    38.075% |                  — |
| 3          |       674 |      159 |    23.591% |                  — |
| 4-7        |      2153 |      139 |     6.456% |                  — |
| 8+         |      1782 |        4 |     0.224% |                  — |

### Stratified by max operand magnitude band

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| subnormal  |       110 |       42 |    38.182% |                  — |
| very small (e≤-3) |       567 |      221 |    38.977% |                  — |
| small (e=-2..0) |       989 |      214 |    21.638% |                  — |
| medium (e=1..3) |      1425 |      191 |    13.404% |                  — |
| large (e=4..6) |      1897 |      210 |    11.070% |                  — |
| largest (e=7-8) |      1465 |      142 |     9.693% |                  — |

### Stratified by sign relationship (finite only)

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| same-sign  |      3263 |      462 |    14.159% |                  — |
| opposite-sign |      3190 |      558 |    17.492% |                  — |

### Stratified by subnormal involvement

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| none (neither subnormal) |      5716 |      940 |    16.445% |                  — |
| operand-only (a or b is subnormal) |       678 |       72 |    10.619% |                  — |
| result-only subnormal |        20 |        2 |    10.000% |                  — |
| both ops and result subnormal |        16 |        4 |    25.000% |                  — |
| any subnormal involvement |       737 |       80 |    10.855% |                  — |

### Stratified by rounding-tie tag

| stratum | n holdout | n errors | error rate | train density / pair |
|---------|----------:|---------:|-----------:|---------------------:|
| is a tie   |       610 |       87 |    14.262% |                  — |
| not a tie  |      5843 |      933 |    15.968% |                  — |

## Reading the tables

- **Fit hypothesis**: error rate should correlate inversely with train density.
  Rare strata (low density per pair) should have higher error rates.
- **Structure hypothesis**: error rate should track algorithmic complexity
  irrespective of density. The same-sign vs opposite-sign axis is the cleanest
  test, as it cuts across regimes orthogonally to training density.
