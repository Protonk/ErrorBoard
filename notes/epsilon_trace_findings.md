# ε-trace experiment

Tests whether each vertex's error rate, stratified by *result* mantissa
value m_c, traces the formal ε(m) = log₂(1+m) − m function.

Restricted to normal-result cases (biased exponent ≥ 1, not NaN).

**ε(m) values across the 8 mantissa bins**:

| m_c | 0/8 | 1/8 | 2/8 | 3/8 | 4/8 | 5/8 | 6/8 | 7/8 |
|-----|----:|----:|----:|----:|----:|----:|----:|----:|
| ε   | 0.000 | 0.045 | 0.072 | 0.084 | 0.085 | 0.075 | 0.057 | 0.032 |

Peak at m_c=3/8 or 4/8. If model errors trace ε, error rate should peak there.

### V1: early borderline (iter 5k)

| m_c   | m (fraction) | ε(m)   | n     | n_err | error rate |
|-------|-------------:|-------:|------:|------:|-----------:|
| 0/8   |        0.000 | 0.0000 |   971 |   381 |     39.24% |
| 1/8   |        0.125 | 0.0449 |   716 |   341 |     47.63% |
| 2/8   |        0.250 | 0.0719 |   922 |   218 |     23.64% |
| 3/8   |        0.375 | 0.0844 |   717 |   314 |     43.79% |
| 4/8   |        0.500 | 0.0850 |   922 |   304 |     32.97% |
| 5/8   |        0.625 | 0.0754 |   667 |   276 |     41.38% |
| 6/8   |        0.750 | 0.0574 |   903 |   224 |     24.81% |
| 7/8   |        0.875 | 0.0319 |   549 |   227 |     41.35% |

**Pearson(ε, error_rate) = -0.227**
Peak m_c: 1/8 (error rate 47.63%); ε-peak is at m_c=3/8 or 4/8.

### V2: mid borderline (iter 10k)

| m_c   | m (fraction) | ε(m)   | n     | n_err | error rate |
|-------|-------------:|-------:|------:|------:|-----------:|
| 0/8   |        0.000 | 0.0000 |   971 |   294 |     30.28% |
| 1/8   |        0.125 | 0.0449 |   716 |   272 |     37.99% |
| 2/8   |        0.250 | 0.0719 |   922 |   138 |     14.97% |
| 3/8   |        0.375 | 0.0844 |   717 |   227 |     31.66% |
| 4/8   |        0.500 | 0.0850 |   922 |   228 |     24.73% |
| 5/8   |        0.625 | 0.0754 |   667 |   240 |     35.98% |
| 6/8   |        0.750 | 0.0574 |   903 |   215 |     23.81% |
| 7/8   |        0.875 | 0.0319 |   549 |   188 |     34.24% |

**Pearson(ε, error_rate) = -0.279**
Peak m_c: 1/8 (error rate 37.99%); ε-peak is at m_c=3/8 or 4/8.

### V3: late borderline (iter 20k)

| m_c   | m (fraction) | ε(m)   | n     | n_err | error rate |
|-------|-------------:|-------:|------:|------:|-----------:|
| 0/8   |        0.000 | 0.0000 |   971 |   125 |     12.87% |
| 1/8   |        0.125 | 0.0449 |   716 |   155 |     21.65% |
| 2/8   |        0.250 | 0.0719 |   922 |    75 |      8.13% |
| 3/8   |        0.375 | 0.0844 |   717 |   146 |     20.36% |
| 4/8   |        0.500 | 0.0850 |   922 |    60 |      6.51% |
| 5/8   |        0.625 | 0.0754 |   667 |   134 |     20.09% |
| 6/8   |        0.750 | 0.0574 |   903 |    79 |      8.75% |
| 7/8   |        0.875 | 0.0319 |   549 |   127 |     23.13% |

**Pearson(ε, error_rate) = -0.186**
Peak m_c: 7/8 (error rate 23.13%); ε-peak is at m_c=3/8 or 4/8.

### V4: saturated (L4-E128, iter 20k)

| m_c   | m (fraction) | ε(m)   | n     | n_err | error rate |
|-------|-------------:|-------:|------:|------:|-----------:|
| 0/8   |        0.000 | 0.0000 |   971 |     4 |      0.41% |
| 1/8   |        0.125 | 0.0449 |   716 |     9 |      1.26% |
| 2/8   |        0.250 | 0.0719 |   922 |    13 |      1.41% |
| 3/8   |        0.375 | 0.0844 |   717 |     4 |      0.56% |
| 4/8   |        0.500 | 0.0850 |   922 |     1 |      0.11% |
| 5/8   |        0.625 | 0.0754 |   667 |     1 |      0.15% |
| 6/8   |        0.750 | 0.0574 |   903 |     8 |      0.89% |
| 7/8   |        0.875 | 0.0319 |   549 |    20 |      3.64% |

**Pearson(ε, error_rate) = -0.346**
Peak m_c: 7/8 (error rate 3.64%); ε-peak is at m_c=3/8 or 4/8.

### V5: depth-capped (L1-E128, iter 20k)

| m_c   | m (fraction) | ε(m)   | n     | n_err | error rate |
|-------|-------------:|-------:|------:|------:|-----------:|
| 0/8   |        0.000 | 0.0000 |   971 |   147 |     15.14% |
| 1/8   |        0.125 | 0.0449 |   716 |   173 |     24.16% |
| 2/8   |        0.250 | 0.0719 |   922 |   112 |     12.15% |
| 3/8   |        0.375 | 0.0844 |   717 |   183 |     25.52% |
| 4/8   |        0.500 | 0.0850 |   922 |    51 |      5.53% |
| 5/8   |        0.625 | 0.0754 |   667 |   139 |     20.84% |
| 6/8   |        0.750 | 0.0574 |   903 |    73 |      8.08% |
| 7/8   |        0.875 | 0.0319 |   549 |   134 |     24.41% |

**Pearson(ε, error_rate) = -0.159**
Peak m_c: 3/8 (error rate 25.52%); ε-peak is at m_c=3/8 or 4/8.

## Cross-vertex summary

| vertex | Pearson(ε, err_rate) |
|--------|---------------------:|
| V1 | -0.227 |
| V2 | -0.279 |
| V3 | -0.186 |
| V4 | -0.346 |
| V5 | -0.159 |

## Interpretation guide

- ρ → +1.0: error rate traces ε shape (peak at m=3/8..4/8, low at endpoints)
- ρ → 0:    error rate flat or unrelated to ε
- ρ → −1.0: error rate anti-correlated (peak at endpoints, low in middle)

A cross-vertex pattern where ρ increases from V1 to V4 would suggest that
ε's shape becomes visible only at saturation — earlier models are too noisy
for ε to surface. V5 (depth-capped) vs V4 (saturated) tests the architectural
axis: does depth-1 fail to trace ε even at convergence?