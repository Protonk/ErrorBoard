# m0 attention by Δexp

Probe: c:m0 predictor attention to each operand mantissa bit, stratified
by Δexp = |unbiased_exp(a) − unbiased_exp(b)|. Restricted to normal-normal
finite operand pairs (subnormals, zero, NaN excluded). Sample sizes:

| Δexp bucket | n_samples |
|-------------|----------:|
| 0 | 64 |
| 1 | 64 |
| 2 | 64 |
| 3 | 64 |
| 4-7 | 64 |
| 8+ | 64 |


#### a:m1 attention: c:m0 → a:m1 (attention weight)

| Δexp | n |  V1  |  V2  |  V3  |  V4  |  V5  |
|-----:|--:|-----:|-----:|-----:|-----:|-----:|
|    0 | — | 0.035 | 0.037 | 0.040 | 0.055 | 0.036 |
|    1 | — | 0.035 | 0.035 | 0.039 | 0.049 | 0.033 |
|    2 | — | 0.037 | 0.034 | 0.038 | 0.057 | 0.034 |
|    3 | — | 0.037 | 0.035 | 0.039 | 0.050 | 0.034 |
|  4-7 | — | 0.035 | 0.035 | 0.040 | 0.050 | 0.032 |
|   8+ | — | 0.039 | 0.036 | 0.039 | 0.055 | 0.032 |

#### a:m0 attention: c:m0 → a:m0 (attention weight)

| Δexp | n |  V1  |  V2  |  V3  |  V4  |  V5  |
|-----:|--:|-----:|-----:|-----:|-----:|-----:|
|    0 | — | 0.104 | 0.094 | 0.092 | 0.071 | 0.068 |
|    1 | — | 0.101 | 0.086 | 0.082 | 0.057 | 0.073 |
|    2 | — | 0.098 | 0.083 | 0.075 | 0.060 | 0.066 |
|    3 | — | 0.096 | 0.080 | 0.071 | 0.063 | 0.069 |
|  4-7 | — | 0.097 | 0.079 | 0.069 | 0.069 | 0.066 |
|   8+ | — | 0.094 | 0.077 | 0.066 | 0.064 | 0.070 |

#### b:m1 attention: c:m0 → b:m1 (attention weight)

| Δexp | n |  V1  |  V2  |  V3  |  V4  |  V5  |
|-----:|--:|-----:|-----:|-----:|-----:|-----:|
|    0 | — | 0.030 | 0.036 | 0.036 | 0.055 | 0.054 |
|    1 | — | 0.030 | 0.033 | 0.034 | 0.070 | 0.053 |
|    2 | — | 0.030 | 0.034 | 0.035 | 0.065 | 0.055 |
|    3 | — | 0.032 | 0.036 | 0.038 | 0.071 | 0.061 |
|  4-7 | — | 0.030 | 0.034 | 0.036 | 0.057 | 0.051 |
|   8+ | — | 0.033 | 0.036 | 0.038 | 0.060 | 0.052 |

#### b:m0 attention: c:m0 → b:m0 (attention weight)

| Δexp | n |  V1  |  V2  |  V3  |  V4  |  V5  |
|-----:|--:|-----:|-----:|-----:|-----:|-----:|
|    0 | — | 0.065 | 0.062 | 0.059 | 0.076 | 0.109 |
|    1 | — | 0.065 | 0.063 | 0.059 | 0.084 | 0.110 |
|    2 | — | 0.064 | 0.060 | 0.056 | 0.086 | 0.106 |
|    3 | — | 0.065 | 0.062 | 0.058 | 0.085 | 0.102 |
|  4-7 | — | 0.069 | 0.065 | 0.060 | 0.091 | 0.115 |
|   8+ | — | 0.070 | 0.066 | 0.062 | 0.103 | 0.114 |

#### c:m0 attention ratio (a:m1 + b:m1) / (a:m0 + b:m0)

Higher ratio ⟹ model attends more to operand m1 than m0 when predicting c:m0.
Under the rounding-bit hypothesis (H1), this ratio should peak at Δexp=1 and grow with vertex capability.

| Δexp |  V1  |  V2  |  V3  |  V4  |  V5  |
|-----:|-----:|-----:|-----:|-----:|-----:|
|    0 | 0.39 | 0.47 | 0.51 | 0.75 | 0.50 |
|    1 | 0.39 | 0.46 | 0.51 | 0.84 | 0.47 |
|    2 | 0.41 | 0.48 | 0.56 | 0.84 | 0.52 |
|    3 | 0.43 | 0.50 | 0.60 | 0.82 | 0.55 |
|  4-7 | 0.39 | 0.48 | 0.59 | 0.67 | 0.46 |
|   8+ | 0.44 | 0.51 | 0.60 | 0.69 | 0.46 |

## Interpretation guide

- If the ratio table peaks at Δexp=1 and grows with vertex capability,
  that's the rounding-bit (H1) signature.
- If the ratio is roughly flat across Δexp but varies by vertex,
  that supports the generic 'LSB is hardest' reading (H3).
- If the ratio is flat across both Δexp and vertex, the m0 anomaly from
  the pentagon writeup is not explained by operand-m1 attention shifts,
  and we'd need a different probe direction.