# Probe 1 — regime saturation order vs per-pair training density

Threshold: first iter where holdout accuracy ≥ 95%.
Trajectory data: 5 seeds of L4-E044.
Total training mass: 2,560,000 samples; n_active_regimes = 7; mass per regime: 365,714.

| regime | train_size | per-pair density | density rank | sat median (min..max) | sat rank |
|--------|-----------:|-----------------:|-------------:|----------------------:|---------:|
| special-values     |        922 |            396.7 |            3 |    500 (250..750) |        1 |
| overflow           |        752 |            486.3 |            2 |    500 (500..500) |        2 |
| underflow-to-zero  |          0 |                — |            — |    structurally empty |        — |
| subnormal-result   |        531 |            688.7 |            1 |   4500 (3000..5000) |        4 |
| cancellation       |       1296 |            282.2 |            4 |   6750 (5000..10750) |        5 |
| rounding-tie       |       5490 |             66.6 |            5 |   8250 (7000..10000) |        6 |
| large-dexp         |      33786 |             10.8 |            7 |   2500 (2250..3250) |        3 |
| default            |      16205 |             22.6 |            6 |  17000 (15750..17750) |        7 |

## Spearman rank correlation (density rank, saturation rank): **ρ = +0.429**

Interpretation:
- ρ → +1.0: saturation order matches density order (fit hypothesis)
- ρ → 0:    density and saturation order are unrelated (structural)
- ρ → −1.0: saturation order is opposite of density order (anti-fit)