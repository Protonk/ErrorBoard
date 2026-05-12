# Failure consensus across L4-E044 seeds

Computed per-pair failure count across 20 seeds (L4-E044, iter 20k).
Holdout n = 6554. Mean per-(seed,pair) failure rate = 10.76%.

## Failure-count histogram

For each k ∈ 0..20, count of holdout pairs failed by exactly k of 20 seeds:

| k (seeds failing) | observed | expected (i.i.d. binomial) | ratio obs/exp |
|------------------:|---------:|--------------------------:|--------------:|
| 0 |     3695 |                      672.4 |          5.50 |
| 1 |      592 |                     1621.6 |          0.37 |
| 2 |      462 |                     1857.6 |          0.25 |
| 3 |      358 |                     1344.0 |          0.27 |
| 4 |      265 |                      688.7 |          0.38 |
| 5 |      219 |                      265.8 |          0.82 |
| 6 |      175 |                       80.1 |          2.18 |
| 7 |      146 |                       19.3 |          7.56 |
| 8 |      121 |                        3.8 |         31.96 |
| 9 |       90 |                        0.6 |        147.87 |
| 10 |       98 |                        0.1 |       1213.92 |
| 11 |       70 |                        0.0 |             ∞ |
| 12 |       61 |                        0.0 |             ∞ |
| 13 |       55 |                        0.0 |             ∞ |
| 14 |       42 |                        0.0 |             ∞ |
| 15 |       29 |                        0.0 |             ∞ |
| 16 |       27 |                        0.0 |             ∞ |
| 17 |       30 |                        0.0 |             ∞ |
| 18 |       13 |                        0.0 |             ∞ |
| 19 |        6 |                        0.0 |             ∞ |
| 20 |        0 |                        0.0 |             ∞ |

**Variance of per-pair fail count**: observed 13.210, i.i.d. predicted 1.921 (ratio 6.88). Higher than predicted ⟹ underlying per-pair probability is non-uniform (some pairs much harder than others).

## Three strata

| stratum | n_pairs | fraction |
|---------|--------:|---------:|
| structural core (20/20 fail) | 0 | 0.0% |
| lottery zone (1..19/20 fail) | 2859 | 43.6% |
| structural easy (0/20 fail) | 3695 | 56.4% |

## Where each stratum lives (regime)

| regime | n_total | n_core | n_lottery | n_easy | core% | lottery% |
|--------|--------:|-------:|----------:|-------:|------:|---------:|
| special-values     |     102 |      0 |         0 |    102 |  0.0% |     0.0% |
| overflow           |      84 |      0 |         0 |     84 |  0.0% |     0.0% |
| subnormal-result   |      59 |      0 |         3 |     56 |  0.0% |     5.1% |
| cancellation       |     144 |      0 |        24 |    120 |  0.0% |    16.7% |
| rounding-tie       |     610 |      0 |       532 |     78 |  0.0% |    87.2% |
| large-dexp         |    3754 |      0 |       728 |   3026 |  0.0% |    19.4% |
| default            |    1801 |      0 |      1572 |    229 |  0.0% |    87.3% |

## m_c distribution within strata (normal-result only)

| m_c | n_total | core | lottery | easy | core% | lottery% |
|-----|--------:|-----:|--------:|-----:|------:|---------:|
| 0/8 |     971 |    0 |     469 |  502 |  0.0% |    48.3% |
| 1/8 |     716 |    0 |     353 |  363 |  0.0% |    49.3% |
| 2/8 |     922 |    0 |     411 |  511 |  0.0% |    44.6% |
| 3/8 |     717 |    0 |     320 |  397 |  0.0% |    44.6% |
| 4/8 |     922 |    0 |     353 |  569 |  0.0% |    38.3% |
| 5/8 |     667 |    0 |     302 |  365 |  0.0% |    45.3% |
| 6/8 |     903 |    0 |     371 |  532 |  0.0% |    41.1% |
| 7/8 |     549 |    0 |     275 |  274 |  0.0% |    50.1% |

## Projection to more seeds

If the per-pair failure probabilities are stable across seeds, with N seeds
we resolve per-pair probability to ±√(p(1-p)/N) (1σ).
For the lottery zone (where p ≈ 0.5), this is:

| N seeds | ±1σ per-pair p estimate | distinguishable bins (≈ 3σ apart) |
|--------:|------------------------:|----------------------------------:|
| 5 | ±0.224 | ~1 |
| 10 | ±0.158 | ~2 |
| 20 | ±0.112 | ~2 |
| 50 | ±0.071 | ~4 |
| 100 | ±0.050 | ~6 |

5 seeds give ±0.22 per-pair p resolution — we can distinguish ~1 bin (0 vs ~1).
20 seeds would resolve ~3 bins (e.g., 'low', 'mid', 'high' failure-probability classes).
100 seeds would resolve ~5-6 bins, enough to characterize the lottery distribution
as a continuous shape rather than a binary core-vs-lottery split.