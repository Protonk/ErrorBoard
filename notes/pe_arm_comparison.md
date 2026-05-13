# PE arm comparison: sweep-L4-E048 vs rope-L4-E048

Both at L4-E048, iter 20k, 20 seeds each. Same holdout split (seed=0).

## sweep-L4-E048 — single-arm stats

n_seeds = 20, n_holdout = 6554, mean per-(seed,pair) fail rate = 8.98%.

**Variance of per-pair fail count**: observed 10.009, i.i.d. predicted 1.635 (ratio 6.12). Higher ⟹ heavier-tailed per-pair difficulty distribution.

| stratum | n_pairs | fraction |
|---------|--------:|---------:|
| structural core (20/20 fail) | 2 | 0.0% |
| lottery zone (1..19/20 fail) | 2684 | 41.0% |
| structural easy (0/20 fail) | 3868 | 59.0% |

## rope-L4-E048 — single-arm stats

n_seeds = 20, n_holdout = 6554, mean per-(seed,pair) fail rate = 11.93%.

**Variance of per-pair fail count**: observed 16.109, i.i.d. predicted 2.102 (ratio 7.66). Higher ⟹ heavier-tailed per-pair difficulty distribution.

| stratum | n_pairs | fraction |
|---------|--------:|---------:|
| structural core (20/20 fail) | 8 | 0.1% |
| lottery zone (1..19/20 fail) | 2861 | 43.7% |
| structural easy (0/20 fail) | 3685 | 56.2% |

## Side-by-side diff: sweep-L4-E048 vs rope-L4-E048

| metric | sweep-L4-E048 | rope-L4-E048 | Δ (B − A) |
|--------|------:|------:|-----:|
| mean fail rate | 8.98% | 11.93% | +2.95pp |
| variance ratio (obs/iid) | 6.12 | 7.66 | +1.54 |
| structural core % | 0.0% | 0.1% | +0.1pp |
| lottery zone % | 41.0% | 43.7% | +2.7pp |
| structural easy % | 59.0% | 56.2% | -2.8pp |

### Per-pair p̂ distribution (binned)

| bin | sweep-L4-E048 pairs | rope-L4-E048 pairs |
|-----|-------:|-------:|
| 0/20 | 3868 | 3685 |
| 1/20 | 646 | 553 |
| 2-5/20 | 1262 | 1236 |
| 6-10/20 | 551 | 646 |
| 11-15/20 | 205 | 306 |
| 16-19/20 | 20 | 120 |
| 20/20 | 2 | 8 |

### Per-regime mean fail rate

| regime | n_total | sweep-L4-E048 fail% | rope-L4-E048 fail% | Δ |
|--------|--------:|------:|------:|------:|
| special-values     |     102 |  0.10% |  0.00% | -0.10pp |
| overflow           |      84 |  0.12% |  0.12% | +0.00pp |
| subnormal-result   |      59 |  0.25% |  0.68% | +0.42pp |
| cancellation       |     144 |  1.42% |  1.25% | -0.17pp |
| rounding-tie       |     610 |  9.65% | 13.88% | +4.23pp |
| large-dexp         |    3754 |  2.81% |  3.72% | +0.91pp |
| default            |    1801 | 23.42% | 30.84% | +7.42pp |

### Lottery-zone overlap

- sweep-L4-E048 lottery: 2684 pairs
- rope-L4-E048 lottery: 2861 pairs
- intersection: 2393 pairs
- union: 3152 pairs
- Jaccard: 0.759

Of sweep-L4-E048's lottery, 89.2% is also rope-L4-E048's lottery; of rope-L4-E048's lottery, 83.6% is also sweep-L4-E048's. High overlap ⟹ both PE arms struggle on the same pairs (format-driven). Low overlap ⟹ each PE arm has its own lottery (arch-driven).