# Three-arm comparison: bit vs RoPE-bit vs SEM

All L4-E048, iter 20k, 20 seeds each. Same holdout split (seed=0, n=6554).

## learned-PE bit — single-arm stats

n_seeds=20  n_holdout=6554  mean_fail=8.98%  var_ratio=6.12  core=0.0%  lottery=41.0%  easy=59.0%

## RoPE bit — single-arm stats

n_seeds=20  n_holdout=6554  mean_fail=11.93%  var_ratio=7.66  core=0.1%  lottery=43.7%  easy=56.2%

## learned-PE SEM — single-arm stats

n_seeds=20  n_holdout=6554  mean_fail=0.68%  var_ratio=2.92  core=0.0%  lottery=7.0%  easy=93.0%

## Cross-arm headline table

| arm | mean fail % | var ratio | core % | lottery % | easy % | p̂ ≥ 0.8 pairs | p̂ = 1.0 pairs |
|-----|------:|------:|------:|------:|------:|------:|------:|
| learned-PE bit | 8.98% | 6.12 | 0.03% | 41.0% | 59.0% | 22 | 2 |
| RoPE bit | 11.93% | 7.66 | 0.12% | 43.7% | 56.2% | 128 | 8 |
| learned-PE SEM | 0.68% | 2.92 | 0.00% | 7.0% | 93.0% | 0 | 0 |

## Per-pair p̂ distribution (20 seeds)

| bin | learned-PE bit | RoPE bit | learned-PE SEM |
|-----|------|------|------|
| 0/20 | 3868 | 3685 | 6092 |
| 1/20 | 646 | 553 | 258 |
| 2-5/20 | 1262 | 1236 | 186 |
| 6-10/20 | 551 | 646 | 17 |
| 11-15/20 | 205 | 306 | 1 |
| 16-19/20 | 20 | 120 | 0 |
| 20/20 | 2 | 8 | 0 |

## Per-regime mean fail rate

| regime | n | learned-PE bit | RoPE bit | learned-PE SEM |
|-----|------|------|------|------|
| special-values | 102 | 0.10% | 0.00% | 0.00% |
| overflow | 84 | 0.12% | 0.12% | 0.00% |
| subnormal-result | 59 | 0.25% | 0.68% | 0.34% |
| cancellation | 144 | 1.42% | 1.25% | 0.03% |
| rounding-tie | 610 | 9.65% | 13.88% | 0.58% |
| large-dexp | 3754 | 2.81% | 3.72% | 0.32% |
| default | 1801 | 23.42% | 30.84% | 1.58% |

## Pairwise lottery-zone Jaccard

Lottery zone = pairs failed by 1..n_seeds-1 of the seeds.

| | learned-PE bit | RoPE bit | learned-PE SEM |
|---|---|---|---|
| learned-PE bit | 1.000 | 0.759 | 0.155 |
| RoPE bit | 0.759 | 1.000 | 0.148 |
| learned-PE SEM | 0.155 | 0.148 | 1.000 |

## Heavy-tail (p̂ ≥ 0.8) overlap across arms

- **learned-PE bit** heavy tail = 22 pairs; of those:
  - 19 (86.4%) are also heavy-tail in **RoPE bit**
  - 0 (0.0%) are also heavy-tail in **learned-PE SEM**
- **RoPE bit** heavy tail = 128 pairs; of those:
  - 19 (14.8%) are also heavy-tail in **learned-PE bit**
  - 0 (0.0%) are also heavy-tail in **learned-PE SEM**
- **learned-PE SEM** heavy tail empty.
