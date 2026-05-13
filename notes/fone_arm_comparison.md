# Four-arm comparison: bit vs RoPE-bit vs SEM vs FoNE

All L4-E048, iter 20k, 20 seeds each. Same holdout split.

## Cross-arm headline table

| arm | mean fail % | var ratio | core % | lottery % | easy % | p̂ ≥ 0.8 | p̂ = 1.0 |
|-----|------:|------:|------:|------:|------:|------:|------:|
| learned-PE bit | 8.98% | 6.12 | 0.03% | 41.0% | 59.0% | 22 | 2 |
| RoPE bit | 11.93% | 7.66 | 0.12% | 43.7% | 56.2% | 128 | 8 |
| learned-PE SEM | 0.68% | 2.92 | 0.00% | 7.0% | 93.0% | 0 | 0 |
| learned-PE FoNE | 14.59% | 9.38 | 0.55% | 41.9% | 57.6% | 242 | 36 |

## Per-pair p̂ distribution (20 seeds)

| bin | learned-PE bit | RoPE bit | learned-PE SEM | learned-PE FoNE |
|-----|------|------|------|------|
| 0/20 | 3868 | 3685 | 6092 | 3773 |
| 1/20 | 646 | 553 | 258 | 452 |
| 2-5/20 | 1262 | 1236 | 186 | 950 |
| 6-10/20 | 551 | 646 | 17 | 684 |
| 11-15/20 | 205 | 306 | 1 | 453 |
| 16-19/20 | 20 | 120 | 0 | 206 |
| 20/20 | 2 | 8 | 0 | 36 |

## Per-regime mean fail rate

| regime | n | learned-PE bit | RoPE bit | learned-PE SEM | learned-PE FoNE |
|-----|------|------|------|------|------|
| special-values | 102 | 0.10% | 0.00% | 0.00% | 0.00% |
| overflow | 84 | 0.12% | 0.12% | 0.00% | 0.00% |
| subnormal-result | 59 | 0.25% | 0.68% | 0.34% | 0.76% |
| cancellation | 144 | 1.42% | 1.25% | 0.03% | 1.53% |
| rounding-tie | 610 | 9.65% | 13.88% | 0.58% | 22.79% |
| large-dexp | 3754 | 2.81% | 3.72% | 0.32% | 6.75% |
| default | 1801 | 23.42% | 30.84% | 1.58% | 31.14% |

## Pairwise lottery-zone Jaccard

Lottery zone = pairs failed by 1..n_seeds-1 of the seeds.

| | learned-PE bit | RoPE bit | learned-PE SEM | learned-PE FoNE |
|---|---|---|---|---|
| learned-PE bit | 1.000 | 0.759 | 0.155 | 0.608 |
| RoPE bit | 0.759 | 1.000 | 0.148 | 0.629 |
| learned-PE SEM | 0.155 | 0.148 | 1.000 | 0.137 |
| learned-PE FoNE | 0.608 | 0.629 | 0.137 | 1.000 |

## Heavy-tail (p̂ ≥ 0.8) overlap across arms

- **learned-PE bit** heavy tail = 22 pairs; of those:
  - 19 (86.4%) are also heavy-tail in **RoPE bit**
  - 0 (0.0%) are also heavy-tail in **learned-PE SEM**
  - 5 (22.7%) are also heavy-tail in **learned-PE FoNE**
- **RoPE bit** heavy tail = 128 pairs; of those:
  - 19 (14.8%) are also heavy-tail in **learned-PE bit**
  - 0 (0.0%) are also heavy-tail in **learned-PE SEM**
  - 26 (20.3%) are also heavy-tail in **learned-PE FoNE**
- **learned-PE SEM** heavy tail empty.
- **learned-PE FoNE** heavy tail = 242 pairs; of those:
  - 5 (2.1%) are also heavy-tail in **learned-PE bit**
  - 26 (10.7%) are also heavy-tail in **RoPE bit**
  - 0 (0.0%) are also heavy-tail in **learned-PE SEM**
