# Multiplication severity: ε_mult = m_a · m_b stratification

Per-seed Pearson(ε_mult, mean |log Δ|) at L4-E048, 5 seeds per arm.
Severity binned by m_a · m_b in 8 bins. Reports two correlations:
  - full: includes EXACT_RESULT pairs (which have ε_mult = 0 and contribute
    severity 0 for FP-native arms — censors the signal toward 0)
  - rounding-only: excludes EXACT_RESULT and SPECIAL_VALUES regimes

Predictions from `notes/epsilon_under_multiplication.md`:
  - FP-native (bit, SEM): +0.4 to +0.8 rounding-only ρ
  - FoNE F1: -0.1 to +0.2
  - FoNE F2: 0.0 to +0.3

## Per-seed ρ table

| arm | seed | ρ (full) | ρ (rounding-only) | n_err total |
|-----|-----:|--------:|------------------:|------------:|
| bit | 0 | +0.550 | +0.547 | 3253 |
| bit | 1 | -0.213 | -0.216 | 3248 |
| bit | 2 | -0.395 | -0.300 | 3262 |
| bit | 3 | +0.154 | +0.154 | 3220 |
| bit | 4 | +0.356 | -0.056 | 3243 |
| SEM | 0 |   nan |   nan | 3253 |
| SEM | 1 |   nan |   nan | 3216 |
| SEM | 2 |   nan |   nan | 3217 |
| SEM | 3 |   nan |   nan | 3214 |
| SEM | 4 |   nan |   nan | 3199 |
| FoNE F1 | 0 | +0.197 | -0.172 | 4856 |
| FoNE F1 | 1 | +0.452 | +0.388 | 4664 |
| FoNE F1 | 2 | -0.538 | -0.496 | 3548 |
| FoNE F1 | 3 | +0.613 | +0.051 | 3460 |
| FoNE F1 | 4 | +0.462 | -0.011 | 3326 |
| FoNE F2 | 0 | -0.285 | -0.480 | 83 |
| FoNE F2 | 1 | -0.783 | -0.774 | 1034 |
| FoNE F2 | 2 | -0.772 | -0.748 | 187 |
| FoNE F2 | 3 | -0.453 | -0.967 | 216 |
| FoNE F2 | 4 | -0.287 | -0.310 | 320 |

## Mean ρ per arm

| arm | mean ρ (full) | mean ρ (rounding-only) |
|-----|------:|------:|
| bit | +0.090 | +0.026 |
| SEM | +nan | +nan |
| FoNE F1 | +0.237 | -0.048 |
| FoNE F2 | -0.516 | -0.656 |