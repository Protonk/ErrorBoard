# Failure-pair overlap (cross-vertex + cross-seed)

Jaccard similarity of failure-pair sets across checkpoints. 1.0 = identical
failure sets; 0.0 = disjoint failure sets.

## Failure counts

| checkpoint | n_failures |
|------------|-----------:|
| V1 | 2324 |
| V2 | 1822 |
| V3 | 902 |
| V4 | 63 |
| V5 | 1021 |
| L4E044-s1 | 716 |
| L4E044-s2 | 542 |
| L4E044-s3 | 793 |
| L4E044-s4 | 588 |

## Cross-vertex overlap (Jaccard)

| | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|
| V1 | 1.000 | 0.519 | 0.308 | 0.019 | 0.339 |
| V2 | 0.519 | 1.000 | 0.324 | 0.022 | 0.360 |
| V3 | 0.308 | 0.324 | 1.000 | 0.027 | 0.340 |
| V4 | 0.019 | 0.022 | 0.027 | 1.000 | 0.027 |
| V5 | 0.339 | 0.360 | 0.340 | 0.027 | 1.000 |

## Cross-seed overlap, L4-E044 iter 20000 (Jaccard)

| | V3 | L4E044-s1 | L4E044-s2 | L4E044-s3 | L4E044-s4 |
|---|---:|---:|---:|---:|---:|
| V3 | 1.000 | 0.278 | 0.226 | 0.306 | 0.260 |
| L4E044-s1 | 0.278 | 1.000 | 0.188 | 0.253 | 0.196 |
| L4E044-s2 | 0.226 | 0.188 | 1.000 | 0.205 | 0.219 |
| L4E044-s3 | 0.306 | 0.253 | 0.205 | 1.000 | 0.234 |
| L4E044-s4 | 0.260 | 0.196 | 0.219 | 0.234 | 1.000 |

## All-vs-all (for direct comparison)

| | V1 | V2 | V3 | V4 | V5 | L4E044-s1 | L4E044-s2 | L4E044-s3 | L4E044-s4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | 1.000 | 0.519 | 0.308 | 0.019 | 0.339 | 0.231 | 0.185 | 0.258 | 0.202 |
| V2 | 0.519 | 1.000 | 0.324 | 0.022 | 0.360 | 0.249 | 0.211 | 0.273 | 0.225 |
| V3 | 0.308 | 0.324 | 1.000 | 0.027 | 0.340 | 0.278 | 0.226 | 0.306 | 0.260 |
| V4 | 0.019 | 0.022 | 0.027 | 1.000 | 0.027 | 0.032 | 0.047 | 0.026 | 0.024 |
| V5 | 0.339 | 0.360 | 0.340 | 0.027 | 1.000 | 0.254 | 0.250 | 0.298 | 0.249 |
| L4E044-s1 | 0.231 | 0.249 | 0.278 | 0.032 | 0.254 | 1.000 | 0.188 | 0.253 | 0.196 |
| L4E044-s2 | 0.185 | 0.211 | 0.226 | 0.047 | 0.250 | 0.188 | 1.000 | 0.205 | 0.219 |
| L4E044-s3 | 0.258 | 0.273 | 0.306 | 0.026 | 0.298 | 0.253 | 0.205 | 1.000 | 0.234 |
| L4E044-s4 | 0.202 | 0.225 | 0.260 | 0.024 | 0.249 | 0.196 | 0.219 | 0.234 | 1.000 |

## Reading

- **V3 vs V1, V2** (same model, training trajectory): high Jaccard reflects
  that V1's failure set shrinks toward V3's; V3 ⊂ V1, V2 mostly.
- **V3 vs L4E044-s{1..4}** (same architecture, different seeds): tests whether
  the failure set is *seed-determined* or *architecture-determined*.
  High Jaccard ⟹ architecture wins; low Jaccard ⟹ lottery.
- **V3 vs V4** (deeper-wider): V3's failures should mostly be a superset of V4's.
- **V3 vs V5** (depth-1 vs depth-4): if low Jaccard, depth produces qualitatively
  different failure patterns.
- **L4E044 seed pairs**: matrix off-diagonal entries say whether different seeds
  converge on the same failures.
