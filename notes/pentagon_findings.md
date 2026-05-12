# Pentagon inspection findings — regime 'cancellation', n=64

## Vertices

| label | path | description | L | E | params |
|-------|------|-------------|--:|--:|-------:|
| V1 | `runs/sweep-L4-E044-s0/checkpoint_005000.pt` | early borderline (iter 5k) | 4 | 44 | 95,568 |
| V2 | `runs/sweep-L4-E044-s0/checkpoint_010000.pt` | mid borderline (iter 10k) | 4 | 44 | 95,568 |
| V3 | `runs/sweep-L4-E044-s0/checkpoint_020000.pt` | late borderline (iter 20k) | 4 | 44 | 95,568 |
| V4 | `runs/sweep-L4-E128-s0/checkpoint_020000.pt` | saturated (L4-E128, iter 20k) | 4 | 128 | 794,112 |
| V5 | `runs/sweep-L1-E128-s0/checkpoint_020000.pt` | depth-capped (L1-E128, iter 20k) | 1 | 128 | 203,520 |

## Per-bit commutativity  f(a, b) == f(b, a)

| bit  | V1 | V2 | V3 | V4 | V5 |
|------|------|------|------|------|------|
| sign | 0.750 | 0.953 | 1.000 | 1.000 | 1.000 |
| e3   | 0.875 | 1.000 | 0.953 | 1.000 | 0.953 |
| e2   | 0.969 | 0.922 | 1.000 | 1.000 | 1.000 |
| e1   | 0.875 | 0.859 | 0.953 | 1.000 | 0.984 |
| e0   | 0.734 | 0.891 | 1.000 | 1.000 | 0.984 |
| m2   | 0.859 | 0.984 | 1.000 | 1.000 | 0.984 |
| m1   | 0.969 | 0.984 | 1.000 | 1.000 | 1.000 |
| m0   | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Full-result agreement (all 8 bits):
| metric    | V1 | V2 | V3 | V4 | V5 |
|-----------|------|------|------|------|------|
| full-comm | 0.328 | 0.641 | 0.906 | 1.000 | 0.906 |

## Park Thm 5 — operand-a/b residual cosine sim at final layer (doubling inputs)

| bit  | V1 | V2 | V3 | V4 | V5 |
|------|------|------|------|------|------|
| sign | +0.97 | +0.98 | +0.99 | +0.99 | +0.99 |
| e3   | +0.97 | +0.98 | +0.99 | +0.99 | +0.98 |
| e2   | +0.98 | +0.99 | +0.99 | +0.99 | +0.99 |
| e1   | +0.96 | +0.97 | +0.99 | +0.99 | +0.96 |
| e0   | +0.98 | +0.98 | +0.99 | +0.98 | +0.99 |
| m2   | +0.99 | +0.99 | +0.99 | +0.99 | +1.00 |
| m1   | +0.99 | +0.99 | +0.99 | +0.98 | +1.00 |
| m0   | +0.41 | +0.58 | +0.39 | +0.24 | -0.03 |

## Direct logit attribution by block (mean over batch, sum over 8 result bits)

| vertex | embed | blk0 | blk1 | blk2 | blk3 |   final |
|--------|------|------|------|------|------|---------|
| V1     | -1.6 | +10.1 | +7.5 | +7.9 | +36.6 | +60.4  |
| V2     | -0.6 | +10.2 | +5.4 | +7.7 | +44.7 | +67.3  |
| V3     | -0.1 | +9.5 | +6.5 | +10.4 | +48.0 | +74.3  |
| V4     | +0.3 | +21.9 | +3.5 | +4.2 | +50.6 | +80.5  |
| V5     | +14.8 | +97.3 |    —   |    —   |    —   | +112.1  |

## Final-layer attention sharpness (max weight at result-predictor positions)

| vertex | n_layer | mean_max_attn |  uniform_floor |
|--------|--------:|--------------:|---------------:|
| V1     |       4 |         0.296 |          0.037 |
| V2     |       4 |         0.283 |          0.037 |
| V3     |       4 |         0.265 |          0.037 |
| V4     |       4 |         0.306 |          0.037 |
| V5     |       1 |         0.150 |          0.037 |


Per-vertex full dumps: `/home/adam/Desktop/science/ErrorBoard/runs/pentagon/V*_full.txt`
Raw numpy arrays:      `/home/adam/Desktop/science/ErrorBoard/runs/pentagon/V*.npz`