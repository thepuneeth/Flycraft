# Frozen VNC throughput report

CPU-only throughput plumbing. No Stage 2 training started.

- Device: `cpu`; PyTorch `2.13.0`; torch threads: `4`.
- Core: `n=11000`, `nnz=485074`, density `0.00401`; dense fp32 estimate `484.0 MB`.
- Chosen batch: **B=128**; smallest measured end-to-end batch within 90% of best throughput (`1741.25` env steps/s; best `1934.72`).

## CPU sparse matmul

| B | total ms | per-env ms |
|---:|---:|---:|
| 1 | 2.543 | 2.5432 |
| 8 | 2.890 | 0.3613 |
| 32 | 3.557 | 0.1111 |
| 128 | 8.517 | 0.0665 |
| 512 | 30.182 | 0.0589 |

## End-to-end vector environment + batched policy

| B | batch steps/s | environment steps/s | wall s |
|---:|---:|---:|---:|
| 1 | 93.97 | 93.97 | 0.319 |
| 8 | 69.46 | 555.64 | 0.432 |
| 32 | 43.44 | 1390.08 | 0.691 |
| 128 | 14.52 | 1858.76 | 2.066 |
| 512 | 3.78 | 1934.72 | 7.939 |

## Required checks

- Batched-vs-single core max abs difference: `0`; backward max abs difference: `0`; tolerance: `1e-05`.
- Vector auto-reset probe: done mask `[True, False]`; reset column max abs `0.0`; nonterminal column unchanged: `True`.
- Frozen invariants: W hash pre/post equal `True`; `W.grad is None` `True`; W parameter `False`; command map requires grad `False`; all pass `True`.
- Three arms: `connectome, random, rewired`; same B/device/budget assertion: `True` (`128`, `cpu`, `5,000,000` steps/arm).
- MPS: refused by `FrozenVNC`; no dense fallback. CUDA not benchmarked on this host.

## Projected wall clock per arm

Using selected B and measured end-to-end throughput; projections are arithmetic, not runs.

| budget | hours/arm |
|---:|---:|
| 5,000,000 | 0.747 h |
| 50,000,000 | 7.472 h |

Stage 2 remains owner-gated and unrun.
