# MPS verdict + the throughput task (agent instructions)

**Short answer on MPS: do not use it.** Measured on this machine, MPS is
slower than CPU for our core at every batch size tested. The real win is
**batching the core on CPU**, which needs no GPU at all. Numbers below are
measured, not estimated — reproduce them before trusting them.

---

## 1. What was measured

Core: `n=11000`, `nnz=485074`, density `0.00401`. Dense float32 would be
`484 MB`.

Single matvec:

| path | time |
|---|---|
| CPU sparse CSR | **2.563 ms** |
| CPU dense | 7.893 ms |
| MPS dense | 5.864 ms |
| MPS dense + per-step CPU round trip | 6.531 ms |

Batched (`W @ V`, V is `n × B`), per-environment cost:

| B | CPU sparse | MPS dense | per-env CPU | per-env MPS |
|--:|--:|--:|--:|--:|
| 1 | 2.70 ms | 6.28 ms | 2.703 ms | 6.275 ms |
| 8 | 3.13 ms | 7.01 ms | 0.391 ms | 0.876 ms |
| 32 | 3.79 ms | 6.50 ms | 0.118 ms | 0.203 ms |
| 128 | 9.26 ms | 11.08 ms | **0.072 ms** | 0.087 ms |
| 512 | 36.82 ms | 55.35 ms | 0.072 ms | 0.108 ms |

**CPU sparse wins everywhere.** MPS never leads.

Two further facts that settle it:
- Sparse CSR cannot be constructed on MPS at all:
  `Could not run 'new_compressed_tensor' from the 'mps:0' device.`
  (pytorch/pytorch#129842). MPS would require the dense path.
- The dense path is not numerically identical: max abs difference between
  CPU-sparse and MPS-dense on the same input was `1.221e-03`. Switching
  backends mid-programme would change registered results.

`embody_policy.py` therefore refuses MPS at construction. **Leave that guard
in place.** Do not add a silent dense/MPS fallback.

---

## 2. The finding that actually matters

Per environment step, with the 4-command bottleneck policy:

- MuJoCo physics (`env.step`, incl. resets): **0.210 ms**
- Policy step (4 core substeps): **10.952 ms**
- **Core is 98.1% of the cost.** Physics is nearly free.
- Single environment: **90 steps/sec** → `5,000,000` steps is **15.5 h per
  arm**, **46.5 h for three arms**.

Batching the core to `B=128` drops per-env core cost from `2.703 ms` to
`0.072 ms` per matvec — about **37×**. Projected per-env step becomes roughly
`0.29 ms` core + `0.21 ms` physics ≈ `0.5 ms`, i.e. **~2000 steps/sec**, and
5M steps per arm falls from 15.5 h to well under an hour.

This is the fix for the compute-honesty problem in `EMBODY_3WAY_SPEC_V2.md`
§3.2. It removes the "the null might just be starvation" objection **without
waiting for CUDA**.

---

## 3. Your task

Build a **batched vectorised environment + batched core**, CPU only, so the
three arms can run at field-comparable sample counts.

### 3.1 Batch the core
- `FrozenVNC.forward` currently takes a single `node_state` of shape `(n,)`
  and a current of shape `(n,)`. Extend it to accept `(n, B)` and return
  `(n, B)`, keeping the single-vector path working unchanged.
- The sparse matmul already supports this: `torch.sparse.mm(W, V)` with V of
  shape `(n, B)`. `_FixedSparseMM` needs its backward checked for the batched
  case.
- Everything else in the rate equation is elementwise; broadcast the
  per-neuron buffers (`decay`, `dt_tau`, `threshold`, `fr_cap`,
  `activation_gain`) over the batch dimension.
- **Invariant:** batched output must match looping the single-vector path,
  to within float tolerance. Assert this in the self-check with an explicit
  tolerance, and report the observed max difference.

### 3.2 Batch the environment
- Wrap `B` independent `HexapodEnv` instances behind one object exposing
  `reset() -> (B, obs_dim)` and `step(actions: (B, 18)) -> (obs, reward, done)`.
- Auto-reset any environment that terminates, and make sure the corresponding
  column of `node_state` is reset with it. **Getting this wrong silently
  carries brain state across episode boundaries** — assert it explicitly.
- MuJoCo is CPU and cheap here (`0.210 ms`); a simple Python loop over the B
  instances is fine. Do not add multiprocessing before measuring — physics is
  1.9% of the cost.

### 3.3 Measure, then pick B
- Sweep `B` in `{1, 8, 32, 128, 512}` and report measured steps/sec end to end
  (not just the matmul). Pick the smallest B that reaches the throughput
  plateau; the matmul plateau is at `B=128`, but confirm end-to-end.
- Report projected wall-clock per arm at 5M steps, and at 50M.

### 3.4 Keep every registered invariant
- Frozen core: `core.W.grad is None`, not an `nn.Parameter`, hash equal
  pre/post. `command_map.requires_grad is False`.
- All three arms use the same B, same device, same budget.
- CPU or CUDA only. **No MPS. No silent dense fallback.**
- Do not change any constant in `EMBODY_3WAY_SPEC_V2.md` to make things fit.

---

## 4. When CUDA arrives

Re-run the same sweep on CUDA before committing to a device. CUDA sparse CSR
is supported, and the batched sparse matmul should scale further. Record the
device in the results JSON. Whichever device is chosen, **all three arms run
on it** — never mix.

---

## 5. Report back

1. Reproduced benchmark table (yours, not mine) — CPU sparse across B, and
   end-to-end steps/sec.
2. Chosen B, with the measured plateau that justifies it.
3. Batched-vs-single-path equivalence: max abs difference observed.
4. Proof the per-environment `node_state` reset is correct.
5. Projected wall-clock per arm at 5M and 50M steps.
6. Confirmation that all frozen invariants still hold.

Do not start Stage 2 training. This task is throughput plumbing only; the
owner checkpoint in `EMBODY_3WAY_SPEC_V2.md` §5.5 still stands. Do not commit.
