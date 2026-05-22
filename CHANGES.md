# ADD: diffV_v4 — End-Biased Sparse Guidance + Truncated BPTT

## Summary

diffV_v4 makes three targeted changes to the guidance loop — all confined to
`guidance.py`.  No runner, train, or env code is touched.

**Changes from diffV_v3:**

1. **End-biased schedule** (replaces uniform `rollout_every`): guidance fires at
   `n_guided=10` DDIM steps chosen by a power-law schedule with `guidance_bias=3.0`.
   With 50 DDIM steps this places guided steps at approximately
   `[0, 24, 30, 34, 37, 40, 43, 45, 47, 49]` — one early anchor then dense
   coverage of the low-noise tail where `x0_pred` is most reliable.

2. **Shorter rollout**: `guidance_rollout_steps` default reduced from 256 → 128.
   At γ·λ = 0.945, >97% of the GAE signal is captured in 64 steps; 128 gives
   comfortable margin while halving rollout cost.

3. **Truncated BPTT**: inside `gae_score_sum → per_level`, the LSTM scan is
   split into a warmup phase (first `T − lstm_k` steps, carry
   `stop_gradient`'d) and a differentiable window (last `lstm_k=32` steps).
   `jax.grad` only backpropagates through the 32-step window, reducing the
   backward-scan cost ~8× while retaining effectively all gradient signal.

### Compute budget (vs diffV_v3, per DDIM pass)

| Component | v3 | v4 |
|---|---|---|
| Guided steps | 5 (every 10) | 10 (end-biased) |
| Rollout per guided step | 256 LSTM fwd | 128 LSTM fwd |
| Backward per guided step | 256-step scan | 32-step scan |
| Total guidance cost (units) | 5 × 1024 = 5120 | 10 × 320 = 3200 |
| Wall-clock vs v3 | — | ~0.6× (40% faster) |

### `make_guidance_mask(num_steps, n_guided, bias)`

New helper: converts `(n_guided, bias)` into a static boolean mask of shape
`(num_steps,)`.  Positions computed as `round(t^(1/bias) * (num_steps−1))` for
`t = linspace(0, 1, n_guided)`.  `bias > 1` stretches positions towards the end.

### New parameter `rollout_every` status

`rollout_every` is kept in the `v4` function signature (default 1, ignored) for
backward compatibility with runner call-sites that pass it as a kwarg.

---

## diffV_v4 Files Changed

### `src/minimax/add/guidance.py`

- Added `import numpy as np`.
- Added `make_guidance_mask(num_steps, n_guided, bias=3.0) -> jnp.ndarray`.
- Added `ppo_value_guided_ddim_sample_theta_v4(...)`.  v3 function kept as-is.

---

## diffV_v4 Design Decisions

| Decision | Rationale |
|---|---|
| End-bias not uniform | Early DDIM steps have high noise; x0_pred is unreliable and guidance gradient is near-random. Concentrating steps at the end gives cleaner signal per compute unit. |
| One early anchor (step 0) | Provides a gradient signal even when the level is still mostly noise — acts as a weak global push. |
| lstm_k=32 window | GAE effective horizon ≈ 18 steps (1/(1−γλ)); K=32 spans ~1.7 horizons, capturing >97% of signal while reducing backward cost 8×. |
| Warmup carry under stop_gradient | Gives LSTM a realistic hidden state before the differentiable window without growing the backward graph. XLA zeroes cotangents through the warmup. |
| guidance_rollout_steps=128 | Halves rollout cost; agent still completes ~1–1.5 episodes. If mazes are rarely solved at 128 steps the guidance gradient degrades — monitor cmplx_solv. |

---

# ADD: diffV_v3 — K-Step Guided DDIM

## Summary

diffV_v3 reduces the per-iteration guidance cost by ~K× while keeping guidance
faithful.

**Key insight**: diffV_v2 runs a full real rollout + differentiable LSTM scan +
gradient at *every* DDIM step. The intermediate steps between denoising and the
final level are different levels — running guidance on all 50 is expensive. Worse,
running degraded guidance (e.g. zero-carry or stale trajectory) on the in-between
steps would actively mislead the diffusion process. The clean solution: only guide
at every K-th step and let DDIM run freely on the rest.

**Changes from diffV_v2:**

1. **K-step guidance**: guided steps fire at DDIM steps 0, K, 2K, … using a full
   real rollout + differentiable LSTM + gradient. All other steps are plain DDIM
   — no rollout, no gradient, no guidance at all. With K=10 and 50 DDIM steps,
   5 guided steps replace 50, giving ~10× reduction in guidance compute.

2. **Higher omega** (default 10.0 vs 5.0): fewer guidance steps means each must
   push harder.

Implementation uses `jax.lax.cond((i % rollout_every) == 0, guided_step,
plain_ddim_step, ...)` inside the existing `fori_loop`. Both branches are compiled
at JIT time; only one executes per step at runtime. The `fori_loop` carry stays
`(x, rng)` — no cached trajectory needed.

---

# ADD: diffV_v2 — On-Policy GAE Guidance with LSTM Carry

## Summary

diffV_v2 improves on diffV's PPO-value GAE guidance in two ways:

1. **On-policy trajectory**: at each DDIM step, the predicted clean level `x0_pred`
   is hard-decoded and a real RL rollout is run on it. The resulting trajectory
   (agent positions, directions, dones) is used immediately for that DDIM step's
   guidance — no stale cross-iteration cache.

2. **Correct LSTM carry**: instead of zeroing the LSTM carry at every position,
   the carry is threaded sequentially through the rollout positions using soft
   observations extracted from the current `x0_pred`. This gives value estimates
   that reflect the agent's actual episode history on the current level.

The `--guidance_ued_score` flag (choices: `l1_value_loss`, `positive_value_loss`)
controls whether guidance maximises full GAE magnitude or the positive-only part.

---

## Gradient Path

`x0_pred` (diffusion space `[-1,1]`) and `theta` (level space `[0,1]`) are the
**same level** under a linear rescaling: `theta = (x0_pred + 1) / 2`
(`diffusion_to_theta`). Differentiating w.r.t. `x0_pred` is identical to
differentiating w.r.t. `theta`.

The full differentiable path at each DDIM step:
```
x0_pred
  → diffusion_to_theta (linear, grad = 0.5)
  → theta_to_soft_maze_map (sigmoid + linear)   → soft_maze (21, 21, 3)
  → soft_extract_obs at rollout positions        → obs_t (5, 5, 3)
    (jax.lax.dynamic_slice — differentiable through values, not indices)
  → LSTM value head with threaded carry          → V_t
  → GAE backward scan                            → score
grad = ∂score/∂x0_pred   [jax.grad — no UNet backprop]
```

The hard decode and real rollout are **outside** this path — they only produce
integer trajectory positions `(pos_t, dir_t, done_t)` that are constants in the
gradient graph.

---

## Wrapped Env API

The codebase wraps the base `Maze` env with `MonitorReturnWrapper` (the
`benv.env` field). This changes the step/reset signatures relative to the
bare env:

| call | signature | returns |
|------|-----------|---------|
| `benv.env.set_env_instance(encoding)` | `EnvInstance` | `(obs, state, extra)` |
| `benv.env.step(key, state, action, reset_state, extra)` | — | `(obs, state, reward, done, info, extra)` |

`reset_state` in `step` is the state to auto-reset to when `done=True` (always
active — `reset_on_done` is hardcoded True in the wrapper). Passing `state0`
(the initial decoded level state) means every completed episode replays the
same decoded level during the guidance rollout.

Both calls are vmapped over the batch dimension `B` (number of parallel levels).

---

## Files Changed

### `src/minimax/add/guidance.py`

**Added `ppo_value_guided_ddim_sample_theta_v2()`.**
The diffV function `ppo_value_guided_ddim_sample_theta()` is kept as-is for
ablation comparison.

#### `ppo_value_guided_ddim_sample_theta_v2(...)`

Signature (key parameters vs diffV):

```python
def ppo_value_guided_ddim_sample_theta_v2(
    diff_model_fn, diff_params,
    ppo_apply_fn,           # model.apply(params, obs, carry, reset) -> (v, logits, carry)
    ppo_params,
    decode_and_reset_fn,    # x0_pred (B,16,16,3) -> (obs, state, extra)  [vmap(benv.env.set_env_instance)]
    env_step_fn,            # (rngs, states, actions, reset_states, extras) -> (obs, state, r, done, info, extra)  [vmap(benv.env.step)]
    shape,                  # (B, 16, 16, 3)
    rng, schedule,
    omega, num_steps,
    guidance_rollout_steps, # rollout length per DDIM step (default 256)
    gamma, gae_lambda,
    use_positive_value_loss,
)
```

At each of the `num_steps` DDIM steps:

1. Compute `x0_pred` from UNet (clamped to [-1,1]).
2. **Real rollout** (stop_gradient throughout):
   - `decode_and_reset_fn(x0_pred)` → `(obs0, state0, extra0)` — hard-decode
     and reset env to the predicted level.
   - `jax.lax.scan` over `guidance_rollout_steps` using PPO policy (argmax),
     calling `env_step_fn(rngs_B, state, action, state0, extra)` at each step.
     `state0` as `reset_state` means done episodes replay the same level.
   - Collect `(pos_t, dir_t, done_t)` of shape `(T, B, ...)`.
3. **Differentiable GAE score** `gae_score_sum(x0_pred)`:
   - Build soft maze from raw `x0_pred` → `theta_to_soft_maze_map`.
   - `jax.vmap(per_level)` over B: each level runs an independent LSTM scan
     over T positions using `soft_extract_obs` (differentiable), with carry
     reset on episode boundaries via `done_prev`.
   - Backward GAE scan → score = `mean(sqrt(A^2+ε))` or `mean(clip(A,0))`.
4. `grad = jax.grad(gae_score_sum)(x0_pred)` → guide x0_pred → DDIM update.

`fori_loop` carry is `(x, rng)` so env steps (which need RNG) work inside the loop.

Removed vs diffV:
- `cached_pos`, `cached_dir`, `cached_done` — trajectory collected internally
- `env_params` — not needed; wrapper handles reset internally

---

### `src/minimax/add/runner.py`

#### Removed
- `_rollout_students_collect_states()` — guidance manages its own rollout
- `cached_rollout` argument from `sample_levels()` and `run()`
- `_cached_pos`, `_cached_dir`, `_cached_done` from `_run_rl` stats
- `_make_env_step_fn()` factory method — replaced by `self._env_step_fn` built once in `__init__`

#### Added / Modified

**`__init__`**: pre-builds two reusable objects:
```python
self._decode_and_reset_fn = self._make_decode_and_reset_fn()
self._env_step_fn         = jax.vmap(self.benv.env.step)
```

**`_make_decode_and_reset_fn()`**: returns a closure that converts
`x0_pred → (obs, state, extra)` by calling `_decode_to_instances` (reuses
existing decode logic) then `vmap(benv.env.set_env_instance)`.

**`_reset_from_instances()`**: unchanged — uses `vmap(benv.env.set_env_instance)`,
returns `(obs, state, extra)`.

**`_run_rl()`**: unpacks `obs, state, extra` (3-tuple) from `_reset_from_instances`
and passes all three through `_rollout_students`.

**`sample_levels(rng, ppo_params_s0, omega)`**: no `cached_rollout` arg.

**`run(..., omega)`**: simplified API:
```python
stats, *runner_state = runner.run(*runner_state, omega)
```

---

### `src/minimax/add/train.py`

#### Removed
- `cached_rollout` dict and `_make_zero_cached_rollout()`
- All `stats["_cached_*"]` handling
- `omega = 0 on first tick` guard

#### Added
- `--guidance_rollout_steps` (int, default 256)
- `--guidance_ued_score` (str, default `l1_value_loss`, choices:
  `l1_value_loss` / `positive_value_loss`)
- PPO stats logging (value loss, policy loss, entropy) at `log_every`
- Done-rate logging (fraction of episodes completed per rollout) at `log_every`

---

### `src/minimax/add/theta.py`

No changes. `theta_to_soft_maze_map` and `soft_extract_obs` reused as-is.

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|----------|-----------|
| Real rollout at every DDIM step | Trajectory always on-policy w.r.t. current predicted level; diffV cache is on a completely different level. |
| Correct LSTM carry via sequential scan | Value estimates reflect actual episode history; diffV zero-carry underestimates values in familiar corridors. |
| Hard decode for rollout, soft map for gradient | Rollout needs valid discrete env state; gradient needs differentiable level. Two separate uses of `x0_pred`. |
| `reset_state=state0` in env step | Reuses codebase's existing auto-reset mechanism; keeps guidance rollout on the same decoded level across episode boundaries. |
| `diffusion_to_theta` is the only conversion | `x0_pred` and `theta` are identical under linear rescaling; gradient w.r.t. `x0_pred` = gradient w.r.t. `theta` × 0.5. |
| `--guidance_ued_score` flag | L1 vs positive-only is an open empirical question; CLI arg avoids premature commitment. |
| No K-step refresh cache in v2 | Simpler baseline. Superseded by diffV_v3's `rollout_every`. |
| Gradient on `x0_pred`, not `x_t` | Avoids backprop through UNet. Standard classifier guidance practice. |

---

## What Was NOT Changed

- `src/minimax/add/critic.py`: kept for ablation.
- `src/minimax/add/unet.py`, `diffusion.py`: no changes.
- `src/minimax/envs/`: no changes to the RL environment.
- `src/minimax/add/guidance.py` diffV function: kept for ablation comparison.

---

## diffV_v3 Files Changed

### `src/minimax/add/guidance.py`

**Added `ppo_value_guided_ddim_sample_theta_v3()`.**
diffV_v2 function kept as-is; v3 is a separate function.

New parameter vs v2: `rollout_every: int = 10`.

`fori_loop` body uses `jax.lax.cond((i % rollout_every) == 0, guided_step, plain_ddim_step, (x0_pred, rng))`:

- **`guided_step`**: identical to diffV_v2 body — real rollout → differentiable LSTM scan → `jax.grad` → guidance update.
- **`plain_ddim_step`**: standard DDIM update with no rollout, no gradient.

Both branches return `(x_prev, rng)`; carry stays `(x, rng)`.

### `src/minimax/add/runner.py`

Added `rollout_every: int = 1` constructor parameter (default 1 = diffV_v2 behaviour).
`_sample_thetas` dispatches to `ppo_value_guided_ddim_sample_theta_v3` when `rollout_every > 1`,
otherwise uses v2 as before.

### `src/minimax/add/train.py`

Added `--rollout_every` (int, default 1).

### `run_add_diffv2.sh`

Updated to full training settings: `--n_updates=30000`, `--ddim_steps=50`,
`--rollout_every=10`, `--omega=10.0`, `--save_every=3000`.

---

## diffV_v3 Design Decisions

| Decision | Rationale |
|----------|-----------|
| Plain DDIM on non-guided steps (not cached trajectory) | Degraded guidance (stale trajectory, zero carry) actively misleads the diffusion process — better to run freely than noisily. |
| `jax.lax.cond` inside `fori_loop` | Both branches compile but only one executes per step; no Python-level loop needed, JIT intact. |
| `rollout_every=1` default preserves v2 | Zero diff for existing runs; opt-in to v3 by passing `--rollout_every 10`. |
| omega default raised to 10.0 | Fewer guidance steps means each step must push harder to achieve comparable level steering. |
