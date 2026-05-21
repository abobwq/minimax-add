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

## Files Changed

### `src/minimax/add/guidance.py`

**Added `ppo_value_guided_ddim_sample_theta_v2()`.**
The diffV function `ppo_value_guided_ddim_sample_theta()` is kept as-is for
ablation comparison.

#### `ppo_value_guided_ddim_sample_theta_v2(...)`

At each of the 50 DDIM steps:

1. Compute `x0_pred` from UNet (clamped, same as diffV).
2. **Real rollout** (outside grad scope, `stop_gradient` throughout):
   - Hard-decode `x0_pred` → `EnvInstance` via `vmap(decode_level)`
   - Reset env to decoded instances via `env_set_instance_fn`
   - Run `guidance_rollout_steps` steps via `jax.lax.scan` using PPO policy
     (argmax actions) → collect `(pos_t, dir_t, done_t)` shape `(T, B, ...)`
3. **Differentiable GAE score** `gae_score_sum(x0_pred)`:
   - Build soft maze: `diffusion_to_theta(x0_pred)` → `theta_to_soft_maze_map`
   - Thread LSTM carry via `jax.lax.scan` over T rollout positions:
     ```
     carry_0 = zeros
     for t in range(T):
         obs_t = soft_extract_obs(soft_maze, pos_t, dir_t)       [differentiable]
         V_t, carry_{t+1} = LSTM(carry_t, obs_t, reset=done_{t-1}) [differentiable]
     ```
     `ScannedRNN` zeros carry internally when `reset=True` — same mechanism as
     the normal RL rollout outside the diffusion loop.
   - Compute soft GAE with `V_t`, `r_soft_t` (goal channel at next pos), `done_t`
   - Score = `mean(sqrt(A_t^2 + 1e-6))` (L1) or `mean(clip(A_t, 0))` (positive)
4. `grad = jax.grad(gae_score_sum)(x0_pred)` → guide x0_pred → DDIM update.

New parameters vs diffV:
- `env_set_instance_fn`: `vmap(env.set_env_instance)` — resets env to decoded level
- `env_step_fn`: `vmap(env.step)` — advances env one step
- `env_params`: passed to env step
- `guidance_rollout_steps`: length of guidance rollout (default 256)
- `use_positive_value_loss`: False → L1, True → positive-only (default False)

Removed parameters vs diffV:
- `cached_pos`, `cached_dir`, `cached_done` — no longer passed in externally

---

### `src/minimax/add/runner.py`

#### Removed
- `_rollout_students_collect_states()` — no longer needed; guidance manages its
  own rollout internally
- `cached_rollout` argument from `sample_levels()` and `run()`
- `_cached_pos`, `_cached_dir`, `_cached_done` from `_run_rl` stats

#### Modified: `_sample_thetas()`
Calls `ppo_value_guided_ddim_sample_theta_v2`, passing:
- `env_set_instance_fn = jax.vmap(self.benv.env.set_env_instance)`
- `env_step_fn = jax.vmap(self.benv.env.step)`
- `env_params = self.benv.env.params`
- `guidance_rollout_steps` from constructor arg

#### Modified: `sample_levels(rng, ppo_params_s0, omega)`
Removed `cached_rollout` argument.

#### Modified: `run(..., omega)`
Removed `cached_rollout` argument. API is now simpler:
```python
stats, *runner_state = runner.run(*runner_state, omega)
```

#### Modified: `_run_rl()`
Reverts to calling `_rollout_students` (standard DRRunner method).
`_mean_return` computation unchanged.

---

### `src/minimax/add/train.py`

#### Removed
- `cached_rollout` dict and `_make_zero_cached_rollout()`
- All `stats["_cached_*"]` handling
- `omega = 0 on first tick` guard (guidance no longer needs a warm-up cache)

#### Added
- `--guidance_rollout_steps` (int, default 256): rollout length inside guidance
- `--guidance_ued_score` (str, default `l1_value_loss`, choices:
  `l1_value_loss` / `positive_value_loss`): controls GAE score variant

#### Changed
- `runner.run(*runner_state, omega)` — no `cached_rollout` arg
- New args passed into `ADDRunner.__init__`

---

### `src/minimax/add/theta.py`

No changes. `theta_to_soft_maze_map` and `soft_extract_obs` from diffV are
reused as-is.

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|----------|-----------|
| Real rollout at every DDIM step | Trajectory is always on-policy w.r.t. the current predicted level. diffV's stale cross-iteration cache is on a completely different level. |
| Correct LSTM carry via sequential scan | Value estimates reflect actual episode history. diffV's zero-carry approximation underestimates values in familiar corridors. |
| Hard decode for rollout, soft map for gradient | Rollout needs a valid discrete env state. Gradient needs a differentiable level representation. These are two separate uses of `x0_pred`. |
| `diffusion_to_theta` is the only conversion | `x0_pred` and `theta` are the same level under linear rescaling `(x+1)/2`. Gradient w.r.t. `x0_pred` equals gradient w.r.t. `theta` scaled by 0.5. |
| `--guidance_ued_score` flag | L1 vs positive-only value loss is an open empirical question; keeping it as a CLI arg avoids premature commitment. |
| No K-step refresh cache | Simpler to implement and debug. Can add K-step caching later if the per-step rollout cost is prohibitive. |
| Gradient on x0_pred, not x_t | Avoids backprop through UNet. Standard classifier guidance practice. |

---

## What Was NOT Changed

- `src/minimax/add/critic.py`: kept for reference / ablation.
- `src/minimax/add/unet.py`, `diffusion.py`: no changes.
- `src/minimax/envs/`: no changes to the RL environment.
- `src/minimax/add/guidance.py` diffV function: kept for ablation comparison.
