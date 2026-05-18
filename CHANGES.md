# ADD: PPO-Value GAE Guidance — Change Log

## Summary

Replaced the categorical `EnvCritic` guidance mechanism with a fully
differentiable PPO-value GAE signal.  The core idea: instead of training a
separate 128-channel UNet encoder to predict return distributions, we reuse
the live PPO value head and compute a differentiable approximation of the
PLR `L1_VALUE_LOSS` score (mean |GAE advantage|) directly from the predicted
clean level `x0_pred`.

This eliminates the replay-buffer warmup lag, closes the representation gap
between the guidance critic and the actual agent, and aligns the guidance
objective with PLR/ACCEL's established difficulty metric.

---

## Files Changed

### `src/minimax/add/theta.py`

**Added two new functions at the bottom of the file.**

#### `theta_to_soft_maze_map(theta, sharpness=20.0)`

Converts a theta image `(16, 16, 3)` into a soft, differentiable padded maze
map `(21, 21, 3)` that matches the encoding produced by `make_maze_map` with
`pad_obs=True` and `normalize_obs=True`.

Why: The PPO model was trained on observations extracted from the integer
maze_map.  To make the value function differentiable w.r.t. theta (the level
image), we need a soft version of this map where every operation is smooth.

Key design choices:
- Channel 0 (walls): taken directly from `theta[..., 0]`.
- Channel 1 (agent): soft-thresholded with sigmoid at 0.75 (separating agent
  marker at 1.0 from direction marker at 0.5).  Agent tile omitted from the
  final obs (see_agent=False replaces it with empty in get_obs()).
- Channel 2 (goal): taken directly from `theta[..., 2]`.
- Each pixel is a weighted soft mixture of wall/empty/goal tile values,
  normalized by /10 to match `normalize_obs=True`.
- Padded to (21, 21, 3) with wall tiles, matching the `padding=4` layout
  used by the maze env's `make_maze_map(pad_obs=True)`.

#### `soft_extract_obs(soft_padded_map, agent_pos_xy, agent_dir_idx, view_size=5)`

Extracts a differentiable 5×5 egocentric observation from the soft padded
maze map, replicating the exact window selection and rotation from `get_obs()`
in `maze.py`.

Why: `jax.lax.dynamic_slice` is differentiable through *values* (not indices),
so slicing the soft maze map at cached trajectory positions propagates gradients
through the map values back to theta.

Key design choice — `_OBS_ROW_OFFSETS / _OBS_COL_OFFSETS`:
  These are per-direction start offsets (relative to the agent's padded
  position) derived by tracing `get_obs()`'s bounds computation for each of
  the four directions.  They allow a single `dynamic_slice` call that produces
  exactly the same 5×5 window as the real observation.

---

### `src/minimax/add/guidance.py`

**Added `ppo_value_guided_ddim_sample_theta()`.**

Added imports for `theta_to_soft_maze_map` and `soft_extract_obs` from theta.py.

#### `ppo_value_guided_ddim_sample_theta(...)`

New primary guidance function.  At each DDIM step:

1. Compute `x0_pred` from `x_t` and `eps_pred` (clamped, same as before).
2. Inside `gae_score_sum(x0)` (differentiated w.r.t. `x0`):
   a. Convert x0 → theta via `diffusion_to_theta` (linear, differentiable).
   b. Build soft padded maze maps via `theta_to_soft_maze_map` (differentiable).
   c. Extract egocentric observations at all `(T_sub+1, B)` cached positions
      using `soft_extract_obs` (differentiable through map values).
   d. Flatten to `(1, N, 5, 5, 3)` (T=1, B=N=(T_sub+1)*B) and call the PPO
      model once with zero LSTM carry (zero-carry approximation: each cached
      position is evaluated as episode start).
   e. Extract soft wall/goal probs at next positions via `dynamic_slice`.
   f. Compute soft TD residuals with soft next-state value (wall-bounce
      handling: if wall blocks movement, agent stays in place).
   g. Run GAE backward scan per level (fully differentiable).
   h. Return `sum(mean(sqrt(A_t^2 + 1e-6)))` — smooth L1 of advantages.
3. `grad_score = jax.grad(gae_score_sum)(x0_pred)`.
4. Guide: `eps_guided = eps_clean - sqrt(1-ab_t) * omega * grad_score`.
5. Compute guided `x0_guided` and `eps_final` for DDIM update.

**Guidance objective rationale (vs. EnvCritic):**
- PLR's `L1_VALUE_LOSS = mean(|A_t^GAE|)` is the standard difficulty metric
  used by PLR and ACCEL.  Full magnitude (not just positive part) captures
  both overestimation and underestimation of level difficulty.
- Differentiating through the PPO value head (rather than a separate critic)
  means guidance adapts immediately to the agent's current capability without
  replay-buffer lag.
- No new NN is trained; no warmup buffer required.

**Zero-carry approximation:** The LSTM carry is zeroed for all evaluations,
treating each cached position as if it were episode start.  This is a
necessary approximation to avoid re-running the full LSTM sequence during
guidance.  In practice, the early-episode value estimate is the dominant
signal for whether a level is challenging.

---

### `src/minimax/add/runner.py`

**Replaced `EnvCritic`-based sampling with PPO-value guidance.**

#### Removed
- `from minimax.add.critic import EnvCritic, batch_rollout_to_targets`
- `self.critic_model = EnvCritic()` in `__init__`
- `init_critic_params()` method
- `alpha` parameter (regret CVaR threshold, only used by EnvCritic)

#### Added: `_rollout_students_collect_states()`

A modified version of `DRRunner._rollout_students()` that also stacks per-step
agent positions and directions via the scan's `ys` output:
```
stacked_pos: (T, n_students, n_parallel, 2)  — agent_pos before each action
stacked_dir: (T, n_students, n_parallel)      — agent_dir_idx before each action
```

Why collect *before* the transition: the observation used to compute the value
at step t is from the state *before* the action, so the guidance should
evaluate V(s_t) at that position.

#### Modified: `sample_levels()`

Signature changed from `(rng, critic_params, omega)` to
`(rng, ppo_params_s0, omega, cached_rollout)`.

- `ppo_params_s0`: student-0 PPO params (student vmap dim stripped via
  `jax.tree.map(lambda p: p[0], train_state.params)`).
- `cached_rollout`: dict with keys `"pos"`, `"dir"`, `"done"` from `_run_rl`.

#### Modified: `_run_rl()`

After the rollout, sub-samples `_T_SUB=32` evenly-spaced steps from the
256-step trajectory:
```python
cached_pos  (T_sub+1, n_parallel, 2) — positions including the final next-pos
cached_dir  (T_sub+1, n_parallel)
cached_done (T_sub, n_parallel)
```
Returns these in `stats["_cached_pos/_dir/_done"]` for the training loop to
cache and pass back on the next iteration.

The `_targets` / `_n_episodes` stats (two-hot return targets for CriticBuffer)
are removed.  `_mean_return` is recomputed from rewards directly.

#### Modified: `run()`

Signature changed: `(critic_params, omega)` → `(cached_rollout, omega)`.
PPO params are extracted from `train_state` at call time.

---

### `src/minimax/add/train.py`

**Removed all EnvCritic infrastructure; added rollout cache management.**

#### Removed
- `from minimax.add.critic import CriticBuffer, make_critic_train_step, train_critic`
- `from minimax.add.critic import EnvCritic` (via runner)
- `critic_params`, `critic_optimizer`, `critic_opt_state`, `critic_buffer`
- `critic_step`, `train_critic()` call in the training loop
- `--critic_lr`, `--critic_weight_decay`, `--critic_grad_clip`,
  `--critic_buffer_size`, `--critic_train_iters`, `--critic_batch_size` CLI args
- `--alpha` arg (regret CVaR)

#### Added
- `_make_zero_cached_rollout()`: creates a zero-filled `cached_rollout` dict
  for the first iteration (where omega=0 so guidance is inactive).
- `cached_rollout` dict, updated each iteration from `stats["_cached_pos/_dir/_done"]`.
- `omega = args.omega if tick > 0 else 0.0`: guidance is disabled on the first
  iteration (no trajectory cache available).

#### Changed
- `save_checkpoint` no longer stores critic params/opt_state.
- Logging: removed `critic_loss` line; kept `mean_return` and `guided/unguided`.

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|----------|-----------|
| Zero LSTM carry | Avoids re-running the full LSTM sequence during guidance. Approximates V(s_0) at each cached position. The first few steps of an episode dominate the difficulty signal. |
| T_sub=32 | 32 positions × 32 levels × 50 DDIM steps = 51,200 PPO forward passes per training iteration. Each pass is one CNN + LSTM step (no scan), so throughput is acceptable on TPU/GPU. |
| Gradient on x0_pred, not x_t | Avoids backprop through the UNet. Keeps compilation time manageable and is standard classifier guidance practice. |
| Full GAE magnitude | PLR's `L1_VALUE_LOSS = mean(|A_t^GAE|)` — full magnitude captures both overestimation and underestimation. `POSITIVE_VALUE_LOSS` only captures one direction. |
| Soft next-state handling | When the sampled next position has high wall probability, the agent likely stayed in place. `V_next_soft = (1-p_wall)*V(pos_{t+1}) + p_wall*V(pos_t)` interpolates smoothly. |
| No structural validity term | The GAE signal naturally selects levels where the agent's value function is miscalibrated, which already encourages navigable structure. Adding a separate path-length term can be done later. |

---

## What Was NOT Changed

- `src/minimax/add/critic.py`: kept as-is for reference / ablation.
- `src/minimax/add/unet.py`: no changes.
- `src/minimax/add/diffusion.py`: no changes.
- `src/minimax/envs/`: no changes to the actual RL environment.
- `run_add_seeds.sh`: the `--alpha` and `--critic_*` flags now have no effect
  (argparse will silently accept them for backward compatibility of shell scripts).
  You should remove them from the script for clarity.
