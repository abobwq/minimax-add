"""Guided DDIM sampling for ADD.

Two guidance modes:
  guided_ddim_sample_theta         — original EnvCritic regret guidance
  ppo_value_guided_ddim_sample_theta — new PPO-value GAE guidance (replaces
                                       EnvCritic; uses the live PPO value head
                                       and a differentiable soft maze map)
"""

from typing import Callable

import jax
import jax.numpy as jnp

from minimax.add.diffusion import (
    DiffusionSchedule,
    _make_ddim_timesteps,
    diffusion_to_theta,
)
from minimax.add.critic import regret
from minimax.add.theta import (
    theta_to_soft_maze_map,
    soft_extract_obs,
)


ModelFn = Callable  # (params, x_t, t_batch) -> output


def guided_ddim_sample(
    diff_model_fn: ModelFn,
    diff_params,
    critic_model_fn: ModelFn,
    critic_params,
    shape: tuple,
    rng: jax.Array,
    schedule: DiffusionSchedule,
    omega: float = 5.0,
    alpha: float = 0.15,
    num_steps: int = 50,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """Regret-guided DDIM sampling. Returns x_0 in diffusion space.

    Follows the paper's DDIM guidance flow: clamp x0 before guidance to get a
    consistent eps baseline, apply the regret gradient, then leave the guided
    x0 unclamped so guidance can push beyond [-1, 1]. Re-derive eps from the
    guided x0 to maintain algebraic consistency in the DDIM step.
    """
    T = schedule.betas.shape[0]
    timesteps = _make_ddim_timesteps(T, num_steps)

    alpha_bars_ext = jnp.concatenate([jnp.array([1.0]), schedule.alpha_bars])

    x = jax.random.normal(rng, shape)

    def body(i, x):
        t_cur = timesteps[i]
        t_prev = jnp.where(i < num_steps - 1, timesteps[i + 1], -1)

        ab_t = alpha_bars_ext[t_cur + 1]
        ab_prev = alpha_bars_ext[t_prev + 1]

        B = shape[0]
        t_batch = jnp.full((B,), t_cur, dtype=jnp.int32)

        sqrt_ab_t = jnp.sqrt(ab_t)
        sqrt_1m_ab_t = jnp.sqrt(1.0 - ab_t)

        # Unconditional noise prediction.
        eps_pred = diff_model_fn(diff_params, x, t_batch)

        # Predict x0 from raw eps, clamp to [-1, 1].
        x0_pred = (x - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t
        x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        # Re-derive eps from clamped x0 (consistent baseline for guidance).
        eps_clean = (x - sqrt_ab_t * x0_pred) / sqrt_1m_ab_t

        # Regret gradient w.r.t. x_t (critic params frozen).
        def regret_sum(x_t):
            logits = critic_model_fn(critic_params, x_t, t_batch)
            return regret(logits, alpha, num_bins, min_return, max_return).sum()

        grad_regret = jax.grad(regret_sum)(x)

        # Classifier-guidance: shift cleaned eps by the regret gradient.
        eps_guided = eps_clean - sqrt_1m_ab_t * omega * grad_regret

        # Predict guided x0 (NOT clamped — guidance may push beyond [-1, 1]).
        x0_guided = (x - sqrt_1m_ab_t * eps_guided) / sqrt_ab_t

        # Re-derive eps from guided x0 for algebraic consistency.
        eps_final = (x - sqrt_ab_t * x0_guided) / sqrt_1m_ab_t

        # DDIM deterministic update (eta=0).
        x_prev = (
            jnp.sqrt(ab_prev) * x0_guided
            + jnp.sqrt(1.0 - ab_prev) * eps_final
        )
        return x_prev

    x = jax.lax.fori_loop(0, num_steps, body, x)
    return x


def guided_ddim_sample_theta(
    diff_model_fn: ModelFn,
    diff_params,
    critic_model_fn: ModelFn,
    critic_params,
    shape: tuple,
    rng: jax.Array,
    schedule: DiffusionSchedule,
    omega: float = 5.0,
    alpha: float = 0.15,
    num_steps: int = 50,
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """Regret-guided DDIM sampling, returning result in [0, 1] (theta space)."""
    x = guided_ddim_sample(
        diff_model_fn=diff_model_fn,
        diff_params=diff_params,
        critic_model_fn=critic_model_fn,
        critic_params=critic_params,
        shape=shape,
        rng=rng,
        schedule=schedule,
        omega=omega,
        alpha=alpha,
        num_steps=num_steps,
        num_bins=num_bins,
        min_return=min_return,
        max_return=max_return,
    )
    return jnp.clip(diffusion_to_theta(x), 0.0, 1.0)


def ppo_value_guided_ddim_sample_theta(
    diff_model_fn: ModelFn,
    diff_params,
    ppo_apply_fn: ModelFn,          # Flax model.apply(params, obs, carry, reset)
    ppo_params,                     # student-0 Flax params (no vmap student dim)
    shape: tuple,                   # (B, 16, 16, 3)
    rng: jax.Array,
    schedule: DiffusionSchedule,
    cached_pos: jnp.ndarray,        # (T_sub+1, B, 2) inner (col, row) coords
    cached_dir: jnp.ndarray,        # (T_sub+1, B) int {0,1,2,3}
    cached_done: jnp.ndarray,       # (T_sub, B) float — 1 if episode ended
    view_size: int = 5,
    ppo_hidden_dim: int = 256,
    omega: float = 5.0,
    num_steps: int = 50,
    gamma: float = 0.995,
    gae_lambda: float = 0.95,
    # Kept for API signature parity with the critic variant (unused here):
    num_bins: int = 100,
    min_return: float = 0.0,
    max_return: float = 1.0,
) -> jnp.ndarray:
    """PPO-value GAE guided DDIM sampling, returning theta in [0, 1].

    At each DDIM step, instead of using a separate EnvCritic, we:
      1. Predict the clean level x0_pred from the current noisy x_t.
      2. Build a soft, differentiable maze map from x0_pred.
      3. Extract egocentric observations at trajectory positions cached from
         the previous RL rollout.
      4. Evaluate the live PPO value head (frozen params, zero LSTM carry) at
         each cached position.
      5. Compute a differentiable GAE score = mean(sqrt(A_t^2 + eps)) over
         the trajectory — the full-magnitude version of PLR's L1 value loss.
      6. Guide x0_pred toward high-GAE (hard) levels via the gradient.

    Cached trajectory positions come from the immediately preceding PPO
    rollout (supplied by ADDRunner._run_rl via stats["_cached_pos/dir/done"]).
    When no cache is available (first iteration), omega is set to 0 by the
    caller, so guidance is skipped and unguided DDIM is used.
    """
    T = schedule.betas.shape[0]
    timesteps = _make_ddim_timesteps(T, num_steps)

    alpha_bars_ext = jnp.concatenate([jnp.array([1.0]), schedule.alpha_bars])

    x = jax.random.normal(rng, shape)

    B        = shape[0]
    T_sub_p1 = cached_pos.shape[0]   # T_sub + 1
    T_sub    = T_sub_p1 - 1

    def body(i, x):
        t_cur  = timesteps[i]
        t_prev = jnp.where(i < num_steps - 1, timesteps[i + 1], -1)

        ab_t   = alpha_bars_ext[t_cur + 1]
        ab_prev = alpha_bars_ext[t_prev + 1]

        sqrt_ab_t    = jnp.sqrt(ab_t)
        sqrt_1m_ab_t = jnp.sqrt(1.0 - ab_t)

        t_batch  = jnp.full((B,), t_cur, dtype=jnp.int32)
        eps_pred = diff_model_fn(diff_params, x, t_batch)

        # Predicted clean level — clamped for the unguided baseline.
        x0_pred = (x - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t
        x0_pred = jnp.clip(x0_pred, -1.0, 1.0)

        # Re-derive eps from clamped x0 as a consistent baseline.
        eps_clean = (x - sqrt_ab_t * x0_pred) / sqrt_1m_ab_t

        # --- Differentiable GAE score ---
        def gae_score_sum(x0):
            # x0: (B, 16, 16, 3) in diffusion space [-1, 1]
            thetas = jax.vmap(diffusion_to_theta)(x0)   # (B, 16, 16, 3) in [0,1]

            # Build soft padded maze maps (differentiable via sigmoid + linear).
            soft_maps = jax.vmap(theta_to_soft_maze_map)(thetas)   # (B, 21, 21, 3)

            # Extract egocentric obs at all (T_sub+1) cached positions.
            # cached_pos: (T_sub+1, B, 2), swap to (B, T_sub+1, 2) for vmap.
            pos_BT = jnp.swapaxes(cached_pos, 0, 1)  # (B, T_sub+1, 2)
            dir_BT = jnp.swapaxes(cached_dir, 0, 1)  # (B, T_sub+1)

            def level_obs(soft_map, pos_seq, dir_seq):
                return jax.vmap(
                    lambda p, d: soft_extract_obs(soft_map, p, d, view_size)
                )(pos_seq, dir_seq)   # (T_sub+1, 5, 5, 3)

            obs_all = jax.vmap(level_obs)(soft_maps, pos_BT, dir_BT)
            # obs_all: (B, T_sub+1, view_size, view_size, 3)

            # Evaluate PPO value for all (T_sub+1)*B positions in one batch.
            # Flatten to (1, N, ...) with T=1 so each position uses zero carry.
            N          = B * T_sub_p1
            flat_imgs  = obs_all.reshape(1, N, view_size, view_size, 3)
            flat_dirs  = dir_BT.reshape(N)[None].astype(jnp.int32)   # (1, N)
            ppo_obs    = {"image": flat_imgs, "agent_dir": flat_dirs}
            zero_carry = (
                jnp.zeros((N, ppo_hidden_dim)),
                jnp.zeros((N, ppo_hidden_dim)),
            )
            zero_reset = jnp.zeros((1, N), dtype=jnp.bool_)

            v_raw, _, _ = ppo_apply_fn(ppo_params, ppo_obs, zero_carry, zero_reset)
            # v_raw: (1, N, 1) → (T_sub+1, B)
            values = v_raw[0, :, 0].reshape(B, T_sub_p1).T  # (T_sub+1, B)

            # Soft wall / goal probability at each next position.
            # next positions: cached_pos[1:] shape (T_sub, B, 2)
            pos_next_BT = jnp.swapaxes(cached_pos[1:], 0, 1)  # (B, T_sub, 2)

            def level_wg(theta_b, pos_next_seq):
                # theta_b: (16,16,3), pos_next_seq: (T_sub, 2) in (col, row)
                def single(pos_xy):
                    col = pos_xy[0].astype(jnp.int32)
                    row = pos_xy[1].astype(jnp.int32)
                    wall_p = jax.lax.dynamic_slice(
                        theta_b[:, :, 0], (row + 1, col + 1), (1, 1)
                    ).squeeze()
                    goal_p = jax.lax.dynamic_slice(
                        theta_b[:, :, 2], (row + 1, col + 1), (1, 1)
                    ).squeeze()
                    return wall_p, goal_p
                wps, gps = jax.vmap(single)(pos_next_seq)
                return wps, gps   # each (T_sub,)

            wall_BT, goal_BT = jax.vmap(level_wg)(thetas, pos_next_BT)
            # wall_BT, goal_BT: (B, T_sub) → transpose to (T_sub, B)
            p_wall = wall_BT.T
            r_soft  = goal_BT.T

            # Soft next-state value: if wall blocks movement, agent stays.
            V_cur      = values[:T_sub]   # (T_sub, B)
            V_next_raw = values[1:]       # (T_sub, B)
            V_next_soft = (1.0 - p_wall) * V_next_raw + p_wall * V_cur

            not_done = 1.0 - cached_done  # (T_sub, B)
            deltas   = r_soft + gamma * not_done * V_next_soft - V_cur  # (T_sub, B)

            # GAE backward scan, independently per level.
            def gae_per_level(delta_l, nd_l):
                # delta_l, nd_l: (T_sub,)
                def scan_fn(gae_next, td_nd):
                    d, nd = td_nd
                    gae = d + gamma * gae_lambda * nd * gae_next
                    return gae, gae

                _, adv_rev = jax.lax.scan(
                    scan_fn, jnp.float32(0.),
                    (delta_l[::-1], nd_l[::-1]),
                )
                advantages = adv_rev[::-1]   # (T_sub,)
                return jnp.mean(jnp.sqrt(advantages ** 2 + 1e-6))

            scores = jax.vmap(gae_per_level)(deltas.T, not_done.T)   # (B,)
            return scores.sum()

        grad_score = jax.grad(gae_score_sum)(x0_pred)

        # Classifier-guidance: shift cleaned eps by the GAE gradient,
        # same formulation as the EnvCritic variant.
        eps_guided = eps_clean - sqrt_1m_ab_t * omega * grad_score

        # Guided x0 (unclamped — guidance may push beyond [-1, 1]).
        x0_guided = (x - sqrt_1m_ab_t * eps_guided) / sqrt_ab_t

        # Re-derive eps from guided x0 for DDIM consistency.
        eps_final = (x - sqrt_ab_t * x0_guided) / sqrt_1m_ab_t

        x_prev = (
            jnp.sqrt(ab_prev) * x0_guided
            + jnp.sqrt(1.0 - ab_prev) * eps_final
        )
        return x_prev

    x = jax.lax.fori_loop(0, num_steps, body, x)
    return jnp.clip(diffusion_to_theta(x), 0.0, 1.0)
