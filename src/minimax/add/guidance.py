"""Regret-guided DDIM sampling for ADD."""

from typing import Callable

import jax
import jax.numpy as jnp

from minimax.add.diffusion import (
    DiffusionSchedule,
    _make_ddim_timesteps,
    diffusion_to_theta,
)
from minimax.add.critic import categorical_mean


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

        # Path-length gradient w.r.t. x_t (critic params frozen).
        def path_length_sum(x_t):
            logits = critic_model_fn(critic_params, x_t, t_batch)
            return categorical_mean(logits, num_bins, min_return, max_return).sum()

        grad_path_length = jax.grad(path_length_sum)(x)

        # Classifier-guidance: shift cleaned eps by the path-length gradient.
        eps_guided = eps_clean - sqrt_1m_ab_t * omega * grad_path_length

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
