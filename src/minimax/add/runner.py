"""ADDRunner: DRRunner that generates levels via a pretrained diffusion model.

Guidance mode: PPO-value GAE guidance (replaces the original EnvCritic).
  - sample_levels() accepts a cached_rollout dict from the previous iteration.
  - _run_rl() sub-samples agent positions/directions/dones from the rollout and
    returns them in stats["_cached_pos/_dir/_done"] for the next iteration.
  - omega=0 disables guidance (unguided DDIM); used for the first iteration
    before any rollout cache is available.

Changes from the original EnvCritic version:
  - Removed: EnvCritic, init_critic_params, critic-related imports
  - Added:   _rollout_students_collect_states (collects per-step positions)
  - Modified: sample_levels signature (cached_rollout replaces critic_params)
  - Modified: _run_rl returns _cached_pos/_dir/_done instead of _targets
  - Modified: run() forwards cached_rollout to sample_levels
"""

import pickle
from functools import partial

import jax
import jax.numpy as jnp

from minimax.runners.dr_runner import DRRunner
from minimax.envs.maze.common import EnvInstance

from minimax.add.theta import decode_level
from minimax.add.unet import UNet
from minimax.add.diffusion import make_schedule
from minimax.add.guidance import ppo_value_guided_ddim_sample_theta

# Sub-sample this many trajectory steps for guidance (out of n_rollout_steps=256).
# Lower → faster DDIM compilation; 32 gives 32×32=1024 value evals per step.
_T_SUB = 32


class ADDRunner(DRRunner):
    def __init__(
        self,
        *,
        diffusion_ckpt_path: str,
        ddim_steps: int = 50,
        unet_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.diff_model = UNet(**(unet_kwargs or {}))
        self.schedule = make_schedule()
        self.ddim_steps = ddim_steps

        with open(diffusion_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.diff_params = jax.device_put(ckpt["ema_params"])

    # ------------------------------------------------------------------
    # Rollout helpers
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _rollout_students_collect_states(
        self,
        rng,
        train_state,
        state,
        start_state,
        obs,
        carry,
        done,
        reset_state=None,
        extra=None,
        ep_stats=None,
    ):
        """Like DRRunner._rollout_students but also stacks per-step agent states.

        Returns everything _rollout_students returns, plus:
          stacked_pos: (n_rollout_steps, n_students, n_parallel, 2) — agent_pos
          stacked_dir: (n_rollout_steps, n_students, n_parallel)    — agent_dir_idx

        Positions are collected BEFORE each transition (i.e. the position from
        which the action was taken), so stacked_pos[t] is the state at step t.
        """
        rollout = self.student_rollout.reset()
        rngs = jax.random.split(rng, self.n_rollout_steps)

        def _scan_rollout(scan_carry, rng_step):
            rollout, state, start_state, obs, carry, done, extra, ep_stats, train_state = scan_carry

            # Capture current-step agent state before the transition.
            cur_pos = state.agent_pos       # (n_students, n_parallel, 2) [col, row]
            cur_dir = state.agent_dir_idx   # (n_students, n_parallel)

            next_scan_carry = self._get_transition(
                rng_step,
                self.student_pop,
                jax.lax.stop_gradient(train_state.params),
                rollout,
                state,
                start_state,
                obs,
                carry,
                done,
                reset_state,
                extra,
            )
            (rollout, next_state, next_start_state,
             next_obs, next_carry, done, info, extra) = next_scan_carry

            ep_stats = self._update_ep_stats(ep_stats, done, info)

            return (
                rollout, next_state, next_start_state, next_obs, next_carry,
                done, extra, ep_stats, train_state,
            ), (cur_pos, cur_dir)

        (rollout, state, start_state, obs, carry, done, extra, ep_stats,
         train_state), (stacked_pos, stacked_dir) = jax.lax.scan(
            _scan_rollout,
            (rollout, state, start_state, obs, carry, done, extra, ep_stats, train_state),
            rngs,
            length=self.n_rollout_steps,
        )
        # stacked_pos: (T, n_students, n_parallel, 2)
        # stacked_dir: (T, n_students, n_parallel)

        return rollout, state, start_state, obs, carry, extra, ep_stats, train_state, stacked_pos, stacked_dir

    # ------------------------------------------------------------------
    # Level sampling (guided DDIM)
    # ------------------------------------------------------------------

    def _sample_thetas(self, rng, n_levels, ppo_params_s0, omega, cached_rollout):
        """PPO-value guided DDIM sample. omega=0 → unguided."""
        def diff_model_fn(params, x, t):
            return self.diff_model.apply(params, x, t)

        def ppo_apply_fn(params, obs, carry, reset):
            return self.student_pop.agent.model.apply(params, obs, carry, reset)

        return ppo_value_guided_ddim_sample_theta(
            diff_model_fn=diff_model_fn,
            diff_params=self.diff_params,
            ppo_apply_fn=ppo_apply_fn,
            ppo_params=ppo_params_s0,
            shape=(n_levels, 16, 16, 3),
            rng=rng,
            schedule=self.schedule,
            cached_pos=cached_rollout["pos"],    # (T_sub+1, B, 2)
            cached_dir=cached_rollout["dir"],    # (T_sub+1, B)
            cached_done=cached_rollout["done"],  # (T_sub, B)
            omega=omega,
            num_steps=self.ddim_steps,
        )

    def _decode_to_instances(self, thetas):
        wall_maps, agent_pos_rc, goal_pos_rc, agent_dirs = jax.vmap(decode_level)(thetas)
        agent_pos_xy = agent_pos_rc[:, ::-1].astype(jnp.uint32)
        goal_pos_xy = goal_pos_rc[:, ::-1].astype(jnp.uint32)
        return EnvInstance(
            agent_pos=agent_pos_xy,
            agent_dir_idx=agent_dirs.astype(jnp.uint8),
            goal_pos=goal_pos_xy,
            wall_map=wall_maps.astype(jnp.bool_),
        )

    def _reset_from_instances(self, rng, instances, n_parallel, n_eval):
        instances_repeated = jax.tree.map(
            lambda x: jnp.repeat(x, n_eval, axis=0), instances
        )
        return jax.vmap(self.benv.env.set_env_instance)(instances_repeated)

    @partial(jax.jit, static_argnums=(0,))
    def sample_levels(self, rng, ppo_params_s0, omega, cached_rollout):
        """Guided DDIM sampling, compiled separately from the RL rollout.

        ppo_params_s0: student-0 Flax params (vmap student dim stripped).
        cached_rollout: dict with keys "pos" (T_sub+1, B, 2), "dir" (T_sub+1, B),
                        "done" (T_sub, B) — output of _run_rl from the prior step.
        """
        rng, *diff_rngs = jax.random.split(rng, self.n_students + 1)

        def _sample(rng):
            thetas = self._sample_thetas(rng, self.n_parallel, ppo_params_s0, omega, cached_rollout)
            instances = self._decode_to_instances(thetas)
            return thetas, instances

        return jax.vmap(_sample)(jnp.array(diff_rngs))

    # ------------------------------------------------------------------
    # RL rollout + PPO update
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _run_rl(
        self,
        rng,
        train_state,
        state,
        start_state,
        obs,
        carry,
        extra,
        ep_stats,
        all_thetas,
        all_instances,
    ):
        """RL rollout + PPO on pre-sampled levels.

        Returns the standard runner_state plus stats that include:
          _cached_pos  (T_sub+1, n_parallel, 2) — for next iteration's guidance
          _cached_dir  (T_sub+1, n_parallel)
          _cached_done (T_sub, n_parallel)
          _thetas      (n_parallel, 16, 16, 3)  — for logging
          _mean_return float
        """
        rollout_batch_shape = (self.n_students, self.n_parallel * self.n_eval)

        obs, state, extra = jax.vmap(
            lambda inst: self._reset_from_instances(None, inst, self.n_parallel, self.n_eval)
        )(all_instances)

        ep_stats = self.rolling_stats.reset_stats(batch_shape=rollout_batch_shape)
        rollout_start_state = state

        done = jnp.zeros(rollout_batch_shape, dtype=jnp.bool_)
        reset_state = state

        rng, subrng = jax.random.split(rng)
        (rollout, state, start_state, obs, carry, extra, ep_stats,
         train_state, stacked_pos, stacked_dir) = self._rollout_students_collect_states(
            subrng, train_state, state, start_state, obs, carry, done,
            reset_state, extra, ep_stats,
        )
        # stacked_pos: (T, n_students, n_parallel, 2), stacked_dir: (T, n_students, n_parallel)
        # Use student-0's trajectory for guidance.
        pos_T  = stacked_pos[:, 0, :, :]   # (T, n_parallel, 2)
        dir_T  = stacked_dir[:, 0, :]      # (T, n_parallel)

        # Dones from rollout storage: shape (n_students, T, n_parallel * n_eval)
        # Slice student 0, first eval copy.
        dones_T = rollout["dones"][0, :, :self.n_parallel]  # (T, n_parallel)

        # Sub-sample T_sub evenly-spaced steps.
        T = self.n_rollout_steps
        sub_idx = jnp.linspace(0, T - 2, _T_SUB, dtype=jnp.int32)  # T-2 so idx+1 < T

        cached_pos = jnp.concatenate([
            pos_T[sub_idx],            # (T_sub, n_parallel, 2)
            pos_T[sub_idx[-1] + 1][None],  # (1, n_parallel, 2) — final next pos
        ], axis=0)   # (T_sub+1, n_parallel, 2)

        cached_dir = jnp.concatenate([
            dir_T[sub_idx],
            dir_T[sub_idx[-1] + 1][None],
        ], axis=0)   # (T_sub+1, n_parallel)

        cached_done = dones_T[sub_idx].astype(jnp.float32)   # (T_sub, n_parallel)

        train_batch = self.student_rollout.get_batch(
            rollout,
            self.student_pop.get_value(
                jax.lax.stop_gradient(train_state.params), obs, carry
            ),
        )

        rng, subrng = jax.random.split(rng)
        train_state, update_stats = self.student_pop.update(
            subrng, train_state, train_batch
        )

        if self.track_env_metrics:
            env_metrics = self.benv.get_env_metrics(rollout_start_state)
        else:
            env_metrics = None

        stats = self._compile_stats(update_stats, ep_stats, env_metrics)
        stats.update(dict(n_updates=train_state.n_updates[0]))

        # Logging payloads.
        stats["_thetas"] = all_thetas[0]

        # Mean episodic return from the rollout (student 0).
        rewards_s0 = rollout["rewards"][0]   # (T, n_parallel * n_eval)
        dones_s0   = rollout["dones"][0]     # (T, n_parallel * n_eval)
        ep_returns = (rewards_s0 * dones_s0).sum(axis=0)
        n_eps = dones_s0.sum(axis=0).clip(1)
        stats["_mean_return"] = (ep_returns / n_eps).mean()

        # Trajectory cache for next iteration's guidance.
        stats["_cached_pos"]  = cached_pos   # (T_sub+1, n_parallel, 2)
        stats["_cached_dir"]  = cached_dir   # (T_sub+1, n_parallel)
        stats["_cached_done"] = cached_done  # (T_sub, n_parallel)

        train_state = train_state.increment()

        return (
            stats,
            rng,
            train_state,
            state,
            start_state,
            obs,
            carry,
            extra,
            ep_stats,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        rng,
        train_state,
        state,
        start_state,
        obs,
        carry,
        extra,
        ep_stats,
        cached_rollout,   # replaces (critic_params, omega) from the old API
        omega,
    ):
        """Full ADD step: guided DDIM sampling then RL rollout + PPO.

        cached_rollout: dict{"pos", "dir", "done"} from the prior _run_rl call,
                        or a zero-filled placeholder on the first iteration.
        omega: 0.0 until the first rollout completes, then args.omega.
        """
        if self.n_devices > 1:
            rng = jax.random.fold_in(rng, jax.lax.axis_index("device"))

        # Extract student-0 params (strip vmap student dimension).
        ppo_params_s0 = jax.tree.map(lambda p: p[0], train_state.params)

        rng, sample_rng = jax.random.split(rng)
        all_thetas, all_instances = self.sample_levels(
            sample_rng, ppo_params_s0, omega, cached_rollout
        )

        result = self._run_rl(
            rng, train_state, state, start_state, obs, carry, extra, ep_stats,
            all_thetas, all_instances,
        )
        self.n_updates += 1
        return result
