"""ADDRunner: DRRunner that generates levels via a pretrained diffusion model.

Guidance mode: PPO-value GAE guidance v2 (on-policy, correct LSTM carry).
  - At each DDIM step, x0_pred is hard-decoded and a real rollout is run on the
    decoded level.  The trajectory positions are used to extract soft observations
    from the current x0_pred for differentiable LSTM-threaded GAE guidance.
  - omega=0 disables guidance (unguided DDIM).

Changes from diffV:
  - Removed: _rollout_students_collect_states, cached_rollout API
  - Added:   decode_and_reset_fn, env_step_fn passed into guidance
  - Uses:    ppo_value_guided_ddim_sample_theta_v2

Hybrid DR/diffusion sampling (dr_frac):
  - A fixed fraction (default 0.5) of levels per rollout are drawn from
    domain randomisation (random Maze reset, walls uniform in [0, dr_max_walls]).
  - The remaining fraction uses guided DDIM as normal.
  - The split is deterministic given the shape: n_dr = round(n_parallel*dr_frac),
    n_diff = n_parallel - n_dr.  Both branches run every tick; DR is cheap.
"""

import pickle
from functools import partial

import jax
import jax.numpy as jnp

from minimax.runners.dr_runner import DRRunner
from minimax.envs.maze.common import EnvInstance
from minimax.envs.maze.maze import Maze

from minimax.add.theta import decode_level
from minimax.add.unet import UNet
from minimax.add.diffusion import make_schedule, diffusion_to_theta
from minimax.add.guidance import (
    ppo_value_guided_ddim_sample_theta_v2,
    ppo_value_guided_ddim_sample_theta_v3,
    ppo_value_guided_ddim_sample_theta_v4,
)


class ADDRunner(DRRunner):
    def __init__(
        self,
        *,
        diffusion_ckpt_path: str,
        ddim_steps: int = 50,
        guidance_rollout_steps: int = 256,
        rollout_every: int = 1,       # 1 = every step (diffV_v2 behaviour); >1 = v3 K-step
        use_positive_value_loss: bool = False,
        dr_frac: float = 0.5,         # fraction of levels drawn from DR each tick
        dr_max_walls: int = 60,       # DR wall budget: uniform in [0, dr_max_walls]
        unet_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.diff_model = UNet(**(unet_kwargs or {}))
        self.schedule = make_schedule()
        self.ddim_steps = ddim_steps
        self.guidance_rollout_steps = guidance_rollout_steps
        self.rollout_every = rollout_every
        self.use_positive_value_loss = use_positive_value_loss

        # Hybrid DR / diffusion split (resolved once at construction, static at JIT time).
        self._n_dr   = max(0, round(self.n_parallel * dr_frac))
        self._n_diff = self.n_parallel - self._n_dr

        with open(diffusion_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.diff_params = jax.device_put(ckpt["ema_params"])

        # Pre-build these once: both are used inside sample_levels (JIT'd).
        # benv.env is MonitorReturnWrapper; its step/set_env_instance return
        # (obs, state, reward, done, info, extra) / (obs, state, extra).
        self._decode_and_reset_fn = self._make_decode_and_reset_fn()
        self._env_step_fn         = jax.vmap(self.benv.env.step)

        # Bare maze env used only for DR resets (no wrappers, no agent).
        if self._n_dr > 0:
            ep = self.env.params
            self._dr_env = Maze(
                height=ep.height,
                width=ep.width,
                n_walls=dr_max_walls,
                sample_n_walls=True,
                agent_view_size=ep.agent_view_size,
                see_through_walls=ep.see_through_walls,
                see_agent=ep.see_agent,
                normalize_obs=ep.normalize_obs,
            )

    # ------------------------------------------------------------------
    # Level sampling (guided DDIM + DR hybrid)
    # ------------------------------------------------------------------

    def _make_decode_and_reset_fn(self):
        """x0_pred (B,16,16,3) -> (obs, state, extra) via wrapped set_env_instance."""
        env = self.benv.env

        def fn(x0_pred):
            thetas    = jax.vmap(diffusion_to_theta)(x0_pred)
            instances = self._decode_to_instances(thetas)
            return jax.vmap(env.set_env_instance)(instances)

        return fn

    def _sample_thetas(self, rng, n_levels, ppo_params_s0, omega):
        def diff_model_fn(params, x, t):
            return self.diff_model.apply(params, x, t)

        def ppo_apply_fn(params, obs, carry, reset):
            return self.student_pop.agent.model.apply(params, obs, carry, reset)

        guidance_fn = (
            ppo_value_guided_ddim_sample_theta_v4
            if self.rollout_every > 1
            else ppo_value_guided_ddim_sample_theta_v2
        )
        kwargs = dict(
            diff_model_fn=diff_model_fn,
            diff_params=self.diff_params,
            ppo_apply_fn=ppo_apply_fn,
            ppo_params=ppo_params_s0,
            decode_and_reset_fn=self._decode_and_reset_fn,
            env_step_fn=self._env_step_fn,
            shape=(n_levels, 16, 16, 3),
            rng=rng,
            schedule=self.schedule,
            omega=omega,
            num_steps=self.ddim_steps,
            guidance_rollout_steps=self.guidance_rollout_steps,
            use_positive_value_loss=self.use_positive_value_loss,
        )
        if self.rollout_every > 1:
            kwargs["rollout_every"] = self.rollout_every
        return guidance_fn(**kwargs)

    def _sample_dr_instances(self, rng, n_levels):
        """Sample n_levels random maze instances via plain env reset (no diffusion)."""
        rngs = jax.random.split(rng, n_levels)
        _, states = jax.vmap(self._dr_env.reset_env)(rngs)
        return EnvInstance(
            agent_pos=states.agent_pos,
            agent_dir_idx=states.agent_dir_idx,
            goal_pos=states.goal_pos,
            wall_map=states.wall_map,
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
    def sample_levels(self, rng, ppo_params_s0, omega):
        """Hybrid level sampling: n_diff guided DDIM + n_dr random DR per student."""
        rng, *student_rngs = jax.random.split(rng, self.n_students + 1)

        # _n_diff / _n_dr are Python ints — branches resolved at trace time.
        def _sample(rng):
            rng, diff_rng, dr_rng = jax.random.split(rng, 3)

            # DR dummy thetas are 16×16 to match diffusion theta space.
            _THETA_HW = 16
            if self._n_diff > 0 and self._n_dr > 0:
                diff_thetas = self._sample_thetas(diff_rng, self._n_diff, ppo_params_s0, omega)
                diff_inst   = self._decode_to_instances(diff_thetas)
                dr_inst     = self._sample_dr_instances(dr_rng, self._n_dr)
                dr_thetas   = jnp.zeros((self._n_dr, _THETA_HW, _THETA_HW, 3))
                thetas    = jnp.concatenate([diff_thetas, dr_thetas], axis=0)
                instances = jax.tree.map(
                    lambda a, b: jnp.concatenate([a, b], axis=0),
                    diff_inst, dr_inst,
                )
            elif self._n_dr == 0:
                thetas    = self._sample_thetas(diff_rng, self._n_diff, ppo_params_s0, omega)
                instances = self._decode_to_instances(thetas)
            else:
                instances = self._sample_dr_instances(dr_rng, self._n_dr)
                thetas    = jnp.zeros((self._n_dr, _THETA_HW, _THETA_HW, 3))

            return thetas, instances

        return jax.vmap(_sample)(jnp.array(student_rngs))

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
        rollout_batch_shape = (self.n_students, self.n_parallel * self.n_eval)

        obs, state, extra = jax.vmap(
            lambda inst: self._reset_from_instances(None, inst, self.n_parallel, self.n_eval)
        )(all_instances)

        ep_stats = self.rolling_stats.reset_stats(batch_shape=rollout_batch_shape)
        rollout_start_state = state

        done = jnp.zeros(rollout_batch_shape, dtype=jnp.bool_)
        reset_state = state

        rng, subrng = jax.random.split(rng)
        rollout, state, start_state, obs, carry, extra, ep_stats, train_state = \
            self._rollout_students(
                subrng, train_state, state, start_state, obs, carry, done,
                reset_state, extra, ep_stats,
            )

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

        stats["_thetas"] = all_thetas[0]

        rewards_s0 = rollout["rewards"][0]
        dones_s0   = rollout["dones"][0]
        ep_returns = (rewards_s0 * dones_s0).sum(axis=0)
        n_eps = dones_s0.sum(axis=0).clip(1)
        stats["_mean_return"] = (ep_returns / n_eps).mean()

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
        omega,
    ):
        """Full ADD step: guided DDIM sampling then RL rollout + PPO."""
        if self.n_devices > 1:
            rng = jax.random.fold_in(rng, jax.lax.axis_index("device"))

        ppo_params_s0 = jax.tree.map(lambda p: p[0], train_state.params)

        rng, sample_rng = jax.random.split(rng)
        all_thetas, all_instances = self.sample_levels(
            sample_rng, ppo_params_s0, omega
        )

        result = self._run_rl(
            rng, train_state, state, start_state, obs, carry, extra, ep_stats,
            all_thetas, all_instances,
        )
        self.n_updates += 1
        return result
