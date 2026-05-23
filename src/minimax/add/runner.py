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
import minimax.util.graph as graph_util

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
        x0_clamp: float = 3.0,        # guidance stabilization: clip x0_guided to [−x0_clamp, x0_clamp]
        use_grad_norm: bool = False,  # normalize guidance gradient per-level to unit L2
        grad_compress_p: float = 1.0,  # power compression exponent before normalization (1=off)
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

        self.dr_frac  = dr_frac
        self.x0_clamp = x0_clamp
        self.use_grad_norm = use_grad_norm
        self.grad_compress_p = grad_compress_p

        with open(diffusion_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.diff_params = jax.device_put(ckpt["ema_params"])

        # Pre-build these once: both are used inside sample_levels (JIT'd).
        # benv.env is MonitorReturnWrapper; its step/set_env_instance return
        # (obs, state, reward, done, info, extra) / (obs, state, extra).
        self._decode_and_reset_fn = self._make_decode_and_reset_fn()
        self._env_step_fn         = jax.vmap(self.benv.env.step)

        # Bare maze env used only for DR resets (no wrappers, no agent).
        if dr_frac > 0:
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
            x0_clamp=self.x0_clamp,
            use_grad_norm=self.use_grad_norm,
            grad_compress_p=self.grad_compress_p,
        )
        if self.rollout_every > 1:
            kwargs["rollout_every"] = self.rollout_every
        return guidance_fn(**kwargs)  # returns (thetas, stats)

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

    @partial(jax.jit, static_argnums=(0, 4))
    def sample_levels(self, rng, ppo_params_s0, omega, use_dr: bool):
        """Per-update level sampling: all guided DDIM or all DR (decided before call)."""
        rng, *student_rngs = jax.random.split(rng, self.n_students + 1)

        _zero_stats = {
            "grad_norms":  jnp.zeros(self.ddim_steps),
            "clamp_fracs": jnp.zeros(self.ddim_steps),
            "eff_ks":      jnp.zeros(self.ddim_steps),
        }

        # use_dr is a Python bool (static) — branch resolved at trace time.
        def _sample(rng):
            rng, diff_rng, dr_rng = jax.random.split(rng, 3)
            if use_dr:
                instances = self._sample_dr_instances(dr_rng, self.n_parallel)
                thetas    = jnp.zeros((self.n_parallel, 16, 16, 3))
                stats     = _zero_stats
            else:
                thetas, stats = self._sample_thetas(diff_rng, self.n_parallel, ppo_params_s0, omega)
                instances = self._decode_to_instances(thetas)
            return thetas, instances, stats

        return jax.vmap(_sample)(jnp.array(student_rngs))

    # ------------------------------------------------------------------
    # RL rollout + PPO update
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0, 11))
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
        use_dr: bool,
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

        if use_dr:
            # DR update: log solvability from real instances (not thetas)
            dr_wm  = all_instances.wall_map[0]
            dr_ap  = all_instances.agent_pos[0]
            dr_gp  = all_instances.goal_pos[0]
            dr_pls = jax.vmap(graph_util.shortest_path_len)(dr_wm, dr_ap, dr_gp)
            stats["_dr_n_solv"]  = (dr_pls > 0).sum()
            stats["_dr_n_walls"] = dr_wm.sum(axis=(-1, -2)).mean()
            stats["_thetas"]     = jnp.zeros((0, 16, 16, 3))  # empty → skip complexity log
        else:
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

        rng, dr_rng, sample_rng = jax.random.split(rng, 3)
        use_dr = self.dr_frac > 0 and bool(jax.random.bernoulli(dr_rng, self.dr_frac))
        all_thetas, all_instances, diff_stats = self.sample_levels(
            sample_rng, ppo_params_s0, omega, use_dr
        )

        result = self._run_rl(
            rng, train_state, state, start_state, obs, carry, extra, ep_stats,
            all_thetas, all_instances, use_dr
        )
        self.n_updates += 1

        # Merge guidance stats (student 0) into the rl stats dict.
        stats_dict, *rest = result
        stats_dict["_diff_grad_norms"]  = diff_stats["grad_norms"][0]
        stats_dict["_diff_clamp_fracs"] = diff_stats["clamp_fracs"][0]
        stats_dict["_diff_grad_eff_ks"] = diff_stats["eff_ks"][0]
        return (stats_dict, *rest)
