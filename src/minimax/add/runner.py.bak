"""ADDRunner: DRRunner that generates levels via a pretrained diffusion model.

Always compiles the guided DDIM path. The caller passes critic_params and
omega as regular arguments to run(). Set omega=0 to disable guidance (the
regret gradient gets zeroed out, equivalent to unguided DDIM).
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
from minimax.add.guidance import guided_ddim_sample_theta
from minimax.add.critic import EnvCritic, batch_rollout_to_targets


class ADDRunner(DRRunner):
    def __init__(
        self,
        *,
        diffusion_ckpt_path: str,
        ddim_steps: int = 50,
        alpha: float = 0.15,
        unet_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.diff_model = UNet(**(unet_kwargs or {}))
        self.critic_model = EnvCritic()
        self.schedule = make_schedule()
        self.ddim_steps = ddim_steps
        self.alpha = alpha

        with open(diffusion_ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        self.diff_params = jax.device_put(ckpt["ema_params"])

    def init_critic_params(self, rng):
        """Initialize critic params. Call once before the first run()."""
        return self.critic_model.init(
            rng, jnp.ones((1, 16, 16, 3)), jnp.array([0])
        )

    def _sample_thetas(self, rng, n_levels, critic_params, omega):
        """Guided DDIM sample. omega=0 disables guidance."""
        def model_fn(params, x, t):
            return self.diff_model.apply(params, x, t)

        def critic_fn(params, x, t):
            return self.critic_model.apply(params, x, t)

        return guided_ddim_sample_theta(
            diff_model_fn=model_fn,
            diff_params=self.diff_params,
            critic_model_fn=critic_fn,
            critic_params=critic_params,
            shape=(n_levels, 16, 16, 3),
            rng=rng,
            schedule=self.schedule,
            omega=omega,
            alpha=self.alpha,
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
        critic_params,
        omega,
    ):
        if self.n_devices > 1:
            rng = jax.random.fold_in(rng, jax.lax.axis_index("device"))

        rollout_batch_shape = (self.n_students, self.n_parallel * self.n_eval)

        rng, *diff_rngs = jax.random.split(rng, self.n_students + 1)

        def _sample_thetas_and_instances(rng):
            thetas = self._sample_thetas(rng, self.n_parallel, critic_params, omega)
            instances = self._decode_to_instances(thetas)
            return thetas, instances

        all_thetas, all_instances = jax.vmap(_sample_thetas_and_instances)(
            jnp.array(diff_rngs)
        )

        obs, state, extra = jax.vmap(
            lambda inst: self._reset_from_instances(None, inst, self.n_parallel, self.n_eval)
        )(all_instances)

        ep_stats = self.rolling_stats.reset_stats(batch_shape=rollout_batch_shape)
        rollout_start_state = state

        done = jnp.zeros(rollout_batch_shape, dtype=jnp.bool_)
        reset_state = state

        rng, subrng = jax.random.split(rng)
        rollout, state, start_state, obs, carry, extra, ep_stats, train_state = (
            self._rollout_students(
                subrng,
                train_state,
                state,
                start_state,
                obs,
                carry,
                done,
                reset_state,
                extra,
                ep_stats,
            )
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
        targets, n_episodes, mean_return = batch_rollout_to_targets(
            rollout["rewards"][0], rollout["dones"][0]
        )
        stats["_targets"] = targets
        stats["_n_episodes"] = n_episodes
        stats["_mean_return"] = mean_return

        train_state = train_state.increment()
        self.n_updates += 1

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
