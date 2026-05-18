"""Stage 3: Full ADD training — RL with regret-guided diffusion.

Usage:
    tpu-device 0 python -m minimax.add.train --diffusion_ckpt checkpoints/diffusion/final.pkl

The full ADD loop:
  1. ADDRunner.run() samples levels via guided DDIM (omega=0 until critic ready).
  2. PPO rollouts collect per-level episodic returns.
  3. Environment critic is trained on (level, return) pairs.
  4. Updated critic params are passed to the next run() call.
"""

import argparse
import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import minimax.envs as envs
import minimax.models as models
import minimax.agents as agents
from minimax.runners.eval_runner import EvalRunner

from minimax.add.runner import ADDRunner
from minimax.add.theta import decode_level
from minimax.add.diffusion import make_schedule
import minimax.util.graph as graph_util
from minimax.add.critic import (
    CriticBuffer, make_critic_train_step, train_critic, returns_to_categorical,
)


@jax.jit
def compute_complexity_metrics(thetas):
    """Compute wall count and shortest path from a batch of theta images.

    Uses minimax's JIT-compiled APSP shortest path, vmapped over the batch.
    Returns (n_walls, path_lengths) as JAX arrays, shape (B,).
    Path length is 0 for unsolvable levels.
    """
    wall_maps, agent_pos_rc, goal_pos_rc, _ = jax.vmap(decode_level)(thetas)
    n_walls = wall_maps.sum(axis=(1, 2))
    # decode_level returns (row, col); minimax graph uses (col, row)
    agent_pos_xy = agent_pos_rc[:, ::-1].astype(jnp.uint32)
    goal_pos_xy = goal_pos_rc[:, ::-1].astype(jnp.uint32)
    path_lengths = jax.vmap(graph_util.shortest_path_len)(
        wall_maps, agent_pos_xy, goal_pos_xy,
    )
    return n_walls, path_lengths


PATH_LEN_MAX = 100.0   # normalize path length to [0, 1]; clips paths > 100

EVAL_ENV_NAMES = [
    "Maze-FourRooms",
    "Maze-SixteenRooms",
    "Maze-SixteenRooms2",
    "Maze-Labyrinth",
    "Maze-Labyrinth2",
    "Maze-StandardMaze",
    "Maze-StandardMaze2",
    "Maze-StandardMaze3",
    "Maze-SmallCorridor",
    "Maze-LargeCorridor",
    "Maze-Crossing",
    "Maze-PerfectMaze",
]


def main():
    parser = argparse.ArgumentParser(description="Full ADD training (Stage 3)")
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--omega", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--unet_attn_res", type=int, nargs="+", default=None,
                        help="Override UNet attention_resolutions (e.g. 4 2 for v1)")
    parser.add_argument("--unet_no_scale_shift", action="store_true",
                        help="Disable FiLM (use_scale_shift_norm=False, for v1/v2 ckpts)")
    parser.add_argument("--unet_num_heads", type=int, default=None,
                        help="Fix UNet attention head count to a constant "
                             "(matches PyTorch reference, =4 for v4+). "
                             "Default None preserves legacy max(1, channels // 64) "
                             "needed by v1/v2/v3 checkpoints.")

    parser.add_argument("--n_parallel", type=int, default=32)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=5)
    parser.add_argument("--ppo_minibatches", type=int, default=1)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.0)

    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--critic_weight_decay", type=float, default=0.05)
    parser.add_argument("--critic_grad_clip", type=float, default=1.0,
                        help="Global grad norm clip for critic (0 to disable)")
    parser.add_argument("--critic_buffer_size", type=int, default=1600)
    parser.add_argument("--critic_train_iters", type=int, default=5)
    parser.add_argument("--critic_batch_size", type=int, default=128)
    parser.add_argument("--n_updates", type=int, default=30000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt_dir = os.path.join(args.ckpt_dir, f"add_s{args.seed}")
    os.makedirs(ckpt_dir, exist_ok=True)

    steps_per_update = args.n_parallel * args.rollout_steps
    total_steps = args.n_updates * steps_per_update
    print(f"JAX devices: {jax.devices()}")
    print(f"Full ADD | {total_steps:,} total env steps | omega={args.omega}")
    print(f"  replay=ON | targets=distributional"
          f" | critic_lr={args.critic_lr}"
          f" | wd={args.critic_weight_decay}"
          f" | grad_clip={args.critic_grad_clip}")

    # --- Environment and agent setup ---
    env_kwargs = dict(
        height=13, width=13, n_walls=60, see_through_walls=True,
        agent_view_size=5, max_episode_steps=250, normalize_obs=True,
        sample_n_walls=True, replace_wall_pos=True,
    )
    dummy_env, _ = envs.make("Maze", env_kwargs=env_kwargs)
    n_actions = dummy_env.action_space().n

    student_model = models.make(
        env_name="Maze", model_name="default_student_cnn",
        output_dim=n_actions, recurrent_arch="lstm",
    )
    student_agent = agents.PPOAgent(
        model=student_model, n_epochs=args.ppo_epochs,
        n_minibatches=args.ppo_minibatches, clip_eps=args.ppo_clip,
        entropy_coef=args.entropy_coef,
    )

    # --- ADD runner (handles diffusion sampling + RL rollouts) ---
    unet_kwargs = {}
    if args.unet_attn_res is not None:
        unet_kwargs["attention_resolutions"] = tuple(args.unet_attn_res)
    if args.unet_no_scale_shift:
        unet_kwargs["use_scale_shift_norm"] = False
    if args.unet_num_heads is not None:
        unet_kwargs["num_heads"] = args.unet_num_heads

    runner = ADDRunner(
        diffusion_ckpt_path=args.diffusion_ckpt,
        ddim_steps=args.ddim_steps,
        alpha=args.alpha,
        unet_kwargs=unet_kwargs or None,
        env_name="Maze",
        env_kwargs=env_kwargs,
        student_agents=[student_agent],
        n_students=1,
        n_parallel=args.n_parallel,
        n_eval=1,
        n_rollout_steps=args.rollout_steps,
        lr=args.lr,
        discount=args.discount,
        gae_lambda=args.gae_lambda,
        track_env_metrics=False,
    )

    eval_runner = EvalRunner(
        pop=runner.student_pop,
        env_names=EVAL_ENV_NAMES,
        env_kwargs={"normalize_obs": True},
        n_episodes=args.eval_episodes,
    )

    # --- Critic setup ---
    rng = jax.random.PRNGKey(args.seed)
    rng, critic_rng = jax.random.split(rng)
    critic_params = runner.init_critic_params(critic_rng)
    critic_opt_parts = []
    if args.critic_grad_clip > 0:
        critic_opt_parts.append(optax.clip_by_global_norm(args.critic_grad_clip))
    critic_opt_parts.append(optax.adamw(args.critic_lr, weight_decay=args.critic_weight_decay))
    critic_optimizer = optax.chain(*critic_opt_parts)
    critic_opt_state = critic_optimizer.init(critic_params)
    critic_buffer = CriticBuffer(capacity=args.critic_buffer_size)
    critic_step = make_critic_train_step(runner.critic_model, critic_optimizer)
    schedule = make_schedule()

    n_cp = sum(p.size for p in jax.tree.leaves(critic_params))
    print(f"Critic parameters: {n_cp:,}")
    print("Loaded diffusion checkpoint")

    def save_checkpoint(tick, train_steps, runner_state, critic_params, critic_opt_state):
        path = os.path.join(ckpt_dir, f"step_{train_steps:09d}.pkl")
        data = {
            "tick": tick,
            "train_steps": train_steps,
            "rl_params": jax.device_get(runner_state[1].params),
            "critic_params": jax.device_get(critic_params),
            "critic_opt_state": jax.device_get(critic_opt_state),
            "args": vars(args),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  saved {path}")

    # --- Training loop ---
    runner_state = runner.reset(rng)
    t0 = time.time()
    tick = 0
    train_steps = 0

    while tick < args.n_updates:
        rng, rng_critic = jax.random.split(rng)

        # omega=0 disables guidance; omega>0 enables it once critic is trained.
        omega = jnp.array(args.omega if critic_buffer.is_full else 0.0)

        # run() takes critic_params and omega as traced args.
        stats, *runner_state = runner.run(
            *runner_state, critic_params, omega,
        )
        train_steps += steps_per_update
        tick += 1

        # Collect thetas and compute path-length targets (cheat critic).
        thetas_np = np.array(jax.device_get(stats["_thetas"]))        # (n_parallel, 16, 16, 3)
        mean_return = float(jax.device_get(stats["_mean_return"]))

        if not np.any(np.isnan(thetas_np)):
            _, path_lengths = compute_complexity_metrics(jnp.array(thetas_np))
            path_lengths_norm = jnp.clip(path_lengths / PATH_LEN_MAX, 0.0, 1.0)
            targets_np = np.array(returns_to_categorical(path_lengths_norm, num_bins=100))
            critic_buffer.add(thetas_np, targets_np)

        # Train critic if buffer is full.
        critic_loss = 0.0
        if critic_buffer.is_full:
            critic_params, critic_opt_state, critic_loss = train_critic(
                critic_step=critic_step,
                critic_params=critic_params,
                opt_state=critic_opt_state,
                buffer=critic_buffer,
                rng=rng_critic,
                schedule=schedule,
                num_iterations=args.critic_train_iters,
                batch_size=args.critic_batch_size,
            )

        # Logging.
        if tick % args.log_every == 0:
            elapsed = time.time() - t0
            sps = train_steps / elapsed
            guided_str = "guided" if float(omega) > 0 else "unguided"
            print(
                f"update {tick:>6d}/{args.n_updates} | "
                f"steps {train_steps:>10,} | "
                f"return {mean_return:.3f} | "
                f"critic {critic_loss:.4f} | "
                f"{guided_str} | "
                f"{sps:.0f} sps"
            )

        if (tick + 1) % args.eval_every == 0:
            rng, rng_eval = jax.random.split(rng)
            params = runner_state[1].params
            eval_stats = eval_runner.run(rng_eval, params)

            solved_rates = {}
            for k, v in eval_stats.items():
                if "solved_rate" in k:
                    env_name = k.split(":")[-1]
                    solved_rates[env_name] = float(v)

            if solved_rates:
                mean_solved = np.mean(list(solved_rates.values()))
                print(f"  eval | mean solved {mean_solved:.1%}")
                for name, rate in sorted(solved_rates.items()):
                    print(f"    {name}: {rate:.1%}")

            # Complexity metrics on the most recent batch of generated levels.
            n_walls, path_lens = compute_complexity_metrics(jnp.array(thetas_np))
            solvable = path_lens > 0
            n_solv = int(solvable.sum())
            path_solv = path_lens[solvable]
            pl_str = f"{float(path_solv.mean()):.1f}" if n_solv > 0 else "n/a"
            print(f"  complexity | walls {float(n_walls.mean()):.1f}±{float(n_walls.std()):.1f} | path {pl_str} | solv {n_solv}/{len(n_walls)}")

        if tick % args.save_every == 0:
            save_checkpoint(tick, train_steps, runner_state, critic_params, critic_opt_state)

    save_checkpoint(tick, train_steps, runner_state, critic_params, critic_opt_state)
    total_time = time.time() - t0
    print(f"Training complete in {total_time/3600:.1f}h, {train_steps:,} steps")


if __name__ == "__main__":
    main()
