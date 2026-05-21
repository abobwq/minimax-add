"""Stage 3: Full ADD training — RL with on-policy PPO-value GAE guided diffusion (diffV_v2).

Usage:
    tpu-device 0 python -m minimax.add.train --diffusion_ckpt checkpoints/diffusion/final.pkl

The full ADD loop:
  1. ADDRunner.run() samples levels via guided DDIM.
     At each DDIM step: decode x0_pred → real rollout → differentiable LSTM-
     threaded GAE score → gradient guides x0_pred toward harder levels.
  2. PPO rollouts collect per-level episodic returns and update the policy.
"""

import argparse
import os
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

import minimax.envs as envs
import minimax.models as models
import minimax.agents as agents
from minimax.runners.eval_runner import EvalRunner

from minimax.add.runner import ADDRunner
from minimax.add.theta import decode_level
from minimax.add.diffusion import make_schedule
import minimax.util.graph as graph_util


@jax.jit
def compute_complexity_metrics(thetas):
    """Compute wall count and shortest path from a batch of theta images."""
    wall_maps, agent_pos_rc, goal_pos_rc, _ = jax.vmap(decode_level)(thetas)
    n_walls = wall_maps.sum(axis=(1, 2))
    agent_pos_xy = agent_pos_rc[:, ::-1].astype(jnp.uint32)
    goal_pos_xy = goal_pos_rc[:, ::-1].astype(jnp.uint32)
    path_lengths = jax.vmap(graph_util.shortest_path_len)(
        wall_maps, agent_pos_xy, goal_pos_xy,
    )
    return n_walls, path_lengths


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
    parser = argparse.ArgumentParser(description="Full ADD training (diffV_v2, on-policy guidance)")
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--omega", type=float, default=5.0)
    parser.add_argument("--guidance_rollout_steps", type=int, default=256,
                        help="Rollout length inside each DDIM guidance step")
    parser.add_argument("--guidance_ued_score", type=str, default="l1_value_loss",
                        choices=["l1_value_loss", "positive_value_loss"],
                        help="GAE score variant for guidance")
    parser.add_argument("--unet_attn_res", type=int, nargs="+", default=None)
    parser.add_argument("--unet_no_scale_shift", action="store_true")
    parser.add_argument("--unet_num_heads", type=int, default=None)

    parser.add_argument("--n_parallel", type=int, default=32)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=5)
    parser.add_argument("--ppo_minibatches", type=int, default=1)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.0)

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
    print(f"ADD diffV_v2 | {total_steps:,} total env steps | omega={args.omega}")
    print(f"  guidance_rollout_steps={args.guidance_rollout_steps}"
          f" | ued_score={args.guidance_ued_score}")

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
        guidance_rollout_steps=args.guidance_rollout_steps,
        use_positive_value_loss=(args.guidance_ued_score == "positive_value_loss"),
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

    def save_checkpoint(tick, train_steps, runner_state):
        path = os.path.join(ckpt_dir, f"step_{train_steps:09d}.pkl")
        data = {
            "tick": tick,
            "train_steps": train_steps,
            "rl_params": jax.device_get(runner_state[1].params),
            "args": vars(args),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  saved {path}")

    rng = jax.random.PRNGKey(args.seed)
    runner_state = runner.reset(rng)
    t0 = time.time()
    tick = 0
    train_steps = 0

    while tick < args.n_updates:
        rng, _ = jax.random.split(rng)
        omega = jnp.array(args.omega)

        stats, *runner_state = runner.run(*runner_state, omega)
        train_steps += steps_per_update
        tick += 1

        stats_cpu = jax.device_get(stats)
        mean_return = float(stats_cpu["_mean_return"])
        thetas_np = np.array(stats_cpu["_thetas"])

        if tick % args.log_every == 0:
            elapsed = time.time() - t0
            sps = train_steps / elapsed
            guided_str = "guided" if float(omega) > 0 else "unguided"

            # Core stats
            print(
                f"update {tick:>6d}/{args.n_updates} | "
                f"steps {train_steps:>10,} | "
                f"return {mean_return:.3f} | "
                f"{guided_str} | "
                f"{sps:.0f} sps"
            )

            # PPO diagnostics (keys from PPOAgent: total_loss, actor_loss, value_loss, entropy, grad_norm)
            ppo_parts = []
            for key, label, fmt in [
                ("value_loss", "val_loss", ".4f"),
                ("actor_loss", "act_loss", ".4f"),
                ("entropy",    "ent",      ".4f"),
                ("grad_norm",  "gnorm",    ".3f"),
            ]:
                if key in stats_cpu:
                    ppo_parts.append(f"{label}={float(stats_cpu[key]):{fmt}}")
            if ppo_parts:
                print(f"  ppo  | {' | '.join(ppo_parts)}")

            # Level complexity on the most recent batch of generated levels
            n_walls, path_lens = compute_complexity_metrics(jnp.array(thetas_np))
            solvable = path_lens > 0
            n_solv = int(solvable.sum())
            path_solv = path_lens[solvable]
            pl_str = f"{float(path_solv.mean()):.1f}±{float(path_solv.std()):.1f}" if n_solv > 0 else "n/a"
            print(
                f"  level| walls {float(n_walls.mean()):.1f}±{float(n_walls.std()):.1f} | "
                f"path {pl_str} | solv {n_solv}/{len(n_walls)}"
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

            # Complexity logged at eval too (fuller path length summary)
            n_walls, path_lens = compute_complexity_metrics(jnp.array(thetas_np))
            solvable = path_lens > 0
            n_solv = int(solvable.sum())
            path_solv = path_lens[solvable]
            pl_str = f"{float(path_solv.mean()):.1f}" if n_solv > 0 else "n/a"
            print(f"  complexity | walls {float(n_walls.mean()):.1f}±{float(n_walls.std()):.1f} | path {pl_str} | solv {n_solv}/{len(n_walls)}")

        if tick % args.save_every == 0:
            save_checkpoint(tick, train_steps, runner_state)

    save_checkpoint(tick, train_steps, runner_state)
    total_time = time.time() - t0
    print(f"Training complete in {total_time/3600:.1f}h, {train_steps:,} steps")


if __name__ == "__main__":
    main()
