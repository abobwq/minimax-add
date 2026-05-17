"""Stage 2: Train RL agent using diffusion-generated levels (ADD w/o guidance).

Usage:
    tpu-device 0 python -m minimax.add.train_rl --diffusion_ckpt checkpoints/diffusion/final.pkl

Compare against standard DR by running with minimax's DR runner directly.
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
from minimax.runners.dr_runner import DRRunner
from minimax.runners.paired_runner import PAIREDRunner
from minimax.runners.plr_runner import PLRRunner
from minimax.runners.eval_runner import EvalRunner
from minimax.util.rl import AgentPop

from minimax.add.runner import ADDRunner


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


def make_runner(args):
    env_kwargs = dict(
        height=13,
        width=13,
        n_walls=args.n_walls,
        see_through_walls=True,
        agent_view_size=5,
        max_episode_steps=250,
        normalize_obs=False,
        sample_n_walls=True,
        replace_wall_pos=True,
    )

    # PLR uses fixed wall count and normalized obs (paper config).
    if args.runner == "plr":
        env_kwargs.update(normalize_obs=True, sample_n_walls=False)

    dummy_env, _ = envs.make("Maze", env_kwargs=env_kwargs)
    n_actions = dummy_env.action_space().n

    student_model = models.make(
        env_name="Maze",
        model_name="default_student_cnn",
        output_dim=n_actions,
        recurrent_arch="lstm",
    )

    student_agent = agents.PPOAgent(
        model=student_model,
        n_epochs=args.ppo_epochs,
        n_minibatches=args.ppo_minibatches,
        clip_eps=args.ppo_clip,
        entropy_coef=args.entropy_coef,
    )

    runner_kwargs = dict(
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
        track_env_metrics=True,
    )

    if args.runner == "add":
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
            unet_kwargs=unet_kwargs or None,
            **runner_kwargs,
        )
    elif args.runner == "dr":
        runner = DRRunner(**runner_kwargs)
    elif args.runner == "paired":
        # Teacher MDP kwargs — fixed constants from the PAIRED paper config.
        # n_walls follows --n_walls so student and teacher environments are consistent.
        ued_env_kwargs = dict(
            height=13,
            width=13,
            n_walls=args.n_walls,
            noise_dim=50,
            replace_wall_pos=True,
            fixed_n_wall_steps=True,
            first_wall_pos_sets_budget=False,
            set_agent_dir=False,
            normalize_obs=True,
        )

        # Instantiate UED env just to query teacher action/obs space dimensions.
        ued_env, _, _ = envs.make("Maze", env_kwargs=env_kwargs, ued_env_kwargs=ued_env_kwargs)
        max_teacher_steps = ued_env.ued_max_episode_steps()

        teacher_model = models.make(
            env_name="Maze",
            model_name="default_teacher_cnn",
            output_dim=ued_env.ued_action_space().n,
            recurrent_arch="lstm",
            recurrent_hidden_dim=256,
            hidden_dim=32,
            n_hidden_layers=1,
            n_conv_filters=128,
            n_scalar_embeddings=max_teacher_steps,
            max_scalar=max_teacher_steps,
            scalar_embed_dim=10,
        )

        teacher_agent = agents.PPOAgent(
            model=teacher_model,
            n_epochs=args.ppo_epochs,
            n_minibatches=args.ppo_minibatches,
            clip_eps=args.ppo_clip,
            entropy_coef=args.teacher_entropy_coef,
        )

        runner = PAIREDRunner(
            env_name="Maze",
            env_kwargs=env_kwargs,
            ued_env_kwargs=ued_env_kwargs,
            student_agents=[student_agent],
            teacher_agents=[teacher_agent],
            n_students=2,
            n_parallel=args.n_parallel,
            n_eval=1,
            n_rollout_steps=args.rollout_steps,
            lr=args.lr,
            discount=args.discount,
            gae_lambda=args.gae_lambda,
            teacher_discount=args.teacher_discount,
            teacher_gae_lambda=args.teacher_gae_lambda,
            ued_score=args.ued_score,
            track_env_metrics=True,
        )
    elif args.runner == "plr":
        runner = PLRRunner(
            replay_prob=args.plr_replay_prob,
            buffer_size=args.plr_buffer_size,
            staleness_coef=args.plr_staleness_coef,
            temp=args.plr_temp,
            use_score_ranks=args.plr_use_score_ranks,
            min_fill_ratio=args.plr_min_fill_ratio,
            use_robust_plr=args.plr_use_robust_plr,
            use_parallel_eval=args.plr_use_parallel_eval,
            ued_score=args.plr_ued_score,
            force_unique=args.plr_force_unique,
            **runner_kwargs,
        )
    else:
        raise ValueError(f"Unknown runner: {args.runner}")

    eval_runner = EvalRunner(
        pop=runner.student_pop,
        env_names=EVAL_ENV_NAMES,
        env_kwargs={},
        n_episodes=args.eval_episodes,
    )

    return runner, eval_runner


def main():
    parser = argparse.ArgumentParser(description="Train RL agent (Stage 2)")
    parser.add_argument("--runner", type=str, default="add", choices=["add", "dr", "paired", "plr"])
    parser.add_argument("--diffusion_ckpt", type=str, default="checkpoints/diffusion/final.pkl")
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--unet_attn_res", type=int, nargs="+", default=None,
                        help="Override UNet attention_resolutions (e.g. 4 2 for v1).")
    parser.add_argument("--unet_no_scale_shift", action="store_true",
                        help="Disable FiLM (use_scale_shift_norm=False, for v1/v2 ckpts).")
    parser.add_argument("--unet_num_heads", type=int, default=None,
                        help="Fix UNet attention head count to a constant "
                             "(matches PyTorch reference, =4 for v4+). "
                             "Default None preserves legacy max(1, channels // 64) "
                             "needed by v1/v2/v3 checkpoints.")

    parser.add_argument("--n_walls", type=int, default=25)
    parser.add_argument("--n_parallel", type=int, default=32)
    parser.add_argument("--rollout_steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--discount", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=5)
    parser.add_argument("--ppo_minibatches", type=int, default=1)
    parser.add_argument("--ppo_clip", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.0)

    # PLR-specific hyperparameters (ignored for other runners)
    parser.add_argument("--plr_ued_score", type=str, default="max_mc")
    parser.add_argument("--plr_replay_prob", type=float, default=0.5)
    parser.add_argument("--plr_buffer_size", type=int, default=4000)
    parser.add_argument("--plr_staleness_coef", type=float, default=0.5)
    parser.add_argument("--plr_temp", type=float, default=0.3)
    parser.add_argument("--plr_min_fill_ratio", type=float, default=0.5)
    parser.add_argument("--plr_use_score_ranks", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plr_use_robust_plr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plr_use_parallel_eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plr_force_unique", action=argparse.BooleanOptionalAction, default=True)

    # PAIRED-specific hyperparameters (ignored for other runners)
    parser.add_argument("--teacher_entropy_coef", type=float, default=0.05)
    parser.add_argument("--teacher_gae_lambda", type=float, default=0.98)
    parser.add_argument("--teacher_discount", type=float, default=0.995)
    parser.add_argument("--ued_score", type=str, default="relative_regret")

    parser.add_argument("--n_updates", type=int, default=30000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rl")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    steps_per_update = args.n_parallel * args.rollout_steps
    total_steps = args.n_updates * steps_per_update
    run_name = args.run_name or f"{args.runner}_s{args.seed}"
    print(f"JAX devices: {jax.devices()}")
    print(f"Runner: {args.runner}, run: {run_name}, {total_steps:,} total env steps")

    ckpt_dir = os.path.join(args.ckpt_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    runner, eval_runner = make_runner(args)

    rng = jax.random.PRNGKey(args.seed)
    runner_state = runner.reset(rng)

    if args.runner == "add":
        rng, critic_rng = jax.random.split(rng)
        critic_params = runner.init_critic_params(critic_rng)
    else:
        critic_params = None

    t0 = time.time()
    tick = 0
    train_steps = 0

    while tick < args.n_updates:
        evaluate = (tick + 1) % args.eval_every == 0

        if critic_params is not None:
            stats, *runner_state = runner.run(*runner_state, critic_params, 0.0)
        else:
            stats, *runner_state = runner.run(*runner_state)

        train_steps += steps_per_update
        tick += 1

        if tick % args.log_every == 0:
            elapsed = time.time() - t0
            sps = train_steps / elapsed
            # DR/ADD: stats has "return"; PAIRED: stats has "mean_return_a0", "mean_return_a1"
            if "return" in stats:
                mean_return = float(stats["return"])
            else:
                student_returns = [float(v) for k, v in stats.items() if k.startswith("mean_return_a")]
                mean_return = float(np.mean(student_returns)) if student_returns else 0.0
            print(
                f"update {tick:>6d}/{args.n_updates} | "
                f"steps {train_steps:>10,} | "
                f"return {mean_return:.3f} | "
                f"{sps:.0f} sps"
            )

        if evaluate:
            rng_eval = runner_state[0]
            rng_eval, subrng = jax.random.split(rng_eval)
            params = runner_state[1].params
            eval_stats = eval_runner.run(subrng, params)

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

            walls = stats.get("env/n_walls")
            path = stats.get("env/shortest_path_length")
            passable = stats.get("env/passable_ratio")
            if walls is not None and path is not None and passable is not None:
                p_frac = float(passable)
                walls_str = f"{float(walls):.1f}" if p_frac > 0 else "n/a"
                path_str = f"{float(path):.1f}" if p_frac > 0 else "n/a"
                print(
                    f"  complexity | walls {walls_str} | path {path_str} | "
                    f"passable {p_frac:.1%} (means over passable levels)"
                )

        if tick % args.ckpt_every == 0 or tick == args.n_updates:
            ckpt_path = os.path.join(ckpt_dir, f"step_{tick:06d}.pkl")
            ckpt = {"params": jax.device_get(runner_state[1].params), "step": tick}
            if args.runner == "paired":
                # runner_state[6] is ued_train_state (teacher)
                ckpt["teacher_params"] = jax.device_get(runner_state[6].params)
            with open(ckpt_path, "wb") as f:
                pickle.dump(ckpt, f)
            print(f"  checkpoint saved: {ckpt_path}")

    total_time = time.time() - t0
    print(f"Training complete in {total_time/3600:.1f}h, {train_steps:,} steps")


if __name__ == "__main__":
    main()
