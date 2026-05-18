"""Re-evaluate saved checkpoints on a specified set of test environments.

Usage:
    tpu-device 0 python -m minimax.add.reeval --env_set paper
    tpu-device 0 python -m minimax.add.reeval --env_set original
    tpu-device 0 python -m minimax.add.reeval --env_set both

Loads final checkpoints from checkpoints/rl/*/step_030000.pkl and evaluates
each on the requested test environment set(s).
"""

import argparse
import glob
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np

import minimax.envs as envs
import minimax.models as models
import minimax.agents as agents
from minimax.runners.eval_runner import EvalRunner
from minimax.util.rl import AgentPop


# Paper set: matches ADD paper Table 3 (FourRooms instead of LabyrinthFlipped)
PAPER_ENVS = [
    "Maze-FourRooms",
    "Maze-SixteenRooms", "Maze-SixteenRooms2",
    "Maze-Labyrinth", "Maze-Labyrinth2",
    "Maze-StandardMaze", "Maze-StandardMaze2", "Maze-StandardMaze3",
    "Maze-SmallCorridor", "Maze-LargeCorridor",
    "Maze-Crossing", "Maze-PerfectMaze",
]

# Original set: what our training runs evaluated on
ORIGINAL_ENVS = [
    "Maze-SixteenRooms", "Maze-SixteenRooms2",
    "Maze-Labyrinth", "Maze-Labyrinth2", "Maze-LabyrinthFlipped",
    "Maze-StandardMaze", "Maze-StandardMaze2", "Maze-StandardMaze3",
    "Maze-SmallCorridor", "Maze-LargeCorridor",
    "Maze-Crossing", "Maze-PerfectMaze",
]


def make_agent_pop():
    """Create the same agent architecture used during training."""
    env_kwargs = dict(
        height=13, width=13, n_walls=25, see_through_walls=True,
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
        model=student_model, n_epochs=5, n_minibatches=1,
        clip_eps=0.2, entropy_coef=0.0,
    )
    return AgentPop(student_agent, n_agents=1)


def find_checkpoints(ckpt_dir, step="030000"):
    """Find all final checkpoints, grouped by run name."""
    pattern = os.path.join(ckpt_dir, f"*/step_{step}.pkl")
    paths = sorted(glob.glob(pattern))
    runs = {}
    for p in paths:
        run_name = os.path.basename(os.path.dirname(p))
        if run_name == "add_s0":
            continue
        runs[run_name] = p
    return runs


def evaluate_checkpoint(pop, params, env_names, n_episodes, rng):
    """Run evaluation and return per-env solved rates."""
    eval_runner = EvalRunner(
        pop=pop,
        env_names=env_names,
        env_kwargs={"normalize_obs": True},
        n_episodes=n_episodes,
    )
    eval_stats = eval_runner.run(rng, params)

    results = {}
    for k, v in eval_stats.items():
        if "solved_rate" in k:
            env_name = k.split(":")[-1]
            results[env_name] = float(v)
    return results


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoints")
    parser.add_argument("--env_set", type=str, default="both",
                        choices=["paper", "original", "both"])
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/rl")
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--output", type=str, default="logs/stage2/reeval.json")
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}")

    env_sets = {}
    if args.env_set in ("paper", "both"):
        env_sets["paper"] = PAPER_ENVS
    if args.env_set in ("original", "both"):
        env_sets["original"] = ORIGINAL_ENVS

    runs = find_checkpoints(args.ckpt_dir)
    print(f"Found {len(runs)} checkpoints")
    for name in runs:
        print(f"  {name}")

    pop = make_agent_pop()

    # Initialize params from a dummy observation to get the right tree structure
    rng = jax.random.PRNGKey(0)
    dummy_env, _ = envs.make("Maze", env_kwargs=dict(
        height=13, width=13, n_walls=25, see_through_walls=True,
        agent_view_size=5, max_episode_steps=250, normalize_obs=True,
    ))
    dummy_obs, _ = dummy_env.reset_env(rng)
    dummy_obs_batched = jax.tree.map(lambda x: x[None], dummy_obs)
    rng, subrng = jax.random.split(rng)
    template_params = pop.init_params(subrng, dummy_obs_batched)

    all_results = {}

    for run_name, ckpt_path in runs.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {run_name}")

        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        saved_params = ckpt["params"]

        # Wrap in the same vmap structure as training (n_agents=1)
        if "params" in saved_params:
            params = saved_params
        else:
            params = template_params
            params = jax.tree.map(lambda t, s: s, params, saved_params)

        run_results = {}
        for set_name, env_names in env_sets.items():
            rng, subrng = jax.random.split(rng)
            t0 = time.time()
            results = evaluate_checkpoint(pop, params, env_names, args.n_episodes, subrng)
            dt = time.time() - t0

            mean_solved = np.mean(list(results.values())) if results else 0
            run_results[set_name] = results
            print(f"  {set_name}: mean solved = {mean_solved:.1%} ({dt:.1f}s)")
            for env, rate in sorted(results.items()):
                print(f"    {env}: {rate:.0%}")

        all_results[run_name] = run_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
