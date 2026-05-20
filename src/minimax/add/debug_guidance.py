#!/usr/bin/env python3
"""
Guided-diffusion debugging harness — no PPO / RL rollouts.

Stages:
  1  Toy wall-count objective: verify guidance sign / scaling
  2  Theta-space gradient ascent: verify objective gradient alone
  3  Oracle soft-path guided DDIM sweep (omega x guidance_fraction grid)
  4  x0 vs xt guidance-mode comparison
  5  Learned critic calibration  (--critic_ckpt required)
  6  Diagnosis table

Usage:
    python -m minimax.add.debug_guidance \\
        --diffusion_ckpt /path/to/diffusion.pkl \\
        --objective soft_path

    python -m minimax.add.debug_guidance \\
        --diffusion_ckpt /path/to/diffusion.pkl \\
        --critic_ckpt   /path/to/critic.pkl \\
        --objective critic_path \\
        --stages 1,2,3,4,5,6

Pass --stages 1,3 to run only selected stages.
First run will trigger JIT compilation; subsequent runs are faster.
"""

import argparse
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC  = os.path.normpath(os.path.join(_HERE, '..', '..'))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from minimax.add.diffusion import (
    DiffusionSchedule,
    _make_ddim_timesteps,
    diffusion_to_theta,
    make_schedule,
    q_sample,
    theta_to_diffusion,
)
from minimax.add.theta import decode_level, GRID_SIZE
from minimax.add.unet import UNet
import minimax.util.graph as graph_util


# ---------------------------------------------------------------------------
# Objective functions  (theta: (B,16,16,3) -> (B,) scalar, differentiable)
# ---------------------------------------------------------------------------

def wall_count_obj(theta: jnp.ndarray) -> jnp.ndarray:
    """Mean wall probability in inner 13×13.  Simple differentiable toy."""
    return theta[:, 1:14, 1:14, 0].mean(axis=(1, 2))


def _soft_path_cost_single(theta: jnp.ndarray) -> jnp.ndarray:
    """Differentiable soft path-cost for one (16,16,3) sample.

    Assigns cost -log(1 - wall_prob) to each inner cell, then runs
    Bellman relaxation from agent to goal.  The result is the
    minimum-cost path, which is large when all direct routes are
    blocked by high-wall-probability cells.

    Gradient ascent on this objective places walls on the current
    shortest route → forces detours → longer hard BFS paths.

    Agent / goal positions are decoded via argmax (zero-gradient);
    the gradient flows only through channel 0 (walls).
    """
    wall_prob = jnp.clip(theta[1:14, 1:14, 0], 1e-6, 1.0 - 1e-6)
    cell_cost = -jnp.log(1.0 - wall_prob)          # (13,13)

    _, agent_rc, goal_rc, _ = decode_level(jnp.clip(theta, 0.0, 1.0))

    INF = 500.0
    dist = jnp.full((GRID_SIZE, GRID_SIZE), INF)
    dist = dist.at[agent_rc[0], agent_rc[1]].set(
        cell_cost[agent_rc[0], agent_rc[1]]
    )

    def relax(dist, _):
        nbr = jnp.minimum(
            jnp.minimum(
                jnp.pad(dist[1:,  :],  ((0, 1), (0, 0)), constant_values=INF),
                jnp.pad(dist[:-1, :],  ((1, 0), (0, 0)), constant_values=INF),
            ),
            jnp.minimum(
                jnp.pad(dist[:, 1:],   ((0, 0), (0, 1)), constant_values=INF),
                jnp.pad(dist[:, :-1],  ((0, 0), (1, 0)), constant_values=INF),
            ),
        )
        return jnp.minimum(dist, nbr + cell_cost), None

    # 26 relaxation steps covers the worst-case path in a 13×13 grid.
    dist, _ = jax.lax.scan(relax, dist, None, length=26)
    d = dist[goal_rc[0], goal_rc[1]]
    return jnp.where(d < INF / 2, d, INF / 2)


def soft_path_obj(theta: jnp.ndarray) -> jnp.ndarray:
    """Batched soft path cost.  (B,16,16,3) -> (B,)."""
    return jax.vmap(_soft_path_cost_single)(theta)


# ---------------------------------------------------------------------------
# Hard (non-differentiable) BFS metrics
# ---------------------------------------------------------------------------

def compute_hard_metrics(thetas: jnp.ndarray) -> dict:
    wall_maps, agent_rc, goal_rc, _ = jax.vmap(decode_level)(thetas)
    n_walls   = np.array(wall_maps.sum(axis=(1, 2)))
    agent_xy  = agent_rc[:, ::-1].astype(jnp.uint32)
    goal_xy   = goal_rc[:, ::-1].astype(jnp.uint32)
    path_lens = np.array(jax.vmap(graph_util.shortest_path_len)(wall_maps, agent_xy, goal_xy))
    solvable  = path_lens > 0
    return {
        "n_walls":       n_walls,
        "path_lens":     path_lens,
        "solvable":      solvable,
        "mean_walls":    float(n_walls.mean()),
        "mean_path":     float(path_lens[solvable].mean()) if solvable.any() else 0.0,
        "max_path":      int(path_lens.max()),
        "pct_gt15":      float((path_lens > 15).mean() * 100),
        "pct_gt20":      float((path_lens > 20).mean() * 100),
        "pct_solvable":  float(solvable.mean() * 100),
    }


def _print_metrics(m: dict, prefix: str = "  "):
    print(f"{prefix}walls={m['mean_walls']:.1f}  "
          f"path mean={m['mean_path']:.1f} max={m['max_path']}  "
          f">15={m['pct_gt15']:.0f}%  >20={m['pct_gt20']:.0f}%  "
          f"solvable={m['pct_solvable']:.0f}%")


# ---------------------------------------------------------------------------
# Instrumented DDIM (Python for-loop; enables per-step logging)
# ---------------------------------------------------------------------------

def debug_guided_ddim(
    diff_model_fn,
    diff_params,
    obj_fn,                          # (theta: (B,16,16,3)) -> (B,)
    shape: tuple,
    rng: jax.Array,
    schedule: DiffusionSchedule,
    omega: float = 5.0,
    num_steps: int = 50,
    guidance_mode: str = "x0",       # "x0" | "xt"
    guidance_fraction: float = 1.0,  # guide last X fraction of denoising steps
    verbose: bool = False,
) -> tuple:
    """Instrumented DDIM.  Returns (theta_final, step_logs).

    guidance_mode="x0":
        Compute grad of obj_fn(theta(x0_pred)) wrt x0_pred_clipped.
        Normalise per-sample by RMS.  Add directly to x0_pred_clipped.
        Re-derive eps from the guided x0.

    guidance_mode="xt":
        Freeze eps (stop_gradient), compute grad of obj_fn through
        x0_pred(x_t, eps_fixed) wrt x_t.
        Subtract scaled grad from eps  (standard classifier-guidance).

    At every guided step logs: t, f_before, f_after, df, grad_norm,
    x0_min/max, theta_min/max.
    """
    T          = schedule.betas.shape[0]
    timesteps  = _make_ddim_timesteps(T, num_steps)
    ab_ext     = jnp.concatenate([jnp.array([1.0]), schedule.alpha_bars])

    x          = jax.random.normal(rng, shape)
    B          = shape[0]
    step_logs  = []
    guide_from = int(num_steps * (1.0 - guidance_fraction))

    for i in range(num_steps):
        t_cur  = int(timesteps[i])
        t_prev = int(timesteps[i + 1]) if i < num_steps - 1 else -1

        ab_t    = ab_ext[t_cur  + 1]
        ab_prev = ab_ext[t_prev + 1]
        s_ab    = jnp.sqrt(ab_t)
        s_1mab  = jnp.sqrt(1.0 - ab_t)
        t_batch = jnp.full((B,), t_cur, dtype=jnp.int32)

        eps_pred      = diff_model_fn(diff_params, x, t_batch)
        x0_pred       = (x - s_1mab * eps_pred) / s_ab
        x0_clipped    = jnp.clip(x0_pred, -1.0, 1.0)
        eps_clean     = (x - s_ab * x0_clipped) / s_1mab

        theta_before  = jnp.clip(diffusion_to_theta(x0_clipped), 0.0, 1.0)
        f_before      = float(obj_fn(theta_before).mean())

        use_guidance  = (omega != 0.0) and (i >= guide_from)
        grad_for_log  = jnp.zeros_like(x0_clipped)

        if use_guidance:
            if guidance_mode == "x0":
                def _score_x0(x0_in):
                    th = jnp.clip(diffusion_to_theta(jnp.clip(x0_in, -1.0, 1.0)), 0.0, 1.0)
                    return obj_fn(th).sum()
                grad_x0      = jax.grad(_score_x0)(x0_clipped)
                rms          = jnp.sqrt((grad_x0 ** 2).mean(axis=(1, 2, 3), keepdims=True) + 1e-8)
                x0_guided    = x0_clipped + omega * (grad_x0 / rms)
                eps_final    = (x - s_ab * x0_guided) / s_1mab
                grad_for_log = grad_x0
            else:  # "xt"
                eps_fixed = jax.lax.stop_gradient(eps_pred)
                def _score_xt(x_in):
                    x0_tmp = jnp.clip((x_in - s_1mab * eps_fixed) / s_ab, -1.0, 1.0)
                    th     = jnp.clip(diffusion_to_theta(x0_tmp), 0.0, 1.0)
                    return obj_fn(th).sum()
                grad_xt      = jax.grad(_score_xt)(x)
                eps_final    = eps_clean - s_1mab * omega * grad_xt
                x0_guided    = (x - s_1mab * eps_final) / s_ab
                grad_for_log = grad_xt
        else:
            x0_guided = x0_clipped
            eps_final = eps_clean

        s_ab_prev   = jnp.sqrt(ab_prev)
        s_1mab_prev = jnp.sqrt(1.0 - ab_prev)
        x = s_ab_prev * x0_guided + s_1mab_prev * eps_final

        theta_after = jnp.clip(diffusion_to_theta(jnp.clip(x0_guided, -1.0, 1.0)), 0.0, 1.0)
        f_after     = float(obj_fn(theta_after).mean())
        grad_norm   = float(jnp.linalg.norm(grad_for_log))

        log = dict(
            t=t_cur, f_before=f_before, f_after=f_after,
            df=f_after - f_before, grad_norm=grad_norm,
            x0_min=float(x0_clipped.min()), x0_max=float(x0_clipped.max()),
            theta_min=float(theta_before.min()), theta_max=float(theta_before.max()),
            guided=use_guidance,
        )
        step_logs.append(log)

        if verbose:
            flag = "*" if use_guidance else " "
            print(f"    {flag}t={t_cur:4d}  f: {f_before:.4f}->{f_after:.4f}"
                  f"  df={log['df']:+.5f}  |g|={grad_norm:.4f}")

    theta_final = jnp.clip(diffusion_to_theta(jnp.clip(x, -1.0, 1.0)), 0.0, 1.0)
    return theta_final, step_logs


# ---------------------------------------------------------------------------
# Stage 1 — Toy wall-count guidance sanity check
# ---------------------------------------------------------------------------

def run_stage1(diff_model_fn, diff_params, schedule, n_samples, num_steps, seed):
    print("\n" + "=" * 70)
    print("STAGE 1: Toy wall-count objective — guidance sign / scaling")
    print("=" * 70)
    print("Expected: positive omega raises wall count; negative lowers it.")

    rng     = jax.random.PRNGKey(seed)
    shape   = (n_samples, 16, 16, 3)
    results = {}

    for omega in [0.0, 0.5, 2.0, 5.0, -0.5, -2.0]:
        rng, sub = jax.random.split(rng)
        verbose  = (omega == 2.0)
        if verbose:
            print(f"\n  omega={omega:+.1f}  (per-step output):")
        theta, logs = debug_guided_ddim(
            diff_model_fn, diff_params,
            obj_fn=wall_count_obj,
            shape=shape, rng=sub, schedule=schedule,
            omega=omega, num_steps=num_steps,
            guidance_mode="x0", guidance_fraction=1.0,
            verbose=verbose,
        )
        m        = compute_hard_metrics(theta)
        f_final  = float(wall_count_obj(theta).mean())
        g_logs   = [l for l in logs if l["guided"]]
        mean_df  = float(np.mean([l["df"] for l in g_logs])) if g_logs else 0.0
        print(f"  omega={omega:+.1f}  f_wall={f_final:.4f}"
              f"  mean_walls={m['mean_walls']:.1f}"
              f"  mean_df/step={mean_df:+.5f}")
        _print_metrics(m, prefix="    ")
        results[omega] = m

    w0   = results.get(0.0,  {}).get("mean_walls", 0)
    wpos = results.get(2.0,  {}).get("mean_walls", 0)
    wneg = results.get(-2.0, {}).get("mean_walls", 0)
    pos_ok = wpos > w0 + 0.5
    neg_ok = wneg < w0 - 0.5

    print(f"\n  Stage 1 result:")
    print(f"    omega=+2: walls={wpos:.1f}  baseline={w0:.1f}  "
          f"{'PASS' if pos_ok else 'FAIL — sign/scaling/injection broken'}")
    print(f"    omega=-2: walls={wneg:.1f}  baseline={w0:.1f}  "
          f"{'PASS' if neg_ok else 'FAIL — sign/scaling/injection broken'}")

    return results, pos_ok and neg_ok


# ---------------------------------------------------------------------------
# Stage 2 — Theta-space gradient ascent
# ---------------------------------------------------------------------------

def run_stage2(diff_model_fn, diff_params, schedule, n_samples, num_steps,
               obj_fn, obj_name, seed):
    print("\n" + "=" * 70)
    print(f"STAGE 2: Theta-space gradient ascent  (objective: {obj_name})")
    print("=" * 70)
    print("Expected: soft objective and hard BFS both improve.")

    rng   = jax.random.PRNGKey(seed)
    shape = (n_samples, 16, 16, 3)

    # Sample unguided levels as starting points.
    rng, sub = jax.random.split(rng)
    theta0, _ = debug_guided_ddim(
        diff_model_fn, diff_params, wall_count_obj,
        shape, sub, schedule, omega=0.0, num_steps=num_steps,
        guidance_mode="x0", guidance_fraction=1.0, verbose=False,
    )

    m0     = compute_hard_metrics(theta0)
    f0     = float(obj_fn(theta0).mean())
    print(f"\n  Before ascent:  f_soft={f0:.4f}")
    _print_metrics(m0, prefix="    ")

    # Normalised gradient ascent on theta ∈ [0,1].
    n_steps = 50
    eta     = 0.05
    theta   = theta0

    @jax.jit
    def step(th):
        val, g = jax.value_and_grad(lambda t: obj_fn(t).sum())(th)
        g_norm  = jnp.linalg.norm(g) + 1e-8
        return jnp.clip(th + eta * g / g_norm, 0.0, 1.0), val

    for i in range(n_steps):
        theta, _ = step(theta)
        if (i + 1) % 10 == 0:
            f_cur = float(obj_fn(theta).mean())
            print(f"  step {i+1:3d}: f_soft={f_cur:.4f}")

    m_after    = compute_hard_metrics(theta)
    f_after    = float(obj_fn(theta).mean())
    soft_ok    = f_after > f0 + 1e-3
    hard_ok    = m_after["mean_path"] > m0["mean_path"] + 0.5

    print(f"\n  After {n_steps} ascent steps:")
    print(f"    f_soft: {f0:.4f} -> {f_after:.4f}  "
          f"{'improved' if soft_ok else 'DID NOT IMPROVE'}")
    print(f"    hard BFS mean: {m0['mean_path']:.1f} -> {m_after['mean_path']:.1f}  "
          f"{'improved' if hard_ok else 'DID NOT IMPROVE — soft/hard misalignment?'}")
    _print_metrics(m_after, prefix="    ")

    return {"soft_improved": soft_ok, "hard_improved": hard_ok,
            "m0": m0, "m_after": m_after}


# ---------------------------------------------------------------------------
# Stage 3 — Oracle soft-path guided DDIM sweep
# ---------------------------------------------------------------------------

def run_stage3(diff_model_fn, diff_params, schedule, n_samples, num_steps,
               obj_fn, obj_name, seed):
    print("\n" + "=" * 70)
    print(f"STAGE 3: Oracle guided DDIM omega sweep  (objective: {obj_name})")
    print("=" * 70)

    rng   = jax.random.PRNGKey(seed)
    shape = (n_samples, 16, 16, 3)

    omegas = [0.0, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    fracs  = [1.0, 0.5, 0.25]
    fl     = {1.0: "all", 0.5: "last50%", 0.25: "last25%"}

    hdr = (f"{'omega':>7}  {'frac':>8}  {'mean_path':>9}  {'max_path':>8}  "
           f"{'%>15':>6}  {'%>20':>6}  {'solvable%':>9}  {'f_soft':>8}  {'walls':>6}")
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))

    results           = {}
    baseline_mean_path = None

    for omega in omegas:
        for frac in (fracs if omega > 0 else [1.0]):
            rng, sub = jax.random.split(rng)
            theta, _ = debug_guided_ddim(
                diff_model_fn, diff_params, obj_fn, shape, sub, schedule,
                omega=omega, num_steps=num_steps,
                guidance_mode="x0", guidance_fraction=frac, verbose=False,
            )
            m      = compute_hard_metrics(theta)
            f_soft = float(obj_fn(theta).mean())
            if omega == 0.0:
                baseline_mean_path = m["mean_path"]
            fl_label = fl.get(frac, f"{frac:.0%}")
            print(f"  {omega:>7.1f}  {fl_label:>8}  {m['mean_path']:>9.1f}  "
                  f"{m['max_path']:>8d}  {m['pct_gt15']:>6.1f}  {m['pct_gt20']:>6.1f}  "
                  f"{m['pct_solvable']:>9.1f}  {f_soft:>8.4f}  {m['mean_walls']:>6.1f}")
            results[(omega, frac)] = m

    best = max(
        ((k, v) for k, v in results.items() if k[0] > 0),
        key=lambda kv: kv[1]["mean_path"],
        default=(None, None),
    )
    ddim_ok = False
    if best[0] is not None:
        ob, fb = best
        ddim_ok = fb["mean_path"] > (baseline_mean_path or 0) + 1.0
        print(f"\n  Best guided: omega={ob[0]}, frac={ob[1]}"
              f"  mean_path={fb['mean_path']:.1f}  baseline={baseline_mean_path:.1f}"
              f"  -> {'PASS' if ddim_ok else 'FAIL'}")

    return results, ddim_ok


# ---------------------------------------------------------------------------
# Stage 4 — x0 vs xt guidance-mode comparison
# ---------------------------------------------------------------------------

def run_stage4(diff_model_fn, diff_params, schedule, n_samples, num_steps,
               obj_fn, obj_name, seed):
    print("\n" + "=" * 70)
    print(f"STAGE 4: x0 vs xt guidance comparison  (objective: {obj_name})")
    print("=" * 70)
    print("Both modes should raise f_soft with increasing omega.")

    rng    = jax.random.PRNGKey(seed)
    shape  = (n_samples, 16, 16, 3)
    omegas = [0.0, 2.0, 10.0]

    hdr = (f"{'mode':>6}  {'omega':>7}  {'mean_path':>9}  {'max_path':>8}  "
           f"{'%>15':>6}  {'f_soft':>8}  {'solvable%':>9}")
    print(f"\n  {hdr}")
    print("  " + "-" * len(hdr))

    results = {}
    for mode in ["x0", "xt"]:
        for omega in omegas:
            rng, sub = jax.random.split(rng)
            theta, _ = debug_guided_ddim(
                diff_model_fn, diff_params, obj_fn, shape, sub, schedule,
                omega=omega, num_steps=num_steps,
                guidance_mode=mode, guidance_fraction=1.0, verbose=False,
            )
            m      = compute_hard_metrics(theta)
            f_soft = float(obj_fn(theta).mean())
            print(f"  {mode:>6}  {omega:>7.1f}  {m['mean_path']:>9.1f}  "
                  f"{m['max_path']:>8d}  {m['pct_gt15']:>6.1f}  "
                  f"{f_soft:>8.4f}  {m['pct_solvable']:>9.1f}")
            results[(mode, omega)] = {"metrics": m, "f_soft": f_soft}

        f0  = results[(mode, 0.0)]["f_soft"]
        f10 = results[(mode, 10.0)]["f_soft"]
        ok  = f10 > f0 + 1e-3
        print(f"    [{mode}] f_soft omega=0->{10}: {f0:.4f}->{f10:.4f}  "
              f"{'PASS' if ok else 'FAIL — guidance not moving objective'}")

    x0_ok = results[("x0", 10.0)]["f_soft"] > results[("x0", 0.0)]["f_soft"] + 1e-3
    return results, x0_ok


# ---------------------------------------------------------------------------
# Stage 5 — Learned critic calibration
# ---------------------------------------------------------------------------

def run_stage5(critic_model_fn, critic_params, schedule, seed):
    from minimax.add.critic import categorical_mean as crit_mean
    from minimax.add.theta import encode_level

    print("\n" + "=" * 70)
    print("STAGE 5: Learned critic calibration")
    print("=" * 70)
    print("Expected: critic prediction tracks true BFS (not saturated at ~15).")

    # Build a small set of levels with known (approximate) BFS lengths.
    def make_theta(wm_np, agent_rc, goal_rc):
        return encode_level(jnp.array(wm_np),
                            jnp.array(agent_rc, dtype=jnp.int32),
                            jnp.array(goal_rc,  dtype=jnp.int32),
                            jnp.array(0))

    # 1. Open grid corner-to-corner → BFS = 24
    wm_open = np.zeros((13, 13), dtype=bool)

    # 2. Zigzag-1: one horizontal wall, open at right → BFS ≈ 24-30
    wm_zig1 = np.zeros((13, 13), dtype=bool)
    wm_zig1[6, :12] = True       # row 6, cols 0-11 = wall; col 12 open

    # 3. Zigzag-2: two walls → BFS ≈ 36-44
    wm_zig2 = wm_zig1.copy()
    wm_zig2[9, 1:]  = True       # row 9, cols 1-12 = wall; col 0 open

    # 4. Three walls → BFS ≈ 50+
    wm_zig3 = wm_zig2.copy()
    wm_zig3[3, :12] = True       # row 3, cols 0-11 = wall; col 12 open

    test_specs = [
        ("open(~24)",   wm_open, np.array([0, 0]), np.array([12, 12])),
        ("zig1(~30+)",  wm_zig1, np.array([0, 0]), np.array([12, 12])),
        ("zig2(~44+)",  wm_zig2, np.array([0, 0]), np.array([12, 12])),
        ("zig3(~58+)",  wm_zig3, np.array([0, 0]), np.array([12, 12])),
    ]

    rng = jax.random.PRNGKey(seed)

    print(f"\n  {'level':<14}  {'true_BFS':>8}  {'t':>5}  "
          f"{'critic_pred':>11}  {'error':>7}  {'grad_norm':>10}")
    print("  " + "-" * 63)

    records, preds_at_0 = [], []
    for name, wm, agent_rc, goal_rc in test_specs:
        theta = make_theta(wm, agent_rc, goal_rc)
        x0    = theta_to_diffusion(theta[None])

        wall_j   = jnp.array(wm)
        agent_xy = jnp.array([agent_rc[1], agent_rc[0]], dtype=jnp.uint32)
        goal_xy  = jnp.array([goal_rc[1], goal_rc[0]],  dtype=jnp.uint32)
        true_bfs = int(graph_util.shortest_path_len(wall_j, agent_xy, goal_xy))

        for t_val in [0, 100, 500, 999]:
            rng, sub = jax.random.split(rng)
            t_arr  = jnp.array([t_val])
            noise  = jax.random.normal(sub, x0.shape)
            x_t    = q_sample(x0, t_arr, noise, schedule)

            logits  = critic_model_fn(critic_params, x_t, t_arr)
            cp      = float(crit_mean(logits, num_bins=100).mean()) * 100

            def _obj(x):
                return crit_mean(critic_model_fn(critic_params, x, t_arr),
                                 num_bins=100).sum()
            gn = float(jnp.linalg.norm(jax.grad(_obj)(x_t)))

            print(f"  {name:<14}  {true_bfs:>8}  {t_val:>5}  "
                  f"{cp:>11.1f}  {cp - true_bfs:>+7.1f}  {gn:>10.4f}")
            records.append(dict(name=name, true_bfs=true_bfs, t=t_val,
                                critic_pred=cp, grad_norm=gn))
            if t_val == 0:
                preds_at_0.append((true_bfs, cp))

    preds_at_0_arr = np.array(preds_at_0)
    max_pred  = preds_at_0_arr[:, 1].max()
    max_true  = preds_at_0_arr[:, 0].max()
    saturated = (max_pred < 18.0) and (max_true > 20)
    print(f"\n  At t=0: max critic_pred={max_pred:.1f}  max true_BFS={max_true}")
    print(f"  Critic saturated at ~15: "
          f"{'YES — critic cannot extrapolate; gradient useless above ~15' if saturated else 'NO'}")

    return records, not saturated


# ---------------------------------------------------------------------------
# Stage 6 — Diagnosis
# ---------------------------------------------------------------------------

def run_stage6(s1_ok, s2_soft_ok, s2_hard_ok, s3_ok, s4_x0_ok, s5_ok):
    print("\n" + "=" * 70)
    print("STAGE 6: DIAGNOSIS")
    print("=" * 70)

    diagnoses = [
        (
            not s1_ok,
            "Stage 1 FAILED  — toy wall guidance broken",
            "Guidance sign / injection / scaling is wrong.\n"
            "      Check: omega sign convention in eps update, x0_guided formula,\n"
            "             re-derived eps direction, DDIM step.",
        ),
        (
            s1_ok and not s2_soft_ok,
            "Stage 1 OK, Stage 2 soft objective did NOT improve",
            "The objective gradient is broken for the chosen objective.\n"
            "      For soft_path: check cell_cost sign, INF sentinel, scan length.\n"
            "      For wall_count: this should always work; check vmap.",
        ),
        (
            s1_ok and s2_soft_ok and not s2_hard_ok,
            "Stage 2 soft OK but hard BFS did NOT improve",
            "Soft/hard objective misalignment.\n"
            "      The gradient steers theta but decoded hard BFS does not follow.\n"
            "      Try: more ascent steps, larger eta, or a validity penalty.",
        ),
        (
            s1_ok and s2_soft_ok and s2_hard_ok and not s3_ok,
            "Stage 2 OK but oracle DDIM guidance fails",
            "DDIM guidance injection / schedule is the bottleneck.\n"
            "      Try: larger omega, guidance_fraction=0.25 (late-only),\n"
            "           fewer diffusion steps so guidance has more relative weight.",
        ),
        (
            s3_ok and s5_ok is False,
            "Oracle DDIM works but learned critic is SATURATED",
            "Critic predicts ~15 for all hard levels — gradient is near-zero above 15.\n"
            "      Fix: retrain critic on longer levels (path > 15),\n"
            "           or replace critic with oracle soft_path for guidance.",
        ),
        (
            s3_ok and s5_ok is True,
            "Oracle DDIM works AND critic is not saturated",
            "The bottleneck may be omega scale or normalisation with the learned critic.\n"
            "      Try: sweep larger omega with the critic.",
        ),
    ]

    any_printed = False
    for active, title, detail in diagnoses:
        if active:
            print(f"\n  [!] {title}")
            print(f"      -> {detail}")
            any_printed = True

    if not any_printed:
        print("\n  All stages passed or skipped — no clear single bottleneck detected.")

    print("\n  Summary:")
    checks = [
        ("Stage 1: toy guidance sign OK",   s1_ok),
        ("Stage 2: soft objective improves", s2_soft_ok),
        ("Stage 2: hard BFS improves",       s2_hard_ok),
        ("Stage 3: oracle DDIM works",       s3_ok),
        ("Stage 4: x0-guidance works",       s4_x0_ok),
        ("Stage 5: critic not saturated",    s5_ok),
    ]
    for label, ok in checks:
        tag = "PASS" if ok is True else ("FAIL" if ok is False else "SKIP")
        print(f"    {tag:4s}  {label}")


# ---------------------------------------------------------------------------
# Checkpoint loaders
# ---------------------------------------------------------------------------

def load_diffusion(path):
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    params = jax.device_put(ckpt["ema_params"])
    model  = UNet(num_heads=4)
    fn     = jax.jit(lambda p, x, t: model.apply(p, x, t))
    return fn, params


def load_critic(path):
    from minimax.add.critic import EnvCritic
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    params = jax.device_put(ckpt["critic_params"])
    model  = EnvCritic()
    fn     = jax.jit(lambda p, x, t: model.apply(p, x, t))
    return fn, params


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--diffusion_ckpt", required=True)
    parser.add_argument("--critic_ckpt",    default=None,
                        help="Required for --objective critic_path and stage 5")
    parser.add_argument("--objective", default="soft_path",
                        choices=["wall_count", "soft_path", "critic_path"])
    parser.add_argument("--stages",    default="1,2,3,4,5,6",
                        help="Comma-separated stages to run, e.g. 1,2,3")
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    run = set(int(s.strip()) for s in args.stages.split(","))

    if args.objective == "critic_path" and args.critic_ckpt is None:
        parser.error("--critic_ckpt is required for --objective critic_path")

    print(f"JAX devices: {jax.devices()}")
    print(f"Diffusion ckpt: {args.diffusion_ckpt}")
    diff_fn, diff_params = load_diffusion(args.diffusion_ckpt)

    crit_fn = crit_params = None
    if args.critic_ckpt:
        print(f"Critic ckpt:    {args.critic_ckpt}")
        crit_fn, crit_params = load_critic(args.critic_ckpt)

    schedule = make_schedule()

    # Select primary objective for stages 2-4.
    if args.objective == "wall_count":
        obj_fn, obj_name = wall_count_obj, "wall_count"
    elif args.objective == "soft_path":
        obj_fn, obj_name = soft_path_obj, "soft_path"
    else:  # critic_path
        from minimax.add.critic import categorical_mean as _cm

        def _crit_obj(theta):
            x = theta_to_diffusion(theta)
            t = jnp.zeros(theta.shape[0], dtype=jnp.int32)
            return _cm(crit_fn(crit_params, x, t), num_bins=100) * 100

        obj_fn, obj_name = _crit_obj, "critic_path(t=0)"

    # Run stages and collect pass/fail booleans for stage 6.
    s1_ok = s2_soft = s2_hard = s3_ok = s4_ok = s5_ok = None

    if 1 in run:
        _, s1_ok = run_stage1(diff_fn, diff_params, schedule,
                               args.n_samples, args.ddim_steps, args.seed)

    if 2 in run:
        r2      = run_stage2(diff_fn, diff_params, schedule,
                              args.n_samples, args.ddim_steps,
                              obj_fn, obj_name, args.seed + 1)
        s2_soft = r2["soft_improved"]
        s2_hard = r2["hard_improved"]

    if 3 in run:
        _, s3_ok = run_stage3(diff_fn, diff_params, schedule,
                               args.n_samples, args.ddim_steps,
                               obj_fn, obj_name, args.seed + 2)

    if 4 in run:
        _, s4_ok = run_stage4(diff_fn, diff_params, schedule,
                               args.n_samples, args.ddim_steps,
                               obj_fn, obj_name, args.seed + 3)

    if 5 in run:
        if crit_fn is not None:
            _, s5_ok = run_stage5(crit_fn, crit_params, schedule, args.seed + 4)
        else:
            print("\nStage 5 skipped — no --critic_ckpt provided.")

    if 6 in run:
        run_stage6(
            s1_ok    = s1_ok    if s1_ok    is not None else True,
            s2_soft_ok = s2_soft if s2_soft is not None else True,
            s2_hard_ok = s2_hard if s2_hard is not None else True,
            s3_ok    = s3_ok    if s3_ok    is not None else True,
            s4_x0_ok = s4_ok    if s4_ok    is not None else True,
            s5_ok    = s5_ok,
        )


if __name__ == "__main__":
    main()
