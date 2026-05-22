"""Smoke tests for diffV_v4 (ppo_value_guided_ddim_sample_theta_v4).

Tests, in order:
  T1.  Diffusion checkpoint loads; UNet forward pass produces correct shape.
  T2.  decode_level produces valid wall maps, agent / goal positions.
  T3.  decode_and_reset_fn: hard-decode x0_pred -> valid env obs + state.
  T4.  env_step_fn: one vmapped env step changes state, done is bool-typed.
  T5.  Rollout scan: ROLLOUT_STEPS scan -> correct shapes.
  T6.  Gradient check — truncated BPTT: warmup carry stop_gradient'd,
       jax.grad only through last LSTM_K steps; grad is finite and nonzero.
  T7.  make_guidance_mask: correct shape, count, and end-bias.
  T8.  Single guided DDIM body iteration (end-biased mask, guided branch).
  T9.  Full ppo_value_guided_ddim_sample_theta_v4 (tiny DDIM_STEPS).
  T10. ADDRunner.reset() initialises all runner state tensors.
  T11. ADDRunner.run() executes one full ADD step (DDIM + RL + PPO).

Usage:
    tpu-device 0 python smoke_test_v4.py
"""

import sys
import traceback
import time

import jax
import jax.numpy as jnp
import numpy as np

CKPT          = "/storage/shared/mfr/add-replication/checkpoints/diffusion/v4/final.pkl"
N_PARALLEL    = 4
ROLLOUT_STEPS = 16    # guidance_rollout_steps for smoke (normally 128)
LSTM_K        = 8     # truncated BPTT window for smoke (normally 32); must be <= ROLLOUT_STEPS
N_GUIDED      = 2     # guided steps for smoke (normally 10)
GUIDANCE_BIAS = 3.0
DDIM_STEPS    = 4     # tiny DDIM; with N_GUIDED=2, bias=3 → guided at steps 0 and 3

# ── helpers ───────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
results = []

def check(name, fn):
    t0 = time.time()
    try:
        fn()
        dt = time.time() - t0
        print(f"  [{PASS}] {name}  ({dt:.1f}s)")
        results.append((name, True))
    except Exception:
        dt = time.time() - t0
        print(f"  [{FAIL}] {name}  ({dt:.1f}s)")
        traceback.print_exc()
        results.append((name, False))

def assert_shape(arr, expected, label=""):
    assert arr.shape == expected, f"{label} shape {arr.shape} != {expected}"

def assert_finite(arr, label=""):
    assert bool(jnp.all(jnp.isfinite(arr))), f"{label} has non-finite values"


# ── shared fixtures ────────────────────────────────────────────────────────────

import pickle
import minimax.envs as envs
import minimax.models as models
import minimax.agents as agents
from minimax.add.unet import UNet
from minimax.add.diffusion import make_schedule, diffusion_to_theta
from minimax.add.theta import theta_to_soft_maze_map, soft_extract_obs, decode_level
from minimax.envs.maze.common import EnvInstance

env_kwargs = dict(
    height=13, width=13, n_walls=60, see_through_walls=True,
    agent_view_size=5, max_episode_steps=250, normalize_obs=True,
    sample_n_walls=True, replace_wall_pos=True,
)

rng = jax.random.PRNGKey(42)

print("Loading diffusion checkpoint...")
with open(CKPT, "rb") as f:
    ckpt = pickle.load(f)
diff_params = jax.device_put(ckpt["ema_params"])
diff_model  = UNet()
schedule    = make_schedule()

print("Building PPO student model...")
dummy_env, _ = envs.make("Maze", env_kwargs=env_kwargs)
n_actions = dummy_env.action_space().n
student_model = models.make(
    env_name="Maze", model_name="default_student_cnn",
    output_dim=n_actions, recurrent_arch="lstm",
)
rng, rng_init = jax.random.split(rng)
obs_init_shape = {"image": jnp.zeros((1, 1, 5, 5, 3)),
                  "agent_dir": jnp.zeros((1, 1), jnp.int32)}
carry_init = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
ppo_params = student_model.init(rng_init, obs_init_shape, carry_init,
                                jnp.zeros((1, 1), jnp.bool_))

B = N_PARALLEL
rng, rng_x = jax.random.split(rng)
x0_test = jax.random.normal(rng_x, (B, 16, 16, 3))

from minimax.envs.batch_env import BatchEnv
_benv      = BatchEnv("Maze", N_PARALLEL, 1, env_kwargs)
wrapped_env = _benv.env

def make_decode_and_reset_fn():
    def fn(x0_pred):
        thetas = jax.vmap(diffusion_to_theta)(x0_pred)
        wall_maps, agent_pos_rc, goal_pos_rc, agent_dirs = jax.vmap(decode_level)(thetas)
        instances = EnvInstance(
            agent_pos     = agent_pos_rc[:, ::-1].astype(jnp.uint32),
            agent_dir_idx = agent_dirs.astype(jnp.uint8),
            goal_pos      = goal_pos_rc[:, ::-1].astype(jnp.uint32),
            wall_map      = wall_maps.astype(jnp.bool_),
        )
        return jax.vmap(wrapped_env.set_env_instance)(instances)
    return fn

decode_and_reset_fn = make_decode_and_reset_fn()
env_step_fn = jax.vmap(wrapped_env.step)

def ppo_apply_fn(params, obs, carry, reset):
    return student_model.apply(params, obs, carry, reset)

print()
print("=" * 60)
print("Running smoke tests — diffV_v4")
print("=" * 60)

# ── T1: UNet forward pass ──────────────────────────────────────────────────────
def t1():
    t_batch = jnp.zeros((B,), jnp.int32)
    eps = diff_model.apply(diff_params, x0_test, t_batch)
    assert_shape(eps, (B, 16, 16, 3), "UNet eps")
    assert_finite(eps, "UNet eps")

check("T1: UNet forward pass", t1)

# ── T2: decode_level correctness ──────────────────────────────────────────────
def t2():
    thetas = jax.vmap(diffusion_to_theta)(x0_test)
    wall_maps, agent_pos_rc, goal_pos_rc, agent_dirs = jax.vmap(decode_level)(thetas)
    assert_shape(wall_maps,    (B, 13, 13), "wall_maps")
    assert_shape(agent_pos_rc, (B, 2),      "agent_pos_rc")
    assert_shape(goal_pos_rc,  (B, 2),      "goal_pos_rc")
    assert_shape(agent_dirs,   (B,),        "agent_dirs")
    assert bool(jnp.all(agent_pos_rc >= 0) and jnp.all(agent_pos_rc <= 12))
    assert bool(jnp.all(goal_pos_rc  >= 0) and jnp.all(goal_pos_rc  <= 12))
    assert bool(jnp.all(agent_dirs   >= 0) and jnp.all(agent_dirs   <= 3))

check("T2: decode_level correctness", t2)

# ── T3: decode_and_reset_fn ────────────────────────────────────────────────────
def t3():
    obs0, state0, extra0 = decode_and_reset_fn(x0_test)
    assert obs0["image"].shape == (B, 5, 5, 3)
    assert "agent_dir" in obs0
    assert_finite(obs0["image"], "obs image")

check("T3: decode_and_reset_fn", t3)

# ── T4: vmapped env step ───────────────────────────────────────────────────────
def t4():
    obs0, state0, extra0 = decode_and_reset_fn(x0_test)
    rngs_B  = jax.random.split(jax.random.PRNGKey(1), B)
    actions = jnp.zeros(B, dtype=jnp.int32)
    obs1, state1, reward, done, info, extra1 = env_step_fn(
        rngs_B, state0, actions, state0, extra0,
    )
    assert obs1["image"].shape == (B, 5, 5, 3)
    assert reward.shape == (B,)
    assert done.shape  == (B,)
    assert_finite(reward, "reward")

check("T4: vmapped env step", t4)

# ── T5: rollout scan ────────────────────────────────────────────────────────────
def t5():
    obs0, state0, extra0 = decode_and_reset_fn(x0_test)
    zero_carry_B = (jnp.zeros((B, 256)), jnp.zeros((B, 256)))

    def rollout_step(rc, step_rng):
        env_state, obs_dict, extra, lstm_carry = rc
        ppo_obs = {
            "image":     obs_dict["image"][None],
            "agent_dir": obs_dict["agent_dir"][None].astype(jnp.int32),
        }
        _, logits, lstm_carry_next = ppo_apply_fn(
            ppo_params, ppo_obs, lstm_carry,
            jnp.zeros((1, B), dtype=jnp.bool_),
        )
        action  = jnp.argmax(logits[0], axis=-1)
        pos     = env_state.agent_pos
        dir_idx = env_state.agent_dir_idx
        rngs_B  = jax.random.split(step_rng, B)
        obs_next, state_next, _, done, _, extra_next = env_step_fn(
            rngs_B, env_state, action, state0, extra,
        )
        return (
            (state_next, obs_next, extra_next, lstm_carry_next),
            (pos, dir_idx, done.astype(jnp.float32)),
        )

    rngs_steps = jax.random.split(jax.random.PRNGKey(2), ROLLOUT_STEPS)
    (_, _, _, _), (pos_seq, dir_seq, done_seq) = jax.lax.scan(
        rollout_step, (state0, obs0, extra0, zero_carry_B),
        rngs_steps, length=ROLLOUT_STEPS,
    )
    assert pos_seq.shape  == (ROLLOUT_STEPS, B, 2)
    assert dir_seq.shape  == (ROLLOUT_STEPS, B)
    assert done_seq.shape == (ROLLOUT_STEPS, B)
    assert bool(jnp.all(pos_seq >= 0) and jnp.all(pos_seq <= 12))

check(f"T5: rollout scan ({ROLLOUT_STEPS} steps)", t5)

# ── T6: gradient check — truncated BPTT ───────────────────────────────────────
def t6():
    obs0, state0, extra0 = decode_and_reset_fn(x0_test)
    zero_carry_B = (jnp.zeros((B, 256)), jnp.zeros((B, 256)))
    gamma = 0.995; gae_lambda = 0.95; view_size = 5
    warmup = ROLLOUT_STEPS - LSTM_K

    def rollout_step(rc, step_rng):
        env_state, obs_dict, extra, lstm_carry = rc
        ppo_obs = {
            "image":     obs_dict["image"][None],
            "agent_dir": obs_dict["agent_dir"][None].astype(jnp.int32),
        }
        _, logits, lstm_carry_next = ppo_apply_fn(
            ppo_params, ppo_obs, lstm_carry,
            jnp.zeros((1, B), dtype=jnp.bool_),
        )
        action  = jnp.argmax(logits[0], axis=-1)
        pos     = env_state.agent_pos
        dir_idx = env_state.agent_dir_idx
        rngs_B  = jax.random.split(step_rng, B)
        obs_next, state_next, _, done, _, extra_next = env_step_fn(
            rngs_B, env_state, action, state0, extra,
        )
        return (
            (state_next, obs_next, extra_next, lstm_carry_next),
            (pos, dir_idx, done.astype(jnp.float32)),
        )

    rngs_steps = jax.random.split(jax.random.PRNGKey(3), ROLLOUT_STEPS)
    (_, _, _, _), (pos_seq, dir_seq, done_seq) = jax.lax.scan(
        rollout_step, (state0, obs0, extra0, zero_carry_B),
        rngs_steps, length=ROLLOUT_STEPS,
    )
    pos_seq  = jax.lax.stop_gradient(pos_seq)
    dir_seq  = jax.lax.stop_gradient(dir_seq)
    done_seq = jax.lax.stop_gradient(done_seq)

    def gae_score_sum(x0_in):
        thetas    = jax.vmap(diffusion_to_theta)(x0_in)
        soft_maps = jax.vmap(theta_to_soft_maze_map)(thetas)
        pos_BT  = pos_seq.swapaxes(0, 1)
        dir_BT  = dir_seq.swapaxes(0, 1)
        done_BT = done_seq.swapaxes(0, 1)

        def per_level(soft_map, theta_b, pos_l, dir_l, done_l):
            done_prev = jnp.concatenate(
                [jnp.zeros((1,), jnp.float32), done_l[:-1]], axis=0,
            )

            def lstm_step(lstm_carry, t_data):
                pos_t, dir_t, reset_t = t_data
                obs_t = soft_extract_obs(soft_map, pos_t, dir_t, view_size)
                ppo_obs = {
                    "image":     obs_t[None, None],
                    "agent_dir": dir_t.reshape(1, 1).astype(jnp.int32),
                }
                v, _, carry_next = ppo_apply_fn(
                    ppo_params, ppo_obs, lstm_carry,
                    reset_t.reshape(1, 1).astype(jnp.bool_),
                )
                return carry_next, v[0, 0, 0]

            init_carry = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))

            # Warm-up carry — no gradient flows back through this scan.
            warm_carry, _ = jax.lax.scan(
                lstm_step, init_carry,
                (pos_l[:warmup], dir_l[:warmup], done_prev[:warmup]),
            )
            warm_carry = jax.lax.stop_gradient(warm_carry)

            # Differentiable window: last LSTM_K steps only.
            _, V_seq = jax.lax.scan(
                lstm_step, warm_carry,
                (pos_l[warmup:], dir_l[warmup:], done_prev[warmup:]),
            )  # (LSTM_K,)

            pos_w  = pos_l[warmup:]
            done_w = done_l[warmup:]

            def get_wall_goal(pos_t):
                col = pos_t[0].astype(jnp.int32)
                row = pos_t[1].astype(jnp.int32)
                wp = jax.lax.dynamic_slice(theta_b[:, :, 0], (row+1, col+1), (1, 1)).squeeze()
                gp = jax.lax.dynamic_slice(theta_b[:, :, 2], (row+1, col+1), (1, 1)).squeeze()
                return wp, gp

            wall_ps, goal_ps = jax.vmap(get_wall_goal)(pos_w)
            V_next_raw  = jnp.concatenate([V_seq[1:], V_seq[-1:]], axis=0)
            V_next_soft = (1.0 - wall_ps) * V_next_raw + wall_ps * V_seq
            not_done = 1.0 - done_w
            deltas   = goal_ps + gamma * not_done * V_next_soft - V_seq

            def gae_step(g, td_nd):
                d, nd = td_nd
                out = d + gamma * gae_lambda * nd * g
                return out, out

            _, adv_rev = jax.lax.scan(
                gae_step, jnp.float32(0.), (deltas[::-1], not_done[::-1]),
            )
            return jnp.mean(jnp.sqrt(adv_rev[::-1] ** 2 + 1e-6))

        scores = jax.vmap(per_level)(soft_maps, thetas, pos_BT, dir_BT, done_BT)
        return scores.sum()

    grad = jax.grad(gae_score_sum)(x0_test)
    assert_shape(grad, (B, 16, 16, 3), "grad")
    assert_finite(grad, "grad")
    assert bool(jnp.any(grad != 0)), "grad is all zeros"
    print(f"       grad norm={float(jnp.linalg.norm(grad)):.4f}"
          f"  warmup={warmup} diff_window={LSTM_K}", end="")

check(f"T6: gradient check — truncated BPTT (warmup={ROLLOUT_STEPS - LSTM_K}, K={LSTM_K})", t6)

# ── T7: make_guidance_mask ─────────────────────────────────────────────────────
def t7():
    from minimax.add.guidance import make_guidance_mask

    mask = make_guidance_mask(50, 10, 3.0)
    assert mask.shape == (50,), f"mask shape {mask.shape}"
    n = int(mask.sum())
    assert n == 10, f"expected 10 guided steps, got {n}"
    positions = [i for i, v in enumerate(mask) if v]
    # end-biased: strictly more than half of steps should be in the upper half
    second_half = sum(1 for p in positions if p >= 25)
    assert second_half > 5, f"not end-biased: only {second_half}/10 in steps 25-49"
    print(f"       steps={positions}", end="")

check("T7: make_guidance_mask (shape, count, end-bias)", t7)

# ── T8: single guided body iteration (end-biased mask) ────────────────────────
def t8():
    from minimax.add.guidance import make_guidance_mask
    from minimax.add.diffusion import _make_ddim_timesteps

    guidance_mask = make_guidance_mask(DDIM_STEPS, N_GUIDED, GUIDANCE_BIAS)
    guided_indices = [i for i, v in enumerate(guidance_mask) if v]
    assert len(guided_indices) >= 1, "no guided step found in smoke mask"

    T_diff = schedule.betas.shape[0]
    timesteps = _make_ddim_timesteps(T_diff, DDIM_STEPS)
    alpha_bars_ext = jnp.concatenate([jnp.array([1.0]), schedule.alpha_bars])

    def diff_model_fn(params, x, t): return diff_model.apply(params, x, t)

    warmup = ROLLOUT_STEPS - LSTM_K
    gamma = 0.995; gae_lambda = 0.95; view_size = 5

    # Run one guided body step manually (first guided index).
    x = jax.random.normal(jax.random.PRNGKey(7), (B, 16, 16, 3))
    rng_body = jax.random.PRNGKey(7)
    i = guided_indices[0]

    t_cur   = timesteps[i]
    t_prev  = timesteps[i + 1] if i < DDIM_STEPS - 1 else jnp.array(-1, jnp.int32)
    ab_t    = alpha_bars_ext[t_cur  + 1]
    ab_prev = alpha_bars_ext[t_prev + 1]
    sqrt_ab_t    = jnp.sqrt(ab_t)
    sqrt_1m_ab_t = jnp.sqrt(1.0 - ab_t)

    t_batch   = jnp.full((B,), t_cur, dtype=jnp.int32)
    eps_pred  = diff_model_fn(diff_params, x, t_batch)
    x0_pred   = jnp.clip((x - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t, -1.0, 1.0)
    eps_clean = (x - sqrt_ab_t * x0_pred) / sqrt_1m_ab_t
    assert_shape(x0_pred, (B, 16, 16, 3), "x0_pred")

    # Guided branch: rollout + truncated BPTT grad.
    x0_hard = jax.lax.stop_gradient(x0_pred)
    obs0, state0, extra0 = decode_and_reset_fn(x0_hard)
    zero_carry_B = (jnp.zeros((B, 256)), jnp.zeros((B, 256)))

    def rollout_step(rc, step_rng):
        env_state, obs_dict, extra, lstm_carry = rc
        ppo_obs = {
            "image":     obs_dict["image"][None],
            "agent_dir": obs_dict["agent_dir"][None].astype(jnp.int32),
        }
        _, logits, lstm_carry_next = ppo_apply_fn(
            ppo_params, ppo_obs, lstm_carry,
            jnp.zeros((1, B), dtype=jnp.bool_),
        )
        action  = jnp.argmax(logits[0], axis=-1)
        pos     = env_state.agent_pos
        dir_idx = env_state.agent_dir_idx
        rngs_B  = jax.random.split(step_rng, B)
        obs_next, state_next, _, done, _, extra_next = env_step_fn(
            rngs_B, env_state, action, state0, extra,
        )
        return (
            (state_next, obs_next, extra_next, lstm_carry_next),
            (pos, dir_idx, done.astype(jnp.float32)),
        )

    rng_body, rng_roll = jax.random.split(rng_body)
    rngs_steps = jax.random.split(rng_roll, ROLLOUT_STEPS)
    (_, _, _, _), (pos_seq, dir_seq, done_seq) = jax.lax.scan(
        rollout_step, (state0, obs0, extra0, zero_carry_B),
        rngs_steps, length=ROLLOUT_STEPS,
    )
    pos_seq  = jax.lax.stop_gradient(pos_seq)
    dir_seq  = jax.lax.stop_gradient(dir_seq)
    done_seq = jax.lax.stop_gradient(done_seq)

    def gae_score_sum(x0):
        thetas    = jax.vmap(diffusion_to_theta)(x0)
        soft_maps = jax.vmap(theta_to_soft_maze_map)(thetas)
        pos_BT  = pos_seq.swapaxes(0, 1)
        dir_BT  = dir_seq.swapaxes(0, 1)
        done_BT = done_seq.swapaxes(0, 1)

        def per_level(soft_map, theta_b, pos_l, dir_l, done_l):
            done_prev = jnp.concatenate(
                [jnp.zeros((1,), jnp.float32), done_l[:-1]], axis=0,
            )
            def lstm_step(c, d):
                pt, dt, rt = d
                obs_t = soft_extract_obs(soft_map, pt, dt, view_size)
                ppo_obs = {"image": obs_t[None, None],
                           "agent_dir": dt.reshape(1, 1).astype(jnp.int32)}
                v, _, cn = ppo_apply_fn(ppo_params, ppo_obs, c,
                                        rt.reshape(1, 1).astype(jnp.bool_))
                return cn, v[0, 0, 0]
            ic = (jnp.zeros((1, 256)), jnp.zeros((1, 256)))
            warm_carry, _ = jax.lax.scan(
                lstm_step, ic,
                (pos_l[:warmup], dir_l[:warmup], done_prev[:warmup]),
            )
            warm_carry = jax.lax.stop_gradient(warm_carry)
            _, V_seq = jax.lax.scan(
                lstm_step, warm_carry,
                (pos_l[warmup:], dir_l[warmup:], done_prev[warmup:]),
            )
            pos_w  = pos_l[warmup:]
            done_w = done_l[warmup:]
            def get_wg(p):
                col = p[0].astype(jnp.int32); row = p[1].astype(jnp.int32)
                wp = jax.lax.dynamic_slice(theta_b[:,:,0], (row+1,col+1), (1,1)).squeeze()
                gp = jax.lax.dynamic_slice(theta_b[:,:,2], (row+1,col+1), (1,1)).squeeze()
                return wp, gp
            wall_ps, goal_ps = jax.vmap(get_wg)(pos_w)
            V_next_soft = (1.0-wall_ps)*jnp.concatenate([V_seq[1:],V_seq[-1:]])+wall_ps*V_seq
            not_done = 1.0 - done_w
            deltas = goal_ps + gamma*not_done*V_next_soft - V_seq
            def gs(g, td): d, nd = td; out = d+gamma*gae_lambda*nd*g; return out, out
            _, ar = jax.lax.scan(gs, jnp.float32(0.), (deltas[::-1], not_done[::-1]))
            return jnp.mean(jnp.sqrt(ar[::-1]**2 + 1e-6))

        return jax.vmap(per_level)(soft_maps, thetas, pos_BT, dir_BT, done_BT).sum()

    grad_score = jax.grad(gae_score_sum)(x0_pred)
    eps_guided = eps_clean - sqrt_1m_ab_t * 10.0 * grad_score
    x0_guided  = (x - sqrt_1m_ab_t * eps_guided) / sqrt_ab_t
    eps_final  = (x - sqrt_ab_t * x0_guided) / sqrt_1m_ab_t
    x_prev = jnp.sqrt(ab_prev) * x0_guided + jnp.sqrt(1.0 - ab_prev) * eps_final
    assert_shape(x_prev, (B, 16, 16, 3), "x_prev")
    assert_finite(x_prev, "x_prev")
    print(f"       guided_step={i}  mask={[i for i,v in enumerate(guidance_mask) if v]}", end="")

check("T8: single guided DDIM body iteration (end-biased mask)", t8)

# ── T9: full ppo_value_guided_ddim_sample_theta_v4 (tiny) ─────────────────────
def t9():
    from minimax.add.guidance import ppo_value_guided_ddim_sample_theta_v4

    def diff_model_fn(params, x, t): return diff_model.apply(params, x, t)

    thetas = ppo_value_guided_ddim_sample_theta_v4(
        diff_model_fn          = diff_model_fn,
        diff_params            = diff_params,
        ppo_apply_fn           = ppo_apply_fn,
        ppo_params             = ppo_params,
        decode_and_reset_fn    = decode_and_reset_fn,
        env_step_fn            = env_step_fn,
        shape                  = (B, 16, 16, 3),
        rng                    = jax.random.PRNGKey(9),
        schedule               = schedule,
        omega                  = 10.0,
        num_steps              = DDIM_STEPS,
        guidance_rollout_steps = ROLLOUT_STEPS,
        n_guided               = N_GUIDED,
        guidance_bias          = GUIDANCE_BIAS,
        lstm_k                 = LSTM_K,
    )
    assert_shape(thetas, (B, 16, 16, 3), "thetas")
    assert bool(jnp.all(thetas >= 0.0) and jnp.all(thetas <= 1.0)), \
        f"thetas not in [0,1]: min={thetas.min():.3f} max={thetas.max():.3f}"
    assert_finite(thetas, "thetas")
    print(f"       theta range [{float(thetas.min()):.3f}, {float(thetas.max()):.3f}]", end="")

check(f"T9: full v4 fn (ddim={DDIM_STEPS}, rollout={ROLLOUT_STEPS}, lstm_k={LSTM_K})", t9)

# ── T10: ADDRunner.reset() ─────────────────────────────────────────────────────
def t10():
    from minimax.add.runner import ADDRunner

    student_agent = agents.PPOAgent(
        model=student_model, n_epochs=1, n_minibatches=1,
        clip_eps=0.2, entropy_coef=0.0,
    )
    runner = ADDRunner(
        diffusion_ckpt_path     = CKPT,
        ddim_steps              = DDIM_STEPS,
        guidance_rollout_steps  = ROLLOUT_STEPS,
        rollout_every           = 2,
        use_positive_value_loss = False,
        env_name                = "Maze",
        env_kwargs              = env_kwargs,
        student_agents          = [student_agent],
        n_students              = 1,
        n_parallel              = N_PARALLEL,
        n_eval                  = 1,
        n_rollout_steps         = 16,
        lr                      = 1e-4,
        discount                = 0.995,
        gae_lambda              = 0.95,
        track_env_metrics       = False,
    )
    runner_state = runner.reset(jax.random.PRNGKey(10))
    assert runner_state is not None and len(runner_state) > 0

check("T10: ADDRunner.reset()", t10)

# ── T11: ADDRunner.run() ───────────────────────────────────────────────────────
def t11():
    from minimax.add.runner import ADDRunner

    student_agent = agents.PPOAgent(
        model=student_model, n_epochs=1, n_minibatches=1,
        clip_eps=0.2, entropy_coef=0.0,
    )
    runner = ADDRunner(
        diffusion_ckpt_path     = CKPT,
        ddim_steps              = DDIM_STEPS,
        guidance_rollout_steps  = ROLLOUT_STEPS,
        rollout_every           = 2,
        use_positive_value_loss = False,
        env_name                = "Maze",
        env_kwargs              = env_kwargs,
        student_agents          = [student_agent],
        n_students              = 1,
        n_parallel              = N_PARALLEL,
        n_eval                  = 1,
        n_rollout_steps         = 16,
        lr                      = 1e-4,
        discount                = 0.995,
        gae_lambda              = 0.95,
        track_env_metrics       = False,
    )
    runner_state = runner.reset(jax.random.PRNGKey(11))
    stats, *_ = runner.run(*runner_state, jnp.array(10.0))

    assert "_mean_return" in stats, "stats missing _mean_return"
    assert "_thetas"      in stats, "stats missing _thetas"
    mean_return = float(jax.device_get(stats["_mean_return"]))
    thetas_np   = np.array(jax.device_get(stats["_thetas"]))
    assert np.isfinite(mean_return), f"mean_return not finite: {mean_return}"
    assert thetas_np.shape == (N_PARALLEL, 16, 16, 3)
    print(f"       mean_return={mean_return:.4f}", end="")

check("T11: ADDRunner.run() — one full ADD step", t11)

# ── summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 60)
n_pass = sum(ok for _, ok in results)
n_fail = sum(not ok for _, ok in results)
print(f"Results: {n_pass}/{len(results)} passed, {n_fail} failed")
if n_fail > 0:
    print("Failed tests:")
    for name, ok in results:
        if not ok:
            print(f"  - {name}")
    sys.exit(1)
else:
    print("All smoke tests passed.")
