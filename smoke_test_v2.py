"""Smoke tests for diffV_v2 (ppo_value_guided_ddim_sample_theta_v2).

Tests, in order:
  T1. Diffusion checkpoint loads; UNet forward pass produces correct shape.
  T2. decode_level produces valid wall maps, agent / goal positions.
  T3. decode_and_reset_fn: hard-decode x0_pred -> valid env obs + state.
  T4. env_step_fn: one vmapped env step changes state, done is bool-typed.
  T5. Rollout scan: full guidance_rollout_steps scan -> correct shapes.
  T6. Gradient check: jax.grad(gae_score_sum)(x0_pred) finite & nonzero.
  T7. Single guided DDIM body(): one complete body iteration runs and
      returns (x_prev, rng) with correct shapes.
  T8. Full ppo_value_guided_ddim_sample_theta_v2 with num_steps=3 (tiny).
  T9. ADDRunner.reset() initialises all runner state tensors.
  T10. ADDRunner.run() executes one full ADD step (DDIM + RL + PPO).

Usage:
    tpu-device 0 python smoke_test_v2.py
"""

import sys
import traceback
import time

import jax
import jax.numpy as jnp
import numpy as np

CKPT = "/storage/shared/mfr/add-replication/checkpoints/diffusion/v4/final.pkl"
N_PARALLEL = 4      # small batch for speed
ROLLOUT_STEPS = 32  # much shorter than 256 for smoke test
DDIM_STEPS = 3      # tiny DDIM for T8


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
from minimax.add.theta import theta_to_soft_maze_map, soft_extract_obs
from minimax.add.theta import decode_level
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
obs_init_shape = {"image": jnp.zeros((1,1,5,5,3)),
                  "agent_dir": jnp.zeros((1,1), jnp.int32)}
carry_init = (jnp.zeros((1,256)), jnp.zeros((1,256)))
ppo_params = student_model.init(rng_init, obs_init_shape, carry_init,
                                jnp.zeros((1,1), jnp.bool_))

B = N_PARALLEL
rng, rng_x = jax.random.split(rng)
x0_test = jax.random.normal(rng_x, (B, 16, 16, 3))

print()
print("=" * 60)
print("Running smoke tests")
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
    thetas = jax.vmap(diffusion_to_theta)(x0_test)     # (B,16,16,3) in [0,1]
    wall_maps, agent_pos_rc, goal_pos_rc, agent_dirs = jax.vmap(decode_level)(thetas)
    assert_shape(wall_maps,    (B, 13, 13), "wall_maps")
    assert_shape(agent_pos_rc, (B, 2),      "agent_pos_rc")
    assert_shape(goal_pos_rc,  (B, 2),      "goal_pos_rc")
    assert_shape(agent_dirs,   (B,),        "agent_dirs")
    # Positions must be in [0,12]
    assert bool(jnp.all(agent_pos_rc >= 0) and jnp.all(agent_pos_rc <= 12)), \
        f"agent pos out of range: {agent_pos_rc}"
    assert bool(jnp.all(goal_pos_rc  >= 0) and jnp.all(goal_pos_rc  <= 12)), \
        f"goal pos out of range: {goal_pos_rc}"
    # Directions must be in [0,3]
    assert bool(jnp.all(agent_dirs >= 0) and jnp.all(agent_dirs <= 3)), \
        f"agent dirs out of range: {agent_dirs}"

check("T2: decode_level correctness", t2)

# ── T3: decode_and_reset_fn ────────────────────────────────────────────────────
# Use the wrapped env (MonitorReturnWrapper) — same as benv.env in DRRunner.
# set_env_instance returns (obs, state, extra); step returns (obs, state, reward, done, info, extra).
from minimax.envs.batch_env import BatchEnv
_benv = BatchEnv("Maze", N_PARALLEL, 1, env_kwargs)
wrapped_env = _benv.env   # MonitorReturnWrapper around Maze

def make_decode_and_reset_fn():
    def fn(x0_pred):
        thetas = jax.vmap(diffusion_to_theta)(x0_pred)
        wall_maps, agent_pos_rc, goal_pos_rc, agent_dirs = jax.vmap(decode_level)(thetas)
        instances = EnvInstance(
            agent_pos      = agent_pos_rc[:, ::-1].astype(jnp.uint32),
            agent_dir_idx  = agent_dirs.astype(jnp.uint8),
            goal_pos       = goal_pos_rc[:, ::-1].astype(jnp.uint32),
            wall_map       = wall_maps.astype(jnp.bool_),
        )
        return jax.vmap(wrapped_env.set_env_instance)(instances)  # (obs, state, extra)
    return fn

decode_and_reset_fn = make_decode_and_reset_fn()
env_step_fn = jax.vmap(wrapped_env.step)  # (key, state, action, reset_state, extra) -> (obs, state, r, done, info, extra)

def t3():
    obs0, state0, extra0 = decode_and_reset_fn(x0_test)
    img = obs0["image"]
    assert img.shape == (B, 5, 5, 3), f"obs image shape {img.shape}"
    assert "agent_dir" in obs0, "obs missing agent_dir key"
    assert_finite(img, "obs image")

check("T3: decode_and_reset_fn", t3)

# ── T4: vmapped env step ───────────────────────────────────────────────────────
def t4():
    obs0, state0, extra0 = decode_and_reset_fn(x0_test)
    rng_step = jax.random.PRNGKey(1)
    rngs_B   = jax.random.split(rng_step, B)
    actions  = jnp.zeros(B, dtype=jnp.int32)
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

    def ppo_apply_fn(params, obs, carry, reset):
        return student_model.apply(params, obs, carry, reset)

    def rollout_step(rc, step_rng):
        env_state, obs_dict, extra, lstm_carry = rc
        ppo_obs = {
            "image":     obs_dict["image"][None],
            "agent_dir": obs_dict["agent_dir"][None].astype(jnp.int32),
        }
        reset = jnp.zeros((1, B), dtype=jnp.bool_)
        _, logits, lstm_carry_next = ppo_apply_fn(ppo_params, ppo_obs, lstm_carry, reset)
        action = jnp.argmax(logits[0], axis=-1)
        pos     = env_state.agent_pos
        dir_idx = env_state.agent_dir_idx
        rngs_B  = jax.random.split(step_rng, B)
        obs_next, state_next, _, done, _, extra_next = env_step_fn(
            rngs_B, env_state, action, state0, extra,
        )
        return (state_next, obs_next, extra_next, lstm_carry_next), (pos, dir_idx, done.astype(jnp.float32))

    rng_roll = jax.random.PRNGKey(2)
    rngs_steps = jax.random.split(rng_roll, ROLLOUT_STEPS)
    (_, _, _, _), (pos_seq, dir_seq, done_seq) = jax.lax.scan(
        rollout_step, (state0, obs0, extra0, zero_carry_B), rngs_steps,
        length=ROLLOUT_STEPS,
    )
    assert pos_seq.shape  == (ROLLOUT_STEPS, B, 2), f"pos_seq {pos_seq.shape}"
    assert dir_seq.shape  == (ROLLOUT_STEPS, B),    f"dir_seq {dir_seq.shape}"
    assert done_seq.shape == (ROLLOUT_STEPS, B),    f"done_seq {done_seq.shape}"
    assert bool(jnp.all(pos_seq >= 0) and jnp.all(pos_seq <= 12)), "pos out of range"

check("T5: rollout scan (guidance_rollout_steps)", t5)

# ── T6: gradient check ─────────────────────────────────────────────────────────
def t6():
    obs0, state0, extra0 = decode_and_reset_fn(x0_test)
    zero_carry_B = (jnp.zeros((B, 256)), jnp.zeros((B, 256)))
    gamma = 0.995; gae_lambda = 0.95; view_size = 5

    def ppo_apply_fn(params, obs, carry, reset):
        return student_model.apply(params, obs, carry, reset)

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
        action = jnp.argmax(logits[0], axis=-1)
        pos     = env_state.agent_pos
        dir_idx = env_state.agent_dir_idx
        rngs_B  = jax.random.split(step_rng, B)
        obs_next, state_next, _, done, _, extra_next = env_step_fn(
            rngs_B, env_state, action, state0, extra,
        )
        return (state_next, obs_next, extra_next, lstm_carry_next), (pos, dir_idx, done.astype(jnp.float32))

    rng_roll = jax.random.PRNGKey(3)
    rngs_steps = jax.random.split(rng_roll, ROLLOUT_STEPS)
    (_, _, _, _), (pos_seq, dir_seq, done_seq) = jax.lax.scan(
        rollout_step, (state0, obs0, extra0, zero_carry_B), rngs_steps,
        length=ROLLOUT_STEPS,
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
            done_prev = jnp.concatenate([jnp.zeros((1,), jnp.float32), done_l[:-1]])

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
            _, V_seq = jax.lax.scan(lstm_step, init_carry, (pos_l, dir_l, done_prev))
            V_next_raw = jnp.concatenate([V_seq[1:], V_seq[-1:]])

            def get_wall_goal(pos_t):
                col = pos_t[0].astype(jnp.int32)
                row = pos_t[1].astype(jnp.int32)
                wp = jax.lax.dynamic_slice(theta_b[:,:,0], (row+1,col+1), (1,1)).squeeze()
                gp = jax.lax.dynamic_slice(theta_b[:,:,2], (row+1,col+1), (1,1)).squeeze()
                return wp, gp

            wall_ps, goal_ps = jax.vmap(get_wall_goal)(pos_l)
            V_next_soft = (1.0 - wall_ps)*V_next_raw + wall_ps*V_seq
            not_done = 1.0 - done_l
            deltas   = goal_ps + gamma*not_done*V_next_soft - V_seq

            def gae_step(g, td_nd):
                d, nd = td_nd
                out = d + gamma*gae_lambda*nd*g
                return out, out

            _, adv_rev = jax.lax.scan(gae_step, jnp.float32(0.), (deltas[::-1], not_done[::-1]))
            advantages = adv_rev[::-1]
            return jnp.mean(jnp.sqrt(advantages**2 + 1e-6))

        scores = jax.vmap(per_level)(soft_maps, thetas, pos_BT, dir_BT, done_BT)
        return scores.sum()

    grad = jax.grad(gae_score_sum)(x0_test)
    assert_shape(grad, (B, 16, 16, 3), "grad")
    assert_finite(grad, "grad")
    assert bool(jnp.any(grad != 0)), "grad is all zeros"
    print(f"       grad norm={float(jnp.linalg.norm(grad)):.4f}", end="")

check("T6: gradient check (real PPO LSTM + real env rollout)", t6)

# ── T7: single guided DDIM body iteration ─────────────────────────────────────
def t7():
    from minimax.add.guidance import ppo_value_guided_ddim_sample_theta_v2
    from minimax.add.diffusion import _make_ddim_timesteps

    T_diff = schedule.betas.shape[0]
    timesteps = _make_ddim_timesteps(T_diff, 50)
    alpha_bars_ext = jnp.concatenate([jnp.array([1.0]), schedule.alpha_bars])

    rng_body = jax.random.PRNGKey(7)
    x_init   = jax.random.normal(rng_body, (B, 16, 16, 3))

    def diff_model_fn(params, x, t): return diff_model.apply(params, x, t)
    def ppo_apply_fn(params, obs, carry, reset): return student_model.apply(params, obs, carry, reset)

    # Manually run one body step (i=0)
    i = 0
    x, rng_carry = x_init, rng_body
    t_cur  = timesteps[i]
    t_prev = timesteps[i + 1]
    ab_t   = alpha_bars_ext[t_cur  + 1]
    ab_prev = alpha_bars_ext[t_prev + 1]
    sqrt_ab_t    = jnp.sqrt(ab_t)
    sqrt_1m_ab_t = jnp.sqrt(1.0 - ab_t)

    t_batch  = jnp.full((B,), t_cur, dtype=jnp.int32)
    eps_pred = diff_model_fn(diff_params, x, t_batch)
    x0_pred  = jnp.clip((x - sqrt_1m_ab_t * eps_pred) / sqrt_ab_t, -1.0, 1.0)
    eps_clean = (x - sqrt_ab_t * x0_pred) / sqrt_1m_ab_t
    assert_shape(x0_pred, (B, 16, 16, 3), "x0_pred")

    # decode_and_reset_fn
    x0_hard = jax.lax.stop_gradient(x0_pred)
    obs0, state0, extra0 = decode_and_reset_fn(x0_hard)
    zero_carry_B = (jnp.zeros((B, 256)), jnp.zeros((B, 256)))

    # 4-step rollout inside body
    T_roll = 4
    def rollout_step(rc, step_rng):
        env_state, obs_dict, extra, lstm_carry = rc
        ppo_obs = {"image": obs_dict["image"][None],
                   "agent_dir": obs_dict["agent_dir"][None].astype(jnp.int32)}
        _, logits, lstm_carry_next = ppo_apply_fn(
            ppo_params, ppo_obs, lstm_carry,
            jnp.zeros((1, B), dtype=jnp.bool_))
        action = jnp.argmax(logits[0], axis=-1)
        pos = env_state.agent_pos; dir_idx = env_state.agent_dir_idx
        rngs_B = jax.random.split(step_rng, B)
        obs_next, state_next, _, done, _, extra_next = env_step_fn(
            rngs_B, env_state, action, state0, extra)
        return (state_next, obs_next, extra_next, lstm_carry_next), (pos, dir_idx, done.astype(jnp.float32))

    rng_carry, rng_roll = jax.random.split(rng_carry)
    rngs_steps = jax.random.split(rng_roll, T_roll)
    (_, _, _, _), (pos_seq, dir_seq, done_seq) = jax.lax.scan(
        rollout_step, (state0, obs0, extra0, zero_carry_B), rngs_steps, length=T_roll)
    pos_seq  = jax.lax.stop_gradient(pos_seq)
    dir_seq  = jax.lax.stop_gradient(dir_seq)
    done_seq = jax.lax.stop_gradient(done_seq)

    # compute grad
    gamma = 0.995; gae_lambda = 0.95; view_size = 5
    def gae_score_sum(x0):
        thetas = jax.vmap(diffusion_to_theta)(x0)
        soft_maps = jax.vmap(theta_to_soft_maze_map)(thetas)
        pos_BT  = pos_seq.swapaxes(0, 1)
        dir_BT  = dir_seq.swapaxes(0, 1)
        done_BT = done_seq.swapaxes(0, 1)
        def per_level(sm, th, pl, dl, donl):
            done_prev = jnp.concatenate([jnp.zeros((1,), jnp.float32), donl[:-1]])
            def ls(c, d):
                pt, dt, rt = d
                obs_t = soft_extract_obs(sm, pt, dt, view_size)
                ppo_obs = {"image": obs_t[None,None], "agent_dir": dt.reshape(1,1).astype(jnp.int32)}
                v, _, cn = ppo_apply_fn(ppo_params, ppo_obs, c, rt.reshape(1,1).astype(jnp.bool_))
                return cn, v[0,0,0]
            ic = (jnp.zeros((1,256)), jnp.zeros((1,256)))
            _, V_seq = jax.lax.scan(ls, ic, (pl, dl, done_prev))
            V_next = jnp.concatenate([V_seq[1:], V_seq[-1:]])
            not_done = 1.0 - donl
            deltas = gamma*not_done*V_next - V_seq
            def gs(g, td): d, nd = td; out = d + gamma*gae_lambda*nd*g; return out, out
            _, ar = jax.lax.scan(gs, jnp.float32(0.), (deltas[::-1], not_done[::-1]))
            adv = ar[::-1]
            return jnp.mean(jnp.sqrt(adv**2 + 1e-6))
        scores = jax.vmap(per_level)(soft_maps, thetas, pos_BT, dir_BT, done_BT)
        return scores.sum()

    grad_score = jax.grad(gae_score_sum)(x0_pred)
    eps_guided = eps_clean - sqrt_1m_ab_t * 5.0 * grad_score
    x0_guided  = (x - sqrt_1m_ab_t * eps_guided) / sqrt_ab_t
    eps_final  = (x - sqrt_ab_t * x0_guided) / sqrt_1m_ab_t
    x_prev = jnp.sqrt(ab_prev) * x0_guided + jnp.sqrt(1.0 - ab_prev) * eps_final
    assert_shape(x_prev, (B, 16, 16, 3), "x_prev")
    assert_finite(x_prev, "x_prev")

check("T7: single guided DDIM body iteration", t7)

# ── T8: full ppo_value_guided_ddim_sample_theta_v2 (tiny) ─────────────────────
def t8():
    from minimax.add.guidance import ppo_value_guided_ddim_sample_theta_v2

    def diff_model_fn(params, x, t): return diff_model.apply(params, x, t)
    def ppo_apply_fn(params, obs, carry, reset): return student_model.apply(params, obs, carry, reset)

    rng_t8 = jax.random.PRNGKey(8)
    thetas = ppo_value_guided_ddim_sample_theta_v2(
        diff_model_fn      = diff_model_fn,
        diff_params        = diff_params,
        ppo_apply_fn       = ppo_apply_fn,
        ppo_params         = ppo_params,
        decode_and_reset_fn = decode_and_reset_fn,
        env_step_fn        = env_step_fn,
        shape              = (B, 16, 16, 3),
        rng                = rng_t8,
        schedule           = schedule,
        omega              = 5.0,
        num_steps          = DDIM_STEPS,
        guidance_rollout_steps = ROLLOUT_STEPS,
    )
    assert_shape(thetas, (B, 16, 16, 3), "thetas")
    assert bool(jnp.all(thetas >= 0.0) and jnp.all(thetas <= 1.0)), \
        f"thetas not in [0,1]: min={thetas.min():.3f} max={thetas.max():.3f}"
    assert_finite(thetas, "thetas")
    print(f"       theta range [{float(thetas.min()):.3f}, {float(thetas.max()):.3f}]", end="")

check(f"T8: full guidance fn (num_steps={DDIM_STEPS}, rollout={ROLLOUT_STEPS})", t8)

# ── T8b: full ppo_value_guided_ddim_sample_theta_v3 (K-step, tiny) ────────────
def t8b():
    from minimax.add.guidance import ppo_value_guided_ddim_sample_theta_v3

    def diff_model_fn(params, x, t): return diff_model.apply(params, x, t)
    def ppo_apply_fn(params, obs, carry, reset): return student_model.apply(params, obs, carry, reset)

    rng_t8b = jax.random.PRNGKey(88)
    # rollout_every=2 with num_steps=3: guided at step 0 and 2, plain DDIM at step 1
    thetas = ppo_value_guided_ddim_sample_theta_v3(
        diff_model_fn       = diff_model_fn,
        diff_params         = diff_params,
        ppo_apply_fn        = ppo_apply_fn,
        ppo_params          = ppo_params,
        decode_and_reset_fn = decode_and_reset_fn,
        env_step_fn         = env_step_fn,
        shape               = (B, 16, 16, 3),
        rng                 = rng_t8b,
        schedule            = schedule,
        omega               = 10.0,
        num_steps           = DDIM_STEPS,
        guidance_rollout_steps = ROLLOUT_STEPS,
        rollout_every       = 2,
    )
    assert_shape(thetas, (B, 16, 16, 3), "thetas_v3")
    assert bool(jnp.all(thetas >= 0.0) and jnp.all(thetas <= 1.0)), \
        f"thetas not in [0,1]: min={thetas.min():.3f} max={thetas.max():.3f}"
    assert_finite(thetas, "thetas_v3")
    print(f"       theta range [{float(thetas.min()):.3f}, {float(thetas.max()):.3f}]", end="")

check(f"T8b: v3 K-step guidance (rollout_every=2, num_steps={DDIM_STEPS})", t8b)

# ── T9: ADDRunner.reset() ──────────────────────────────────────────────────────
def t9():
    from minimax.add.runner import ADDRunner
    import minimax.agents as agents

    student_agent = agents.PPOAgent(
        model=student_model, n_epochs=1, n_minibatches=1,
        clip_eps=0.2, entropy_coef=0.0,
    )
    runner = ADDRunner(
        diffusion_ckpt_path        = CKPT,
        ddim_steps                 = DDIM_STEPS,
        guidance_rollout_steps     = ROLLOUT_STEPS,
        use_positive_value_loss    = False,
        env_name                   = "Maze",
        env_kwargs                 = env_kwargs,
        student_agents             = [student_agent],
        n_students                 = 1,
        n_parallel                 = N_PARALLEL,
        n_eval                     = 1,
        n_rollout_steps            = 16,  # short
        lr                         = 1e-4,
        discount                   = 0.995,
        gae_lambda                 = 0.95,
        track_env_metrics          = False,
    )
    rng_r = jax.random.PRNGKey(9)
    runner_state = runner.reset(rng_r)
    assert runner_state is not None, "reset returned None"
    assert len(runner_state) > 0, "runner_state is empty"

check("T9: ADDRunner.reset()", t9)

# ── T10: ADDRunner.run() — one full ADD step ───────────────────────────────────
def t10():
    from minimax.add.runner import ADDRunner
    import minimax.agents as agents

    student_agent = agents.PPOAgent(
        model=student_model, n_epochs=1, n_minibatches=1,
        clip_eps=0.2, entropy_coef=0.0,
    )
    runner = ADDRunner(
        diffusion_ckpt_path        = CKPT,
        ddim_steps                 = DDIM_STEPS,
        guidance_rollout_steps     = ROLLOUT_STEPS,
        use_positive_value_loss    = False,
        env_name                   = "Maze",
        env_kwargs                 = env_kwargs,
        student_agents             = [student_agent],
        n_students                 = 1,
        n_parallel                 = N_PARALLEL,
        n_eval                     = 1,
        n_rollout_steps            = 16,
        lr                         = 1e-4,
        discount                   = 0.995,
        gae_lambda                 = 0.95,
        track_env_metrics          = False,
    )
    rng_r = jax.random.PRNGKey(10)
    runner_state = runner.reset(rng_r)
    rng_r, _ = jax.random.split(rng_r)
    omega = jnp.array(5.0)

    stats, *runner_state_new = runner.run(*runner_state, omega)

    assert "_mean_return" in stats, "stats missing _mean_return"
    assert "_thetas"      in stats, "stats missing _thetas"
    mean_return = float(jax.device_get(stats["_mean_return"]))
    thetas_np   = np.array(jax.device_get(stats["_thetas"]))
    assert np.isfinite(mean_return), f"mean_return not finite: {mean_return}"
    assert thetas_np.shape == (N_PARALLEL, 16, 16, 3), f"thetas shape {thetas_np.shape}"
    print(f"       mean_return={mean_return:.4f}", end="")

check("T10: ADDRunner.run() — one full ADD step (v2)", t10)

# ── T11: ADDRunner.run() with rollout_every=2 (v3) ────────────────────────────
def t11():
    from minimax.add.runner import ADDRunner
    import minimax.agents as agents

    student_agent = agents.PPOAgent(
        model=student_model, n_epochs=1, n_minibatches=1,
        clip_eps=0.2, entropy_coef=0.0,
    )
    runner = ADDRunner(
        diffusion_ckpt_path        = CKPT,
        ddim_steps                 = DDIM_STEPS,
        guidance_rollout_steps     = ROLLOUT_STEPS,
        rollout_every              = 2,
        use_positive_value_loss    = False,
        env_name                   = "Maze",
        env_kwargs                 = env_kwargs,
        student_agents             = [student_agent],
        n_students                 = 1,
        n_parallel                 = N_PARALLEL,
        n_eval                     = 1,
        n_rollout_steps            = 16,
        lr                         = 1e-4,
        discount                   = 0.995,
        gae_lambda                 = 0.95,
        track_env_metrics          = False,
    )
    rng_r = jax.random.PRNGKey(11)
    runner_state = runner.reset(rng_r)
    omega = jnp.array(10.0)

    stats, *runner_state_new = runner.run(*runner_state, omega)

    assert "_mean_return" in stats, "stats missing _mean_return"
    assert "_thetas"      in stats, "stats missing _thetas"
    mean_return = float(jax.device_get(stats["_mean_return"]))
    thetas_np   = np.array(jax.device_get(stats["_thetas"]))
    assert np.isfinite(mean_return), f"mean_return not finite: {mean_return}"
    assert thetas_np.shape == (N_PARALLEL, 16, 16, 3), f"thetas shape {thetas_np.shape}"
    print(f"       mean_return={mean_return:.4f}", end="")

check("T11: ADDRunner.run() — one full ADD step (v3, rollout_every=2)", t11)

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
