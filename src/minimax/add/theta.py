"""
Encoding and decoding between maze levels and diffusion-model images.

The diffusion model operates on theta in R^{16x16x3}. This module converts
between (wall_map, agent_pos, goal_pos, agent_dir) and theta images.

Soft maze utilities:
  theta_to_soft_maze_map  — differentiable theta → padded maze_map (21,21,3)
  soft_extract_obs        — differentiable egocentric 5×5 obs crop
Both are used by the PPO-value guided DDIM in guidance.py.
"""

import jax
import jax.numpy as jnp


GRID_SIZE = 13
IMG_SIZE = 16

# Padded maze_map dimensions (matches make_maze_map with pad_obs=True, view_size=5).
_MAZE_PAD = 4           # view_size - 1
_MAZE_H = 13 + 2 * _MAZE_PAD   # 21
_MAZE_W = 13 + 2 * _MAZE_PAD   # 21

# Normalized tile values (/10) for MiniGrid encoding used by maze env:
#   wall:  OBJECT=2, COLOR_grey=5,   state=0  → [0.2, 0.5, 0.0]
#   empty: OBJECT=1, COLOR=0,        state=0  → [0.1, 0.0, 0.0]
#   goal:  OBJECT=8, COLOR_green=1,  state=0  → [0.8, 0.1, 0.0]
# (agent tile is omitted: see_agent=False replaces it with empty in obs)
_WALL_TILE  = jnp.array([0.2, 0.5, 0.0], dtype=jnp.float32)
_EMPTY_TILE = jnp.array([0.1, 0.0, 0.0], dtype=jnp.float32)
_GOAL_TILE  = jnp.array([0.8, 0.1, 0.0], dtype=jnp.float32)

# Per-direction start offsets (in padded maze_map frame) relative to the
# agent's padded position [row+pad, col+pad], so that dynamic_slice gives the
# same 5×5 window as get_obs() in maze.py.
# Derived by tracing get_obs() for each of the four directions.
# Layout: row_offset, col_offset for dir in {0=right, 1=down, 2=left, 3=up}.
_OBS_ROW_OFFSETS = jnp.array([-2,  0, -2, -4], dtype=jnp.int32)
_OBS_COL_OFFSETS = jnp.array([ 0, -2, -4, -2], dtype=jnp.int32)

# Direction offsets as (row_offset, col_offset).
# 0=right, 1=down, 2=left, 3=up.
DIR_OFFSETS = jnp.array([
    [0, 1],   # right
    [1, 0],   # down
    [0, -1],  # left
    [-1, 0],  # up
], dtype=jnp.int32)


def encode_level(
    wall_map: jnp.ndarray,   # (13, 13) bool
    agent_pos: jnp.ndarray,  # (2,) inner coords (row, col)
    goal_pos: jnp.ndarray,   # (2,) inner coords (row, col)
    agent_dir: jnp.ndarray,  # scalar int in {0,1,2,3}
) -> jnp.ndarray:            # (16, 16, 3) float in [0, 1]
    theta = jnp.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=jnp.float32)

    # Channel 0: walls. Place inner grid walls at padded coords.
    theta = theta.at[1:14, 1:14, 0].set(wall_map.astype(jnp.float32))

    # Paint thick wall border in padding (matching ADD paper encoding).
    theta = theta.at[0, :, 0].set(1.0)
    theta = theta.at[:, 0, 0].set(1.0)
    theta = theta.at[14:, :, 0].set(1.0)
    theta = theta.at[:, 14:, 0].set(1.0)

    # Channel 1: agent start (1.0) and direction marker (0.5).
    agent_padded = agent_pos + 1
    theta = theta.at[agent_padded[0], agent_padded[1], 1].set(1.0)
    offset = DIR_OFFSETS[agent_dir]
    dir_padded = agent_padded + offset
    theta = theta.at[dir_padded[0], dir_padded[1], 1].set(0.5)

    # Channel 2: goal.
    goal_padded = goal_pos + 1
    theta = theta.at[goal_padded[0], goal_padded[1], 2].set(1.0)

    return theta


def decode_level(
    theta: jnp.ndarray,  # (16, 16, 3) float in [0, 1]
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Returns (wall_map, agent_pos, goal_pos, agent_dir) in inner coords."""
    inner = theta[1:14, 1:14]  # (13, 13, 3)

    # Walls: threshold at 0.5. No forced border — minimax handles grid
    # boundaries via position clamping, matching the ADD paper and DR generator.
    wall_map = inner[:, :, 0] > 0.5

    # Goal: brightest blue pixel in inner region.
    goal_flat = jnp.argmax(inner[:, :, 2].ravel())
    goal_row, goal_col = jnp.divmod(goal_flat, GRID_SIZE)
    goal_pos = jnp.array([goal_row, goal_col], dtype=jnp.int32)

    # Agent: brightest green pixel in inner region.
    agent_flat = jnp.argmax(inner[:, :, 1].ravel())
    agent_row, agent_col = jnp.divmod(agent_flat, GRID_SIZE)
    agent_pos = jnp.array([agent_row, agent_col], dtype=jnp.int32)

    # Direction: check 4 neighbours of agent in PADDED image's channel 1.
    # The neighbour with the highest value (the 0.5 marker) gives direction.
    agent_padded_r = agent_row + 1
    agent_padded_c = agent_col + 1
    neighbour_vals = jnp.array([
        theta[agent_padded_r + DIR_OFFSETS[0, 0],
              agent_padded_c + DIR_OFFSETS[0, 1], 1],
        theta[agent_padded_r + DIR_OFFSETS[1, 0],
              agent_padded_c + DIR_OFFSETS[1, 1], 1],
        theta[agent_padded_r + DIR_OFFSETS[2, 0],
              agent_padded_c + DIR_OFFSETS[2, 1], 1],
        theta[agent_padded_r + DIR_OFFSETS[3, 0],
              agent_padded_c + DIR_OFFSETS[3, 1], 1],
    ])
    agent_dir = jnp.argmax(neighbour_vals)

    # Clear walls at agent and goal positions.
    wall_map = wall_map.at[agent_row, agent_col].set(False)
    wall_map = wall_map.at[goal_row, goal_col].set(False)

    return wall_map, agent_pos, goal_pos, agent_dir


def sample_random_level(
    rng: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sample a random maze level matching the paper's distribution.

    Returns (wall_map, agent_pos, goal_pos, agent_dir) in inner coords.
    """
    rng_nwalls, rng_walls, rng_agent, rng_goal, rng_dir = jax.random.split(rng, 5)

    # Sample walls on the full 13x13 grid (169 cells), matching the ADD paper.
    # No explicit border — border is painted in the 16x16 padding by encode_level.
    grid_size_sq = GRID_SIZE * GRID_SIZE  # 169

    n_walls = jax.random.randint(rng_nwalls, (), 0, 60)
    wall_indices = jax.random.randint(rng_walls, (60,), 0, grid_size_sq)
    mask = (jnp.arange(60) < n_walls).astype(jnp.int32)

    wall_flat = jnp.zeros(grid_size_sq, dtype=jnp.int32)
    wall_flat = wall_flat.at[wall_indices].add(mask)
    wall_map = (wall_flat > 0).reshape(GRID_SIZE, GRID_SIZE)

    # Agent: uniform random from all 169 cells.
    agent_flat = jax.random.randint(rng_agent, (), 0, grid_size_sq)
    agent_row = agent_flat // GRID_SIZE
    agent_col = agent_flat % GRID_SIZE
    agent_pos = jnp.array([agent_row, agent_col], dtype=jnp.int32)

    # Goal: uniform random from 168 cells (excluding agent position).
    goal_flat = jax.random.randint(rng_goal, (), 0, grid_size_sq - 1)
    goal_flat = jnp.where(goal_flat >= agent_flat, goal_flat + 1, goal_flat)
    goal_row = goal_flat // GRID_SIZE
    goal_col = goal_flat % GRID_SIZE
    goal_pos = jnp.array([goal_row, goal_col], dtype=jnp.int32)

    # Clear walls at agent and goal positions.
    wall_map = wall_map.at[agent_row, agent_col].set(False)
    wall_map = wall_map.at[goal_row, goal_col].set(False)

    # Random direction.
    agent_dir = jax.random.randint(rng_dir, (), 0, 4)

    return wall_map, agent_pos, goal_pos, agent_dir


def sample_random_theta(rng: jnp.ndarray) -> jnp.ndarray:
    """Sample a random level and encode it as a theta image. Vmappable."""
    wall_map, agent_pos, goal_pos, agent_dir = sample_random_level(rng)
    return encode_level(wall_map, agent_pos, goal_pos, agent_dir)


# ---------------------------------------------------------------------------
# Differentiable soft maze utilities (used by PPO-value guided DDIM)
# ---------------------------------------------------------------------------

def theta_to_soft_maze_map(theta: jnp.ndarray, sharpness: float = 20.0) -> jnp.ndarray:
    """Differentiable theta (16,16,3) → padded soft maze_map (21,21,3).

    Produces a differentiable approximation of the integer maze_map built by
    make_maze_map(pad_obs=True), normalized by /10 to match normalize_obs=True.
    Used to extract differentiable egocentric observations for PPO-value guidance.

    The agent tile is omitted (treated as empty) because see_agent=False in the
    training env replaces the agent cell with empty in get_obs().
    """
    inner = theta[1:14, 1:14]          # (13, 13, 3) — strip theta border padding

    wall_p  = inner[..., 0]            # wall probability (channel 0)
    # Channel 1 encodes agent (1.0) and direction marker (0.5); threshold at 0.75.
    agent_p = jax.nn.sigmoid(sharpness * (inner[..., 1] - 0.75))
    goal_p  = inner[..., 2]            # goal probability (channel 2)
    empty_p = jnp.clip(1.0 - wall_p - agent_p - goal_p, 0.0, 1.0)

    # Weighted soft tile: each pixel is a mixture of tile types.
    # ch0 = object-type channel / 10, ch1 = color channel / 10, ch2 = 0
    ch0 = 0.2 * wall_p + 0.1 * empty_p + 0.8 * goal_p + 1.0 * agent_p
    ch1 = 0.5 * wall_p + 0.0 * empty_p + 0.1 * goal_p + 0.0 * agent_p
    ch2 = jnp.zeros_like(ch0)
    inner_soft = jnp.stack([ch0, ch1, ch2], axis=-1)   # (13, 13, 3)

    # Initialize padded map with wall tile then write inner grid.
    padded = jnp.tile(_WALL_TILE[None, None, :], (_MAZE_H, _MAZE_W, 1))   # (21,21,3)
    padded = padded.at[_MAZE_PAD:-_MAZE_PAD, _MAZE_PAD:-_MAZE_PAD, :].set(inner_soft)

    # Re-apply surrounding wall border at wall_start = pad-1 = 3 (matches
    # make_maze_map's explicit border, which sits just inside the padding).
    ws = _MAZE_PAD - 1          # 3
    we = GRID_SIZE + _MAZE_PAD  # 17
    padded = padded.at[ws, ws:we + 1, :].set(_WALL_TILE)   # top
    padded = padded.at[we, ws:we + 1, :].set(_WALL_TILE)   # bottom
    padded = padded.at[ws:we + 1, ws, :].set(_WALL_TILE)   # left
    padded = padded.at[ws:we + 1, we, :].set(_WALL_TILE)   # right

    return padded   # (21, 21, 3)


def soft_extract_obs(
    soft_padded_map: jnp.ndarray,   # (21, 21, 3)
    agent_pos_xy: jnp.ndarray,      # (2,) in (col, row) = (x, y) inner coords
    agent_dir_idx: jnp.ndarray,     # scalar int {0,1,2,3}
    view_size: int = 5,
) -> jnp.ndarray:                   # (view_size, view_size, 3)
    """Differentiable egocentric obs extraction matching maze.py get_obs().

    Uses jax.lax.dynamic_slice (differentiable through values) to crop the
    soft padded maze map at the agent's position, then applies the same rot90
    rotation that get_obs() uses.  Positions are integer (non-differentiable
    indices from cached trajectory); values in the map are differentiable w.r.t.
    theta.

    agent_pos_xy: (col, row) — matches EnvState.agent_pos convention.
    """
    col = agent_pos_xy[0].astype(jnp.int32)
    row = agent_pos_xy[1].astype(jnp.int32)

    # Padded map position of agent: [row + pad, col + pad].
    # Slice start relative to that, derived from _OBS_ROW/COL_OFFSETS.
    pad = _MAZE_PAD
    row_start = row + pad + _OBS_ROW_OFFSETS[agent_dir_idx]
    col_start = col + pad + _OBS_COL_OFFSETS[agent_dir_idx]

    raw = jax.lax.dynamic_slice(
        soft_padded_map,
        (row_start, col_start, 0),
        (view_size, view_size, 3),
    )   # (view_size, view_size, 3)

    # Apply the same rotation as get_obs() in maze.py.
    obs = (
        (agent_dir_idx == 0) * jnp.rot90(raw, 1)
        + (agent_dir_idx == 1) * jnp.rot90(raw, 2)
        + (agent_dir_idx == 2) * jnp.rot90(raw, 3)
        + (agent_dir_idx == 3) * jnp.rot90(raw, 4)
    )
    return obs   # (view_size, view_size, 3)
