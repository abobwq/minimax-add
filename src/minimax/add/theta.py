"""
Encoding and decoding between maze levels and diffusion-model images.

The diffusion model operates on theta in R^{16x16x3}. This module converts
between (wall_map, agent_pos, goal_pos, agent_dir) and theta images.
"""

import jax
import jax.numpy as jnp


GRID_SIZE = 13
IMG_SIZE = 16

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

    # Walls: threshold at 0.99. No forced border — minimax handles grid
    # boundaries via position clamping, matching the ADD paper and DR generator.
    wall_map = inner[:, :, 0] > 0.99

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
