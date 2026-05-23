# ADD Guidance: Gradient Flow Analysis

How the GAE score gradient reaches `x0_pred`, and why it is sparse.

---

## What is `delta_t`?

`delta_t` is the **one-step TD error** (Bellman residual) at trajectory step `t`:

```
delta_t = r_t  +  γ · V(s_{t+1})  −  V(s_t)
```

Intuitively: how much better (or worse) was this step than the value function
predicted?  Positive delta = the agent did better than expected (found reward or
moved to a higher-value state).  GAE accumulates these errors backward in time
to form advantages `A_t`.  The guidance score `mean(sqrt(A_t^2 + ε))` is high
when the agent faces high-variance, hard-to-predict situations — i.e., the level
is hard.

In the soft maze each term is:

```
r_t        = goal_ps[t]       soft probability of reaching the goal at next cell
V(s_{t+1}) = V_next_soft[t]   LSTM value at next cell, adjusted for wall blocking
V(s_t)     = V_seq[t]         LSTM value at current cell
```

Wall adjustment: if there is a wall at the next cell the agent cannot move, so
its effective next state is the current state:

```
V_next_soft[t] = (1 − wall_ps[t]) · V_next_raw[t]    ← agent moves freely
               +      wall_ps[t]  · V_seq[t]          ← wall blocks, agent stays

where  V_next_raw[t] = V_seq[t+1]   (value one step ahead)
```

Full TD error:

```
delta_t = goal_ps[t]  +  γ · not_done[t] · V_next_soft[t]  −  V_seq[t]
```

---

## Three Gradient Paths into `theta` (= `x0_pred`)

`theta` is the soft level representation (shape 16×16×3, channel 0 = wall
probability, channel 2 = goal probability).  The guidance score depends on
`theta` through three distinct paths.

---

### Path 1 — Reward signal (goal channel)

```
theta[row_{t+1}, col_{t+1}, channel=2]
    │
    │  index read (gather): one scalar from one array element
    ▼
goal_ps[t]
    │
    │  additive term
    ▼
delta_t  =  goal_ps[t]  + ...
```

- `goal_ps[t]` is a **direct array read** of a single element of `theta`.
  There is no function applied — the value literally is that element.
- `∂delta_t / ∂goal_ps[t] = 1`  (goal reward enters delta additively).
- Gradient at `theta` = 1 at cell `(row_{t+1}, col_{t+1}, 2)`, 0 everywhere else.
- **Independent of the LSTM** — exists even with a random or frozen value head.
- **Sparse**: one theta element per trajectory step, accumulating over all `t`
  that visit or neighbour the same cell.

---

### Path 2 — Wall blocking signal (wall channel)

```
theta[row_{t+1}, col_{t+1}, channel=0]
    │
    │  index read (gather): one scalar from one array element
    ▼
wall_ps[t]
    │
    │  gates the next-state value
    ▼
V_next_soft[t]  =  (1 − wall_ps[t]) · V_next_raw[t]  +  wall_ps[t] · V_seq[t]
    │
    │  multiplicative term in delta
    ▼
delta_t  =  ...  +  γ · not_done[t] · V_next_soft[t]  −  ...
```

- `wall_ps[t]` is also a **direct array read** of a single element of `theta`.
- `∂delta_t / ∂wall_ps[t] = γ · not_done[t] · (V_seq[t] − V_next_raw[t])`
- The gradient magnitude equals the **LSTM value contrast** across the wall:
  how much better (or worse) the current state is compared to the next.
  - Early training: LSTM values are near-constant → `V_seq[t] ≈ V_next_raw[t]`
    → gradient ≈ 0.
  - Later training: value contrast grows → path 2 becomes meaningful.
- **Entangled with LSTM**: unlike path 1, the magnitude of this gradient depends
  on LSTM quality.  It cannot be treated as fully independent.
- **Sparse**: same structure as path 1 — one theta element per step.

---

### Path 3 — Value signal (LSTM)

`V_seq[t]` is the LSTM value estimate at step `t`.  It reaches `theta` through:

```
theta  (16×16×3)
    │
    │  theta_to_soft_maze_map: padding + sigmoid → soft_map (21×21×3)
    ▼
soft_map
    │
    │  soft_extract_obs: differentiable 5×5 gather at position pos_t
    │  → egocentric observation (5×5×3)  at each of the T trajectory steps
    ▼
obs_t  (5×5×3 window of soft_map, centred on agent at step t)
    │
    │  LSTM forward pass, lstm_k steps with truncated BPTT
    ▼
V_seq[t]
```

`V_seq[t]` then enters `delta` via **two sub-paths**:

```
Sub-path A — direct subtraction:

    V_seq[t]
        │  −1 coefficient
        ▼
    delta_t  =  ...  −  V_seq[t]

Sub-path B — as next-state value of the previous step:

    V_seq[t]  ≡  V_next_raw[t−1]     (because V_next_raw[t] = V_seq[t+1])
        │
        │  (1 − wall_ps[t−1]) coefficient
        ▼
    V_next_soft[t−1]
        │
        │  γ · not_done[t−1] coefficient
        ▼
    delta_{t−1}  =  ...  +  γ · not_done[t−1] · V_next_soft[t−1]  −  ...
```

Properties of path 3:
- **Dense-ish within a step**: `soft_extract_obs` reads a 5×5×3 = 75 element
  window of `soft_map`, which maps back to roughly a 3×3 patch in `theta`.
  Over `T` trajectory steps, many overlapping patches accumulate.
- **Small magnitude**: gradient must survive backpropagation through `lstm_k=32`
  LSTM steps.  Vanishing gradients through the LSTM carry significantly reduce
  the signal that reaches `theta`.
- **Not blocked at i=0** (first DDIM step): the rollout still produces a
  trajectory and the LSTM still runs, but `x0_pred` is derived from near-pure
  noise at that step, so the decoded level and resulting trajectory are noisy.

---

## Why the Gradient is Sparse

Paths 1 and 2 are **point lookups** at `pos_{t+1}` — a single theta element per
timestep.  Over a 128-step rollout in a 13×13 maze, the agent visits roughly
10–20 unique `pos_{t+1}` values (many steps revisit the same cells or are
blocked by the same walls).  The gradient from paths 1+2 accumulates at those
10–20 cells — approximately 10–20 elements out of 768 total (≈ 1–3%).

Path 3 is broader but arrives at theta with negligible magnitude after BPTT,
contributing little to the total L2 norm.

**Observed consequence**: with `use_grad_norm=True` and `omega=250`, only ~1.4%
of theta elements are clamped.  Dense gradient (all 768 elements contributing
equally) would give ≈100% clamping; ~11 active elements gives exactly 1.4%.

---

## Implications for Guidance

| Property | Effect |
|----------|--------|
| Sparse gradient (paths 1+2 dominate) | Only goal/wall cells visited by the agent get guided. Most of the level structure is invisible to the gradient. |
| Path 2 gated by LSTM value contrast | Wall guidance is weak early in training, grows as the value function matures. |
| Path 3 magnitude negligible | LSTM-path guidance contributes ≈0 to gradient direction after grad-norm. Cannot be recovered by increasing omega — normalization projects it out. |
| `omega` above saturation threshold | Once active cells are clamped, increasing omega further has no effect. Saturation occurs around omega ≈ 5–10 with `use_grad_norm` and `step = 1/√ᾱ`. |

The structural bottleneck is **coverage**: the score only differentiates through
cells the agent actually visits via point lookups (paths 1+2).  Increasing omega,
changing the step formula, or adjusting the DDIM schedule cannot expand this
coverage — that requires changing what theta elements the score depends on.
