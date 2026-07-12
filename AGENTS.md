# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A 2048 game (pygame UI) plus a set of AI players that attempt to reach the 2048 tile:
classical search agents (random Monte-Carlo rollout, backtracking tree search) and a
reinforcement-learning agent (Double DQN). Python 3.12, managed with `uv`.

> **Python version:** the environment is pinned to **3.12** (`.python-version`). Python 3.14 was
> tried but has no prebuilt wheels yet for `pyzmq`/`numba`/`torch`, so `uv sync` fails on it.

## Commands

The project uses `uv`. The virtualenv may be empty on a fresh checkout — run `uv sync` first.

```bash
uv sync                          # install declared deps into .venv (required before anything runs)
uv run play.py                   # play manually with arrow keys (pygame window); 'q' quits, 'r' restarts
uv run train_dqn.py              # headless Double-DQN training (auto-selects MPS/CUDA/CPU)
uv run train_dqn.py --smoke      # 5-episode sanity run of the whole training pipeline
uv run train_dqn.py --episodes 3000 --eval-every 100   # longer run with periodic greedy eval
uv run evaluate_model.py models/<run>/best_model.pth    # greedy eval + tile distribution vs untrained baseline
tensorboard --logdir runs        # view training curves (Score, MaxTile, Eval/*, Loss, Epsilon)
```

Training is fully **headless** (no pygame window) and uses **torch MPS** on Apple Silicon
automatically (`--device` to override). Checkpoints land in `models/<run>/` (`best_model.pth`,
`final_model.pth`, periodic `checkpoint_ep*.pth`); TensorBoard logs in `runs/<run>/`.

The **n-tuple TD agent** (`train_ntuple.py`) is the *recommended* RL player — it far
outperforms the DQN on 2048 and trains in minutes on CPU (no GPU needed):

```bash
uv run train_ntuple.py                 # ~20k games of afterstate TD-learning
uv run train_ntuple.py --smoke         # 300-game sanity run (reaches 1024 in seconds)
uv run train_ntuple.py --games 40000   # train longer for higher 2048 rates
```

Checkpoints are `models/<run>/best_model.npz` / `final_model.npz` (n-tuple lookup tables).

Benchmark the classical AI players over many seeds (`test-ai.py`):

```bash
uv run test-ai.py backtracking --search_depth 3 --seeds 1-100 --mp
uv run test-ai.py random --searches_per_move 20 --search_length 10 --seeds 1-100
```

Tests:

```bash
uv run pytest tests/                                             # full suite
uv run pytest tests/test_staticboard.py::TestNumpyStaticBoard::test_move   # single test
```

**Dependency gotcha:** `pytest`, `tqdm`, `tabulate`, and `coverage` are used by the tests and
`test-ai.py` but are **not** declared in `pyproject.toml`. The commands above will fail until you
add them, e.g. `uv add --dev pytest tqdm tabulate coverage` (or run ad-hoc with
`uv run --with pytest pytest tests/`).

## Architecture

The central design idea is a **stateless board engine + thin stateful wrapper + pluggable numpy/torch
backend**. Understanding this makes the rest of the code obvious.

### Board engine — `src/game/model/`
- `staticboard.py` defines `StaticBoard`, an ABC whose methods are all **static/pure functions on a
  matrix** (`move`, `collapse_array`, `compute_is_done`, `set_random_cell`, `has_won`, …). There is no
  per-board instance state.
- `staticboardImpl.py` has two interchangeable implementations of that interface:
  - `NumpyStaticBoard` — numpy + numba `@njit`. This is the **default** and is used by the classical AI
    players **and** the DQN environment (the agent converts boards to tensors itself).
  - `TorchStaticBoard` — tensor-based alternative. Correct but slow for RL: scalar per-cell tensor ops
    thrash the accelerator, so it is not used on the training hot path.

  The core move primitive is `collapse_array(arr, reverse)`, applied per row/column. Direction → slice mapping:
  `UP`/`LEFT` use `reverse=True`, `DOWN`/`RIGHT` use `reverse=False` (UP/DOWN slice columns, LEFT/RIGHT slice rows).

### Game controller — `src/game/controller/game.py`
- `Game` holds the only mutable state: `matrix`, `score`, `is_done`. It delegates all board math to a
  `static_board` class chosen at construction (defaults to `NumpyStaticBoard`).
- `Game.clone()` is a deepcopy — classical agents rely on it to simulate candidate futures cheaply.
- `move(action, inplace)` returns `(matrix, score, changed)` and, when the board changed, spawns a
  random tile and recomputes `is_done`.

### Players / agents — `src/agent/`
- `agent.py` — `Player` ABC. It implements the shared `run()` game loop; subclasses only implement
  `get_move()`. `run()` returns `(score, max_val, runtime)`.
- `agentImpl.py` — three players:
  - `RandomGuessAIPlayer` — Monte-Carlo random rollouts per candidate move (~4% win rate).
  - `BacktrackingAIPlayer` — depth-limited tree search maximizing merge score (~32% at depth 5; runtime grows ~4–5× per depth).
  - `DQNPlayer` — **Double DQN** with experience replay + target network. Non-obvious design choices:
    - **Action masking** (`valid_moves_mask`) — only board-changing moves are ever selected, so the game
      always advances (no invalid-move stalls) and no invalid-move penalty is needed.
    - **Device split** — the env runs on CPU (`NumpyStaticBoard`, numba) while only the net lives on
      `device` (MPS/CUDA). Do **not** put the board on MPS: the per-cell scalar tensor ops in
      `TorchStaticBoard` cause constant host↔GPU syncs and dominate runtime.
    - **Headless by default** (`ui=False`); `train()` logs to TensorBoard and `evaluate()` plays greedy
      games. Epsilon decays linearly over `epsilon_decay_steps`.
- `model/dqn.py` — `DQN`, a CNN that **one-hot encodes** the board into 16 channels (tile exponents
  0..15) internally, then two `Conv2d(kernel=2)` layers → two FC layers → 4 Q-values. One-hot (not a
  single log2 scalar channel) is the key change that makes the net learn. No BatchNorm/Dropout, so
  single-sample inference and batched training behave identically.
- `ntuple.py` — **`NTupleNetwork` + afterstate TD-learning: the strongest RL player here.** DQN
  under-performs on 2048 because the *action*-value gaps are tiny next to the *state* value, so
  `Q(s,a)≈V(s)` and greedy ≈ random. Instead this learns `V(afterstate)` — the board right after a
  move's merges, *before* the random spawn — and picks `argmax_a [reward(s,a) + V(afterstate(s,a))]`;
  the reward term separates actions cleanly. `V` is four overlapping 6-cell lookup tables with 8-fold
  dihedral **symmetry weight-sharing**. It exploits that `NumpyStaticBoard.move(..., inplace=False)`
  already returns `(afterstate, merge_reward, changed)`. Tabular/CPU, no GPU — reaches 1024 in seconds
  and 2048 with more training. `train_ntuple.py` drives it; `make_player()` wraps it as a `Player` for
  GUI play.

### View — `src/game/view/gameUI.py`
- `GameUI` renders with pygame. `run()` is the manual-play loop; `update_ui()` is what agents call to
  animate when constructed with `ui=True`.

## Conventions & gotchas

- **Run from the repo root.** Imports are absolute from the `src` package (`from src.game... import`);
  there is no editable install, so scripts and tests must be invoked from the project root.
- **Fixed default seed.** `Game` defaults to `seed=2048` and `restart()` re-seeds the global RNG, so
  runs are deterministic per seed. Benchmarks/tests assert exact scores for specific seeds
  (see `tests/test_game.py`) — changing board or move logic will break those golden values. RL training
  intentionally passes `seed=None` so every episode differs (see `train_dqn.py`).
- **`models/` and `runs/` are gitignored.** Any checkpoints or TensorBoard logs present locally are
  untracked artifacts, not part of the repo.
- **`OldGame/`** is a legacy standalone implementation (`game.py`, `gamestate.py`, `components.py`,
  plus notebooks) kept for reference; the live code is under `src/`. `ai.ipynb` / `exp.ipynb` at the
  root are experiment scratchpads.
- **`.github/workflows/pytest.yml` is stale** — it references a nonexistent `requirements.txt`,
  Python 3.6–3.8, and retired Ubuntu runners. It does not reflect the current `uv` / Python 3.12+ setup.
