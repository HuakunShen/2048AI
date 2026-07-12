# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A 2048 game (pygame UI) plus a set of AI players that attempt to reach the 2048 tile:
classical search agents (random Monte-Carlo rollout, backtracking tree search) and a
Deep Q-Learning (DQN) agent. Python 3.12+, managed with `uv`.

## Commands

The project uses `uv`. The virtualenv may be empty on a fresh checkout — run `uv sync` first.

```bash
uv sync                      # install declared deps into .venv (required before anything runs)
uv run play.py               # play manually with arrow keys (pygame window); 'q' quits, 'r' restarts
uv run train_dqn.py          # train the DQN agent (writes models/ and TensorBoard logs to runs/)
tensorboard --logdir runs    # view DQN training curves
```

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
  - `NumpyStaticBoard` — numpy + numba `@njit`. This is the **default** and is used by the classical AI players.
  - `TorchStaticBoard` — tensor-based, used by the DQN so states feed directly into the network.

  The core move primitive is `collapse_array(arr, reverse)`, applied per row/column. Direction → slice mapping:
  `UP`/`LEFT` use `reverse=True`, `DOWN`/`RIGHT` use `reverse=False` (UP/DOWN slice columns, LEFT/RIGHT slice rows).

### Game controller — `src/game/controller/game.py`
- `Game` holds the only mutable state: `matrix`, `score`, `is_done`. It delegates all board math to a
  `static_board` class chosen at construction (`Game(static_board=TorchStaticBoard)` for DQN).
- `Game.clone()` is a deepcopy — classical agents rely on it to simulate candidate futures cheaply.
- `move(action, inplace)` returns `(matrix, score, changed)` and, when the board changed, spawns a
  random tile and recomputes `is_done`.

### Players / agents — `src/agent/`
- `agent.py` — `Player` ABC. It implements the shared `run()` game loop; subclasses only implement
  `get_move()`. `run()` returns `(score, max_val, runtime)`.
- `agentImpl.py` — three players:
  - `RandomGuessAIPlayer` — Monte-Carlo random rollouts per candidate move (~4% win rate).
  - `BacktrackingAIPlayer` — depth-limited tree search maximizing merge score (~32% at depth 5; runtime grows ~4–5× per depth).
  - `DQNPlayer` — Deep Q-Learning: experience replay + target network, epsilon-greedy, TensorBoard logging.
    Reward shaping and epsilon decay live in `calculate_reward()` and `train()`.
- `model/dqn.py` — `DQN`, a small CNN (two `Conv2d(kernel=2)` layers → two FC layers → 4 action logits).
  The forward pass **log2-transforms the board** (`log2(x+1)`) before the convolutions.

### View — `src/game/view/gameUI.py`
- `GameUI` renders with pygame. `run()` is the manual-play loop; `update_ui()` is what agents call to
  animate when constructed with `ui=True`.

## Conventions & gotchas

- **Run from the repo root.** Imports are absolute from the `src` package (`from src.game... import`);
  there is no editable install, so scripts and tests must be invoked from the project root.
- **Fixed default seed.** `Game` defaults to `seed=2048` and `restart()` re-seeds the global RNG, so
  runs are deterministic per seed. Benchmarks/tests assert exact scores for specific seeds
  (see `tests/test_game.py`) — changing board or move logic will break those golden values.
- **`models/` and `runs/` are gitignored.** Any checkpoints or TensorBoard logs present locally are
  untracked artifacts, not part of the repo.
- **`OldGame/`** is a legacy standalone implementation (`game.py`, `gamestate.py`, `components.py`,
  plus notebooks) kept for reference; the live code is under `src/`. `ai.ipynb` / `exp.ipynb` at the
  root are experiment scratchpads.
- **`.github/workflows/pytest.yml` is stale** — it references a nonexistent `requirements.txt`,
  Python 3.6–3.8, and retired Ubuntu runners. It does not reflect the current `uv` / Python 3.12+ setup.
