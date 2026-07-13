# Experiment Plan — 2048 AI on Linux + RTX 4080

Follow-up to [next_steps.md](next_steps.md), rewritten as a concrete experiment plan for the new
machine. Everything below was planned against the **verified environment** of this box:

| Component | Verified value |
|---|---|
| OS | Linux (kernel 6.17) |
| CPU | 24 hardware threads (`nproc` = 24) |
| GPU | NVIDIA GeForce RTX 4080, 16 GB VRAM, driver 580.159 |
| PyTorch | 2.13.0+cu130, `torch.cuda.is_available() == True` |
| Python | 3.12 via `uv` (venv synced, 71 packages) |

Two consequences of the hardware change:

1. **CPU experiments get much faster.** The n-tuple agent and expectimax search are CPU-bound;
   24 threads means 300-game benchmarks parallelize ~20× with `multiprocessing`.
2. **The GPU is finally worth designing for.** On the Mac, MPS punished small kernels; on CUDA,
   large-batch training is cheap. The right way to use the 4080 is **not** "put the existing DQN on
   CUDA" (it's tiny, the env is the bottleneck) but **vectorized self-play**: thousands of games
   stepped in parallel so the network sees batches of 4–16k positions per forward pass. Track C is
   built around that idea.

Current best result (baseline to beat): **n-tuple TD, 40k games — 80.6% @2048, ~36% @4096,
97.2% @1024, max tile 8192** (greedy 1-ply play).

---

## Track 0 — Infrastructure first (½ day, prerequisite for everything)

Every experiment below needs a trustworthy, comparable evaluation. Do this once, first.

### 0.1 Dev dependencies + CI
- `uv add --dev pytest tqdm tabulate coverage` (tests and `test-ai.py` already import these).
- Rewrite `.github/workflows/pytest.yml`: `ubuntu-latest`, `astral-sh/setup-uv`, Python 3.12,
  `uv sync && uv run pytest tests/`.

### 0.2 Shared benchmark harness — `benchmark.py`
One script that evaluates **any** player type under identical conditions:

```bash
uv run benchmark.py ntuple  --model models/<run>/best_model.npz --games 300 --procs 20
uv run benchmark.py expectimax --model ... --depth 3 --games 300 --procs 20
uv run benchmark.py dqn     --model models/<run>/best_model.pth --games 300
```

- Fixed seed list (e.g. seeds 1..300) so every agent sees the same tile sequences.
- Reports: **reach rate for each of 1024/2048/4096/8192/16384**, mean/median/max score,
  mean moves per game, **moves/second** (cost of the method).
- Appends a row to `docs/results.md` (markdown table) so runs accumulate into one comparison table.
- `multiprocessing.Pool` over games for CPU agents (n-tuple/expectimax pickle fine — LUTs are numpy).

**Exit criterion:** the current 40k-game n-tuple checkpoint re-benchmarked on this machine, one row
in `docs/results.md`. That row is the baseline every experiment compares against.

### 0.3 n-tuple unit tests (small, protects everything downstream)
- Symmetry: `value(board) == value(rot90(board)) == value(fliplr(board))` for random boards.
- Afterstate: `NumpyStaticBoard.move(b, d, inplace=False)` afterstate has no spawned tile.
- TD convergence: on a toy 2-state chain, `update()` drives `value()` to the fixed point.

---

## Track A — Expectimax with n-tuple leaf evaluation ⭐ (1 day, biggest expected jump)

**Hypothesis:** the trained `V` is good but is only used at 1 ply. Literature (Szubert & Jaśkowski
2014; Jaśkowski 2018) shows the same networks jump from ~80% to ~97%+ @2048 and regularly reach
8192–16384 with 2–3-ply expectimax on top.

### Design — new `src/agent/expectimax.py` + player class
Tree over **afterstates** (reuses `NumpyStaticBoard.move(..., inplace=False)` exactly like
`ntuple.py` does):

- **Max node** (our move): `max over valid a of [reward(s,a) + E(afterstate(s,a), depth)]`.
- **Chance node** (tile spawn): for each empty cell, children with tile 2 (p=0.9) and 4 (p=0.1);
  value = probability-weighted average of child max-node values.
- **Leaf:** `net.value(afterstate)`.
- Depth counted in *move plies*; depth=1 must reproduce the current greedy player exactly
  (that's the correctness test).

### Cost control (needed at depth ≥ 3)
Branching per ply ≈ 4 moves × (2 × #empty) chance children — depth 3 is ~10⁴–10⁵ leaves naïvely.
1. **Transposition table**: `dict` keyed on `afterstate.tobytes()` per root search.
2. **Chance-node sampling**: when #empty > K (e.g. 6), evaluate a random/top-K subset of cells and
   renormalize. Late-game boards (few empties, where depth matters most) stay exact.
3. **Probability cutoff**: stop expanding branches whose cumulative probability < 1e-4; return leaf value.
4. **Adaptive depth**: depth 2 when #empty ≥ 8, depth 3 when 4–7, depth 4–5 when ≤ 3
   (endgame is where deep search pays; this is standard practice for 2048 expectimax).

### Experiments
| ID | Config | Games | Measures |
|---|---|---|---|
| A1 | depth=1 (sanity: must equal greedy baseline) | 300 | identical reach rates |
| A2 | depth=2 exact | 300 | reach rates, moves/s |
| A3 | depth=3 + TT + cutoffs | 300 | reach rates, moves/s |
| A4 | adaptive depth (2/3/5 by empties) | 300 | reach rates, moves/s |

24 cores → run the 300 games in a `Pool(20)`; even at 1 s/move a full benchmark finishes overnight,
and A2 should be minutes.

**Success criteria:** A3/A4 ≥ 95% @2048 and ≥ 60% @4096, first 16384 sightings.
**Abort criterion:** if depth 2 doesn't beat greedy at all, the value function is the bottleneck →
prioritize Track B before deeper search.

---

## Track B — Better n-tuple learning (1–2 days, multiplies with Track A)

Same agent, stronger `V`. Each item is an independent experiment against the 0.2 baseline;
all still train on CPU in minutes-to-hours.

### B1. Temporal Coherence (TC) learning — adaptive per-weight step size
Replace the global `alpha` with per-weight `|Σδ| / Σ|δ|` scaling (Beal & Smith; used in all strong
2048 n-tuple work). Two extra float32 arrays per LUT (+128 MB per table set — trivially fits in RAM).
- **Experiment:** 40k games TC vs. current fixed-α run; compare learning curves (score vs. games)
  and final reach rates. Expected: same-or-better final strength, much less α tuning.

### B2. Bigger/more tuples
Current: 4 six-cell patterns × 8 symmetries. Standard stronger sets:
- **B2a:** the Yeh et al. / Jaśkowski **8 × 6-tuple** set (adds column bands + more rectangles).
  Memory: 16⁶ × 4 B = **64 MB per table → 512 MB total**; fine.
- **B2b (optional):** 7-tuples (16⁷ × 4 B = 1 GB per table) — only if B2a plateaus; consider
  float16 LUTs or exponent cap 12 (index base 13) to shrink.
- **Experiment:** 40k and 100k games each, benchmark vs. baseline. Expected @2048 → ~90%+ greedy.

### B3. Multi-stage weights
Separate LUT sets per game stage, switched on board state (e.g. stage 2 once a 2048 tile exists,
stage 3 at 4096 + 2048). Stages train on the boards they see, so late-game values stop being
averaged with early-game ones. This is the technique behind the 32768-tile results (Yeh et al. 2016).
- Depends on B1/B2 choice; multiplies memory by #stages (3 × 512 MB is still fine).
- **Experiment:** 2-stage vs. 3-stage vs. single, 100k games, compare @4096/@8192 rates specifically.

### B4. Just train longer
200k–500k games of the best B1+B2 config. The training loop is pure numpy on one core; if it's the
bottleneck, either `numba.njit` the index/update path or run **lock-free parallel self-play**
(Hogwild-style: N processes sharing the LUTs via `multiprocessing.shared_memory`; TD updates race
benignly, standard for tabular methods). 24 cores → ~20× games/hour.

**Combined goal for Track B:** greedy ≥ 90% @2048; then rerun Track A on top → expect
~97%+ @2048, 8192 regularly, occasional 16384.

---

## Track C — GPU-native methods (the 4080 experiments; 3–7 days, research value)

The DQN post-mortem (dqn_report.md) concluded Q-learning is structurally wrong for 2048
(Q(s,a) ≈ V(s)). These experiments keep the **afterstate** decision rule that provably works and
put the **function approximation and simulation on the GPU**.

### C0. Vectorized GPU game engine — `src/game/model/cudaboard.py` (prerequisite, ~1 day)
The key enabler. A 2048 move decomposes into independent **row** operations, and a row has only
16⁴ = 65,536 states. So:
1. Precompute once (numpy, seconds): `ROW_LUT[65536] → (result_row, reward)` for left-collapse.
2. Represent N boards as an int8 exponent tensor `[N, 4, 4]` on CUDA. A move for all N games =
   pack rows to int32 indices → `torch.gather` into the LUT → unpack. UP/DOWN/RIGHT are
   transposes/flips of LEFT.
3. Batched spawn: multinomial over empty cells per board; batched done-check the same way.

Result: step **thousands of games per GPU kernel call**. Validate against `NumpyStaticBoard` on
10⁵ random boards/moves (exact match required — this test gates everything in Track C).

### C1. Neural afterstate value network (flagship experiment)
n-tuple ideas, neural function approximator:
- Network: `V(afterstate)` — one-hot 16×4×4 input → Conv2d stack (reuse `DQN`'s trunk, scalar
  output head). ~1–5 M params; batch 4096 is nothing for a 4080.
- Acting: for N parallel games, compute all ≤4 afterstates each (GPU engine), evaluate the whole
  `[≤4N]` batch in one forward pass, pick `argmax r + V(after)` per game.
- Learning: **TD(λ) or n-step (n=5) targets** on afterstates, small replay buffer or purely
  on-policy streams from the vectorized games; symmetry augmentation (random dihedral transform
  per sample) to substitute for the LUT weight sharing.
- **Experiments:** C1a 1-step vs. C1b 5-step vs. C1c TD(λ=0.5); 2–5 h wall-clock each at
  ~50k games/hour with 4096 parallel envs.
- **Question answered:** can a CNN match 64 MB of lookup tables? Compare vs. Track B at equal
  game counts. Even a "no" is a publishable-quality note for `docs/`.

### C2. PPO with action masking (modern-RL comparison point)
- Policy+value net on the C0 engine, 4096 parallel envs, GAE(λ), invalid actions masked at the
  logits (mask already exists conceptually via `changed`).
- Reward: merge score, log-scaled (`log2(1+r)`) to tame the non-stationary magnitude that broke DQN.
- **Purpose:** an honest "what does mainstream deep RL do here" row in results.md.
  Expected: mediocre @2048 but strong learning-curve content; ~1 day of work, hours to train.

### C3. Expectimax distillation (behavior cloning → search-free strong player)
- Generate 1–5 M `(board → best move)` labels with the Track A player using the `Pool(20)`
  self-play (this is embarrassingly parallel, runs overnight).
- Train a policy CNN (cross-entropy, symmetry augmentation) on the 4080 — minutes per epoch.
- **Measures:** agreement % with teacher, then benchmark the raw policy (no search) vs. greedy
  n-tuple. Interesting outcome either way: a single forward-pass player at ~depth-3 strength,
  and a warm start for C2's PPO fine-tuning.

### C4 (stretch). n-tuple training itself on GPU
LUTs are 512 MB → fit in VRAM. `value` = `gather` + sum, `update` = `scatter_add`. With the C0
engine, run 10k games in parallel → 10⁶–10⁷ TD updates/s, i.e. **millions of training games/hour**.
Only worth it if B4 shows more games keep helping; if so, this is how to reach the 10⁶-game regime
of the strongest published results.

---

## Track D — Other ways to play (optional, comparison rows)

- **D1. MCTS** over afterstates with `V` for leaf evaluation / rollout truncation — compare vs.
  expectimax at equal time-per-move budget (expectimax usually wins in 2048's low branching, but
  it's a clean experiment).
- **D2. Pure-heuristic expectimax** (monotonicity + smoothness + empty-count + corner-max, the
  classic StackOverflow/nneonneo weights) — the "no learning at all" baseline row; also serves as
  the leaf function ablation for Track A.
- **D3. CMA-ES over the D2 heuristic weights** (or over a tiny value MLP) — gradient-free baseline,
  embarrassingly parallel on 24 cores.

---

## Suggested order & decision points

```
0 (harness) ──► A (expectimax)  ──► B1+B2 (TC + 8 tuples) ──► re-run A on new V
                     │                        │
                     ▼                        ▼
              [≥95% @2048?]            [greedy ≥90% @2048?]
               yes → C3 labels          yes → B3 multi-stage → (B4/C4 long runs)
                     │
                     ▼
              C0 (GPU engine) ──► C1 (afterstate net) ──► C2 (PPO) / C3 (distill)
```

- **Week 1:** Track 0 + Track A + B1/B2. Deliverable: results.md table with greedy vs. depth-2/3
  rows, expected ≥95% @2048.
- **Week 2:** B3/B4 + C0 + C1. Deliverable: GPU engine validated, first neural-afterstate curves
  on TensorBoard, updated dqn_report.md addendum ("what finally made NN+GPU work — or not").
- **Later / optional:** C2, C3, C4, Track D rows.

## Risks & mitigations

- **Expectimax too slow at depth 3** → adaptive depth (A4) and chance sampling are designed in from
  the start; depth 2 alone should already clear 90% @2048.
- **GPU engine correctness** → gated by exact-match test vs. `NumpyStaticBoard` before any RL uses it.
- **Golden-value tests** (`tests/test_game.py` asserts exact scores for fixed seeds) → none of the
  planned work changes board/move logic; the C0 engine is additive. If any engine refactor is ever
  needed, run the full suite before/after.
- **VRAM** — nothing here approaches 16 GB (biggest tensor: 4096 envs × 16×4×4 one-hot ≈ 4 MB;
  LUTs 0.5–1 GB in C4).

---

## Progress log (execution against this plan)

### 2026-07-13 — Track 0 + Track A + Track C0 built & validated

**Track 0 (infrastructure) — DONE**
- `uv add --dev pytest tqdm tabulate coverage`.
- `benchmark.py` — unified harness: fixed seed list, reach rates for
  1024..32768, mean/median/max score, moves/game, moves/s; loads the model once
  per worker (no per-task LUT pickling); appends rows to `docs/results.md`.
- `tests/test_ntuple.py` — 6 tests: dihedral symmetry invariance of `V`,
  afterstate == `move(inplace=False)`, TD update convergence, and the anchor
  **expectimax depth-1 == greedy** (100 random boards).

**Track A (expectimax) — BUILT & VALIDATED**
- `src/agent/expectimax.py` — `ExpectimaxNTuple`: afterstate tree with max
  (our move) / chance (tile spawn) nodes, n-tuple leaf; transposition table,
  adaptive depth, chance-node sampling with a *private* RNG (seeded benchmarks
  stay reproducible). `make_player()` for GUI.
- Result on a **weak 2k-game** model: greedy 0% @2048 → **depth-2 52.5%** @2048.
- Result on a **30%-trained (12k-game)** model: depth-2 lifts @2048 48%→92%,
  @4096 7%→37% — already beating the fully-trained greedy baseline.
- **Definitive result on the final 40k model (82.6% greedy @2048), 300/300/200
  games, identical seeds** — logged in `docs/results.md`:

  | Config | @1024 | @2048 | @4096 | @8192 | mean score | moves/s |
  |---|---|---|---|---|---|---|
  | greedy (d=1) | 97.7% | 79.3% | 29.7% | 0.7% | 42,119 | 107,594 |
  | expectimax d=2 | 99.7% | 96.7% | 83.7% | 19.0% | 76,211 | 5,098 |
  | expectimax d=3 adaptive | 100% | **99.5%** | **96.5%** | **47.0%** | **103,861** | 259 |

  **Exceeds the plan's success criteria** (≥95% @2048, ≥60% @4096). Search takes
  the *same* value function from 79%→99.5% @2048 and 30%→96.5% @4096, with 47%
  reaching 8192 and mean score 2.5×. Cost is steep at depth-3 (endgame depth-4
  search dominates: 259 vs 107k moves/s) — motivating the adaptive schedule and,
  next, a faster leaf (numba-JIT `V`, or the GPU engine). No 16384 yet: that needs
  a stronger value function (Track B: TC + 8×6-tuple + multi-stage).

**Track C0 (vectorized GPU engine) — DONE**
- `src/game/model/vectorboard.py` — row-LUT engine (16⁴ table built by calling
  the *real* `collapse_array`, so semantics are exact). numpy `move_batch`/
  `spawn_batch`/`done_batch`, plus `TorchVectorEngine` (move/spawn/done/self-play
  entirely on CUDA).
- `tests/test_vectorboard.py` — 7 tests: bit-for-bit match vs `NumpyStaticBoard`
  on 100k boards × 4 dirs, full-board edge cases, done-check, spawn, encoding
  roundtrip, and GPU-vs-numpy equivalence.
- **Throughput (measured on this 4080):** per-board 0.37M → numpy vectorized
  7.0M → **GPU 165M board-moves/s**. Full batched self-play (16,384 parallel
  random games): **89,000 games/s ≈ 321M games/hour**. The engine will never be
  the RL bottleneck — the net's forward pass will. Track C1/C2 are unblocked.

**Leaf-eval speedup (numba-JIT `V`) — DONE**
- `NTupleNetwork.LUT` is now one contiguous `[n_tuples, table_size]` array (each
  `LUT[t]` a row view, so `update`/`save`/`load` are unchanged and old checkpoints
  still load). `value()` routes through `_value_njit`, a `@njit(cache=True)` kernel
  doing exponent-decode + index + table-sum in scalar loops (no per-call numpy).
- **Speedup: value() 12.8× (100.8K → 1,292K calls/s); end-to-end depth-2
  expectimax 3.0× (5,098 → 15,111 moves/s @ 20 procs).** Reach rates unchanged
  (96.0% vs 96.7% @2048 — within noise). njit-vs-numpy value differs by <2e-3
  (float32 summation order; never flips a move — depth-1==greedy test still green).
- Benefits every expectimax config and future training (both are `value()`-bound):
  depth-3-adaptive 200-game suite drops from ~58 min toward ~20 min.

### 2026-07-14 — Track B (stronger value function) started

**Built** (all in `src/agent/ntuple.py`, backward-compatible — old checkpoints
still load as 4-tuple):
- **Temporal-coherence (TC) learning** — per-weight adaptive step `α·|E|/A` via
  `_update_tc_njit` (E/A accumulators). Coherent updates step fully, oscillating
  ones are damped.
- **Configurable tuple sets** — `TUPLES_8` (8×6-cell patterns; 512 MB of tables)
  selectable via `--tuples 8`; checkpoints self-describe their pattern set so
  `benchmark.py` loads either transparently.
- **njit'd updates** (`_update_njit`/`_update_tc_njit`) → training ~3× faster
  (74 vs original ~25 early games/s).

**Smoke (800 games each) — TC is a large convergence win:**

  | Config | games/s | @1024 | @2048 | best |
  |---|---|---|---|---|
  | 4-tuple fixed (baseline) | 74 | 35% | 0% | 1024 |
  | 4-tuple TC | 52 | 74% | 8% | 2048 |
  | 8-tuple fixed | 69 | 32% | 0% | 1024 |
  | **8-tuple TC** | 47 | **76%** | **15%** | **4096** |

  TC reaches in 800 games what fixed-α needs ~4000 for; 8-tuple TC already hits
  4096. Tuples alone (8-tuple fixed) barely move early — the gain is TC.

**Running now** (`scratch_logs/track_b.sh`, parallel, 40k games each): **8-tuple
TC** (champion) + **4-tuple TC** (ablation isolating TC vs tuples), then greedy +
depth-2 benchmarks vs the 4-tuple-fixed baseline. Hypothesis: TC lifts greedy
@2048 well above 80% and starts producing 16384.

**Track B final results (40k games each, JIT-accelerated ~1h wall for both in
parallel):**

  | Model | greedy @2048 | greedy @8192 | depth-2 @8192 | depth-2 @16384 |
  |---|---|---|---|---|
  | 4-tuple fixed (baseline) | 79.3% | 0.7% | 19.0% | 0% |
  | 4-tuple TC | 92.7% | 23.3% | 62.7% | 2.0% |
  | **8-tuple TC** | **96.0%** | 23.7% | **68.3%** | **3.7%** |

  TC added +13pp greedy @2048; the 8-tuple set +3pp more. **First 16384 tiles.**

### 2026-07-14 — Track C1: neural afterstate value net — NN THAT WORKS

The DQN post-mortem said Q-learning is structurally wrong for 2048. This tests the
fix: a CNN learning ``V(afterstate)`` and acting ``argmax_a[r + V(afterstate)]`` —
the n-tuple's rule with a neural approximator, trained by online TD on the C0 GPU
engine (`src/agent/afterstate_net.py`, `train_afterstate.py`).

- **It learns** (unlike the DQN, which stalled at baseline). Tiny 205k-param CNN,
  1024 parallel envs, **60k games in 155 s** on the 4080:

  | | @512 | @1024 | @2048 | @4096 |
  |---|---|---|---|---|
  | neural afterstate net (60k games) | 98% | 90% | ~40% | 1–2% |

- Key choices: one-hot exponent input, reward/value scaling (÷2048) so targets are
  O(10), dihedral augmentation (substitutes for the n-tuple's weight sharing),
  online afterstate TD(0) on thousands of GPU games/step (88% GPU util, ~800 g/s).
- **Full 2M-game run revealed instability.** It trained to a **peak of 65% @2048**
  (@600k games, reaching 4096) then **catastrophically collapsed** (~900k-1M: avg
  max tile 1700→258), ending at 1% @2048. Textbook value-function divergence — the
  "deadly triad" (approximation + bootstrapping + online updates). The peak model
  is checkpointed (`best_model.pt`, verified 65%).
- **Fix — target network + Huber loss** (added to `train()`): the bootstrap value
  comes from a lagged copy synced every 250 steps (online net selects the move,
  target net values it); Huber tames large TD errors. A 1.2M-game re-run **stayed
  stable** — oscillating 25-52% @2048 across the whole run, never collapsing (vs
  naive 65%→1%). Reaches 4096 consistently (3-8%). Final 37% @2048.

**Verdict on neural methods:** *viable and now stable, but not competitive with the
tabular n-tuple.* The **afterstate formulation is right** (learns to ~40-50% @2048
and 4096 tiles; DQN's `Q(s,a)` never learned at all), and the **target network is
required** for stable value-net TD here. But the n-tuple's exact table lookups +
8-fold weight sharing are far more sample-efficient — 96% @2048 vs the net's ~50%.
Neural + GPU is a real path, not a dead end, but for *this* game tabular wins
decisively. Plausible ways to close the gap (untested): expectimax on the neural V
(search lifted the n-tuple 48%→92%, so it should help here too), a bigger net,
prioritized replay, or n-step returns.

**Next up (optional):** expectimax over the neural V; Track B multi-stage for 32768;
C2 (PPO) / C3 (distillation) as comparison rows.

## References

- Szubert & Jaśkowski (2014), *Temporal Difference Learning of N-Tuple Networks for the Game 2048*.
- Yeh, Wu et al. (2016), *Multi-Stage Temporal Difference Learning for 2048-like Games*.
- Jaśkowski (2018), *Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight
  Promotion, Redundant Encoding and Carousel Shaping*.
- Beal & Smith (1999), *Temporal Coherence* (adaptive step sizes).
