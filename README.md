# 2048 AI

<img src="README.assets/2048AI.gif" alt="2048AI.gif" width="50%;" />

A [2048](https://en.wikipedia.org/wiki/2048_(video_game)) game (pygame UI) plus a set of AI players that try to reach the 2048 tile — **classical search agents** and **reinforcement-learning agents**. Python 3.12, managed with [`uv`](https://docs.astral.sh/uv/).

## Quick start

```bash
uv sync                                       # install deps into .venv
uv run play.py                                # play manually (arrow keys; 'q' quit, 'r' restart)

# Reinforcement learning
uv run train_ntuple.py                        # train the STRONGEST agent (n-tuple TD; minutes on CPU)
uv run train_dqn.py --smoke                   # sanity-check the DQN pipeline (MPS/CUDA/CPU auto)
uv run evaluate_model.py models/<run>/best_model.pth   # greedy eval vs untrained baseline
tensorboard --logdir runs                     # view training curves

# Classical search agents (benchmark over many seeds)
uv run test-ai.py backtracking --search_depth 3 --seeds 1-100 --mp
uv run test-ai.py random --searches_per_move 20 --search_length 10 --seeds 1-100
```

> The classical benchmark (`test-ai.py`) and the tests need `pytest tqdm tabulate coverage`, which
> are **not** declared in `pyproject.toml` — add them with `uv add --dev pytest tqdm tabulate coverage`.

## Results summary

| Agent | Method | Reach 2048 | Best tile | Speed |
| --- | --- | --- | --- | --- |
| **n-tuple TD** ⭐ | RL — afterstate TD, n-tuple value function | **80.6%** | **8192** | instant |
| `BacktrackingAIPlayer` (depth 5) | search — depth-limited tree search | 32% | 2048 | ~19 s/move |
| `RandomGuessAIPlayer` | search — Monte-Carlo random rollouts | 4% | 2048 | ~2 s/move |
| DQN | RL — value-based (Double/Dueling) | 0% | 512 | instant |

The **n-tuple TD** agent is the clear winner: after 40k games (~96 min on CPU) it reaches 2048 in
80.6% of games (97.2% to 1024, ~36% to 4096, occasionally 8192) — and plays instantly.

## Reinforcement-learning agents

📖 **Full tutorial + experiment log (Chinese, written for people new to neural nets):
[`docs/dqn_report.md`](docs/dqn_report.md).** Roadmap for what's next: [`docs/next_steps.md`](docs/next_steps.md).

### n-tuple network + afterstate TD — the strongest agent

Learns a value function over **afterstates** (the deterministic board right after a move's merges,
before the random tile spawns) and picks actions by `argmax_a [ reward(s,a) + V(afterstate(s,a)) ]`.
`V` is a set of overlapping n-tuple lookup tables with 8-fold dihedral symmetry weight-sharing.
Tabular / CPU (no GPU needed), and dramatically more sample-efficient than DQN on 2048.

```bash
uv run train_ntuple.py --games 40000     # ~96 min -> 80.6% reach-2048
uv run train_ntuple.py --smoke           # 300-game sanity run (reaches 1024 in seconds)
```

Code: [`src/agent/ntuple.py`](src/agent/ntuple.py), [`train_ntuple.py`](train_ntuple.py).

### DQN — investigated in depth (and it struggles here)

A Double/Dueling DQN with 16-channel one-hot board encoding, action masking, experience replay,
target network, reward shaping and reward clipping. Despite a fair, thorough effort (multiple
architectures, ~1.5M steps), **vanilla value-based DQN does not learn to play 2048 well** — it
plateaus near the random baseline and never reaches 2048. The report walks through the whole
debugging journey, including diagnosing a **value-divergence bug** (Q exploded to ~1e5) and fixing
it with reward clipping — after which it's stable but still limited by 2048's exploration /
credit-assignment difficulty (which afterstate methods sidestep).

```bash
uv run train_dqn.py --episodes 5000 --reward-clip 1.0 --empty-weight 0.5
uv run train_dqn.py --resume models/<run>/checkpoint_ep500.pth   # resume training
```

Code: [`src/agent/model/dqn.py`](src/agent/model/dqn.py), `DQNPlayer` in
[`src/agent/agentImpl.py`](src/agent/agentImpl.py), [`train_dqn.py`](train_dqn.py).

## Classical search agents

### RandomGuessAIPlayer

`RandomGuessAIPlayer` has accuracy of 4% — Monte-Carlo random rollouts per candidate move.

```
Within 100 seeds, 4 passed
Total Time Taken: 0:03:15.620000
Average Time Taken: 1.96s
      score  max_val    runtime
seed
44    20300   2048.0  36.838138
65    20532   2048.0  34.245618
72    20324   2048.0  35.080057
81    20224   2048.0  36.138344
```

### BacktrackingAIPlayer

`BacktrackingAIPlayer` has accuracy of 32% — depth-limited tree search maximizing merge score.

| Search Depth | Pass Rate | Time (100 seeds) |
| ------------ | --------- | ---------------- |
| 2            | 4%        | 0:00:22          |
| 3            | 15%       | 0:01:31          |
| 4            | 26%       | 0:06:26          |
| 5            | 32%       | 0:31:10          |

Observation:
Pass Rate and Runtime both increase with Search Depth.
As Search Depth increases by 1, the runtime increases by around 4 times (5 times of its previous depth), and pass rate's increasing rate drops as search depth increases.


<details>
  <summary>Search Depth = 2</summary>

```
Search Depth=2
Accuracy=4%
Total Time Taken: 0:00:23.480000
Average Time Taken: 0.23s
      score  max_val   runtime
seed
52    20180   2048.0  3.376090
86    20336   2048.0  3.415089
91    20612   2048.0  3.161147
97    20172   2048.0  2.773934
```
</details>


<details>
  <summary>Search Depth = 3</summary>

```
Search Depth=3
Accuracy=15%
Total Time Taken: 0:01:31.510000
Average Time Taken: 0.92s
      score  max_val    runtime
seed
0     20344   2048.0  13.649011
4     20464   2048.0  13.875819
6     20204   2048.0  13.452688
10    20740   2048.0  14.429961
14    20560   2048.0  12.330968
20    20444   2048.0  12.224555
22    20472   2048.0  11.956490
29    20160   2048.0  12.149727
45    20380   2048.0  12.408184
49    20396   2048.0  13.272476
52    20380   2048.0  12.438213
60    20264   2048.0  11.940426
95    20372   2048.0   9.455263
96    20276   2048.0   8.376836
98    20356   2048.0   7.601177
```
</details>

<details>
  <summary>Search Depth = 4</summary>

```
Search Depth=3
Accuracy=26%
Total Time Taken: 0:06:26.520000
Average Time Taken: 3.87s
      score  max_val    runtime
seed
4     20208   2048.0  49.506586
5     20420   2048.0  50.247087
10    20312   2048.0  49.936312
13    20328   2048.0  47.755972
16    20348   2048.0  49.977037
17    20220   2048.0  49.544729
20    20508   2048.0  51.539473
28    20336   2048.0  50.380503
35    20312   2048.0  50.914232
37    20236   2048.0  49.910655
38    20848   2048.0  50.354784
45    20188   2048.0  48.015649
46    20352   2048.0  48.977642
50    20204   2048.0  50.954702
58    20956   2048.0  50.587713
62    20184   2048.0  48.263995
63    20416   2048.0  49.810407
65    20308   2048.0  47.298815
68    20344   2048.0  48.631845
69    20284   2048.0  47.222228
73    20256   2048.0  49.400236
75    20300   2048.0  49.068685
79    20300   2048.0  49.949769
81    20324   2048.0  48.920872
85    20332   2048.0  48.574214
88    20340   2048.0  48.564158
```
</details>


<details>
  <summary>Search Depth = 5</summary>

```
Search Depth=5
Accuracy=32%
Total Time Taken: 0:31:10.410000
Average Time Taken: 18.7s
in 100 seeds, the following are passing
        score  max_val     runtime
seed
6     20328   2048.0  247.989939
8     20396   2048.0  260.319733
10    20368   2048.0  243.177021
14    20328   2048.0  241.503823
15    20168   2048.0  246.624875
17    20264   2048.0  238.900709
23    20288   2048.0  249.540942
28    20228   2048.0  249.040510
30    20256   2048.0  245.020730
32    20300   2048.0  246.836974
33    20280   2048.0  245.686801
37    20336   2048.0  256.914075
40    20332   2048.0  255.554091
45    20200   2048.0  246.123966
47    20300   2048.0  250.852761
56    20256   2048.0  253.074194
62    20252   2048.0  244.514475
63    20212   2048.0  245.337596
65    20440   2048.0  245.588092
67    20256   2048.0  246.610915
70    20200   2048.0  237.210614
71    20484   2048.0  248.832584
73    20456   2048.0  251.539340
78    20216   2048.0  247.156880
79    20196   2048.0  235.604769
80    20336   2048.0  242.382276
81    20312   2048.0  252.015181
83    20436   2048.0  253.451177
84    20352   2048.0  242.617993
87    20252   2048.0  243.486014
88    20200   2048.0  242.957299
94    20304   2048.0  210.975342
95    20248   2048.0  177.931139
```
</details>

## Docs

- [`docs/dqn_report.md`](docs/dqn_report.md) — DQN + n-tuple tutorial and full experiment log (Chinese).
- [`docs/next_steps.md`](docs/next_steps.md) — roadmap (n-tuple + expectimax, DQN extensions, …).
- [`AGENTS.md`](AGENTS.md) / `CLAUDE.md` — architecture notes and commands for contributors.
