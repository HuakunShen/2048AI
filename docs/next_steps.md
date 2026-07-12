# 接下来还能做什么（2048 AI 项目 · 后续计划）

这份文件记录当前进度和后续可做的方向，方便随时接着往下推。

## 已完成

- **n-tuple afterstate TD**（`src/agent/ntuple.py` + `train_ntuple.py`）：训练满 40000 局，**80.6% 到 2048、97.2% 到 1024、约 36% 到 4096、最高 8192**。目前最强的智能体。
- **DQN 完整调查**（`src/agent/model/dqn.py`、`src/agent/agentImpl.py`、`train_dqn.py`）：one-hot 编码 + 动作屏蔽 + Double/Dueling + 奖励塑形 + reward clipping。诊断出**价值发散（Q→10⁵）**并修复，但即使修复后仍卡在基线附近（详见 `docs/dqn_report.md` 7.4b–7.4d）。
- **技术报告** `docs/dqn_report.md`（给 NN 初学者的 DQN + n-tuple 教程 + 实验记录）。
- **工具**：`evaluate_model.py`（DQN 贪心评估 + 基线对比）、`train_dqn.py --resume`（断点续训）。
- **环境**：Python 固定到 3.12（3.14 无 wheel）；torch MPS 加速。

## 建议方向（按性价比排序）

### 1. ⭐ n-tuple + Expectimax（最高性价比，冲 4096/8192）
- **思路**：用训好的 n-tuple `V` 当叶子评估函数，做 2–3 层 expectimax 前瞻。2048 有随机冒方块的 chance 节点 → expectimax 对随机取**期望**、对自己动作取 **max**。
- **现成基础**：`BacktrackingAIPlayer`（`src/agent/agentImpl.py`）已是搜索框架；`NumpyStaticBoard.move(..., inplace=False)` 给 afterstate + 合并得分；`NTupleNetwork.value()` 给评估；`NTupleNetwork.load()` 加载存档。
- **做法**：新增 `ExpectimaxNTuplePlayer`（或改造 backtracking，叶子从"合并得分"换成 `reward + V(afterstate)`，并在冒方块层对 2/4、各空格取期望）。
- **预期**：稳定 4096、常摸 8192。**工作量：中（~半天）**。

### 2. 改进 n-tuple 本身（更高 2048 率 / 更大方块）
- **TC-learning**（temporal coherence，自适应每权重步长）——收敛更快更稳。
- **更大/更多 tuple**：7-tuple、或加入覆盖更全的 pattern（当前是 4 个 6-tuple）。内存换性能。
- **multi-staging**：按游戏阶段（如最大方块 <2048 / ≥2048）用不同权重表。
- **工作量：中**。

### 3. 继续压榨 DQN（教学价值高，边际收益递减）
- **n-step 回报**（n=3~5）：显著改善 2048 的信用分配（当前是 1-step）。
- **Prioritized Experience Replay (PER)**：优先重放 TD 误差大的经验，让稀有的高级局面被多学。
- **afterstate 值网络**：把 n-tuple 的思路搬到神经网络 —— 用 CNN 学 `V(afterstate)`，动作选 `argmax_a[r + V(afterstate)]`。这可能是让"神经网络 + GPU"方法在 2048 上真正 work 的路子。
- **Rainbow / 分布式 RL（C51）**：DQN 全家桶。
- **工作量：每项中~大**。

### 4. 其他 RL / ML 范式（学习 + 对比用）
- **PPO**：现代主流策略梯度，通用稳健。
- **进化策略 ES / CMA-ES**：免梯度，直接进化评估函数或策略网络权重，天然并行。
- **行为克隆**：模仿 expectimax 的走法（监督学习），快速热启动，可再接 RL 微调。
- **工作量：每项中**。

### 5. 可视化 / 体验
- **GUI 看 n-tuple 玩**：`src/agent/ntuple.py` 已有 `make_player(net, game, ui=True)`，接 `Player.run()` 即可用 pygame 观战。需先 `net = NTupleNetwork(); net.load('models/ntuple_2048_XXXX/best_model.npz')`。
- **训练曲线对比图**：DQN（平）vs n-tuple（爬升）的 MaxTile/Score 曲线，可用 TensorBoard 或导出画图。
- **工作量：小**。

### 6. 工程 / 项目卫生
- **补 dev 依赖**：`uv add --dev pytest tqdm tabulate coverage`（`tests/` 和 `test-ai.py` 需要，但没在 `pyproject.toml` 里声明）。
- **给 n-tuple 加单元测试**：对称性置换正确性、afterstate = `move(inplace=False)`、`value/update` 的 TD 收敛性。
- **修 `.github/workflows/pytest.yml`**：已过时（引用不存在的 `requirements.txt`、Python 3.6–3.8、退役的 runner）。
- **工作量：小**。

## 快速命令参考

```bash
uv sync
# n-tuple（最强，几分钟见效）
uv run train_ntuple.py --smoke
uv run train_ntuple.py --games 40000
# DQN（含 dueling / reward-clip / 断点续训）
uv run train_dqn.py --smoke
uv run train_dqn.py --episodes 5000 --reward-clip 1.0 --empty-weight 0.5
uv run train_dqn.py --resume models/dqn_2048_XXXX/checkpoint_ep500.pth
uv run evaluate_model.py models/dqn_2048_XXXX/best_model.pth
# 看曲线
tensorboard --logdir runs
```
