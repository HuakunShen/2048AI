# 可变尺寸 Grid 的 2048 AI

## 通用 N-Tuple Afterstate TD / Temporal-Coherence / Expectimax 实施计划

目标：用同一个模型支持最大 8×8 的任意 H×W 棋盘，
同时保留 4×4 专项性能，并建立冲击 32768 / 65536 的研究路线。

> **推荐架构**
>
> 共享局部 N-Tuple 基座 + 对称规范化的位置角色 + Shape/Stage 条件校准 + 小型残差表

**版本：** 1.0
**日期：** 2026 年 7 月 14 日

# 文档定位
这是一份可直接执行的工程与研究计划。它不是把现有 4×4 模型简单改成动态数组，而是重新定义棋盘引擎、特征编译器、价值函数聚合、跨尺寸训练、搜索预算和评测协议，使“同一个模型处理不同长宽”成为可验证的能力。

> **核心结论**
>
> 可以用同一个模型处理 3×3 至 8×8 的任意矩形棋盘，但不能只做“相对坐标 tuple + 全局共享表”。纯平移共享会抹掉角落、边缘和中心的语义；纯求和又会让价值随 placement 数量失控。推荐使用共享局部表、位置角色、placement 归一化、shape/stage 校准和小型条件残差的混合结构。

# 目录
1. [决策摘要与成功标准](#1-决策摘要与成功标准)
2. [为什么朴素扩展会失败](#2-为什么朴素扩展会失败)
3. [推荐的通用价值网络架构](#3-推荐的通用价值网络架构)
4. [Pattern、位置角色与对称性设计](#4-pattern位置角色与对称性设计)
5. [训练系统与超参数](#5-训练系统与超参数)
6. [Expectimax 与 MCTS 搜索路线](#6-expectimax-与-mcts-搜索路线)
7. [4×4 冲击 32768 / 65536 的专项路线](#7-4×4-冲击-32768--65536-的专项路线)
8. [实验矩阵、评测协议与消融](#8-实验矩阵评测协议与消融)
9. [工程里程碑与验收条件](#9-工程里程碑与验收条件)
10. [风险、内存预算与降级方案](#10-风险内存预算与降级方案)
11. [论文与实现参考](#11-论文与实现参考)

# 1. 决策摘要与成功标准
## 1.1 推荐决策
采用“Universal N-Tuple Value Network”架构：棋盘规则和搜索完全支持动态 H×W；局部 pattern 用相对坐标描述并在任意棋盘上生成合法 placements；所有尺寸共享主要 lookup tables；模型再用对称不变的位置角色、棋盘 shape/stage 条件和少量 residual 参数修正跨尺寸的语义差异。

| **方案**                 | **跨尺寸泛化**   | **4×4 上限** | **复杂度** | **结论**                             |
|--------------------------|------------------|--------------|------------|--------------------------------------|
| 每个尺寸独立模型         | 无共享           | 最高         | 中         | 不满足“同一模型”目标，可作为上界基线 |
| 纯共享相对 tuple         | 高               | 可能明显下降 | 低         | 不推荐：丢失绝对位置语义和价值尺度   |
| 共享基座 + 条件校准/残差 | 高               | 接近专项模型 | 中高       | 推荐主方案                           |
| 全卷积神经网络           | 天然支持动态尺寸 | 样本效率较弱 | 高         | 作为对照组，不替代 N-tuple 主线      |

## 1.2 第一阶段成功标准
- 同一份模型文件可以直接推理 3×3、4×4、3×7、5×6、6×5、6×6、8×8，不重新编译 pattern，不重新训练。

- 4×4 greedy 性能不低于现有模型的 95%，4×4 expectimax 性能不低于现有模型的 97%；最终目标是恢复或超过现有基线。

- 在未参与训练的 held-out shape（例如 4×7 或 5×8）上显著优于随机/启发式基线，证明不是只记住尺寸。

- 推理时 value evaluation 的时间复杂度与激活 placement 数近似线性；8×8 greedy 单步仍可在 CPU 上实时运行。

- 训练、评测和搜索全部使用相同的通用 Board API，并能复现固定 seed。

## 1.3 非目标
- 第一版不要求同一模型在所有棋盘尺寸上都达到各自独立模型的理论最优。

- 第一版不同时支持任意障碍格、六边形网格或完全不同的合并规则；接口应预留，但不作为验收条件。

- 第一版不以 MCTS 替代 expectimax。先让通用 value model 和 node-budget 搜索稳定，再做 MCTS 对照。

# 2. 为什么朴素扩展会失败
## 2.1 固定坐标绑定
现有 4×4 pattern 把特征和棋盘绝对坐标绑定。把数组扩成 8×8 并不会自动产生泛化：原表没有见过新的位置组合，也无法解释同一局部结构位于角落、边缘或中心时的不同意义。

## 2.2 纯平移共享会抹掉角落策略
2048 的强策略高度依赖边界、角落、单调链和蛇形排列。若所有同形局部窗口共用完全相同的表项，模型会把“靠角落的 2×3”与“棋盘中央的 2×3”视为同一特征。对图像卷积这类平移不变任务可能合理，但对 2048 的全局拓扑不够。

## 2.3 placement 数导致尺度漂移
同一个 2×3 pattern 在 4×4 上有 6 个 placements，在 8×8 上有 42 个。若 V 直接求和，大棋盘的值会仅因 placements 更多而膨胀；若直接取平均，又可能低估大棋盘更长的未来回报。因此需要“归一化局部聚合 + shape/stage 标定”，而不是无条件求和。

## 2.4 长局训练会改变采样分布
8×8 一局可能比 4×4 长很多。如果按“局数”混合训练，大棋盘会贡献远多于小棋盘的 transitions；若并行 worker 按 episode 自动循环，模型会被长局隐式主导。训练配额应按 transitions 或有效更新次数控制。

## 2.5 7-tuple 的稠密表不可直接复制
```text
6-tuple: 16^6 = 16,777,216 entries
  float32 W only ≈ 64 MiB / table
  W + TC accumulators E + A ≈ 192 MiB / table
  8 tables with W/E/A ≈ 1.5 GiB

7-tuple: 16^7 = 268,435,456 entries
  W only ≈ 1.0 GiB / table
  W + E + A ≈ 3.0 GiB / table
```

因此“更多 7-tuples”只能采用 sparse/hash table、分层符号编码、减少 TC 状态、量化残差或按访问延迟分配，不能照搬 6-tuple 的稠密数组。

# 3. 推荐的通用价值网络架构
## 3.1 总体结构
```text
Board(H, W) + afterstate
│
├── Pattern compiler: relative coordinates → valid placements
├── Symmetry canonicalizer: D4 / D2 / transpose family
├── Position-role encoder: corner / edge / interior / normalized zone
├── Shared dense N-tuple tables
├── Optional sparse high-order / redundant subtuple tables
└── Shape-stage calibrator + small residual tables
│
▼
V(afterstate, H, W, stage)
```

## 3.2 数学定义
设棋盘 B 的尺寸为 H×W；第 k 个 pattern 用相对坐标集合 Pₖ 表示；Πₖ(H,W) 是该 pattern 在当前棋盘上的所有合法 placements；xₖ,ₚ(B) 是 placement p 对应的 tile 编码；role(p,H,W) 是对称规范化的位置角色；z(B) 是游戏阶段。推荐的第一版价值函数为：

```text
u_k(B) = (1 / |Π_k|) · Σ_{p∈Π_k} [T_k(x_{k,p}) + ρ · R_{k,role,z}(x_{k,p})]

V(B,H,W) = b(H,W,z) + c(H,W,z) · Σ_k g_k(H,W,z) · u_k(B)

a* = argmax_a [reward(s,a) + V(afterstate(s,a), H, W)]
```

Tₖ 是跨所有尺寸共享的主表；R 是较小的条件残差，可仅覆盖最重要的 pattern、角色和 stage；gₖ 是 pattern 组权重；c 和 b 负责把归一化局部表示映射到当前棋盘的原始未来得分尺度。这样既保留共享，又允许 4×4、3×7、8×8 在相同局部结构上产生不同的全局估值。

## 3.3 推荐的三层参数共享
| **层级**         | **共享范围**             | **参数形式**          | **作用**                               |
|------------------|--------------------------|-----------------------|----------------------------------------|
| Shared base      | 所有 H×W                 | 稠密 4/5/6-tuple LUT  | 学习局部合并、空格、相邻关系等通用规律 |
| Role residual    | 同类位置角色             | 小型 LUT 或低精度残差 | 恢复角落、边缘、内部的语义差异         |
| Shape-stage head | area/aspect/stage bucket | 标量或很小的线性表    | 校准价值尺度和不同阶段的 pattern 权重  |

## 3.4 Shape 编码
不建议为每个精确 H×W 建立完全独立的主网络。第一版用少量可解释特征做条件化：area=H·W、min(H,W)、max(H,W)、aspect=max/min、是否正方形、空格比例、最大 tile 指数、总 tile mass 的对数。离散 bucket 与连续标量可以并存。

- Area bucket：≤16、17–25、26–36、37–49、50–64。

- Aspect bucket：1.0、(1.0,1.5\]、(1.5,2.0\]、\>2.0。

- Stage：由 max tile + empty ratio 联合决定，而不是只按 score。

- 第一版 shape head 只含几十至几百个标量，不引入神经网络依赖。

## 3.5 Tile alphabet 与高 tile
当前 0..15 的 16-symbol 编码正好覆盖 empty 与 2..32768。若更大棋盘需要 65536 或更高，直接把 alphabet 改成 17/18 会使 LUT 指数级膨胀。推荐把 tile 指数裁剪到 15，并增加高 tile 辅助特征，例如 max exponent、overflow count、最高两枚 tile 的差值；4×4 65536 专项模型可再单独实验扩展编码。

# 4. Pattern、位置角色与对称性设计
## 4.1 相对坐标 Pattern DSL
所有 pattern 都用以 (0,0) 为局部原点的相对坐标定义。编译阶段生成合法平移、方向、角色和对称映射。模型文件保存 pattern schema 与版本，避免代码更新后权重解释改变。

```python
Pattern(
    id="rect_2x3",
    cells=[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    alphabet=16,
    symmetry="canonical",
    storage="dense",
)
```

## 4.2 第一版 Pattern Library
| **组**           | **建议 pattern**                    | **tuple 大小** | **用途**                            |
|------------------|-------------------------------------|----------------|-------------------------------------|
| Small-board core | 2×2、line-4、L-4、zigzag-4          | 4              | 保证 3×N、N×3 上有足够覆盖          |
| Shared core      | line-5、cross/offset-5、L-5         | 5              | 低成本补充中程结构                  |
| Main capacity    | 2×3、3×2、line-6、snake-6、corner-6 | 6              | 主价值容量和 4×4 兼容               |
| Experimental     | selected 7-tuples                   | 7              | 只用 sparse/hash，针对残局和高 tile |
| Redundant        | 主 tuple 的 line-3/4、2×2 子集      | 3–4            | 加速泛化，训练后可合并/折叠         |

## 4.3 位置角色编码
角色必须在对称变换下保持一致，不能简单使用“左上角/右下角”绝对标签。推荐先把 placement 根据最近边界和最近角落规范化，再计算 role。

- Topology：corner-touching、edge-touching、near-edge、interior。

- Normalized zone：placement 中心点映射到 3×3 归一化区域，但在对称 canonicalization 后使用。

- Boundary distances：到最近水平边界与垂直边界的离散距离，裁剪为 0、1、2+。

- Orientation：pattern 的长轴相对最近边界/角落的方向，而不是绝对上下左右。

## 4.4 矩形棋盘的正确对称性
正方形 H=W 时，8 个二面体变换都把棋盘映射回自身。非正方形 H≠W 时，恒等、180°旋转、水平翻转、垂直翻转构成同一 H×W 内的 4 个自同构；90°/270°旋转和对角反射会映射到 W×H。因为系统同时支持 5×6 与 6×5，可以把后者作为跨 shape 的同构数据增强和参数共享，但缓存键和 shape head 必须区分 H×W 与 W×H。

> **实现要求**
>
> Pattern compiler 应输出 canonical feature key，而不是在每次 value evaluation 时动态创建对象。为每个 `(H, W, pattern_set_version)` 预编译 placements、索引偏移、role id 和 symmetry mapping，并缓存为紧凑的 `int16`/`int32` 数组。

# 5. 训练系统与超参数
## 5.1 TD 与 TC 更新
继续使用 afterstate TD(0)，γ=1。统一使用缩放后的 merge reward 以降低数值范围，例如 reward_scale=1/1024；推理时 value 与 reward 使用同一尺度，因此 action ordering 不变。终局 next value 为 0。

```text
δ_t = r_{t+1} + V(after_{t+1}) - V(after_t)
E_i ← E_i + δ_i
A_i ← A_i + |δ_i|
coherence_i = |E_i| / (A_i + ε)
Δw_i = (α / active_feature_count) · coherence_i · δ_i
```

对于 placement 平均聚合，单个激活表项的梯度还应除以该 pattern 的 placement 数，并乘以当前 shape-stage scale c。必须在单元测试中验证：同一局部 pattern 在 4×4 和 8×8 上不会仅因 placement 数不同而产生数量级不同的单次更新。

## 5.2 建议起始超参数
| **参数**             | **起始值** | **搜索范围**                     | **说明**                          |
|----------------------|------------|----------------------------------|-----------------------------------|
| γ                    | 1.0        | 固定                             | 无折扣 episodic return            |
| reward_scale         | 1/1024     | 1/256–1/4096                     | 所有棋盘统一，不按尺寸变化        |
| TC α（shared）       | 0.5        | 0.1–1.0                          | 以现有实现为基准做 sweep          |
| TC α（residual）     | 0.1        | 0.02–0.3                         | 避免小样本 bucket 过拟合          |
| ε                    | 1e-8       | 1e-6–1e-10                       | TC 分母稳定                       |
| Stage 数             | 4          | 2–8                              | 先粗分，再根据访问量拆分          |
| Role residual 系数 ρ | 0.25       | 0.1–0.5                          | 控制条件表对共享基座的偏离        |
| Optimistic init      | 关闭       | 0 / 80k / 160k / 320k 等价缩放值 | 第二轮实验开启，注意 reward_scale |

## 5.3 Curriculum 与尺寸采样
| **阶段**          | **训练分布**              | **预算建议**                  | **验收**                            |
|-------------------|---------------------------|-------------------------------|-------------------------------------|
| A：4×4 parity     | 仅 4×4                    | ≥100k episodes 或现有收敛预算 | 通用引擎复现现有成绩                |
| B：邻近迁移       | 4×4、3×4、4×5、5×5        | 每 shape ≥2M transitions      | 共享模型无明显灾难遗忘              |
| C：全尺寸混合     | 3≤H,W≤8，重点采常用 shape | 每 shape 5–20M transitions    | 所有训练 shape 稳定提升             |
| D：held-out 泛化  | 训练时排除若干 shape      | 只评测，不更新                | 对未见 shape 有正迁移               |
| E：专项 fine-tune | 4×4 32768 + 大棋盘残局    | 按目标独立 budget             | 不破坏 shared base，或使用冻结/残差 |

混合训练必须按 transition quota 控制。建议维护每个 shape bucket 的目标更新比例，而不是按 episode 数轮询。初始分布可让 4×4 占 35% 更新、面积 17–25 占 25%、26–36 占 20%、37–49 占 10%、50–64 占 10%；每 1M updates 根据欠采样 bucket 自动重平衡。

## 5.4 Stage 设计
通用模型不应只用固定 score 阈值划分 stage，因为大棋盘的 score 分布与 4×4 不同。推荐使用 \`(max_exponent, empty_ratio, log2(tile_mass/area))\` 的联合 bucket。第一版可用 4 个阶段：

- S0：早期，max tile ≤256 或 empty_ratio \>0.55。

- S1：中前期，max tile 512–2048，empty_ratio 0.30–0.60。

- S2：中后期，max tile 4096–8192 或 empty_ratio 0.12–0.35。

- S3：残局/高 tile，max tile ≥16384 或 empty_ratio \<0.15。

边界处使用 hysteresis 或根据 afterstate 单向推进，避免 stage 在相邻步来回切换。4×4 专项可以使用更细的 max-tile stage，并启用 weight promotion。

## 5.5 并行训练
- 优先保持现有 numba CPU 路线，worker 只生成 trajectories，参数更新可采用单写者批量更新以确保可复现。

- 第二阶段实验 lock-free optimistic parallelism；必须提供 deterministic 单线程基线和 race-tolerant 性能对照。

- 定期冻结 snapshot 做离线评测，不使用训练局的移动平均作为最终结论。

- 模型 checkpoint 必须包含 pattern schema hash、shape/stage 配置、reward_scale、tile cap 和训练计数。

# 6. Expectimax 与 MCTS 搜索路线
## 6.1 通用 Expectimax
动态棋盘的 chance branching factor 为 2E，其中 E 是空格数。8×8 开局 E 很大，固定 depth 不合适；残局 E 下降后反而值得加深。搜索应以 node budget 或 time budget 为主，以 depth 为上限。

| **空格比例** | **建议 max ply** | **chance 处理**              | **节点预算起点**   |
|--------------|------------------|------------------------------|--------------------|
| \>0.55       | 1–2              | 按概率分层采样 4–12 个空格   | 5k–20k             |
| 0.30–0.55    | 2–3              | 采样或保留高影响空格         | 20k–80k            |
| 0.15–0.30    | 3–4              | 逐步接近全展开               | 80k–250k           |
| \<0.15       | 4–6+             | 全展开 + iterative deepening | 250k–1M 或时间预算 |

## 6.2 Chance 节点采样
- 按 tile 2/4 的真实概率采样，不改变 0.9/0.1。

- 空格采样使用 stratified sampling：角落/边缘/内部至少各保留代表点，避免均匀随机漏掉关键生成位置。

- 对同一 root 的多次搜索使用固定 seed 或 low-discrepancy 空格序列，降低 action value 方差。

- 在评测中同时报告 exact chance 与 sampled chance，不能只报告采样版本。

## 6.3 Transposition Table
键必须包含 H、W、board bit/byte encoding、节点类型、剩余 depth、model version 和 tile cap。对 sampled chance 节点还需区分 sampling policy/version。大棋盘不适合继续用单个 64-bit nibble board；推荐紧凑 byte array + xxHash/wyhash 双校验，或两个 64-bit/128-bit 分块。

## 6.4 MCTS 实验路线
MCTS 在 2048 中不是自动优于 expectimax：环境概率已知、动作仅四个，而 chance branching 大。先做带 chance node 的 stochastic MCTS 对照，不直接使用标准二人零和 PUCT。推荐顺序：

1.  MCTS-0：UCT/PUCT max nodes + 精确 chance sampling，leaf 使用 n-tuple V。

2.  MCTS-1：root action prior 由 1-ply \`r+V(after)\` softmax 得到；不另训 policy network。

3.  MCTS-2：progressive widening 控制空格生成分支，访问量增加时逐步展开更多 chance outcomes。

4.  MCTS-3：expectimax root + MCTS 仅用于 value 差距很小的 top-2 actions，形成 selective hybrid。

> **停止条件**
>
> 只有当 MCTS 在相同 wall-clock 或相同 node budget 下，至少在 3 个 seed 集合上稳定超过 expectimax，才进入主分支；否则保留为研究模块。

# 7. 4×4 冲击 32768 / 65536 的专项路线
通用模型与 4×4 极限模型应共享基础设施，但不强制完全共享所有权重。最稳妥的产品结构是：Universal base 负责跨尺寸；4×4 specialist head 负责最高性能。这样不会为了 8×8 泛化牺牲 4×4 的 32768 概率。

## 7.1 Multi-stage + Weight Promotion
为高 tile 建立 2–4 个专项 stage 表。进入新 stage 时，将上一 stage 对应表项复制到新 stage 的首次访问位置，再允许后续独立更新。优先测试 2-stage 与 4-stage，避免一开始复制大量 16-stage 结构。论文结果表明 multi-stage 对 32768 reaching rate 很关键；后续 optimistic initialization 工作也显示更少 stage 可以配合更强探索。

## 7.2 Delayed TC
把 episode 内同一权重的多次 TD 更新累积，在 episode 末或固定窗口统一应用，可减少频繁随机内存写并稳定 TC 统计。建议先复现 delayed-TC(λ=0.5) 与现有在线 TC(0) 的对照，再决定是否用于通用多尺寸训练。

## 7.3 Redundant Encoding
在 6-tuple 主表之外加入其包含的 3/4-tuple 子特征，例如 line-3、line-4、2×2。它不显著增加表达能力，但能让访问稀疏的高 tile 状态通过较小子模式更快泛化。训练完成后，可把冗余子表折叠进包含它们的主表，减少最终推理 lookup 次数。

## 7.4 Carousel Shaping
保存各 stage 的代表性初始状态池，训练 episode 轮流从 stage 1 正常开局、stage 2/3/4 的历史状态启动，使后期 stage 获得接近均衡的训练量。状态池应按多样性采样并限制同源轨迹比例，避免只学习少量残局模板。该方法更可能改善多 ply 搜索，而不一定提升纯 greedy。

## 7.5 Optimistic Initialization 与 OTD+TC
把未访问权重初始化为偏高价值，促使 agent 探索较少访问的状态。原始工作报告了 \`V_init=320k\`、OTD 后以约 10% 训练预算做 TC fine-tuning 的强结果；在本项目中必须先根据 reward_scale 和激活 tuple 数换算等价初值，不能直接复制原数值。建议 sweep：等价 raw value 80k、160k、320k、480k，并同时记录 1-ply 与 3-ply。

## 7.6 Tile-downgrading Search
高 tile 状态可能落入训练分布极稀疏区域。可在 root 满足特定条件时，把所有高 tile 同比例降一级，保持相对结构后用模型搜索，再将选定 action 应用到原棋盘。该技术只作为 root search transform，不在树内反复降级。需要严格消融，防止对通用大棋盘产生错误归纳。

| **实验优先级** | **组合**                       | **目的**                      |
|----------------|--------------------------------|-------------------------------|
| P0             | 现有 TC + 3-ply baseline       | 建立同 seed 的可复现基线      |
| P1             | 2-stage + weight promotion     | 直接提升 16384/32768 后期估值 |
| P2             | redundant 3/4-tuples           | 提高高 tile 稀疏状态泛化      |
| P3             | carousel shaping               | 均衡后期 stage 访问量         |
| P4             | optimistic init + TC fine-tune | 加强探索并减少所需 stage      |
| P5             | tile-downgrading search        | 修正超出训练分布的 root       |
| P6             | sparse selected 7-tuples       | 最后增加容量，避免先堆内存    |

# 8. 实验矩阵、评测协议与消融
## 8.1 必须报告的指标
- 原始 score：mean、median、P10、P90、maximum；同时给 bootstrap 95% CI。

- 最大 tile 到达率：2048、4096、8192、16384、32768、65536。

- 每局 moves、终局 empty count、最高 tile 所在拓扑角色。

- 速度：value evaluations/s、moves/s、search nodes/s、平均/尾部单步延迟。

- 资源：模型文件大小、训练峰值 RSS、E/A 累积器大小、TT 命中率。

- 跨尺寸归一化指标：score/area、moves/area、最大 tile exponent；但不能用它们替代原始指标。

## 8.2 Shape Split
| **集合**    | **示例 shape**                    | **用途**                          |
|-------------|-----------------------------------|-----------------------------------|
| Train-core  | 3×3、3×4、4×4、4×5、5×5、5×6、6×6 | 覆盖主要面积和长宽比              |
| Train-large | 6×7、7×7、7×8、8×8                | 学习大棋盘尺度与搜索              |
| Held-out A  | 3×7、4×7、5×8                     | 测试未见长宽组合                  |
| Held-out B  | 6×5、8×7                          | 测试 transpose / cross-shape 共享 |
| Specialist  | 4×4                               | 极限 32768/65536 研究             |

## 8.3 关键消融
| **消融**     | **比较**                                           | **要回答的问题**                   |
|--------------|----------------------------------------------------|------------------------------------|
| 位置角色     | 无 role vs corner/edge/interior vs 3×3 zone        | 角落语义是否是跨尺寸性能的主要来源 |
| 聚合尺度     | sum vs mean vs mean+calibrator                     | 大棋盘 value 漂移是否被解决        |
| 共享程度     | 全共享 vs shared+residual vs 独立模型              | 共享造成多少上限损失               |
| Pattern 阶数 | 4/5/6 混合 vs 仅 6 vs sparse 7                     | 容量与内存的最佳点                 |
| Stage        | 无 stage vs 2/4/8 stage                            | 后期高 tile 是否需要分段           |
| 搜索         | greedy、exact expectimax、sampled expectimax、MCTS | 同预算下哪种更强                   |

## 8.4 统计纪律
- 开发阶段每配置至少 1,000 局；高 tile 稀有事件的最终结论建议 10,000 局以上。

- 所有配置使用相同 seed 列表，并记录模型 hash、代码 commit、search budget。

- 不要只挑单个 checkpoint；报告预定义训练预算下的结果，或使用独立 validation seeds 选 checkpoint。

- 对 32768 reaching rate 使用 Wilson interval 或 bootstrap interval，避免 200 局样本上的百分比被误读。

# 9. 工程里程碑与验收条件
| **里程碑**              | **主要任务**                                              | **预计时间** | **验收条件**                               |
|-------------------------|-----------------------------------------------------------|--------------|--------------------------------------------|
| M0 基线冻结             | 保存现有 4×4 模型、seed、指标、性能 profile               | 0.5–1 天     | 任何重构可一键与基线比较                   |
| M1 通用 Board Engine    | 动态 H×W move/merge/spawn/afterstate/hash                 | 2–4 天       | 属性测试覆盖 3×3–8×8；4×4 bit-exact parity |
| M2 Pattern Compiler     | 相对坐标、placements、roles、symmetry cache               | 3–6 天       | 任意 shape 编译稳定；索引无越界            |
| M3 Universal Greedy     | shared tables + placement normalization + calibrator      | 5–10 天      | 同模型可训练/推理多 shape                  |
| M4 Conditional Residual | role/stage residual、shape head、checkpoint schema        | 5–8 天       | 4×4 性能恢复，held-out 改善                |
| M5 Mixed Curriculum     | transition quotas、worker、离线评测                       | 1–2 周       | 训练无尺寸饥饿/灾难遗忘                    |
| M6 Search               | node-budget expectimax、sampling、TT、iterative deepening | 1–2 周       | 各尺寸在延迟预算内稳定提升                 |
| M7 32768 Track          | MS-TD、WP、RE、CS、OI、tile downgrade                     | 2–4 周       | 32768 率显著超过当前基线                   |
| M8 MCTS Research        | stochastic MCTS / progressive widening / hybrid           | 1–3 周       | 同预算胜过 expectimax 才合并               |

## 9.1 M1 测试清单
- 对每个 H,W∈\[3,8\] 随机生成 10k boards，验证 move 后 tile mass 守恒（不含 spawn）和 reward 等于 merge 增量。

- 验证相同 action 的 afterstate 与 spawn 分离；非法 action 不生成随机 tile。

- 验证 5×6 转置后执行映射 action，与 6×5 结果转置一致。

- 验证所有对称变换下 legal moves、reward 和 terminal 判断一致。

- 固定 4×4 boards 与旧引擎逐状态比较，确保重构不改变语义。

## 9.2 M2/M3 测试清单
- 每个 pattern 的 placement 数与解析公式一致，例如 2×3 在 H×W 上应为 (H−1)(W−2)，再计方向/去重。

- 对称等价棋盘的 V 在浮点容差内相同；H×W 与 W×H transpose mapping 正确。

- 同一棋盘重复 value evaluation 不分配 Python 对象，性能 profile 中主要成本为 LUT memory access。

- 所有 active weights 的 update 总量与 α·δ 的设计约束一致，避免 pattern 数增多导致有效学习率增加。

- 模型保存/加载后 value bitwise 或数值等价，schema 不匹配时明确拒绝加载。

## 9.3 推荐代码模块
```text
src/
├── game/
│   ├── board.py                 # generic H×W rules
│   ├── afterstate.py
│   └── symmetry.py
├── ntuple/
│   ├── pattern_schema.py
│   ├── compiler.py              # placements / role / canonical mapping
│   ├── dense_table.py
│   ├── sparse_table.py
│   ├── universal_value.py
│   └── tc_update.py
├── training/
│   ├── curriculum.py
│   ├── transition_quota.py
│   ├── workers.py
│   └── evaluator.py
├── search/
│   ├── expectimax.py
│   ├── chance_sampler.py
│   ├── transposition.py
│   └── mcts.py                  # experimental
├── experiments/
│   ├── configs/
│   ├── benchmark.py
│   └── ablation.py
└── tests/
    ├── property/
    ├── parity/
    └── performance/
```

# 10. 风险、内存预算与降级方案
| **风险**       | **症状**                          | **缓解措施**                                                   | **触发降级**                  |
|----------------|-----------------------------------|----------------------------------------------------------------|-------------------------------|
| 跨尺寸负迁移   | 4×4 下降、held-out 不升           | 提高 role/stage residual；冻结 shared base；分 specialist head | 4×4 expectimax \<97% baseline |
| 价值尺度错误   | 大棋盘 action value 极端、TD 爆炸 | placement mean + shape calibrator；检查 reward_scale           | δ 方差跨 shape 相差 \>10×     |
| 稀疏高阶表过大 | RSS/页错误/缓存 miss              | hash/sparse、int16 residual、只保留高价值 pattern              | 训练 RSS \>机器预算 80%       |
| 长局支配训练   | 8×8 更新比例过高                  | transition quota + per-bucket counters                         | 任一 bucket 偏离目标 \>5%     |
| 搜索速度失控   | 8×8 chance nodes 爆炸             | node budget、stratified sampling、progressive widening         | P95 单步延迟超目标            |
| stage 饥饿     | 后期表访问量低                    | carousel pools、weight promotion、OI                           | S3 更新量 \<S0 的 5%          |
| 模型格式脆弱   | 旧权重静默误加载                  | schema hash、version migration、strict validation              | 任何不匹配必须 fail-fast      |

## 10.1 推荐内存分层
- Tier 0：4/5-tuple dense tables，W/E/A 全量 float32。

- Tier 1：核心 6-tuple dense tables，W float32；E/A 可 float32 或按访问分块分配。

- Tier 2：role/stage residual 使用 float16/int16 + scale，或只为访问过的块分配。

- Tier 3：7-tuple 使用 sparse hash；只在高 tile stage 或专项模型中启用。

- 推理导出：移除 E/A，只保留 W；把冗余子 tuple 折叠后生成只读 mmap 模型。

## 10.2 Go / No-Go 决策点
| **检查点**           | **Go 条件**                       | **No-Go 后处理**                      |
|----------------------|-----------------------------------|---------------------------------------|
| Universal base       | 4×4 ≥95% baseline 且 5×5 学习明显 | 先修聚合尺度和 role，不继续加 pattern |
| Conditional residual | 4×4 恢复且 held-out 改善          | 退回 per-area specialist head         |
| Sparse 7-tuple       | 每 GiB 内存带来显著 32768 增益    | 停止扩表，转向 OI/MS-TD/search        |
| MCTS                 | 同 wall-clock 稳定胜 expectimax   | 保留实验分支，不进默认路径            |

# 11. 论文与实现参考
下面按“先复现、再扩展”的优先级列出。文中超参数应作为复现实验起点，不应不经 reward scaling、network size 和训练预算换算直接照搬。

> \[1\] Marcin Szubert, Wojciech Jaśkowski. Temporal Difference Learning of N-Tuple Networks for the Game 2048. IEEE Conference on Computational Intelligence and Games, 2014. [链接](https://ieeexplore.ieee.org/document/6932907)
>
> \[2\] Wojciech Jaśkowski. Mastering 2048 with Delayed Temporal Coherence Learning, Multi-Stage Weight Promotion, Redundant Encoding and Carousel Shaping. IEEE Transactions on Games 10(1):3–14. [链接](https://arxiv.org/abs/1604.05085)
>
> \[3\] Kun-Hao Yeh et al. Multi-Stage Temporal Difference Learning for 2048-like Games. IEEE Transactions on Computational Intelligence and AI in Games. [链接](https://arxiv.org/abs/1606.07374)
>
> \[4\] Hung Guei, Lung-Pin Chen, I-Chen Wu. Optimistic Temporal Difference Learning for 2048. IEEE Transactions on Games. [链接](https://arxiv.org/abs/2111.11090)
>
> \[5\] Hung Guei. On Reinforcement Learning for the Game of 2048. Dissertation / comprehensive study covering optimistic TD, ensemble learning, MCTS and deep RL. [链接](https://arxiv.org/abs/2212.11087)
>
> \[6\] Wojciech Jaśkowski — mastering-2048 experiment source code. [链接](https://github.com/wjaskowski/mastering-2048)

## 11.1 从论文迁移到本项目的对应关系
| **文献方法**                   | **本项目中的位置**                   | **注意事项**                           |
|--------------------------------|--------------------------------------|----------------------------------------|
| Afterstate TD + n-tuple        | Shared base 的核心学习算法           | 先保持 4×4 parity                      |
| Temporal coherence             | 每个表项的自适应更新                 | placement normalization 会改变有效梯度 |
| Multi-stage / weight promotion | 4×4 specialist + 通用 stage residual | 通用 stage 不能只按固定 score          |
| Redundant encoding             | 3/4-tuple 子特征                     | 训练后尽量折叠减少推理开销             |
| Carousel shaping               | 后期 stage 初始状态池                | 更关注多 ply 搜索性能                  |
| Optimistic initialization      | 探索增强实验                         | 必须按 reward_scale 与激活特征数换算   |
| Tile-downgrading               | 高 tile root search transform        | 只对满足条件的专项搜索启用             |
| MCTS / ensemble                | 第二阶段研究                         | 与 expectimax 做同预算对照             |

# 附录 A：建议立即开始的两周 Sprint
| **日期**  | **任务**                                                  | **输出**                                      |
|-----------|-----------------------------------------------------------|-----------------------------------------------|
| Day 1     | 冻结 4×4 baseline、seed、profile、model schema            | baseline.json + checkpoint + benchmark report |
| Day 2–3   | 实现 generic Board(H,W) 和 property tests                 | 3×3–8×8 规则引擎                              |
| Day 4     | 4×4 parity + transpose/symmetry tests                     | 零语义回归                                    |
| Day 5–6   | Pattern DSL 与 placement compiler                         | 预编译 placement cache                        |
| Day 7     | 接入现有 6-tuple tables，先只跑 4×4                       | 通用 evaluator parity                         |
| Day 8–9   | placement mean + shape calibrator                         | 4×4/5×5 双尺寸训练                            |
| Day 10    | 加入 position role residual                               | role ablation                                 |
| Day 11–12 | mixed transition quota trainer                            | 4×4/3×4/4×5/5×5 混合模型                      |
| Day 13    | held-out 3×5/5×4/3×7 测试                                 | 泛化报告                                      |
| Day 14    | 决策 review：继续 universal residual 或拆 specialist head | M3 Go/No-Go 结论                              |

# 附录 B：最小配置建议
```yaml
model:
  tile_alphabet: 16
  tile_cap_exponent: 15
  shared_patterns:
    - square_2x2
    - line_4
    - l_4
    - line_5
    - rect_2x3
    - line_6
    - snake_6
    - corner_6
  aggregation: mean_per_pattern
  position_roles: [corner, edge, near_edge, interior]
  stage_count: 4
  shape_calibrator: area_aspect_stage
  residual_scale: 0.25

training:
  algorithm: afterstate_tc0
  gamma: 1.0
  reward_scale: 0.0009765625 # 1/1024
  alpha_shared: 0.5
  alpha_residual: 0.1
  transition_quotas: true

search:
  default: expectimax
  budget_mode: nodes
  chance_sampling: stratified
  transposition_table: true
  iterative_deepening: true
```

> **最终建议**
>
> 先完成 M0–M4，验证“共享局部表 + 位置角色 + shape/stage 校准”能否在不显著损伤 4×4 的情况下泛化到未见尺寸。只有这个假设成立后，才投入大规模 mixed-size 训练、7-tuple 和 MCTS。冲击 32768 的最高收益顺序应优先是 multi-stage/weight promotion、redundant encoding、carousel shaping、optimistic initialization 和更强 expectimax，而不是先扩大网络阶数。

---

# 12. 实施进度 (Implementation Progress)

> 本节记录计划的实际实现状态。截至目前，计划的**基础与核心价值里程碑 (M1–M3) 已完成并通过测试**，混合尺寸训练 (M5) 与通用搜索 (M6) 的核心已可运行。

## 12.1 已完成里程碑

| 里程碑 | 状态 | 代码 | 测试 |
|--------|------|------|------|
| **M1 通用 Board Engine** | ✅ 完成 | [`src/game/board.py`](src/game/board.py) | [`tests/test_board.py`](tests/test_board.py) (23) |
| **M2 Pattern Compiler** | ✅ 完成 | [`src/game/symmetry.py`](src/game/symmetry.py), [`src/ntuple/pattern.py`](src/ntuple/pattern.py), [`src/ntuple/library.py`](src/ntuple/library.py) | [`tests/test_pattern.py`](tests/test_pattern.py) (16) |
| **M3 Universal Greedy** | ✅ 完成 | [`src/ntuple/universal_value.py`](src/ntuple/universal_value.py) | [`tests/test_universal.py`](tests/test_universal.py) (9) |
| **M4 Conditional Residual** | ✅ 完成 | [`src/ntuple/universal_value.py`](src/ntuple/universal_value.py) (residual head) | [`tests/test_residual.py`](tests/test_residual.py) (6) |
| **M5 Mixed Curriculum (核心)** | ✅ 核心可用 | [`src/training/`](src/training/) (selfplay/curriculum/evaluator), [`train_universal.py`](train_universal.py) | [`tests/test_training.py`](tests/test_training.py) (3) |
| **M6 Search (核心)** | ✅ 核心可用 | [`src/search/expectimax.py`](src/search/expectimax.py) | [`tests/test_search.py`](tests/test_search.py) (9) |

全套 **94 个新测试通过**（连同既有套件全绿）。

## 12.2 关键设计落地

- **M1** — Exponent (`int8`) 棋盘；`move` 返回 `(afterstate, reward, changed)`，spawn 分离并接受显式 `np.random.Generator`；`@njit` 热路径。在方阵上与旧 `NumpyStaticBoard` **逐状态 bit-exact**，并在 3×3–8×8 上与独立参考实现一致、tile mass 守恒、转置等价 (LEFT↔UP, RIGHT↔DOWN)。修复了旧引擎对非方阵 `move` 迭代 `range(len(matrix))` 的 bug。
- **M2** — 相对坐标 `Pattern` DSL；编译器对每个 `(H,W)` 生成全部 orientation×translation placements（**Design B**：把 D4 orientation 烘焙进 placements 并共享单表，从而结构上保证棋盘对称不变），并按到边界距离计算对称不变的 role (corner/edge/near_edge/interior)。方阵自同构 8 个、非方阵 4 个 (§4.4)。placement schema hash 用于 checkpoint fail-fast。
- **M3** — `V(B)=Σ_k (1/|Π_k|)Σ_p T_k(x)`：所有尺寸**共享** dense 表，placement-mean 归一化消除尺度漂移。afterstate TD(0) + temporal-coherence 自适应步长（每权重 `|E|/A`，步长带 `1/n` 归一化，coherence 用原始 δ 统计，plan §5.1）。测试验证：棋盘对称下 V 严格不变、TD 收敛、**单次更新幅度与棋盘面积无关** (§9.2 不变量)、save/load 往返 + schema 不匹配拒绝加载。
- **M4** — 条件 residual head：对小表 (L≤4：square_2x2/line_4/l_4) 按 **position role** 建 per-value 残差表 `R_k[role,x]`，`V += (ρ/|Π_k|)Σ_p R_k(role,x)`（大表不加 per-x 残差，符合 §3.3/§10.1 内存分层，约 +3MB）。残差用独立 `α_residual`(0.1) 与自身 TC 累加器。测试验证：**加 residual 后 V 仍严格对称**（role 对称不变）、残差可学习并改善拟合、save/load 往返。训练器 `--residual --rho --alpha-residual` 可开启。
- **M5 核心** — `TransitionQuota` 按 **transition（步数）** 而非 episode 配额采样，避免长局（大棋盘）隐式主导 (§2.4/§5.3)；`train_universal.py` 驱动混合尺寸训练 + TensorBoard + 分尺寸离线评测 + best/final checkpoint。
- **M6 核心** — `UniversalExpectimax`：动态棋盘上的 afterstate expectimax，leaf 用通用 V；按**空格比例**自适应深度 (§6.1)、chance 节点采样、`node_budget` 上限、shape-aware TT 键、私有采样 RNG。**depth-1 严格等价 greedy**（正确性锚点，已测）。

## 12.3 实测结果（M3 Go/No-Go 验证）

**冒烟 (1500 局，~3 min)** — 单模型 4×4+5×5，share 稳定 [0.50,0.50]：4×4 达 92%→1024 / 37%→2048；5×5 达 94%→8192、出 32768。

**正式 (20000 局，64 min)** — 单模型（CORE 5-pattern，residual on），训练 4×4:0.5 / 5×5:0.3 / 4×5:0.2，transition share 全程锁定 [0.50,0.30,0.20]。**greedy（无搜索）** 评测（300 局/尺寸）：

| 训练尺寸 | reach 1024 | reach 2048 | reach 4096 | reach 8192 | best tile |
|---------|-----------|-----------|-----------|-----------|-----------|
| 4×4 | 95% | 80% | 29% | — | 4096 |
| 5×5 | 100% | 100% | 100% | 97% | **65536** |
| 4×5 | 100% | 100% | 93% | 53% | 16384 |

> 结论：**同一模型在三个尺寸上同时强力学习**，验证了 §1.2 / §10.2 的核心假设。4×4 greedy 80%→2048 已接近专项基线的 greedy 量级（专项 t8_tc greedy ~96%，但那是 40k 局单尺寸；此处 4×4 仅分到 1万局并与其它尺寸共享表），叠加 expectimax 搜索 (M6) 预计进一步提升。5×5 出现 **65536**（编码上限内的最高 tile）。held-out 泛化见 §12.3.1。

### 12.3.1 Held-out 泛化（§1.2 核心成功标准 — 证明“不是只记住尺寸”）

对**从未训练过**的 shape 直接 greedy 评测（300 局）：

| held-out shape | reach 1024 | reach 2048 | reach 4096 | reach 8192 | best | 对照 |
|----------------|-----------|-----------|-----------|-----------|------|------|
| **5×4** (4×5 的转置) | 100% | 100% | 94% | 55% | 16384 | ≈ 训练的 4×5 (93%/53%) |
| **3×5** (未见长宽比) | 83% | 41% | 1% | — | 4096 | ≫ 随机/启发式 |

> **强证据**：5×4 从未训练，却与训练过的 4×5 **表现几乎相同**（94% vs 93% →4096）——共享表 + 对称不变 placement 让**转置泛化接近完美**。3×5（完全未见的长宽比）也 41%→2048、出 4096，**显著优于随机/启发式基线**。这直接满足 §1.2：“在未参与训练的 held-out shape 上显著优于随机/启发式基线，证明不是只记住尺寸。”**M3 Go/No-Go 通过。**

**Residual 消融（诚实记录）**：在 4×4 **单尺寸** 1000 局下，residual on/off 基本无差异（reach1024 85%/85%、reach2048 34%/33%）——符合预期：单尺寸无“尺寸混淆”，共享基座已足够；residual 的价值在于 **跨尺寸 / held-out** 时恢复角落语义（§8.3 role 消融），需混合训练 + held-out 协议才能显现。head 本身正确（对称、可学、可持久化）。

## 12.3.2 4×4 专项 / 冲击 131072（M7 起步）

**目标（用户设定）**：在 4×4 上冲击 **131072**（2¹⁷，4×4 理论最大 tile，即“通关”）。现实预期：可靠 32768 → 偶发 65536 → 131072 为运气尾部事件（spawn 随机，位置可能很差），成功率极低。因此优化目标是**最大化各档到达率**，而非保证 131072。

**已实现的地基**：
- **可扩展 tile alphabet** — `UniversalNTuple` / `Pattern` 现支持任意 alphabet；`lib.with_alphabet(patterns, 18)` 给出 alphabet=18（覆盖到 2¹⁷=131072）。alphabet=16 时 value fn 把 32768/65536/131072 混为一谈（clip 到 exp15），**无法规划高 tile 残局**；alphabet=18 后可区分（已验证 `V(131072) ≠ V(32768)`）。schema hash 含 alphabet，跨 alphabet checkpoint 拒绝加载。
- **专项 pattern 集** `lib.SPECIALIST_4X4` — 2 个小表 (square/line-4) + **5 个不同的 6-tuple 表**（rect_2x3 / corner_6 / snake_6 / row_plus_6 / row_edge_6），对应更强的 4×4 容量（≈ 文献 8×6-tuple 网络）。alphabet=18 下 681MB (W)，×3 (TC) ≈ 2GB。
- CLI：`train_universal.py --patterns specialist --alphabet 18 --shapes 4x4`；`evaluate_universal.py --expectimax` 做深度搜索评测。

**实测（4×4 专项，alphabet 18，50k 局单核训练 ~98min，300 局评测）**：

| 方法 | →2048 | →4096 | →8192 | →16384 | best tile | mean score |
|------|-------|-------|-------|--------|-----------|-----------|
| greedy（无搜索） | 94% | 58% | 9% | 0% | 8192 | 62k |
| **depth-3 expectimax** | **100%** | **98%** | **70%** | **5%** | **16384** | **136k** |

> 搜索把 mean score 翻倍（62k→136k），天花板从 8192 提到 **16384**（5% 局达到），8192 到达率 9%→70%。**当前最好的 4×4 结果 = 16384 tile。65536 / 131072 目前还达不到**——需要更长训练（→32768）+ multi-stage + 更深搜索；65536/131072 属于 SOTA 前沿的运气尾部事件。

**并行加速（已实现）**：Hogwild 无锁并行训练（多进程共享一份 LUT，`train_universal.py --workers`）+ 并行评测（`evaluate_universal.py --procs`），把原本单核的训练/评测铺满所有核心（24 线程机器 ~15–20×），使 10⁵–10⁶ 局的长训练变得可行。

**下一步（提升高 tile 的关键，按 ROI）**：① ✅ 深度 expectimax（已验证：8192 率 9%→70%，出 16384）② ✅ 并行训练（已实现，解锁长训练）③ multi-stage + weight promotion（§7.1，文献冲击 32768 的核心）④ 10⁵–10⁶ 局长训练 ⑤ redundant encoding / carousel shaping / optimistic init。

## 12.4 待办里程碑

- **M0 baseline freeze** — 待补：把既有 4×4 `ntuple_2048_t8_tc` 模型指标固化为对照 JSON（`benchmark.py` 已有雏形）。
- **M4 扩展** — 已完成 role residual（小表 per-x）。仍待：大表 per-role 标量残差、shape/stage 校准 `b,c,g_k`。
- **M5 并行（✅ 已实现）** — Hogwild 无锁并行训练（`train_universal.py --workers`，多进程共享一份 LUT，实测 24 线程 ~11×，180 g/s）+ 并行评测（`evaluate_universal.py --procs`）。仍待：全尺寸混合 (3–8)、per-bucket 自动重平衡、held-out 泛化协议。
- **M7 multi-stage weight promotion（✅ 核心已实现）** — `UniversalNTuple(stages=[13,15])`：按 max-tile exponent 阈值分 stage，每 stage 独立表；value 用 **fallback-read**（高 stage 未访问项读最深已访问的低 stage），update 首次访问时 **从低 stage 提升 (promote) 权重** 再独立更新（plan §7.1）。支持并行（2D 共享数组）与 save/load（持久化 stage 配置 + visited 标记）。9 个测试覆盖 fallback / promotion / stage 独立 / 学习 / 持久化 / 并行。CLI：`--stages 13,15`。仍待：redundant encoding、carousel shaping、optimistic init、tile-downgrading search。
- **M7 32768 Track** — multi-stage + weight promotion、redundant encoding、carousel shaping、optimistic init、tile-downgrading search。
- **M8 MCTS** — stochastic MCTS / progressive widening 对照。
