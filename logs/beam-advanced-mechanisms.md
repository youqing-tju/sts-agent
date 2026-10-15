# Beam 高级机制与预算调优报告（P0-②）

> 2026-10-15｜对应 `docs/next-chat-handoff.md` §8 的 **P0-②**
> 前置：P0-① 性能优化（`logs/beam-fingerprint-optimization.md`，提速 125×）
> **一句话结论**：CombatSolver 的高级机制**已照搬实现且有测试覆盖**，
> 但在我们当前的搜索规模下**全部无收益**；根因不是机制，而是**启发式评分质量**。

---

## 1. 结论速览

**三个独立维度同时扫描，结果一致：算力越多，结果越差。**

| 维度 | 配置 | 平均楼层 | 通关率 | 相对基线 |
|---|---|---|---|---|
| **基线（当前最优）** | W=2 / depth=6 / nodes=300 | **31.6** | **29.0%** | — |
| 更宽 | W=4 | 29.1 | 16.0% | 🔴 −13.0 pp |
| 更宽 | W=8 | 29.1 | 17.0% | 🔴 −12.0 pp |
| 更深 | depth=12 | 31.2 | 24.0% | −5.0 pp |
| 更深 + 更大预算 | depth=30 / nodes=3000 | 28.5 | 17.0% | 🔴 −12.0 pp |
| 更大预算（同深） | depth=6 / nodes=1500 | 31.6 | 29.0% | **完全无变化** |
| 泳道关闭（W=2） | lane=off | 31.2 | 22.0% | +7.0 pp（**不显著**）|

（n=100 局 / 同 checkpoint `v12_beam_500.pt` / 同 100 种子 / 6 workers / greedy）

**两比例 z 检验**：
- W=2 vs W=4：z ≈ 2.20，**p ≈ 0.028 → 显著**
- W=2 vs 泳道关闭：z ≈ 1.14，**p ≈ 0.26 → 不显著**（不能声称泳道有效）
- depth=6 vs depth=30：z ≈ 2.02，**p ≈ 0.043 → 显著**

---

## 2. P0-②：照搬了什么、放弃了什么

用户 2026-10-15 决策：**「照搬就行，不必保留」**我们的简化实现。

### ✅ 已实现（3 条）

| 机制 | CombatSolver 对应 | 我们的实现 | 测试 |
|---|---|---|---|
| **① 泳道必保席** | `FindBestLane` / `LaneValue`（`...BeamRetentionPolicy.Ranking.cs:590,657`） | `TRAITS` / `lane_values` / `_find_best_lane` / `BeamSolver._select_beam` | 4 项 |
| **② 同分打破序** | `CompareBeamRankOrder`（`Ranking.cs:25`）：分数↓ → 动作数↑ → 进攻进度↓ | `_beam_sort_key` | 1 项 |
| **③ 束宽 Portfolio** | `BeamWidthPortfolio.cs:159-207`：`[W, 2W/3, 3W/2, W+band, W+base]`，**共享一份节点预算** | `portfolio_widths()` / `solve(width_portfolio=True)` | 3 项 |

**泳道的 STS1 映射**（只保留有真实机制对应的）：

| 泳道 | 取值 | 来自 |
|---|---|---|
| `scaling` | `strength*3 + dexterity*2` | Scaling（能力铺垫 = 跨回合价值） |
| `block` | 玩家格挡 | HpInvestment（血量投资） |
| `resource` | `energy*16 + 手牌数 + 抽牌堆数` | Resource + EndTurnDeckCompression |
| `offense` | `−敌人总血量` | 无直接对应（我们补的） |
| `control` | `−敌人意图总伤害` | Control 的**可观测代理**（绑定读不到 Weak/Vulnerable） |

### ❌ 主动放弃（2 条，**不硬凑**）

- **六维支配剪枝**：`TranspositionLabel = (PotionCount, PotionStrategicCost, FutureSoldHp, CumHpLost, ActionCount, Score)`。
  评估后降级 —— 我们的 `fingerprint` 比 CombatSolver 的 `StateKey` **更细**（含手牌顺序、抽牌堆顺序），
  同指纹即真同态，**转置表内再分标签收益有限**。
- **路线级反事实药水策略 / BFWS novelty**：前者依赖药水动作接入（我们尚未接入），后者 CombatSolver **默认关闭**。

### ⚠️ 与 CombatSolver 的**规模差距**（关键背景）

| | CombatSolver（默认档 Medium） | 我们（当前最优） | 倍数 |
|---|---|---|---|
| Beam 宽度 | 60 | 2 | **30×** |
| 最大展开节点 | 120,000 | 300 | **400×** |
| 软时间预算 | 120 s | 每步毫秒级 | ~10⁴× |

> CombatSolver 定位是**给真人辅助的离线求解器**（单场战斗可跑 120 秒）；
> 我们要的是**每局批量训练**（单局 5.6 秒含宏观决策）。
> **为「大预算」设计的多席位/Portfolio 机制，在小预算下无从发挥。**

---

## 3. 三个反直觉现象

### 现象 1：更宽的 beam 更差（「宽度不单调」）

W=2 → 31.6 层 / 29%；W=4 → 29.1 层 / 16%；W=8 → 29.1 层 / 17%。

**这与 CombatSolver 源码注释完全一致**（`BeamWidthPortfolio.cs:91-99`）：
> 「宽度不单调 —— 同根只改宽度结果会**双向变化**，所以把几次不同宽度当抽样。」

→ 这是 **Portfolio 机制存在的理由**，也说明我们最初的 W=2 恰好落在好点上。

### 现象 2：`max_nodes` 从未被用满

`nodes=300` 与 `nodes=1500` 的结果**逐项完全相同**（31.6 层 / 29.00%）。

探针实测（初始战斗状态）：

| depth | 实际节点数 | **规划步数** | ms |
|---|---|---|---|
| **6（我们的配置）** | **23** | **2** | 2.1 |
| 12 | 49 | 12 | 4.0 |
| 20 | 81 | 14 | 5.4 |
| 30 | 128 | 14 | 9.0 |
| 50 | 213 | 14 | 14.1 |

**`max_depth=6` 时搜索只规划 2 步** —— `max_nodes=300` 从头到尾没被触及。
（超过 depth≥12 后 `plan_len` 稳定在 14，说明一场战斗的完整路线约 14 个动作。）

> ⚠️ 这解释了为什么调 `max_nodes` 完全无效 —— **不变量搞错了**。

### 现象 3：前瞻更深反而更差

depth=6（2 步前瞻）29% > depth=12 24% > depth=30 17%。
按 z 检验，depth=6 vs depth=30 **显著**（p≈0.043）。

---

## 4. 根因分析：启发式评分质量

三个现象指向**同一个根因**：

> **`evaluate()` 的评分不准时，更多的算力只会更充分地放大它的误差。**

- 更宽的 beam → 让更多"被误判为高分"的坏节点进入前沿
- 更深的前瞻 → 让误差沿路径累积，越走越偏
- 更大的预算 → 无意义（因为搜索根本没跑满）

**佐证**：
1. 9 个启发式权重是**照抄 CombatSolver 的量级**（HP=30000 等），**从未对 STS1 调过**
   —— 这在 `next-chat-handoff.md` §8 已列为 P1，现在有了实证支撑。
2. 评分里 `−敌人意图伤害 × 6000` 会奖励"消灭敌人"的路线；前瞻越深，
   越容易选中"看起来威胁清零"但实际代价高的分支。
3. **泳道机制失效也有了同一解释**：泳道要保的是"低即时收益、高跨回合价值的**铺垫**节点"，
   而 `max_depth=6` 只前瞻 2 步 —— **第二个回合根本看不到**，"跨回合价值"无从体现。

**因果链**：

```
启发式评分不准
   ↓
多算不如少算（更宽/更深/更大预算全部变差）
   ↓
泳道（为跨回合价值设计）在 2 步前瞻下空转
   ↓
唯一有效的改进方向 = 提升评分质量本身
```

---

## 5. 对本项目的意义

### ✅ 可直接采用的结论

**当前最优战斗配置 = `W=2 / depth=6 / max_nodes=300`**（所有实测中最好，且最快）。
- 训练/评估的默认 CLI 值已经是这个配置 → **无需改动**
- `route_lanes` 默认开，但实测无差异（不显著）→ **保持不变亦可**，它不增加可测成本

### 🔴 对 handoff §3 方法论设想的影响

§3 设想：「训练时随机化 Beam 宽度 → 战斗执行变弱 → RL 靠更强抓牌补偿」。

现在有了新事实：**Beam 的前瞻本来就只有 2 步**（depth=6）。
- W ∈ {1,2,4,8} 扫描的**战斗强度区间确实存在**（通关率 16%~29%）
- 但 `W=1` 尚未实测，且「更弱」的上限有限

→ §3 的对照实验仍可做，但要注意：**"变弱"的杠杆主要在宽度，不在深度**
（深度只会让结果更差且更慢）。

### ⏭ 建议的下一步（按优先级）

| 优先级 | 事项 | 理由 |
|---|---|---|
| **P0** | **调启发式权重**（原 P1 升级） | 三个反常现象的共同根因；这是当前唯一有实证支撑的改进方向 |
| **P0** | 给 `evaluate()` 加**可调权重**（从常量改为参数/配置） | 否则无法实验；当前 9 个权重是模块级常量 |
| P1 | 逆用现象：把 `depth` 从 6 **降到 3~4** 实测 | 若"少算更好"继续成立，可再省一半时间 |
| P1 | Beam 宽度 ↔ RL 抓牌质量对照（§3） | 杠杆用宽度而非深度 |
| P2 | 长期：把手工评分换成**学习出的价值网络** | 这正是 RL 主线的自然延伸 |

> ❌ **不再建议**：加大 Beam 预算（宽度/深度/节点）、继续搬 CombatSolver 剩余机制
> —— 在评分修好之前，这些都无收益。

---

## 6. 复现

```bash
cd "C:/Youqing-Vault/Embodied-AI/07-projects/sts-agent"
export KMP_DUPLICATE_LIB_OK=TRUE

CK=checkpoints/v12_beam_500.pt
COMMON="--checkpoint $CK --seeds 100 --workers 6 --mcts-sim 20000 --combat-solver beam"

# 基线（最优）
.venv/Scripts/python.exe -m evaluation.evaluate $COMMON --beam-width 2 --beam-depth 6 --beam-max-nodes 300
# 更宽
.venv/Scripts/python.exe -m evaluation.evaluate $COMMON --beam-width 4 --beam-depth 6 --beam-max-nodes 300
.venv/Scripts/python.exe -m evaluation.evaluate $COMMON --beam-width 8 --beam-depth 6 --beam-max-nodes 300
# 更深
.venv/Scripts/python.exe -m evaluation.evaluate $COMMON --beam-width 2 --beam-depth 12 --beam-max-nodes 300
.venv/Scripts/python.exe -m evaluation.evaluate $COMMON --beam-width 2 --beam-depth 30 --beam-max-nodes 3000
# 泳道对照
.venv/Scripts/python.exe -m evaluation.evaluate $COMMON --beam-width 4 --beam-depth 6 --beam-max-nodes 300 --no-beam-lanes

# 机制占位率探针 / 节点-深度关系
.venv/Scripts/python.exe scripts/probes/_tmp_lane_seats_probe.py 4 2
```

> ⚠️ 统计口径：**n=100 / 100 种子 / greedy / 6 workers**。
> 100 局下通关率的标准误约 **4~5 pp** → **差异 <10 pp 一律不能声称有效**。

## 7. 代码与测试

- `combat_beam.py`：`TRAITS` / `lane_values` / `_find_best_lane` / `_dominant_by_tiebreak` /
  `_beam_sort_key` / `BeamSolver._select_beam` / `portfolio_widths` / `width_portfolio`
- 透传链路：`env/lightspeed_env.py` → `training/ppo.py`（含 worker 子进程）→
  `training/run_training.py` / `evaluation/evaluate.py`
- CLI：`--no-beam-lanes` / `--beam-lane-budget` / `--beam-portfolio`
- 测试：`tests/test_combat_beam.py` **28 项**（含泳道必保、Portfolio 预算共享、同分打破）
