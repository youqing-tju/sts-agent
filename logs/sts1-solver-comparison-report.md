# STS1 战斗求解器对比纠错报告

> 2026-10-16｜来源：联网检索 + 与本地 vendored 上游源码逐项对比
> **一句话结论**：STS1 生态里**没有**独立的第三方战斗求解器，权威实现就是
> `sts_lightspeed` **自带**的 `BattleScumSearcher2`（MCTS）—— 也就是我们 `combat_brain.py` 在用的那个。
> 真正的**纠错发现**是：**本项目并存两套互不相干的评估函数**，而 10 月转 Beam 时
> 我们用**从未在 STS1 验证过的 STS2 静态启发式**替换掉了**已经验证过的终局评估** ——
> 这正是「多算反而更差」的根源。

---

## 一、STS1 生态盘点（联网检索结果）

| 项目 | 是什么 | 与我们相关度 |
|---|---|---|
| **`gamerpuppy/sts_lightspeed`** | C++17 模拟器 + **自带 tree search**（`BattleScumSearcher2` MCTS） | ⭐⭐⭐ **我们用的就是它** |
| **`daniel-ziegler/sts_lightspeed`** | 上游 fork：把"RNG 重放搜索"改成 **expectimax MCTS**，配自己的 RL | ⭐⭐⭐ **最强对标**（见 §1.1）|
| `cvennevik/slay-the-spire-solver` | C# 规则引擎 + AI（全部动作/状态枚举） | ⭐ 但 **2022 年后停更**，且 README 自述"探索所有可能性不可行" |
| `greamy/SlayTheSpireSolve` | Python 战斗模拟器 + PPO/LSTM | ⭐ 但**只实现了观者**，且是"学打牌"路线 |
| `alexdriedger/SlayTheSpireFightPredictor` | ML 预测战损 / 评估卡牌 | ✗ 不是求解器 |
| `jjoba/sts_cockpit` | 预期胜率 CLI | ✗ 明确声明"战斗中决策不支持" |
| `torch1230/CombatSolver` | **STS2** 战斗路线求解器（我们在 `reference/` 参考的） | ⭐⭐ 但它是**给真人辅助的离线求解器**，预算比我们大 2–3 个数量级 |

### 1.1 最强对标：`daniel-ziegler` fork 的公开成绩

| 指标 | 数值 |
|---|---|
| **A0 击杀心脏** | **~83%**（600 个匹配种子，**1000 次搜索迭代/决策**）|
| A20 击杀心脏 | 18.6% ± 2.4%（n=1000，10000 迭代/决策）|
| 训练 | 512 局/迭代 × 30 worker，A0–A20 统一训练，57 层完整心脏流程 |

⚠️ **与我们 v14 的 53.5% 不是严格可比**：
- 口径可能不同（他们的 "kill the heart" 要求走完第四幕拿钥匙流程；我们的
  `_WIN_OUTCOME = "PLAYER_VICTORY"`，而 `GameOutcome` 只有三态、**不区分是否打心脏**）
- 搜索预算量级不同（他们 1000 次 MCTS 迭代，每次 rollout 到终局；我们 300 节点、前瞻 2 步）
→ **只能作为"上界参考"，不能直接宣称我们落后 X pp**。

---

## 二、🔴 最重要的发现：本项目并存两套互不相干的评估函数

| | **MCTS 路径**（`combat_brain.py` → `CombatBrain`） | **Beam 路径**（`beam_brain.py` → `BeamSolver`） |
|---|---|---|
| 搜索 | MCTS（UCB1）+ **随机 rollout 到终局** | 分层 Beam，**无 rollout** |
| 评估对象 | **终局**（`evaluateEndState`，C++） | **静态中间状态**（`evaluate()`，Python） |
| 评估来源 | `sts_lightspeed` 上游 + **本项目已改进** | **照抄 STS2 `CombatSolver` 的 `SolverWeights` 量级** |
| 药水动作 | ✅ `enumeratePotionActions`（上游有） | ❌ **无** |
| 专家排序 | ✅ `Expert::getPlayOrdering()`（上游有） | ❌ 无 |
| 手牌动作去重 | ✅ 相邻重复牌合并 | ❌ 全枚举 |

**→ 10 月转向 Beam 时，我们抛弃了本项目**已针对 STS1 验证过**的终局评估，
换成了**从未在 STS1 上验证过**的 STS2 静态启发式。**

这与 `logs/beam-advanced-mechanisms.md` 的实验结果**完全自洽**：
一个**没有终局信息**的中间状态评分，越算（更宽/更深）只会越充分地放大它的误差
—— 所以「更宽 / 更深 / 更大预算**全部变差**」。

---

## 三、与上游原版的逐项源码对比（`evaluateEndState`）

上游原版取自 `raw.githubusercontent.com/gamerpuppy/sts_lightspeed/master`，
本地版本见 `third_party/sts_lightspeed/src/sim/search/BattleScumSearcher2.cpp:407`。

### 3.1 胜利分支

| | 上游原版 | 本项目（已改） |
|---|---|---|
| 公式 | `100*(35 + curHp + potionScore − turn*0.01)` | 同左 **− 0.05 × energyWasted** |
| 评价 | — | ✅ **合理改进**：惩罚浪费能量，偏好低损耗胜利 |

**`potionScore = potionCount × 4`** —— ⚠️ **我们两套评估里都没有"药水保留"这一项**
（`combat_beam.evaluate()` 完全没有药水概念）。上游把它计入胜利收益 ✓ **我们缺失**。

### 3.2 失败分支（差距最大）

| 项 | 上游原版 | 本项目 | 倍数 |
|---|---|---|---|
| 血量进度 | `(1 − 非小怪血量比) × 10` | `× 300` | **30×** |
| 存活怪物 | `alive × (−1)` | `alive × (−20)` | **20×** |
| 击杀进度 | **无** | `(总数 − 存活) × 150` | 新增 |
| 基线 | 无（结果约 −20 ~ +40） | **−100** | 新增 |
| 能量惩罚 | `energyWasted × −0.2` | 同 | — |
| 抽牌奖励 | `cardsDrawn × 0.03` | 同 | — |

**判读**：上游失败分支的梯度**极弱**（血量进度只占 10 分，被胜利分 ~3600 完全淹没），
在"随机 rollout 多半死"的 Act3 Boss 战里**没有梯度可学**。
本项目的改动（放大梯度 + 加负基线）**方向正确** —— 本地 commit `ac4cc0a` 的说明也是这么写的
（"修 5/9 Act3-boss 全血死亡"）。**这一项我们是对的，不是错。**

> ⚠️ 但要注意：这条路子的**受益者是 MCTS**（它用 rollout 到终局拿到 `evaluateEndState`）。
> **Beam 完全没用上这个函数** —— 改得再好也与 Beam 无关。

---

## 四、纠错清单

| # | 问题 | 严重度 | 依据 |
|---|---|---|---|
| 1 | **Beam 用静态中间状态启发式，没有终局信息** | 🔴 高 | §二；解释"多算更差" |
| 2 | **Beam 未接入药水动作** | 🔴 高 | 上游 `enumeratePotionActions`（`:261`）|
| 3 | **两套评估函数并存、互不复用** | 🟠 中高 | §二；维护性 + 知识未沉淀 |
| 4 | **Beam 无专家动作排序**（`getPlayOrdering`）| 🟡 中 | 上游 `ExpertKnowledge.cpp:9`（100+ 卡优先级表）|
| 5 | Beam 未做手牌动作去重（上游合并相邻重复牌）| 🟡 低 | `fingerprint` 已按状态去重，收益有限 |

**不是错误的**（我们的改动是改进）：`evaluateEndState` 失败分支的梯度强化、胜利分支的能量惩罚。

---

## 五、建议（按性价比排序）

| 优先级 | 动作 | 理由 |
|---|---|---|
| **P0** | **给 Beam 加"终局估值"**：对 beam 叶节点做 rollout（或直接用 MCTS 的 `BattleScumSearcher2` 做短 rollout）到终局，再用 `evaluateEndState` 打分 | 直接消除"静态启发式"这一根本缺陷；`evaluateEndState` 已经过本项目验证 |
| **P0** | 若上条成本过高 → **至少把 `evaluate()` 的中间状态评分与 `evaluateEndState` 的语义对齐**（补药水项、统一血量/回合语义）| 让两套评估不再"两套世界观" |
| P1 | Beam 接入药水动作 | 上游有、我们没有；药水在 Boss 战是关键资源 |
| P1 | 移植 `Expert::getPlayOrdering` 的动作枚举顺序 | 低成本；上游靠它加速收敛 |
| P2 | 手牌动作去重 | 收益小 |

> ⚠️ **就当前结论而言，最该做的是 P0 第一条** —— 它把 MCTS 的「终局真实收益」
> 与 Beam 的「分层剪枝效率」结合起来，正好补上我们这两条路径各自缺的那一半。

---

## 六、检索与对比的可复现路径

```bash
# 1. 上游原版（对比基准）
#    https://raw.githubusercontent.com/gamerpuppy/sts_lightspeed/master/src/sim/search/BattleScumSearcher2.cpp
# 2. 本地（已被本项目改过，见 git log）
git log --oneline -- third_party/sts_lightspeed/src/sim/search/BattleScumSearcher2.cpp
#    ac4cc0a 失败分支评估加稠密梯度
#    b485a91 ISMCTS 改用"根边统计"聚合 + 评估补能量惩罚
#    330b70e vendor sts_lightspeed with MSVC patches
# 3. 参考实现关键位置
#    evaluateEndState        BattleScumSearcher2.cpp:407
#    enumerateActionsForNode BattleScumSearcher2.cpp:175
#    enumeratePotionActions  BattleScumSearcher2.cpp:261
#    Expert::getPlayOrdering ExpertKnowledge.cpp:9
#    explorationParameter    BattleScumSearcher2.h:38   (3*sqrt(2))
```
