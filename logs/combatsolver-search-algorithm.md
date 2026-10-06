# CombatSolver 搜索主算法解析

> 只读分析。源码根：`reference/CombatSolver/`。引用格式为仓库内相对路径:行号。
>
> **文件命名约定（已核对）**：主求解器是 `partial class`，拆成 40+ 个文件，文件名带
> `CombatBeamSolver.` 前缀。本报告为可读性用了简写，映射如下（其余同理）：
> | 简写 | 真实路径 |
> |---|---|
> | `Phases.cs` | `src/Search/CombatBeamSolver.Phases.cs` |
> | `Expansion.cs` | `src/Search/CombatBeamSolver.Expansion.cs` |
> | `Retention.cs` | `src/Search/CombatBeamSolver.Retention.cs` |
> | `Models.cs` | `src/Search/CombatBeamSolver.Models.cs`（含嵌套类 `SearchRunContext`，:89） |
> | `Ranking.cs` | `src/Search/CombatBeamSolver.BeamRetentionPolicy.Ranking.cs` |
> | `BeamRetentionPolicy.cs` | `src/Search/CombatBeamSolver.BeamRetentionPolicy.cs` |
> | `NoveltySearch.cs` | `src/Search/CombatBeamSolver.NoveltySearch.cs` |
> | `CrossTurnPlanning.cs` | `src/Search/CombatBeamSolver.CrossTurnPlanning.cs` |
> | `CyclePlanning.cs` | `src/Search/CombatBeamSolver.CyclePlanning.cs` |
>
> 无前缀的独立文件（`BeamWidthPortfolio.cs`、`SearchBudgetLedger.cs`、`SearchRequestPipeline.cs`、
> `SolverSearchProfile.cs`、`ContextualRankingModel.cs`、`SolverSettings.cs`）同名即真实路径。
> **结论先行**：这是**确定性、单根、分层 Beam Search（束搜索）**，用「状态去重 + 支配剪枝 + 转置表」代替随机采样，
> 用「多档位宽度组合 + 显式必保通道 + Pareto」保证路线多样性，用「回合层预算切分」控制单回合爆炸。
> BFWS/novelty 只是**默认关闭的可选探索通道**，不是主算法。

---

## ① 全局架构

```
主线程: CombatRootSnapshot(冻结稳定根) → SearchPolicySnapshot(冻结政策)
        SearchRequestPipeline.Run()            (SearchRequestPipeline.cs:19)
        └─ CombatSearchCoordinator 编排多个"成员"(pass)
            每个成员 = 一次 CombatBeamSolver.Solve  ← 组合共享一份节点预算
                    │
  根构造 frontier (Phases.cs:1071-1129): 初始选牌分支 → 若干 root SearchNode
                    │
  ┌────────────── 主循环 Phases.cs:1479 ──────────────┐
  │ 外层 while: 逐回合层推进, frontier 非空且未超预算   │
  │   内层 for playDepth: 同回合内逐出牌深度 (Phases.cs:1525)
  │     Expand(node) 生成子节点 (Expansion.cs:22)      │
  │      → 转置判重 / 支配剪枝 / 记账                   │
  │     Prune(nextPlays) = 选下一层 beam (Retention.cs:174)
  │   回合层预算耗尽 → 强制 EndTurn 收尾                │
  │   回合边界: AnnotateTurnOutcomes → RankFinal → frontier
  └───────────────────────────────────────────────────┘
        → 终局 RankFinal 选完整路线 (BeamRetentionPolicy.cs:480)
        → Terminal 冻结续用戳 → 主线程部署当前回合
```

- **节点 (SearchNode)** = 不可变分叉战斗快照 `SimulationSnapshot` + 父指针 + 动作 + 累计统计（特征见 `Models.cs:343`）。节点代表一个游戏状态。
- **边 = 一个 PlanAction**：出牌 / 用药 / EndTurn / 选牌；由 `Expand` 里的卡牌、药水、回合尾作业分阶段生成（`Expansion.cs:47-158`）。
- **终止条件**：frontier 空、全局 `_run.Expanded >= MaxExpandedNodes`、软时间耗尽、内存无进展截断（`Phases.cs:1479-1485`），或用户接管/采纳路线。
- **去重/剪枝不在 Expand 里**：Expand 只产候选；提交由 coordinator 串行/有序完成，`Transpositions` 与支配关系归 `SearchRunContext` 独占（`Models.cs:112`）。

---

## ② 搜索主循环伪代码（照实际结构）

```
buildRoots() -> frontier                    # 根选牌分支 + 固定前缀对齐
while frontier 非空 and Expanded<MaxNodes and !超时 and !内存截断:
  active = frontier 中非终局;  terminal -> completed
  if active 为空: completed = RankFinal(completed); break
  # ---- 每回合层的预算切分 ----
  turnLayerBudgetMs   = max(250, 剩余时间 / 剩余保留层数)
  turnLayerNodeBudget = max(MinTurnLayerNodes, 剩余节点 / 剩余保留层数)
  for playDepth = 0; active 非空 and Expanded<MaxNodes; playDepth++:
      if 回合层时间/节点耗尽 and 已到回合边界: 强制 EndTurn; break
      nextPlays = []; ended = []
      for parent in active:                  # 串行; 或 DOP 波次并行
          for child in Expand(parent):        # 卡牌/药水/EndTurn 候选
              转置判重 + 支配剪枝
              if child 终局 or 进入新回合: ended += child
              else:                       nextPlays += child
      active = Prune(nextPlays)               # RankBest(...BeamWidth...) + 各必保通道
  ended     = AnnotateTurnOutcomes(ended)
  completed = RankFinal(completed + ended 中终局)   # 完整路线比较
  frontier  = Prune(ended)                            # 非终局者进入下一回合层
return 从 completed / fallback 经 FinalPlanOrdering 选出的最优
```

对照：主循环 `Phases.cs:1479-2040`；`Expand` `Expansion.cs:22`；`Prune` `Retention.cs:174`。

---

## ③ BFWS + novelty 详解（**可选通道，非主搜索**）

`RunNoveltyOpen`（`NoveltySearch.cs:57`）是一条独立的、有界双端 OPEN 的最优优先搜索，用新颖性分层 + 逃逸额度做**局部最优逃逸**，默认关闭（`SKILL.md:198`）。

- **novelty 定义**（`BfwsPackedNovelty.Evaluate.cs:43`）：不是"原子事实计数"，而是 **HSP/FF 风格的 width-1 / width-2 新颖度**。把状态抽成原子事实 `BfwsFact`（`:3`）后：
  - width=1：任一**变化过的**一元事实从未在本分区出现 → `novelty=1`；
  - width=2：任一变化事实**对的组合**从未出现 → `novelty=2`；
  - 都出现过 → `novelty=width+1`（熟悉）。
  事实提取见 `CaptureNoveltyFacts`（`NoveltySearch.cs:202-240`）：能量/星/手牌数/打击潜力/格挡/HP/四牌堆/牌堆顶/Power，标量做 `Log2` 分桶（`:215`）。
- **打包成整数**：事实映射成自增 `int id`；一元元组 `Record((uint)id)`；二元对打包成 `((ulong)(uint)id_i << 32) | (uint)id_j`（`BfwsPackedNovelty.cs:95`），存进按分区隔离的 `HashSet<ulong>`。
- **分区 (partition)** = `(EnemyHp/10, PotionCount)`（`NoveltySearch.cs:106,175`）：不同血量/药水档各自一套历史；父事实复用仅当分区相同且父批完整（`BfwsPackedNovelty.cs:62-67`）。
- **有界 OPEN**（`BfwsBoundedOpen.cs:5`）：`SortedSet` 按 `(novelty, -Score, sequence)` 排序，超容量即弹最差项并立即释放（无墓碑/模拟器图）。
- **逃逸额度**（`BfwsEscapeBudget.cs:5`）：`TryAdmit()` 上限 `FamiliarAllowance`。文件头注释明确：**同一最近新颖祖先的所有熟悉后代共享一份额度，兄弟不补充**——即遇到 novelty>width 的"熟悉"节点不直接丢，而让同一逃逸后代链最多走 N 步翻过局部最优。
- 触发/停止见 `NoveltySearch.cs:92-153`：`MaxActions`/`MaxTurns` 视界 + `MaxNoveltyEntries` 上限触发 `LimitReached` 停止。

---

## ④ 束宽策略与保留策略

### 束宽：多档位**串行**组合（不是并行）
`BeamWidthPortfolio.Run`（`BeamWidthPortfolio.cs:236`）在同一根上按序跑若干成员，**共享一份节点预算**：
```
ProductionMembers 默认 = [基线, 基线×2/3, 基线×3/2, 基线+次段band, 基线+基础分]
```
（`BeamWidthPortfolio.cs:159-207`；比例常量 `:128-129`）。动机（注释 `:91-99`）：**宽度不单调**——同根只改宽度结果会双向变化，于是把几次不同宽度当"抽样"，用既有比较器整条选优。
- 预算共享：第 N 个成员拿剩余预算，扣到 0 即停（`:264,307`）。
- 「次段」成员：普通席位取**分数第 W+1 至 2W 位**，用 `MoveLeadingBandToTail` 把前 W 名挪到队尾再截断（`BeamWidthPortfolio.cs:217`，调用 `BeamRetentionPolicy.cs:1676`）。
- 「基础分」成员：中途排序只用 `node.Score`（`SolverSearchProfile.cs:42`）。
- **准入门控** `RejectRefinement`（`BeamWidthPortfolioGate.cs:77`）：基线须已搜尽、未得"零战损满血胜利"、耗时 < 1/4 时间预算、剩余节点≥基线展开、按宽度×1.5 外推的时间够用，才启动精炼成员。

### 保留策略：`RankBest`（`BeamRetentionPolicy.cs:786`）
排序键 `BeamRankScore`（`Ranking.cs:848`）= `node.Score` + 当前能量 + 持续增益增量 + 铺垫/威胁/压制等加权特征；同分打破 `CompareBeamRankOrder`（`:25`）：**分数↓ → 动作数↑ → 进攻进度↓**。
进 beam 流程（`:1046-1768`）：
1. 先按 `StateKey` 去重取最优（非终局），再按 BeamRank 降序；
2. 装配**必保 (required)** 通道——防守/进攻/能力铺垫/资源/控制/集火/牌堆治理/无药/带药/Power 承诺席位等（`:1240-1575`）；
3. `effectiveLimit=BeamWidth`，截断后把 required 逐个**顶替**最弱普通席位（`:1678-1695`）；
4. 药水/无药**配额**按比例保留领袖（`:1726-1757`）；
5. `DiversifyOrdinaryBeamBoundary`（`:55`）在截线附近做同分战术多样化。
转置表用**六维标签** `TranspositionLabel(PotionCount, PotionStrategicCost, FutureSoldHp, CumHpLost, ActionCount, Score)` 支配（`Models.cs:265`）。

---

## ⑤ 预算模型（含具体档位）

**五个正交预算维度**（不止节点数）：
1. **最大展开状态数** `MaxExpandedNodes`（全局 `_run.Expanded`，`Phases.cs:1483`）；
2. **软时间预算** `SoftTimeBudgetMilliseconds`（`:1649`）；
3. **回合层预算**：节点与时间都按剩余保留层数均分（`:1500-1522`），防单回合无限引擎吃光全局预算（注释 `:1508-1515` 记录 16GB 换页事故）；
4. **单节点分支上限** `MaxCardBranchesPerNode / MaxPileChoiceBranchesPerAction / MaxHandChoiceBranchesPerAction`；
5. **内存预算**：NoGC 区域字节上限 + 逐批预约 + 检查点回收（`SolverSettings.cs:120` `NoGcRegionBudgetGigabytes=16`；`SearchBudgetLedger.cs:10-20` 按请求余量收紧成员 Profile）。

**档位**（`SolverSettings.cs:180-205`，字段序 = Beam宽 / 最大展开 / 牌分支 / 堆选择 / 手选择 / 时间ms）：

| 档位 | BeamWidth | MaxNodes | 牌分支 | 堆选择 | 手选择 | 时间 |
|---|---|---|---|---|---|---|
| Low | 45 | 60,000 | 24 | 12 | 16 | 60 s |
| **Medium（默认）** | **60** | **120,000** | **32** | **18** | **24** | **120 s** |
| High | 90 | 250,000 | 48 | 28 | 36 | 180 s |
| VeryHigh | 135 | 500,000 | 72 | 42 | 54 | 300 s |

**超预算收尾**：不抛异常。当前 `active` 全部强制 EndTurn 推进到回合末，交给 `RankFinal` 发布当前最优（`Phases.cs:1530-1542`、`:1651-1669`）；撞节点上限的成员结果**不参与成员间比较**（`BeamWidthPortfolio.cs:326-335`），只作回退。

---

## ⑥ 跨回合规划机制

- **触发**：`RequiresCrossTurnPlanning` = 已走过 `TurnsWithoutProgress>0`，或"语义状态在模型质量里不可见"（`CrossTurnPlanning.cs:137`）。
- **Probe 状态机**：`CrossTurnProbeState` 逐回合累计 `CompletedTurnTransitions / BestKnownProgressMagnitude / LastTurnImproved / LastTurnSemanticStateChanged`（`:28-36`）；每跨回合用 `MeasureCycleExitQuality` 打分判断是否仍有进展。
- **无进展剪枝** `ShouldPruneCrossTurnNoProgress`（`Expansion.cs:206`）：上限 `noProgressLimit = max(SetupValueHorizonTurns, deckCycleTurns*2)`（`:235`）；Probe 视界被 `SetupValueHorizonTurns` 硬夹（`:240-257`），超限即停。
- **"Cycle（周期）"= 同一回合内语义形状重复出现的动作循环**（如"抽牌→出牌→抽牌"的无限引擎）。`AttachCycleEvidence`（`CyclePlanning.cs:33`）用 `CycleShapeKey` 找回 ≤32 步的结构同形祖先（`:5` `MaximumDetectedCyclePeriodActions=32`），再做动作哈希识别 `CycleSearchState{ShapeKey, SequenceKey, PeriodActions, Repetitions, LastDelta}`；一致的周期允许预测重复 N 次后的收益，并用 `CycleFamilyLedger`/`CycleRegionLedger` **按回合计费**，避免一个组合爆炸吃掉后续回合额度（`Models.cs:130-145`）。
- **CycleExitProbe**：循环附近给 ≤8~32 个动作的"退出探针"，验证能否打破循环取得真实进展（`CyclePlanning.cs:6-7`）。

---

## ⑦ 与 MCTS 的本质区别（为什么确定性搜索可能更强）

1. **无随机 rollout，评价来自完备模拟 + 手工特征**：子节点是一次确定性状态转移，不需采样平均噪声；同一根重跑**逐位可复现**——而 UCT+随机 rollout 的方差本身就是"改进无效"的常见原因。
2. **剪枝是"证明支配"，不是"访问次数统计"**：`Transpositions` 六维支配 + `StateKey` 去重可一次剪掉严格更差的整片子树；MCTS 只能靠多次访问间接淘汰。
3. **显式多样性通道保证关键路线不被均值淹没**：防守/铺垫/控制/Power/无药/带药等**必保席位**直接顶替最弱束位（`BeamRetentionPolicy.cs:1678`）；MCTS 树会系统性偏向高回报分支，易丢"低即时收益、高跨回合价值"的铺垫牌。
4. **宽度组合对抗 Beam 固有偏差**：`[W, 2W/3, 3W/2, W+band, W+base]` 多次确定性搜索取最优，等于用**确定性重启**覆盖不同启发式，比在单树里调 UCT 常数更有方向性。
5. **预算是分层的**：时间/节点都按"回合层"均分（`Phases.cs:1500-1522`），天然抑制单回合组合爆炸；MCTS 若不显式分回合，长引擎牌组会把预算耗在一回合内。**代价**：一切依赖高质量的**状态等价/支配关系**与**启发式评分**——这两处做错会稳定地搜错，而 MCTS 至少有采样兜底。
