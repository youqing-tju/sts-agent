# CombatSolver 剪枝 / 转置 / 估值 / 药水策略解析

> 只读分析对象：`reference/CombatSolver/`（STS2 战斗路线求解器，C#/.NET9）；行号均为相对 `CombatSolver/` 的路径；结论：**确定性**状态空间搜索，转置去重做到 RNG 级别。

---

## ① 状态去重 / 转置机制

**状态指纹 = 字段级 128 位增量哈希**（非整对象序列化）：
- 构造器 `StateFingerprintBuilder`：两条独立混合链 + `RotateLeft`（`src/Search/StateFingerprint.cs:7-76`）。
- `BuildStateKey(...)`（`src/Search/CombatBeamSolver.StateEvaluation.cs:1626-1701`）逐字段 Add：
  turn / 玩家 `CurrentHp·MaxHp·Block·Energy·Phase·Stars` / `shufflesCrossed` / Osty(Hp,MaxHp,可被击中) /
  每个敌人 `(Monster.Id, CombatId, alive, Hp, MaxHp, Block)` /
  **有序牌堆** Hand→Draw→Discard→Exhaust（`AppendPile` :1874）/ Orbs /
  **9 路 RNG 状态**（shuffle、卡生成、药水生成、选牌、能量消耗、目标、球、MonsterAI、Niche）/
  已处理敌人死亡集合 / 历史计数器 / `simulatedCombat.AppendFingerprint`。
  末尾 `ApplyStateKeySalt` 只异或常量、不改相等关系（:1707）。
  → 因为搜索是确定性采样，两节点 StateKey 相同 ⇒ 真正同一后继状态。

**两张转置表**（`src/Search/CombatBeamSolver.Models.cs:112-113`）：
`Transpositions`（行动准入）与 `ExpandedTranspositions`（展开准入），键=StateKey，值=`TranspositionFrontier`。
- `TranspositionLabel = (PotionCount, PotionStrategicCost, FutureSoldHp, CumulativePlayerHpLost, ActionCount, Score)`，
  `Dominates` 判据：**前 5 维全 ≤ 且 Score ≥**（`src/Search/CombatBeamSolver.Transpositions.cs:5-56`）。
- 同一 StateKey 不丢弃，而是**维护一个非支配标签集合**（`TryAccept` 合并：被支配则拒，支配他者则替换，否则并存）。
- 消费点 `TryAcceptTransposition`（`...Expansion.Candidates.cs:875-929`）与 `TryMarkExpandedState`（:931-978）：
  新状态 → 建 frontier；`TryAccept` 成功 → 保留；被支配 → `TranspositionBranchesPruned++`。
  两表合计有 `TranspositionEntryLimit` 条目上限（:865-868）。
- 另有**每步按 StateKey 去重**：`RankBest` 里 `bestByState` 每个 key 只留最优节点
  （`Score` 大者胜，平局 `ActionCount` 小者，`src/Search/CombatBeamSolver.BeamRetentionPolicy.cs:811-820, 841-843`）。

## ② 剪枝规则清单

1. **转置支配剪枝**（①，`Candidates.cs:875-978`）。
2. **beam 截断**：排序后仅保留前 `effectiveLimit`，再用必留节点替换队尾（`BeamRetentionPolicy.cs:1678-1690`）。
3. **多目标 Pareto 支配** `MultiObjectiveDominates`（`BeamRetentionPolicy.cs:756-839`）：
   约 30 个维度（HP、MaxHp、累计失血、LongTermResource、愤怒副本、存活敌数、敌 HP、各种 buff/控制/资源/回合数…），
   **仅当 `EnemyCombatDistributionKey`、`EnemyControlDistributionKey`、`UnorderedPileKey` 三者相等时才可比**；
   用在路由上下文 Pareto（:933-963）与回合边界（:1603-1653，仅保留前 3 名）。
4. **incumbent 界剪枝**：`StrategicHpLowerBound > incumbent.StrategicHpDeficit`（含 turn 平局与 equal-HP 选项，`Retention.cs:419-488, 567-576`）。
5. **智能药水可行性上界**：算不出 Smart 资格的支路直接剪（`src/Search/CombatBeamSolver.SmartPotionBound.cs:51-94`）。
6. **回合层预算停止**：节点/时间配额用尽即把前沿推成 EndTurn 节点（`src/Search/CombatBeamSolver.Phases.cs:1566-1598`）。

## ③ beam 保留的排序键（重点）

两级：**先按 StateKey 去重取最优**，再排序。
- 排序入口 `SortByBeamRank`→`CompareBeamRankOrder`（`BeamRetentionPolicy.cs:756-773, 25-40`）：
  **① BeamRankScore 降序 → ② ActionCount 升序 → ③ OffensiveProgressValue 降序**。
- `BeamRankScore`（`...BeamRetentionPolicy.Ranking.cs:848-930`）=
  `node.Score`（基础分，见 `StateEvaluation.cs:163-197`：`projectedHp*Hp − 累计失血*Hp + MaxHp 增量*Hp − 死亡罚分 + 长期资源/成长折算`，`Hp=30000`，`DeathPenalty=−1e12`）
  `+ min(Energy,cap)*60000`
  `+ min(持久 buff 增量,cap)*50000(普通)/150000(Boss)`
  `+ min(LatentSetup,cap)*12000`
  `+ (Boss) FutureResourceValue*10000`
  `+ min(ReplayPotential,cap)*10000`
  `+ RetainedAttackGrowth*20000 + DelayedDamage*10000 + SandpitRemaining*30000`
  `+ EnemyStrengthSuppression*horizon*30000 + EnemyWeakTurns*expectedHpSaved*30000`
  （权重全在 `src/Search/SolverWeights.cs:8-83`；`BaseScoreOnly` 档案下退化为纯 Score，:851-852）。
- **排序后还有「分道必留」层**（不靠单一分数）：对 8 个 `SearchRouteTraits`
  （Scaling/Resource/Control/RevivalWindow/ReactiveDamage/EndTurnDeckCompression/LongTermResource/HpInvestment）
  按每个药水档各选 1 个泳道冠军（`Ranking.cs:590-672`），加路由上下文 Pareto、家族/选项代表、歧义组合，
  最后才用分数序回填空位（`BeamRetentionPolicy.cs:1580-1690`）。
- 最终 `SortRetained` 用各泳道 `RetentionRank` 的最小值排序、平局 `Score` 降序（`Retention.cs:756-775`）。

## ④ 估值模型设计与 Ironclad 举例

接口 `IPowerCardValuationModel`（`src/Search/PowerCardValuation/IPowerCardValuationModel.cs:11-27`）：
```csharp
Type CardType; PowerCardPool Pool;
PowerCommitmentFamily CommitmentFamily;            // 机制族（全角色共用标签集）
PowerRouteAdmissionPolicy AdmissionPolicy;         // 优先级/最小投影/准入条件
PowerCardValuationRequirements Requirements;       // 需要哪些上下文事实（[Flags] 需求位）
PowerCardValuationResult Evaluate(CardModel card, in PowerCardValuationContext context);
```
**按角色分别实现**：`Cards/{Ironclad,Silent,Defect,Regent,Necrobinder,Colorless}/`，但族标签与准入/优先级是公共层。
输出 `PowerCardValuationReward = {Damage, Prevention, Resource, CardAccess, Scaling, Control}`（`PowerCardValuationContracts.cs:216-245`），
输入 `PowerCardValuationContext` 是约 50 个派生事实的只读结构（:161-214）。
Ironclad 例子（`Cards/Ironclad/IroncladStrengthPowerCardValuationModels.cs`）：
- `Inflame`（:11-16）→ `StatGrowth(card, ctx, 2, 3, perTurn:false)`：一次性 +2/+3 力量。
- `DemonForm`（:17-22）→ `StatGrowth(..., 3, 4, perTurn:true)`：每回合 +3/+4 力量。
- `Rupture`（:23-41）→ `Scaling = 升级值 × 总攻击次数`，仅在 `SelfDamageTriggers>0` 时给分。
- `Cruelty`（:42-61）→ `Damage = 易伤目标攻击伤害 × 25%/50%`，仅 `VulnerableTargetAttackDamage>0` 时给分。
**关键事实**：该接口 docstring 自述（:6-9）这是**合同/文档层，不是生产搜索消费者**；
Reward/Penalty/Timing **未接入**候选准入、beam 保路与终局排序。生产侧走 `PowerCommitmentDescriptor`
（`PowerCommitmentRegistration` + 各卡池开局投影/触发证据），数值 Reward 是"草案"。

## ⑤ 三态药水策略（智能判定逻辑）

- **每瓶独立三态**：`SolverPotionDirective { Smart, Force, Disabled }`（`src/Search/PotionStrategySnapshot.cs:3-8`）；
  全局策略 `SolverPotionPolicy { Disabled, RequireAtLeastOne, Smart }`。
- **新药水默认智能**：`Resolve(slot,id)` 未配置槽位时，只要默认策略不是 Disabled 即返回 `Smart`（`PotionStrategySnapshot.cs:66-71`）。
- **智能判定 = 路线级反事实**（`src/Search/PotionUsePolicy.cs:93-117`），不是回合级阈值：
  `Smart` 允许用瓶当且仅当 `explicitPotionCount==0`，或 `potionRouteWon && !potionFreeWon`（用药把败局翻盘），
  或 `HpSaved >= SmartRequiredHpSaved(strategicCost)`。
  - `HpSaved = max(0, 无药基线 deficit − 用药路线 deficit)`（:57-58）。
  - `SmartRequiredHpSaved` → `RawHpRequiredForPersistentValue`（`ActEndingBossPolicy.cs:99-111`）：
    普通 = 战略成本；`ActClearHeal` 关 = ×5；`RunEnding` 关 ≈ 无穷（终局药水几乎无价值）。
  - 战略成本按药水分档（`PotionValuationRegistry.cs:58-70`）：Token/可再生石 = 0；High = 2×基线；Elevated = 1.5×；其余 = 基线 9（`SolverWeights.cs:99`）。
- 决策在**终局策略门**执行：用药候选要与"无药基线"比较（`CombatBeamSolver.FinalPlanOrdering.cs:217-261`）。
- 搜索中仅用上界提前剪掉注定不合格的支路（`SmartPotionBound.cs:51-94`）。

## ⑥ incumbent 与提前终止

- **incumbent = 目前最优完整胜利**：`(StrategicHpDeficit, CombatEndedTurn, 用药成本)`，
  `TryTightenPrimarySearchIncumbent` 只在"完整胜利 + 满足硬规则 + 零 death save + 药水档吻合"时才收紧
  （`Retention.cs:578-641`）；作为界在 `ApplyPrimaryIncumbentBoundCore` 剪枝（:419-488）；
  同冻结 root 的多次搜索可经 `PrimaryIncumbentTable` 共享（`src/Search/PrimaryIncumbentTable.cs:7-40`）。
- **提前终止 = 达到可接受血量损失**：`MeetsHpTarget`（`Phases.cs:339-346`）要求
  完整胜利 + growth/relic 目标满足 + 偷窃资源恢复 + 零 death save + 用药数 ≤ `earlyStopPotionUses`
  + 累计战斗失血 ≤ `_acceptableBattleHpLoss`。一旦满足即 `acceptableBattleHpLossReached=true`（:348-362），
  立即排空前沿、结束搜索（:2011-2017, 2162-2168）。
- 预算型收尾：节点上限/软时间/内存压力触发时，把当前前沿推成 EndTurn 节点并发布 incumbent（:1526, 1585-1598）。

## ⑦ 对我们的启发（MCTS `BattleScumSearcher2`）

1. **战斗状态做确定性指纹去重**：把 MCTS 节点按有序牌堆 + RNG 状态 + 敌人 HP/block 生成字段级哈希，
   同一指纹的访问统计合并（而非各自 rollout），能大幅省预算并稳定 UCT 估计。
2. **单 UCB 之外加"分道必留席位 + 支配集合"**：进攻/防守/引擎成长/资源各保一个最优代表，
   再对同构状态维护非支配集合，避免随机 rollout 噪声把整条好路线整体挤掉。
3. **用药改判为路线级反事实 + 战略成本分档**：不要固定阈值，而是和"不用药基线"比出 `HpSaved`，
   按药水价值分档设"必须省下多少 HP 才划算"的门槛（终局药水近乎无价值）。
