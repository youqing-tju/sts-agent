# 药水使用调查报告：MCTS 是否会在战斗中使用药水？

**结论先行：会。MCTS（`BattleScumSearcher2`）动作空间包含 `POTION`，且实证 18/18 局都会喝药。原假设（"无用药动作 → 药水永不被消耗 → 评估函数`药水×4`是假奖励"）不成立，战斗层无需为"用药水"而修。**

## 1. C++ 动作空间证据（有 POTION 动作）

- `include/sim/search/Action.h:24-30`：`ActionType` 枚举明确含 `POTION`（与 `CARD`/`END_TURN` 并列）。
- `src/sim/search/BattleScumSearcher2.cpp:178-182`：`enumerateActionsForNode` 在 `PLAYER_NORMAL` 状态下调用 `enumerateCardActions` **+ `enumeratePotionActions`** + `END_TURN`。
- `BattleScumSearcher2.cpp:261-299`：`enumeratePotionActions` 遍历每个非空药水槽，生成 `Action(ActionType::POTION, pIdx[, tIdx])`：无目标药水（直接喝）、需目标药水（按可击怪物枚举）、妖精药水（丢弃）。即**药水动作确实进入搜索树**。
- `src/sim/search/ScumSearchAgent2.cpp:82-99`：`playoutBattle` 每步实例化 `BattleScumSearcher2` 并 `search`，再用 `stepThroughSolution` → `takeAction` 执行最优动作序列（含 POTION）。
- `src/sim/search/Action.cpp:438-444`：POTION 动作执行 `bc.drinkPotion(getSourceIdx(), getTargetIdx())`（target>5 则 `discardPotion`）。
- `src/combat/BattleContext.cpp:2658-2659`：喝药会 `--potionCount`；`:894-895` `exitBattle` 把 `potionCount` 同步回 `GameContext`。

搜索关键词：`POTION` / `potion` / `enumeratePotionActions` / `drinkPotion` / `usePotion` —— 均**找到**药水相关逻辑（非"完全无"）。

## 2. Python 行为验证（决定性证据）

方法：用 `v10_seg3.pt` + greedy 宏观策略 + `LightspeedEnv(mcts_simulations=8000)`，覆写 `_advance`/`step`，在每次 `playout`（=一整场战斗）与每次宏观决策前后读 `gc.potion_count`。战斗内消耗使 `delta<0`，非战斗屏获得使 `delta>0`。

| 指标（18 局汇总） | 值 |
|---|---|
| 总获得药水数 | 179 |
| 总消耗药水数 | 177 |
| 有消耗药水的局数 | 18 / 18 |
| 终局仍持药水局数 | 1（且为败局，仅剩 1 瓶）|
| 峰值持有（全部 ≤3） | 见 raw |

每局"获得≈消耗"，且 18/18 局战斗内出现 `delta<0` —— **绝非"只增不减"**。MCTS 不仅会用，而且用得近乎净尽。
（原始逐局数据：`logs/_potion_probe_raw.json`）

## 3. 交叉验证：是否暴露"用药水"API

- 宏层 `GameAction`（GameContext 级别）**无**药水动作：药水是纯战斗概念，只在 `BattleContext` 内由 MCTS 的 `POTION` 动作消耗。
- Python 绑定仅暴露只读 `gc.potion_count` / `potion_capacity` / `potion_slots`，**不暴露**任何"使用/丢弃药水"接口。
- 含义：若日后要修，杠杆只能在 C++（动作枚举/评估函数），Python 侧无入口。

## 4. 结论与影响面分析

- **MCTS 确实会在战斗中使用药水**，原"能力盲区"假设被证伪。
- 评估函数 `evaluateEndState`（`BattleScumSearcher2.cpp:408,414,424`）中 `potionScore = bc.potionCount * 4`（胜）/ `/ 2`（负）**不是假奖励**——药水被真实消耗，该项能被拿到。
- **残余轻微偏置（非 Bug，无需修）**：该 term 给"终局保留药水"加分，理论上当输赢结果不变时 MCTS 可能偏向留药。但实证消耗率 177/179（≈99%），偏置在实践中可忽略；18 局仅 1 局终局剩 1 瓶。
- 对 `combat-hypothesis-report` 的"死时带药"：本样本败局带药率仅 1/11，远低于其报的 4/13。若其 Act3 样本仍偏高，更可能源于"药水获取过晚/败局已定"，而非"MCTS 不会喝药"。
- **建议**：战斗层**无需**为"用药水"修复。若要进一步压榨残局药水利用率，可选微调 `potionScore` 权重（如胜局改为 0、负局保留小权重），属锦上添花，非必需。
