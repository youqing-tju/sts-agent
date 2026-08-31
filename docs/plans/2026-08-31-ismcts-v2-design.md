# ISMCTS 战斗大脑 v2 设计文档

> 2026-08-31 · 状态：待批准 · 架构决策：信息集 MCTS（Information-Set Monte Carlo Tree Search）

## 1. 问题陈述

v1 战斗大脑把真实游戏状态灌入模拟器后做单次 MCTS 搜索。但 Communication Mod 不发送随机数（RNG）计数器，模拟器重建状态时使用固定 RNG → 搜索器在一个**锁死抽牌顺序的虚构未来**里规划（"下回合抽重锤正好斩杀"），真实游戏的牌序不同 → 斩杀线落空、决策显得"算错伤害"。

目标：决策不依赖任何单一虚构未来，改为在**多个采样未来上取期望最优**。

## 2. 方案：确定性化 + 投票（ISMCTS 的经典实现）

每个战斗决策点：

```
bc0 = BattleContext.init_from_json(真实游戏 JSON)     # 一次重建
for k in 1..K:                                        # K=10 个平行未来
    bc_k = copy(bc0)
    bc_k.randomize_rng(seed_k)                        # 重洗抽牌堆 + 重设所有 RNG
    action_k = BattleSearcher(bc_k, sims=2000).best_action
    votes[card_id(action_k)] += 1                     # 按卡牌身份投票（不按手牌下标）
返回得票最高的动作，按身份匹配翻译成真实游戏命令
```

关键性质：
- 手牌是明牌，K 份拷贝间手牌顺序一致；未知的只有抽牌堆顺序和未来怪物意图 → 只随机化这两者
- 期望最优（期望 = 各未来平均），**不是**最坏情况最优（minimax 过度保守，已否决）
- 耗时与现在持平：10 × 2000 = 20000 总模拟，约 2 秒/步

## 3. 改动清单

### C++ 侧（third_party/sts_lightspeed，一个方法 + 一行绑定）

`include/combat/BattleContext.h` + `src/combat/BattleContext.cpp`：
```cpp
void randomizeRngForDeterminization(std::uint64_t seed);
// 实现：
//   用 seed 派生 4 个独立种子，重设 aiRng / cardRandomRng / miscRng / shuffleRng
//   （monsterHpRng / potionRng 战斗中不参与，不动）
//   用新 shuffleRng 对 cards.drawPile 做 Fisher-Yates 洗牌
```

`bindings/slaythespire.cpp`：
```cpp
.def("randomize_rng", &BattleContext::randomizeRngForDeterminization)
```

### Python 侧（combat_brain.py）

`CombatBrain.decide` 改为 ISMCTS 主循环：
- `determinizations=10`、`sims_per=2000` 两个参数可调
- 投票 key = （动作类型， 模拟器手牌卡 id 归一化名）——手牌顺序在拷贝间稳定，按 id 投票是为防万一
- 平票处理：取得票最高者中第一个出现的
- 降级链：randomize_rng 不可用 → 单次搜索（v1 行为）；任何异常 → 规则出牌

### 测试（TDD）

`tests/test_ismcts.py`：
1. 同一战斗状态、固定 K 个种子 → 投票结果确定性（可复现）
2. K=10 时返回动作合法（CARD/END_TURN/POTION 之一）
3. randomize_rng 后两次搜索的 RNG 序列确实不同（通过抽牌堆顺序或搜索耗时差异间接验证）
4. 降级路径：monkeypatch 使 randomize_rng 抛错 → 回退 v1 单搜索不崩

## 4. 明确不做（YAGNI）

- 不做最坏情况（minimax）目标——过度保守
- 不做怪物意图重采样之外的信息隐藏建模（真实游戏意图可见，comm mod 已发 move_id）
- 不改宏观决策（RL 侧不受影响）
- 不做并行化（K 份串行搜索已够快；GIL 下 C++ 搜索本就单线程）

## 5. 验收标准

1. tests/ 全套绿（含 4 个新测试）
2. 用测试战斗 JSON 跑 K=10 投票，输出各候选得票分布（人工检查合理性）
3. 真实游戏里战斗连贯性不回退（不出牌卡死不得复现——身份匹配和熔断器保留）
