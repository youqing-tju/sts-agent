# ISMCTS v2 实现计划

> 2026-08-31 · 依据 docs/plans/2026-08-31-ismcts-v2-design.md · 两个并行任务 + 一次集成验收

## Task V2-1（C++ 侧）— RNG 重随机化绑定

**Files:**
- Modify: `third_party/sts_lightspeed/include/combat/BattleContext.h`
- Modify: `third_party/sts_lightspeed/src/combat/BattleContext.cpp`
- Modify: `third_party/sts_lightspeed/bindings/slaythespire.cpp`

**实现：**
```cpp
// BattleContext.h 类内声明
void randomizeRngForDeterminization(std::uint64_t seed);

// BattleContext.cpp 实现
void BattleContext::randomizeRngForDeterminization(std::uint64_t seed) {
    // 用 seed 派生独立种子，重设战斗中会影响未来的 RNG
    aiRng          = Random(seed ^ 0x9E3779B97F4A7C15ULL);
    cardRandomRng  = Random(seed ^ 0xC2B2AE3D27D4EB4FULL);
    miscRng        = Random(seed ^ 0x165667B19E3779F9ULL);
    shuffleRng     = Random(seed ^ 0x27D4EB2F165667C5ULL);
    // 用新 shuffleRng 对抽牌堆做 Fisher-Yates 洗牌
    // cards.drawPile 是固定数组 + 有效长度；参考 CardManager/CardPile 的现有洗牌代码风格
}
```
- 注意先查 CardPile/cards.drawPile 的实际类型（fixed_list），按其接口洗牌
- 绑定：`.def("randomize_rng", &BattleContext::randomizeRngForDeterminization)`
- 编译：`third_party/sts_lightspeed/build_bindings.bat`，产物拷贝到项目根
- TDD：写 `tests/test_randomize_rng.py`——同一 bc 拷贝两次、分别 randomize_rng(1)/(2)，搜索或抽牌堆顺序应不同；同 seed 应相同

## Task V2-2（Python 侧）— ISMCTS 投票循环

**Files:**
- Modify: `combat_brain.py`
- Create: `tests/test_ismcts.py`

**实现（combat_brain.py）：**
```python
class CombatBrain:
    def __init__(self, simulation_count=2000, determinizations=10):
        self.sim_count = simulation_count        # 每个平行未来的模拟次数
        self.determinizations = determinizations  # 平行未来数量 K
        ...

    def decide(self, state, gc):
        # 1. bc0 = BattleContext + init_from_json（一次重建）
        # 2. for k in range(K):
        #      bc_k = sts.BattleContext(bc0)          # 拷贝构造（绑定已有）
        #      bc_k.randomize_rng(base_seed + k)      # 重洗抽牌堆+重设 RNG
        #      action = BattleSearcher(bc_k, sim_count).best_action
        #      key = 投票键（action_type, 手牌卡 id 归一化名）
        #      votes[key] += 1; actions[key] = action
        # 3. winner = max(votes)；平票取先出现者
        # 4. 走现有 _to_comm（身份匹配翻译）
        # 降级：无 randomize_rng → v1 单次搜索；异常 → None（规则兜底）
```
- 固定种子源：base_seed 由战斗状态派生（如 seed + floor*1000 + turn），保证同一状态重复决策可复现（防抖动/死循环）
- 日志：打印得票分布（`votes: strike_r=8, defend_r=2`）

**测试（4 个用例）：**
1. 同一状态两次 decide → 结果一致（可复现）
2. K=10 返回合法命令（PLAY/END/POTION）
3. randomize_rng 缺失时降级 v1 不崩（monkeypatch 模拟）
4. 全套回归

## Task V2-3（集成验收，我自己做）

- 两个任务完成后：全套 pytest + 真实战斗 JSON 投票分布人工检查
- 确认耗时 ~2s/步
