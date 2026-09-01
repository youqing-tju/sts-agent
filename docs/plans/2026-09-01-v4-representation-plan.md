# V4 实现计划（维度布局定稿）

> 2026-09-01 · 设计见 2026-09-01-v4-representation-design.md

## 候选维度：368 → 408（定死，所有文件一致）

| 段 | 区间 | 维度 | 内容 |
|---|---|---|---|
| kind | [0,10) | 10 | 不变 |
| card | [10,110) | 100 | type(5)+rarity(5)+**cost(1,真实值)**+upgraded(1)+name-hash(88)，布局不变，仅 cost 接线 |
| room | [110,120) | 10 | 不变 |
| numerics | [120,128) | 8 | 不变 |
| label-hash | [128,368) | 240 | 不变 |
| **map-lookahead** | [368,392) | 24 | 新增：向下 6 层，每层 4 维 one-hot（ELITE/REST/SHOP/OTHER） |
| **card-effects** | [392,408) | 16 | 新增：卡牌效果数值 |

## map-lookahead 规则
- 候选节点 = 下一层 x 坐标（idx1）；从该节点向下 6 层，每层取"离当前 x 最近的子节点"（贪婪单路径近似）
- 每层房间类型 → 4 维 one-hot：ELITE=0, REST=1, SHOP=2, OTHER=3（TREASURE/EVENT/MONSTER/BOSS/NONE 归 OTHER）
- y 越界或 act 结束则置 0
- 用 gc 的 SpireMap + get_room_type(x, y)，纯 Python

## card-effects 16 维（按名查 models/card_data.py 字典）
base_damage, base_block, base_cost, is_power, is_aoe, num_hits, draw_count,
exhausts, self_damage, retain, scales_strength, applies_vulnerable,
applies_weak, gains_block_scaling, is_status, is_curse（归一化到合理范围）
- 非卡牌候选全 0；REWARDS 屏用 gc.get_card_reward()[idx2] 的 id.name 查表
- 字典覆盖铁甲战士（IRONCLAD）常用卡牌（~75 张），数值为**基础值**（未升级），不确定的标 None 置 0

## 需要同步改的文件
- models/candidate_encoder.py：编码逻辑 + 新维度
- models/card_data.py：新增（铁甲卡牌效果字典）
- env/lightspeed_env.py：_DESC_VEC_DIM 368→408 及任何 368 常量
- main.py：SimBrain 的 368 常量
- models/scorer.py：cand_dim 默认 368→408
- training/run_training.py：PPO/REINFORCE 的 Scorer(..., cand_dim=...) 用新维度
- 测试：test_candidate_encoder.py 更新维度断言 + 新增前瞻/字典用例

## 验收
- 单测：维度=408；Bash 字典 damage=8/vulnerable=1；地图前瞻对直通 Boss 路径给出非零计数；cost 非零
- 全套回归绿
