# 真机 V9 跳牌与卡牌偏好探针报告

> 探针对象：`checkpoints/ppo_v9_10000.pt`（obs=477 / cand=347）。仅写 `_tmp_*.py`，未改生产代码。
> 原始数据：`logs/rl_skip_pref_probe.json`。

## 一、实验 A：训练侧 SKIP 候选真实存在性

**方法**：模拟器 20 局（`LightspeedEnv`，每局随机策略推进），遇 `REWARDS` 选牌屏 dump 全部候选 `kind/label/bits`。

| 指标 | 值 |
|---|---|
| REWARDS 选牌屏次数 | 275（20 局累计）|
| 含 SKIP 候选的屏 | 275 / 275（**100%**）|
| 卡牌候选总数 | 385 |
| SKIP 候选总数 | 275 |
| SKIP bits | `0x30000000` → decode：`reward_type=6, idx1/2/3=0` |

**SKIP 向量 vs 卡牌向量差异**（`encode_candidate`）：
- SKIP：`kind` one-hot[3]=1、`numeric[reward_type]=1`（其余 345 维全 0）。
- 卡牌（以 Bash）：`kind`+`card_attr`(3)+`card_onehot`(1)+`semantic`(4) 共 9 维非零。

**结论**：训练侧 REWARDS 屏**每屏都生成 SKIP**，V9 训练时见过。真机"不会跳牌"的根因在 `live_candidates.build_card_reward_candidates` 只构造 3 张卡、漏掉 SKIP——属真机路径缺陷，非训练缺失。

## 二、实验 B：V9 对 SKIP 的打分（真机 60 帧）

**方法**：从 `logs/realtime/0911_*_states.jsonl` 取 `game_state.screen_type=="CARD_REWARD"` 帧（共 150），取 60 帧；每帧构造 3 张卡候选 + 1 个 SKIP 候选（`kind="SKIP", bits=0x30000000`），`init_from_json` 灌入模拟器取 obs，V9 打分。

| 指标 | 值 |
|---|---|
| SKIP 得第一比例 | **0%**（0/60）|
| SKIP 排名分布(0=最高) | 第4(末):47 / 第3:11 / 第2:2 / 第1:0 |
| SKIP 平均排名 | 3.75 / 4 |
| 弱奖励子集（3 张全 common）23 帧，SKIP 平均排名 | **4.0（全末）** |

**结论**：即便给真机 V9 补上 SKIP 候选，它**也不会跳牌**。SKIP 描述向量近乎全零（仅 kind+reward_type 标志），V9 给它的分数恒定偏低——78% 垫底、0% 登顶，连"全 common 该跳"的帧也排最末。结构性不跳牌是 V9 自身打分偏好，并非仅缺候选。

## 三、实验 C：V9 卡牌偏好画像（验证用户观察）

**方法**：31 张候选卡分 6 组；两种 obs 基线（开局=模拟器首决策点 / 中局=真机 floor≥10 非战斗屏）。V9 打分。

**分组平均分（越高越爱）**：

| 分组 | 开局 | 中局 |
|---|---|---|
| 力量 | -12.97 | -14.69 |
| 大攻击 | -12.40 | -14.16 |
| debuff | -13.25 | -15.05 |
| 功能 | -19.02 | -20.67 |
| 基准 | -17.96 | -19.77 |
| **防战** | **-22.01** | **-23.73** |

**Top5**：Reaper(-4.2) / Shockwave(-8.3) / Demon Form(-8.5) / Spot Weakness(-11.8) / Bludgeon(-10.1)。
**Bottom5**：Body Slam(-25.4) / Entrench(-25.2) / Dark Embrace(-23.9) / Juggernaut(-22.8) / Second Wind(-23.5)。

**结论**：**验证用户观察成立**——V9 系统性偏爱 力量/大费高伤/debuff，防战分组分数最低（约比力量低 9 分）。能力牌分化：力量类能力（Demon Form/Inflame）受宠，防战与消耗类能力（Dark Embrace/Feel No Pain/Entrench/Body Slam）垫底，"防战完全不会"确凿，"能力牌完全不会"仅部分成立（力量类能力会拿）。

## 四、字段缺失/不可区分清单（"学不会防战"表征层解释）

防战核心机制在 38 维语义段**确有布尔位**（`block_retain`/`block_multiplier`/`block_on_gain_trigger`/`block_to_damage`/`on_exhaust_trigger` 均在 `EFFECT_KEYS`），故非"零表征"，真缺口是：

1. **无"每回合/持续"维度**：Demon Form（每回合+2 力量）、Metallicize（每回合+3 格挡）、Juggernaut、Feel No Pain 的持续/触发收益只编码成单一布尔位，无幅度也无"复利"标记。`EFFECT_KEYS` 无 recurring/per_turn/duration → Demon Form 与 Inflame（同 `strength_gain=2`）语义段同值，仅费用不同。
2. **防御布尔字段无量级**：`block_retain`(Barricade)、`block_multiplier`(Entrench)、`block_on_gain_trigger`、`block_to_damage`(Body Slam)、`on_exhaust_trigger` 全是 0/1；"格挡翻倍"与"格挡+1"在向量里都记 1.0，无法表达倍率/数值。
3. **DEAD_FIELDS 剔除了 `block_efficiency`、`retain`**：44→38 维时删掉的两个恰与防战相关（格挡效率/保留），表征进一步损失。
4. **无"收益时延/复利"维度**：Power 牌（Barricade/Demon Form/Juggernaut）的延迟复利收益与即时牌仅差 `is_power` 一个标志位，V9 看不到"此卡收益随回合累积"。
5. **流派交叉项依赖 `deck_arch`**：开局牌组无格挡流派时，防战卡交叉项全 0 → 早期拿防战核心卡无任何协同信号；`deck_arch` 是 5 维粗粒度强度，无法表达"已有格挡→Body Slam 更强"的状态耦合。
6. **卡牌 one-hot 与实时状态解耦**：Body Slam「格挡转伤害」威力取决于当前格挡，但候选向量不含当前格挡（属 obs 维非候选维），自身向量只是标志位，V9 无法把它与高格挡状态关联。

## 五、一句话结论

真机 V9 不跳牌有两层原因——`live_candidates` 漏构造 SKIP 是入口缺陷，而即便补上 SKIP，V9 也因 SKIP 向量近乎全零而恒定垫底（0% 登顶）；卡牌偏好上 V9 确证偏爱力量/大伤/debuff、系统性低估防战（分组低约 9 分），根因在候选编码缺"持续/量级/时延/状态耦合"维度，使防战复利收益不可区分。
