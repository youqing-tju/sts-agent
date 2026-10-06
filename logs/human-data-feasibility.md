# 人类 .run → 训练样本：可行性验证报告

> 验证者：team lead（接手 dataset-survey 因 429 中断的任务）
> 结论：**可行**。端到端已产出 1 个形状正确的样本。

## 1. 数据

- 位置：`data/external/Slay-the-Spire-data/runs/<数据集>/<角色>/*.run`
  注意：**按角色分目录**，应按目录名筛角色（`.run` 内的键是 `character_chosen`，不是 `character`）
- 本子集 **106 个 Ironclad**（`ascension_level` 全 20）；仓库总计 749 局
- 关键字段（Run History Plus 版）：

| 字段 | 内容 | 对应决策屏 |
|---|---|---|
| `card_choices[]{picked, not_picked, floor}` | **选项 + 人类选择** | CARD_REWARD |
| `event_choices[]{event_name, player_choice, floor, damage_taken, ...}` | 事件名 + 选择 | EVENT |
| `items_purchased[]` + `item_purchase_floors[]` | 买了什么 + 在哪层 | SHOP |
| `campfire_choices[]{floor, key: REST/SMITH, data}` | 休息/锻造 + 升级目标 | REST |
| `boss_relics[]{picked, not_picked}` | Boss 遗物选择 | BOSS_RELIC |
| `path_taken[]`（如 `['M','M','$','E','?',...]`） | 逐层房间序列 | MAP |
| `current_hp_per_floor[]` / `max_hp_per_floor[]` / `gold_per_floor[]` | **逐层状态轨迹** | 状态重建 |
| `master_deck[]` / `relics_obtained[]` / `neow_bonus` | 最终牌组 / 遗物获得时点 / 起手奖励 | 状态重建 |

## 2. 五项验证（全部通过）

| # | 验证 | 结果 |
|---|---|---|
| 1 | `gc` 字段可写性 | `cur_hp / max_hp / gold / floor_num` **可直接赋值**（55/80/200/12 写读一致）|
| 2 | 模板帧灌入 | 取真机帧 → 改 hp/gold/floor → `gc.init_from_json` **成功**，obs 412 维 |
| 3 | obs 生成 | `enhance_observation` → **477 维 = OBS_DIM** ✓ |
| 4 | 候选编码 | 4 个候选（3 卡 + SKIP）× **369 维 = DESC_VEC_DIM** ✓ |
| 5 | MAP 枚举 | `sts.get_all_actions(gc)` → 3 个动作 `bits=0x0/0x4/0x5`，与真机 `choice_list=['x=0','x=4','x=5']` **一一对应** ✓ |

**产出的样本形状**：`(obs 477, 候选 4×369, 标签=人类 picked)`

## 3. 关键实现要点（本次实测踩到的坑）

1. **`init_from_json` 对 null 极敏感**：`{}` / `{"seed":1}` 均报
   `type must be number, but is null` → **不能从零组装 JSON**。
   ✅ 解法：**拿真机帧当模板 + 覆盖字段**（验证 2 已证明可行）。
2. **CARD_REWARD 屏枚举 = 0**：灌入帧不含奖励内容 → 模拟器认为无可选。
   ✅ 解法：用 `models/live_candidates.build_card_reward_candidates(gs)` **手工构造**
   （真机路径已在用，含 V10 的 SKIP 候选）。
3. **候选卡必须来自 .run 自己的 `picked` + `not_picked`**，不能沿用模板帧的卡
   —— 本次验证实际踩到：模板帧是 Wild Strike/Heavy Blade/Cleave，而该 .run 在
   floor=1 的真实选项是 Evolve / Dual Wield / Havoc。**借模板卡会导致候选与标签错配**。
4. **MAP 的 map 必须换成人类种子的**：用 `sts.SpireMap`（绑定可用）按 seed 生成；
   模板帧里的 map 属于别的种子。
5. 状态"可写 + 模板可覆盖"是最大便利：**不需要重放战斗**（人类数据本就不含逐出牌），
   直接用 `.run` 的逐层轨迹覆盖 hp/gold/floor。

## 4. 剩余风险（未验证项，实现时需处理）

| 风险 | 说明 | 缓解 |
|---|---|---|
| **牌组回滚精度** | `master_deck` 是**最终**牌组；需按 `card_choices` 逐层反推"决策点当时"的牌组 | 校验：回滚到底应 == 铁甲基础牌组(5×Strike/4×Defend/1×Bash) + `neow_bonus` 奖励 |
| **人类水平参差** | 数据来自主播，含送分局/低水平局 | 按 `victory` / `score` / 主播名过滤（如只取胜局或指定高手） |
| 少数遗物不记增删卡 | `Astrolabe` / `Tiny House` 等会改牌组但不留记录 | 跳过含这类遗物的局，或用 `master_deck` 一致性校验剔除 |
| 升级状态 | `campfire_choices` 的 SMITH 会改卡名（`Strike_R+`） | 回滚时同步撤销升级 |

## 5. 推荐实现（三步）

1. **`run_parser`**：`.run` → 决策点列表
   每条 = {screen, floor, 选项列表, 人类选择, 当层 hp/max_hp/gold, 重建牌组/遗物}
2. **`state_builder`**：真机模板帧 + 覆盖字段 → `gc` → `(obs 477, 候选 N×369)`
   - MAP 用 `get_all_actions` 枚举；CARD_REWARD 用 `live_candidates` 手工构造
3. **冻结成数据集**：`npz`（obs / 变长候选 / label idx / 元信息），供 BC 预热训练

**预估规模**：106 局 Ironclad × 约 30~50 个宏观决策点 ≈ **3k~5k 条样本**（单子集）；
若取仓库全部 749 局中的全部 Ironclad，样本量可再提升。
