# 可达性审计报告 · 铁甲战士（Ironclad）/ 进阶 0（A0）/ 单角色爬塔

> 审计对象：`models/relic_kb.py`、`models/potion_kb.py`、`models/event_kb.py`、`models/card_index.py`
> 权威依据（C++ 池表）：`third_party/sts_lightspeed/include/constants/{RelicPools,Potions,Events,CardPools}.h` 与 `src/game/GameContext.cpp`、`src/game/Game.cpp`、`src/game/Map.cpp`
> 只读审计，未修改任何 `models/` / `env/` 文件。
> 注：`RelicPools.h` 第 13–19 行的 `namespace CommonRelicPools`（被 `//` 整段注释）**不生效**；实际生效的是第 27–58 行按角色分拆的四套 `namespace`（Ironclad / Silent / Defect / Watcher）。

---

## 0. 总览结论

| 模块 | KB 总量 | 铁甲可达 | **铁甲永远拿不到（差集）** | 其中建议删除 |
|---|---|---|---|---|
| 遗物 relic_kb (`ALL_RELIC_IDS`) | 180 | 150（130 池 + 起始 1 + 事件 18 + 兜底边界 1） | **49** | **30 确定 + 1 边界** |
| 药水 potion_kb (`_POTION_RAW`) | 33 | 33（= 铁甲药水池全集） | **0** | **0** |
| 事件 event_kb (`EVENT_KB`) | 43 | 43（全部已实现且可达） | **0** | **0** |
| 卡牌 card_index (`CARD_INDEX`) | 110 | 110（75 红 + 35 无色） | **0** | **0** |

**一句话**：只有「遗物」知识库含有铁甲 A0 永远拿不到的内容，共 **31 个枚举名**（30 个确定不可达 + 1 个边界），其余三类知识库（药水 / 事件 / 卡牌）经核实均无越界内容，无需删除。

---

## 1. 遗物（最重要）

### 1.1 铁甲可达遗物全集（计算口径）

铁甲可达 = `RelicPools.h` 中 **`namespace Ironclad`** 五套池（common 33 / uncommon 30 / rare 28 / boss 22 / shop 17 = **130**）∪ **铁甲起始遗物 `BURNING_BLOOD`**（第 21 行 `starterRelics[0]`，第 24 行按 `CharacterClass` 索引）∪ **事件专属遗物（18，见 §1.3）** ∪ 兜底边界 `CIRCLET`（见 §1.4）。

- Ironclad 五套池：§ `RelicPools.h:27-33`（命名空间内 `commonRelicPool`/`uncommonRelicPool`/`rareRelicPool`/`bossRelicPool`/`shopRelicPool`）
- 起始遗物：`RelicPools.h:21` `starterRelics[] = {BURNING_BLOOD, RING_OF_THE_SNAKE, CRACKED_CORE, PURE_WATER}`，铁甲取索引 0 = `BURNING_BLOOD`（`RelicPools.h:23-25`）

### 1.2 差集（49 个）—— 三分类

对 `ALL_RELIC_IDS`（180）做差集得到 49 个，逐条归属如下。

**(a) 异角色专属池（26 个）—— 铁甲池里完全没有，且只出现在其他角色池里 → 确定不可达**

| 归属角色 | 枚举名（26） |
|---|---|
| Silent（10） | `HOVERING_KITE`、`PAPER_KRANE`、`SNECKO_SKULL`、`THE_SPECIMEN`、`TINGSHA`、`TOUGH_BANDAGES`、`TWISTED_FUNNEL`、`WRIST_BLADE`、`RING_OF_THE_SERPENT`、`NINJA_SCROLL` |
| Defect（8） | `EMOTION_CHIP`、`FROZEN_CORE`、`INSERTER`、`DATA_DISK`、`GOLD_PLATED_CABLES`、`NUCLEAR_BATTERY`、`RUNIC_CAPACITOR`、`SYMBIOTIC_VIRUS` |
| Watcher（8） | `CLOAK_CLASP`、`DUALITY`、`GOLDEN_EYE`、`MELANGE`、`VIOLET_LOTUS`、`DAMARU`、`HOLY_WATER`、`TEARDROP_LOCKET` |

> 证据：每个名仅出现在对应角色的 `namespace`（如 `RelicPools.h:36` Silent 池、`RelicPools.h:44` Defect 池、`RelicPools.h:52` Watcher 池），不出现在 `namespace Ironclad`（`RelicPools.h:27-33`）。

**(b) 异角色起始遗物（3 个）—— 在注释掉的 `CommonRelicPools` 命名空间（`RelicPools.h:13-19`，不生效）里，实为其他角色起始 → 确定不可达**

| 枚举名 | 实际归属 |
|---|---|
| `RING_OF_THE_SNAKE` | Silent 起始（`starterRelics[1]`） |
| `CRACKED_CORE` | Defect 起始（`starterRelics[2]`） |
| `PURE_WATER` | Watcher 起始（`starterRelics[3]`） |

> 证据：`RelicPools.h:21` 的 `starterRelics[]` 与 `RelicPools.h:13-19`（注释版 `CommonRelicPools` 恰好列出这三个起始）。

**(c) 事件专属 / 通关解锁（18 个）—— 已核实由事件发放，铁甲 A0 可达 → 保留，不删**

| 遗物 | 发放事件 | 证据（文件:行） |
|---|---|---|
| `BLOODY_IDOL` | Forgotten Altar（拒绝黄金偶像后替换） | `GameContext.cpp:2951` `relics.replaceRelic(GOLDEN_IDOL, BLOODY_IDOL)` |
| `MARK_OF_THE_BLOOM` | Mind Bloom | `GameContext.cpp:3209` `obtainRelic(MARK_OF_THE_BLOOM)` |
| `NECRONOMICON` / `ENCHIRIDION` / `NILRYS_CODEX` | Cursed Tome | `GameContext.cpp:2741` 选书 → `reward.addRelic(book)` |
| `ODD_MUSHROOM` | Hypnotizing Colored Mushrooms | `GameContext.cpp:3237` `reward.addRelic(ODD_MUSHROOM)` |
| `WARPED_TONGS` | Ominous Forge（选项 1） | `GameContext.cpp:2618` `obtainRelic(WARPED_TONGS)` |
| `FACE_OF_CLERIC` / `CULTIST_HEADPIECE` / `GREMLIN_VISAGE` / `SSSERPENT_HEAD`（见下注） | Face Trader（`getRandomFace`） | `GameContext.cpp:2896-2916`（调用 `getRandomFace`）；脸遗物清单 `Game.cpp:329-358` |
| `MUTAGENIC_STRENGTH` | Augmenter（选项 2） | `GameContext.cpp:2873` `obtainRelic(MUTAGENIC_STRENGTH)` |
| `NEOWS_LAMENT` | Neow | `GameContext.cpp:2488` `obtainRelic(NEOWS_LAMENT)` |
| `RED_MASK` | Tomb of Lord Red Mask | `GameContext.cpp:3580` `obtainRelic(RED_MASK)` |
| `GOLDEN_IDOL` | Golden Idol | `GameContext.cpp:3009` `obtainRelic(GOLDEN_IDOL)` |
| `NLOTHS_GIFT` / `NLOTHS_HUNGRY_FACE` | N'loth（及 Face Trader 的脸清单） | `GameContext.cpp:3297/3303` `obtainRelic(NLOTHS_GIFT)`；`Game.cpp:344` 脸清单含 `NLOTHS_HUNGRY_FACE` |
| `SPIRIT_POOP` | Bonfire Spirits（献祭诅咒牌） | `GameContext.cpp:3827` `obtainRelic(SPIRIT_POOP)` |

> 注：`SSSERPENT_HEAD` 虽未在 `GameContext.cpp` 直接出现 `obtainRelic(SSSERPENT_HEAD)`，但它在 `getRandomFace` 的脸清单中（`Game.cpp:348-349`），而 Face Trader 事件会调用 `obtainRelic(getRandomFace(...))`（`GameContext.cpp:2906`），故铁甲可达。
> 以上 18 个事件均位于 `Events.h` 的 Act1/Act2/Act3 池（`Events.h:199/204/209`），对铁甲 A0 全部可达（见 §3 最大 Act 论证）。

**(d) 兜底遗物（2 个）—— 仅当某池为空时由 `returnRandomRelic` 退回，铁甲池恒非空 → 不可达**

| 遗物 | 兜底逻辑 | 证据 | 结论 |
|---|---|---|---|
| `RED_CIRCLET` | `returnRandomRelic(BOSS)` 在 **boss 池为空**时退回 | `GameContext.cpp:1673` | 铁甲 boss 池 22 条永非空 → **确定不可达，建议删除** |
| `CIRCLET` | `returnRandomRelic(RARE)` 在 **rare 池为空**时退回；或集齐全 5 张脸遗物后 `getRandomFace` 兜底（`Game.cpp:353`） | `GameContext.cpp:1657` | 常规不可达；但集齐全脸遗物后理论上可由 Face Trader 发放 → **边界，建议保留或删除均可** |

### 1.3 铁甲可达区间（供对照，不逐条列 150 个）

- 公共 110（五套池 130）+ 起始 `BURNING_BLOOD` + 事件 18（`§1.2(c)`）+ 边界 `CIRCLET` = 150。
- 区间表示（以 `RelicPools.h:28-32` 五套池枚举名全集为准）：`namespace Ironclad` 的全部 130 个 + `BURNING_BLOOD` + `§1.2(c)` 的 18 个事件遗物。

---

## 2. 药水

`Potions.h` 的药水池为 **`potionPool[4][33]`（`Potions.h:247-252`）**，按 `CharacterClass` 分四套行：**行 0 = Ironclad**、行 1 = Silent、行 2 = Defect、行 3 = Watcher，每行 33 个。铁甲可达 = 行 0 的 33 个。`getPotionForClass(cc, idx)`（`Potions.h:255-257`）按角色取池。

逐行比对：行 1/2/3 各自比行 0 多出 3 个**角色专属药水**（共 9 个），其余 30 个为四行共有：

| 专属药水（9，铁甲不可达） | 归属行 |
|---|---|
| `POISON_POTION`、`CUNNING_POTION`、`GHOST_IN_A_JAR` | Silent |
| `FOCUS_POTION`、`POTION_OF_CAPACITY`、`ESSENCE_OF_DARKNESS` | Defect |
| `BOTTLED_MIRACLE`、`STANCE_POTION`、`AMBROSIA` | Watcher |

**核对 `models/potion_kb.py` 的 `_POTION_RAW`（33 条，`potion_kb.py:70`）：** 经归一化比对，33 条与铁甲行 0 的 33 个药水 **1:1 对应**（含显示名差异如 `Elixir`↔`ELIXIR_POTION`、`Fairy in a Bottle`↔`FAIRY_POTION`，已正确映射）。**9 个角色专属药水均未出现在 `_POTION_RAW` 中。**

> 结论：药水知识库**无铁甲不可达内容**，0 条需删除。

---

## 3. 事件

### 3.1 模拟器实际实现的事件全集

`Events.h:13-71` 枚举事件；事件逻辑全在 `GameContext.cpp` 的两个大 `switch`（约 `GameContext.cpp:1017` 与 `:2606` 起）。逐条核对 `EVENT_KB` 的 43 个顶层键（见下），**每个都能在 `GameContext.cpp` 找到对应 `case Event::...` 处理分支**，即全部已实现。

### 3.2 三类“越界”核查

| 核查项 | 结果 | 证据 |
|---|---|---|
| EVENT_KB 里**模拟器根本未实现的事件** | **0 个** | 43 个键逐一映射到 `GameContext.cpp` 的 `case Event::`（如 `NEOW:2606`、`GOLDEN_IDOL:3006`、`FACE_TRADER:2896`、`MINDBLOOM:3177`、`BONFIRE_SPIRITS:1021`、`WE_MEET_AGAIN:3646` 等） |
| 只在 **Act 4（心脏）** 出现的事件 | **0 个** | `Events.h` 仅定义 Act1/2/3 事件池（`Events.h:199/204/209`）+ oneTime + shrines，**没有任何 Act4 事件**；Act4 在 `GameContext.cpp:1335-1336` 直接判定 `PLAYER_VICTORY`，无事件房 |
| 需要 **A15+** 才出现的事件 | **0 个** | `Events.h:195-196` 的 `oneTimeEventsAsc0` 比 `oneTimeEventsAsc15` **多** `NOTE_FOR_YOURSELF`（即 A15 反而**移除**该事件，而非需要 A15 才出现）；A0 下 43 个事件全部可达 |

### 3.3 最大 Act 确认

- Act3 Boss 后调用 `transitionToAct(act+1)`（`GameContext.cpp:1331`）进入 Act4；
- `act == 4` 分支（`GameContext.cpp:1335-1336`）直接将 `outcome = PLAYER_VICTORY`——**Act4 仅为通关结算，无地图、无事件**。
- 故本模拟器对铁甲 A0 的“事件可达最大 Act = 3”，所有 `EVENT_KB` 事件均在 Act1–3 内，全部可达。

> 结论：事件知识库**无铁甲不可达（未实现/Act4/A15）内容**，0 条需删除。

### 3.4 附加发现（非越界，仅供参考）

`EVENT_KB` 仅覆盖 43 个事件，但模拟器**另有 9 个已实现事件未收录**于 `EVENT_KB`：`OLD_BEGGAR`、`CURSED_TOME`、`DESIGNER_IN_SPIRE`、`AUGMENTER`、`KNOWING_SKULL`、`MASKED_BANDITS`、`MYSTERIOUS_SPHERE`、`NLOTH`、`THE_MOAI_HEAD`（均在 `GameContext.cpp` 有 `case` 处理）。这是 KB **覆盖度缺口**（不影响可达性，故不列入删除清单）。

---

## 4. 卡牌

`card_index.py` 的 `CARD_INDEX` = **110**（docstring 声明：`models/card_index.py:1-13`，“红卡按 CardId 枚举顺序 + 无色卡池按枚举名字典序”）。

- `srcColorlessCardPool`（`CardPools.h:67`）= **35**（`CardPools.h:68` `srcColorlessCardPoolSize = 35`）；
- 红卡（Ironclad，`CardColor::RED`，`Cards.h` 的 `cardColors` 表 `Cards.h:425`）= **75**；
- 75（红）+ 35（无色）= **110**，与 `CARD_INDEX` 完全一致。

**越界核查：**

| 检查 | 结果 |
|---|---|
| `CARD_INDEX` 是否等于 `RED ∪ srcColorlessCardPool` | **是**（`expected(110) == CI? True`） |
| 是否混入 GREEN/BLUE/PURPLE/CURSE（其他角色卡） | **否**（0 个） |
| 是否混入非 RED 且不在 `srcColorlessCardPool` 的牌 | **否**（0 个） |

> 注：枚举中 `CardColor::COLORLESS` 共 56 张（`Cards.h` 统计），但 `CARD_INDEX` 只取 `srcColorlessCardPool` 的 35 张（铁甲实际可获得的无色奖励池），其余 21 张为状态/事件专用无色牌，已正确排除——这是**正确行为，不是遗漏**。

> 结论：卡牌知识库**无铁甲不可达/越界内容**，0 条需删除。

---

## 5. 建议删除清单

仅「遗物」需要删除。以下按**理由类别**罗列，共 **30 个确定删除 + 1 个边界（可选删除）**。

### 5.1 异角色专属池（26）—— 类别：异角色专属

**Silent（10）**：`HOVERING_KITE`、`PAPER_KRANE`、`SNECKO_SKULL`、`THE_SPECIMEN`、`TINGSHA`、`TOUGH_BANDAGES`、`TWISTED_FUNNEL`、`WRIST_BLADE`、`RING_OF_THE_SERPENT`、`NINJA_SCROLL`

**Defect（8）**：`EMOTION_CHIP`、`FROZEN_CORE`、`INSERTER`、`DATA_DISK`、`GOLD_PLATED_CABLES`、`NUCLEAR_BATTERY`、`RUNIC_CAPACITOR`、`SYMBIOTIC_VIRUS`

**Watcher（8）**：`CLOAK_CLASP`、`DUALITY`、`GOLDEN_EYE`、`MELANGE`、`VIOLET_LOTUS`、`DAMARU`、`HOLY_WATER`、`TEARDROP_LOCKET`

### 5.2 异角色起始遗物（3）—— 类别：异角色专属（起始）

`RING_OF_THE_SNAKE`（Silent）、`CRACKED_CORE`（Defect）、`PURE_WATER`（Watcher）

### 5.3 兜底不可达（1 确定）—— 类别：兜底不可达

`RED_CIRCLET`（仅 boss 池空时退回，`GameContext.cpp:1673`，铁甲 boss 池非空故不可达）

### 5.4 边界可选（1）—— 类别：兜底边界（集齐全脸遗物后理论上可达）

`CIRCLET`（`GameContext.cpp:1657` 兜底；`Game.cpp:353` 集齐全 5 脸遗物后由 `getRandomFace` 发放）。**建议：保留**（存在理论可达路径）；若坚持“铁甲 A0 常规不可达”口径，可一并删除。

> **合计：确定删除 30 个，边界可选 1 个。药水 / 事件 / 卡牌 删除数均为 0。**

---

## 附：临时脚本说明

审计过程使用的临时脚本（`_tmp_audit.py`、`_tmp_classify.py` 及中间产物 `_tmp_relic_diff.txt`、`_tmp_ironclad_pool.txt`）已置于项目根目录，审计完成后应删除；本报告不写入 `models/` 或 `env/` 任何文件，仅新建本 `logs/audit-reachability.md`。
