# sts-agent 真机战斗行为取证报告

**数据范围**：`logs/realtime/` 三个 run（`0911_013551` / `0911_022209` / `0911_025542`），含 `_decisions.jsonl`、`_states.jsonl` 与根目录 `sts_agent_debug.log`。
**方法**：纯只读流式解析。decisions 提供 PLAY/END 命令与 `kind/degraded`；states 提供 `combat_state`（energy/hand/turn）；debug log 提供 `ISMCTS votes` 票型与回退关键字。debuff 牌判定：卡牌描述含「易伤/虚弱」或属命名集合（Bash/Thunderclap/Clothesline/Shockwave/Uppercut/Intimidate），共检出 debuff 牌 id 见下。

---

## 表1　战斗命令分类统计（覆盖 3 个 run）

| run | 战斗动作 | PLAY(真MCTS) | END 总数 | END=真MCTS(COMBAT,未降级) | END=回退兜底(-,degraded) |
|---|---|---|---|---|---|
| 0911_013551 | 333 | 323 | 92 | 10 (10.9%) | 82 (89.1%) |
| 0911_022209 | 656 | 622 | 181 | 20 (11.0%) | 161 (89.0%) |
| 0911_025542 | 187 | 176 | 49 | 6 (12.2%) | 43 (87.8%) |
| **合计** | **1176** | **1121** | **322** | **36 (11.2%)** | **286 (88.8%)** |

> 所有 PLAY 均为 `kind=COMBAT, degraded=False`（MCTS 主动出牌）；END 近 9 成是 `degraded=True` 的回退兜底。

## 表2　「留 1 费不做事」直接量化（END 时刻 energy + 手牌）

| run | END 含可打牌(energy≥1且手牌有 cost≤energy) | END 时能量≥1但手牌全不可打 | END 时能量=0 | END含可打牌占比 |
|---|---|---|---|---|
| 0911_013551 | 16 | 5 | 71 | 17.4% |
| 0911_022209 | 47 | 12 | 122 | 26.0% |
| 0911_025542 | 13 | 0 | 36 | 26.5% |
| **合计** | **76** | **17** | **229** | **23.6%** |

- 帧侧稳健校验（每个战斗回合末帧）：留能量可打牌回合 98/402 = **24.4%**，与命令侧一致。
- 留费 END 的来源拆分：**76 个中有 53 个（69.7%）是回退兜底，仅 23 个（30.3%）是 MCTS 主动**（即 36 个真 MCTS END 里约有 23 个确实带余能收尾）。

## 表3　「先伤害/防御、后 debuff」顺序证据

检出 debuff 牌 id：Bash(88) Thunderclap(7) Uppercut(15) Clothesline(18) Shockwave(8) Intimidate(1)。
TOP 卡名：Strike_R、Defend_R、Bash 为三大主力。

| run | 同时含攻击+debuff 的回合 | debuff 打出位置分布(第1/2/3/4/5/8个) | debuff 位置中位数 | 手牌同含Bash与攻击时：Bash先打 / 攻击先打 |
|---|---|---|---|---|
| 0911_013551 | 28 | 20/8/2/1/1 | 1 | 13 / 4 (76%) |
| 0911_022209 | 65 | 36/25/8/5/2/1 | 2 | 18 / 9 (67%) |
| 0911_025542 | 24 | 12/8/4/0/1 | 2 | 9 / 1 (90%) |
| **合计** | **117** | **68/41/14/6/3/1** | **1** | **40 / 14 (74%)** |

> 聚合 debuff 位置中位数 = **1**（即 debuff 多为「首打」）。手牌同含 Bash 与攻击时，MCTS 在 **74%** 的回合把 Bash（挂易伤）打在攻击之前。

---

## 结论一（对应观察①「MCTS 常常留 1 费不做事就结束回合」）

「留能量不做事」确实存在：**约 24% 的 END / 战斗回合**以 energy≥1 且手牌仍有可打牌收尾（表2）。但这主要是**命令翻译失败触发规则兜底**造成的——286/322 个 END 是 `degraded=True` 回退命令，留费 END 里 69.7% 属兜底。MCTS 主动 END 仅 36 个，其中约 2/3 也带余能，说明 MCTS 本身收尾偏保守，但「什么都不打就结束」的主体是兜底而非 MCTS 意志。

## 结论二（对应观察②「先打伤害/防御、后挂 debuff」）

**该观察与日志不符**。`combat_brain` 把「先 Bash 挂易伤再 Strike」视为正手：在同时含攻击与 debuff 的 117 个回合里，debuff 位置中位数为 **1**（常为首打）；手牌同含 Bash 与攻击时，MCTS 在 **74%** 的回合先打 Bash。仅少数回合（如 `['Strike_R','Bash','Strike_R'…]`）呈「先伤害后 debuff」。即 MCTS 倾向于**先上易伤/虚弱再输出**，与用户描述相反。

## 结论三（对应观察③「多少是 MCTS 主动、多少是翻译失败/兜底」）

根因清晰。`sts_agent_debug.log` 中 **2329** 条 `ISMCTS votes` 快照里，END_TURN 作为**最高票仅 64 次（2.7%）**，含 END_TURN 的仅 295 次（12.7%），END_TURN 总票占比 4.2%——**MCTS 内部几乎从不投票结束回合**。与此同时真实发出的 END 命令 88.8% 是 `degraded=True` 兜底，且日志含 **596** 次 `cannot be played`、`691` 次 `Selected card`、`fallback` 关键词 **5950** 次。`combat_brain.py:69` 明确记载：「目标翻译失败后退化成 END……表现为连续好几次什么牌都不打就结束回合」。故观察①的「留 1 费不做事」**几乎全部源于目标索引翻译失败（尸体目标→游戏拒打→兜底 END），是翻译层把牌丢掉，而非 MCTS 主动选择**。

---

## 具体回合实例清单（debuff 顺序；id 为卡牌英文 id）

**A. 同回合含攻击+debuff（节选）**
- 013551 turn1: `Strike_R, Bash, Strike_R, Defend_R…` → debuff(Bash)第2位（先伤害后debuff 例）
- 013551 turn4: `Bash, Thunderclap` → debuff 第1位（先debuff 例）
- 013551 turn3: `Thunderclap, Bash` → debuff 第1位
- 013551 turn1: `Bash, Strike_R, Flex` → Bash 先打
- 022209 turn2: `Bash, Pummel` → Bash 先打（易伤先于攻击）
- 022209 turn5: `Defend_R, Strike_R, Thunderclap, Blood for Blood` → debuff 第3位
- 025542 turn2: `Bash, Clothesline` → debuff 先打
- 025542 turn6: `Clothesline, Power Through` → debuff 先打

**B. 手牌同含 Bash 与攻击、Bash 先打（节选）**
- 013551 turn1 `Bash, Strike_R, Flex`；turn3 `Bash, Strike_R`；turn2 `Bash, Strike_R`
- 022209 turn2 `Bash, Pummel`；turn4 `Bash, Sword Boomerang`
- 025542 turn2 `Bash, Clothesline`；turn7 `Bash, Sword Boomerang`

**C. 「留 1 费」END 实例（帧末 energy≥1 且仍有可打牌，degraded 兜底）**
- 013551 多个 `END`（kind=-, degraded=True）：如末帧 energy=1 且手牌剩 `Strike_R(cost1)` 仍 END。
- 022209 36 个留费 END 为兜底；025542 10 个留费 END 为兜底。
- 对应 debug 行：`Selected card cannot be played with the selected target` 累计 596 次 → 触发 `combat_brain` 退化为 END。

**原始聚合数据**：`logs/realtime_forensics.json`（含逐 run 的 kind/degraded 分布、END 能量拆分、debuff 位置、Bash 先打统计、votes 票型与关键字计数）。
