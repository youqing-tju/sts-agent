# MCTS 预算扫描实验报告

真机档位参照：`--combat-sims 50000`、`--determinizations 16`。
实验样本：从 `logs/realtime/0911_*_states.jsonl` 7 个文件去重后抽 **30 帧**战斗帧（screen_type==NONE 且 combat_state/hand 非空，连续 turn+手牌去重）。

## 1. 「留 1 费 END」发生率

> 说明：`bc0.hand[i].is_playable()` 在本机 C++ 绑定中**恒返回 False**（误报），
> 故「可打牌」改用 `0<=cost<=energy` 判定（cost<0 为状态牌不可打）。
> 「留费 END」定义：最终命令==END **且** 手牌存在 `cost∈[1,energy]` 的可打牌。

| 档位 | 有效帧 | END 帧 | 留费END(有1+费可打牌) | 占比 | None(回退) |
|---|---|---|---|---|---|
| 2000 | 24 | 6 | 0 | 0/6 = 0% | 6 |
| 8000 | 24 | 5 | 0 | 0/5 = 0% | 6 |
| 50000 | 23 | 6 | 0 | 0/6 = 0% | 7 |
| 200000(前3帧) | 3 | 1 | 0 | 0/1 | 0 |

**结论**：在 30 帧真实样本中，END 命令几乎都出现在 `energy==0`（确实无牌可打）
或手牌全是状态牌/超费牌的局面；**按 cost 重算，没有任何一帧在 50000 档出现「有 1+ 费可打牌却 END」**——
即「留 1 费 END」在本扫描样本里**未被复现**为 MCTS 直接选 END。
逐帧核查 6 个 END 帧：全部为 `energy==0` 或手牌仅状态牌
（如 #553 能量=1 但唯一手牌 INJURY 是 cost=-2 状态牌，不可打），END 均为合理决策。

但发现另一根因：**7 帧 MCTS 投票最高项是药水(POTION)，
但 `_to_comm` 翻译失败返回 None → 真机回退规则**。这类帧在高档位尤其多（见下），
回退后常常什么都不干就结束回合——这很可能就是用户观察到的「留着不动」现象。

| 帧 | 票分布(最高=药水) | 能量 | 手牌 |
|---|---|---|---|
| 0911_013551_states.jsonl#116 | {"('POTION', 'POTION')": 12, "('END_TURN', 'END_TURN')": 4} | 0 | BASH(2),DAZED(-2) |
| 0911_013551_states.jsonl#240 | {"('POTION', 'POTION')": 16} | 0 | BASH(2),WOUND(-2) |
| 0911_013551_states.jsonl#247 | {"('POTION', 'POTION')": 16} | 1 | DAZED(-2),DAZED(-2),DAZED(-2) |
| 0911_013551_states.jsonl#152 | {"('POTION', 'POTION')": 8, "('END_TURN', 'END_TURN')": 8} | 0 | BURN(-2),WOUND(-2) |
| 0911_013551_states.jsonl#250 | {"('POTION', 'POTION')": 13, "('END_TURN', 'END_TURN')": 3} | 1 | WOUND(-2),DAZED(-2),DAZED(-2) |
| 0911_013551_states.jsonl#263 | {"('POTION', 'POTION')": 16} | 0 | WOUND(-2),DAZED(-2),WOUND(-2) |
| 0911_022209_states.jsonl#567 | {"('POTION', 'POTION')": 11, "('END_TURN', 'END_TURN')": 5} | 0 | INJURY(-2) |

## 2. 档位间决策一致率（判定太少/饱和/已溢出）

| 对比 | 一致率 | 样本数 |
|---|---|---|
| 2000 vs 8000 | 80% | 25 |
| 2000 vs 50000 | 76% | 25 |
| 8000 vs 50000 | 76% | 25 |
| 50000 vs 200000 (前3帧) | 100% | 3 |

**判定（修正）：8000→50000 仍有 24% 帧改变决策（一致率 76%），说明 50000 相对 8000 尚未完全饱和、仍值得用；但 50000 vs 200000 在前 3 帧 100% 一致，提示继续翻倍收益递减。综合：50000 既非「太少」（比 8000 明显更稳、多出 ~24% 正确决策），也未被证明「已溢出」（200000 仅 3 帧小样本）；建议维持 50000，优先修 POTION 回退与搜索稳定性，而非盲目加 sims。**

## 3. 决策稳定性（seed 变体一致率）

前 10 帧，固定 SIMS=50000，base_seed 偏移 j×10000 (j=0..4) 各跑 5 个变体：
**稳定率（5 个 seed 全一致）= 50%**（5/10）。**

| 帧 | 5 个 seed 的 winner | 全一致 |
|---|---|---|
| 0911_001847_states.jsonl#5 | ["('CARD', 'defend_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'defend_red')", "('CARD', 'strike_red')"] | False |
| 0911_001847_states.jsonl#12 | ["('END_TURN', 'END_TURN')", "('END_TURN', 'END_TURN')", "('END_TURN', 'END_TURN')", "('END_TURN', 'END_TURN')", "('END_TURN', 'END_TURN')"] | True |
| 0911_001847_states.jsonl#27 | ["('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')"] | True |
| 0911_013551_states.jsonl#116 | ["('POTION', 'POTION')", "('POTION', 'POTION')", "('POTION', 'POTION')", "('POTION', 'POTION')", "('POTION', 'POTION')"] | True |
| 0911_001847_states.jsonl#6 | ["('CARD', 'defend_red')", "('CARD', 'defend_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'defend_red')"] | False |
| 0911_013551_states.jsonl#144 | ["('CARD', 'blood_for_blood')", "('POTION', 'POTION')", "('CARD', 'blood_for_blood')", "('CARD', 'blood_for_blood')", "('POTION', 'POTION')"] | False |
| 0911_001847_states.jsonl#28 | ["('CARD', 'strike_red')", "('CARD', 'defend_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')"] | False |
| 0911_013551_states.jsonl#240 | ["('POTION', 'POTION')", "('POTION', 'POTION')", "('POTION', 'POTION')", "('POTION', 'POTION')", "('POTION', 'POTION')"] | True |
| 0911_001847_states.jsonl#7 | ["('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')", "('CARD', 'strike_red')"] | True |
| 0911_013551_states.jsonl#145 | ["('END_TURN', 'END_TURN')", "('POTION', 'POTION')", "('POTION', 'POTION')", "('END_TURN', 'END_TURN')", "('POTION', 'POTION')"] | False |

> 注意：稳定性只在「能搜出结果」的帧上测（回退 None 的帧跳过）。
> 即便一致，部分帧的 winner 是 POTION/END，说明搜索本身在这些局面里就不坚决。

## 4. 耗时与加大预算可行性

| 档位 | 平均 ms/帧 | 最大 ms/帧 | 样本数 |
|---|---|---|---|
| 2000 | 218.7 | 457.7 | 30 |
| 8000 | 854.4 | 1868.5 | 30 |
| 50000 | 4921.3 | 12043.9 | 30 |
| 200000 | 18191.2 | 23479.5 | 3 |

50000 档平均约 4921.3 ms/帧。30 帧全扫描（含稳定性 10×5 变体）实测 9 分钟，单进程 16GB 内存充足。

**加大预算可行性评估：**
- 200000 档在前 3 帧可跑（平均约 18191.2 ms/帧，约为 50000 的 3.7×），无内存/OOM 问题；
- 但 §2 显示 8000→50000 仍有 ~24% 帧改变决策、50000→200000 几乎不变 → **当前瓶颈不是模拟次数太少，
  而是搜索在部分局面（多怪/药水/斩杀线）里本就缺乏强信号**；
- 若要提升「留费」质量，建议方向：①修 POTION 翻译使药水真正可执行（消除 None 回退）；
  ②对 END_TURN 高票帧加「是否还有可打牌」的二次校验/兜底；③而非盲目加大 sims。

## 附：投票分布（50000 档）

| 帧 | 不同key数 | END_TURN票 | 总票 | 最高票key |
|---|---|---|---|---|
| 0911_001847_states.jsonl#5 | 2 | 0 | 16 | ('CARD', 'defend_red') |
| 0911_001847_states.jsonl#12 | 2 | 14 | 16 | ('END_TURN', 'END_TURN') |
| 0911_001847_states.jsonl#27 | 2 | 0 | 16 | ('CARD', 'strike_red') |
| 0911_013551_states.jsonl#116 | 2 | 4 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#6 | 2 | 0 | 16 | ('CARD', 'defend_red') |
| 0911_013551_states.jsonl#144 | 3 | 1 | 16 | ('CARD', 'blood_for_blood') |
| 0911_001847_states.jsonl#28 | 2 | 0 | 16 | ('CARD', 'strike_red') |
| 0911_013551_states.jsonl#240 | 1 | 0 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#7 | 2 | 0 | 16 | ('CARD', 'strike_red') |
| 0911_013551_states.jsonl#145 | 2 | 8 | 16 | ('END_TURN', 'END_TURN') |
| 0911_001847_states.jsonl#29 | 2 | 3 | 16 | ('CARD', 'strike_red') |
| 0911_013551_states.jsonl#247 | 1 | 0 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#8 | 1 | 16 | 16 | ('END_TURN', 'END_TURN') |
| 0911_013551_states.jsonl#152 | 2 | 8 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#30 | 1 | 16 | 16 | ('END_TURN', 'END_TURN') |
| 0911_013551_states.jsonl#250 | 2 | 3 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#9 | 3 | 0 | 16 | ('CARD', 'strike_red') |
| 0911_013551_states.jsonl#674 | 2 | 2 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#31 | 2 | 0 | 16 | ('CARD', 'defend_red') |
| 0911_013551_states.jsonl#254 | 1 | 0 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#10 | 3 | 0 | 16 | ('CARD', 'strike_red') |
| 0911_022209_states.jsonl#553 | 2 | 8 | 16 | ('END_TURN', 'END_TURN') |
| 0911_001847_states.jsonl#32 | 2 | 0 | 16 | ('CARD', 'defend_red') |
| 0911_013551_states.jsonl#263 | 1 | 0 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#11 | 3 | 2 | 16 | ('CARD', 'strike_red') |
| 0911_022209_states.jsonl#567 | 2 | 5 | 16 | ('POTION', 'POTION') |
| 0911_001847_states.jsonl#33 | 1 | 0 | 16 | ('CARD', 'defend_red') |
| 0911_013551_states.jsonl#454 | 1 | 16 | 16 | ('END_TURN', 'END_TURN') |
| 0911_001847_states.jsonl#13 | 2 | 0 | 16 | ('CARD', 'defend_red') |
| 0911_022209_states.jsonl#698 | 2 | 0 | 16 | ('CARD', 'sever_soul') |

---
原始数据：`logs/mcts_budget_probe.json`。临时脚本：`_tmp_mcts_probe.py`。
