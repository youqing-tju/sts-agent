# 真机部署启动指南

> 生成：2026-09-10 ｜ 面向操作者：youqing ｜ 自动化侧由我负责监控与分析

---

## 一、已就绪（我已完成的部分）

| 项 | 状态 |
|---|---|
| 权重加载链路 | ✅ `main.py` 参数化，自检确认 `obs=477 cand=347 hidden=(512,256)` |
| 训练/真机一致性 | ✅ 新增 `tests/test_bridge_parity.py`（5 个测试）：观测/流派强度/候选向量**逐元素相同** |
| 协议自检 | ✅ `scripts/selfcheck_bridge.py` 8/8 通过（ready / START / STATE / 畸形 JSON / 防卡死 / 日志） |
| 防卡死守卫 | ✅ 状态指纹连续不变 → 强制换动作 → 仍不行则主动退出（针对历史 5853 次空转） |
| 三层日志 | ✅ 落 `logs/realtime/`：L1 全量状态 / L2 决策 / L3 事件+心跳 |
| Mod 配置 | ✅ 已切到本项目；**原 v2 配置已备份** `config.properties.bak-v2-0910_1640` |

**中文版可直接用**：`init_from_json` 全部按 `id` 字段解析（卡/遗物/药水/怪物），
id 是 ASCII 与语言无关；`name` 字段的 GBK 乱码不影响（我们不用它）。

---

## 二、你要做的（三步）

1. **启动游戏（带 Mod）** — 用游戏目录下的 `mts-launcher.jar` 启动，确保 mod 列表里
   `BaseMod` + `CommunicationMod` 勾选。
   （`%LOCALAPPDATA%\ModTheSpire\mod_lists.json` 里 `CommunicationModCJK.jar` 那一项
   其实**没有对应的 jar 文件**，可以忽略；不放心就把它取消勾选。）

2. **进主菜单 → 开始一局铁甲战士** — 配置里 `runAtGameStart=true`，
   开局后 agent 会自动被拉起。你会在游戏里看到它自己选路/选牌/打仗。

3. **让它自己玩，别干预** — 中途不要手动点卡牌/选项（会和 agent 抢控制权）。
   想停就**直接关游戏**，或者告诉我，我停掉 agent 进程。

**唯一需要你确认的**：窗口里有没有出现类似
`[brain] 就绪 | 权重=... | obs=477 cand=347 | 战斗 ISMCTS 10×2000` 的提示（在 Mod 的日志里）。

---

## 三、时机建议（重要）

V9 训练**还剩约 35 分钟**（4160/10000 局）。两个选择：

| 方案 | 用途 | 建议 |
|---|---|---|
| **现在先跑 1–2 局（用中途权重）** | **只验证桥接是否通**——命令能不能被游戏接受、战斗能不能出牌、日志有没有正常落 | ✅ **推荐先做这个**：桥接验证不需要强模型 |
| **等 V9 训完 + 评估完再正式跑** | 用最强权重做成绩 | V9 训完我会评估（对比 V6 的 36.4%），再告诉你跑 |

---

## 四、我会怎么监控

每局进行中我会看三样东西（你不用管）：

| 我盯的指标 | 数据来源 | 异常阈值 |
|---|---|---|
| 进程是否活着 | L3 心跳（每 60s） | 心跳断 > 2 分钟 → 告警 |
| 是否用上了模型（而非规则降级） | L2 `degraded` 字段比例 | 降级率 > 30% → 说明模拟器路径有问题，立即排查 |
| 是否卡死 | L3 `stall` / `stall_abort` 事件 | 出现 stall → 立刻看 L1 原始状态定位 |

**首局结束我会给你一份**：本局楼层/是否通关、每步决策的屏幕分布、降级次数、
战斗命令成功率、以及任何异常事件的根因分析。

---

## 五、首次运行要重点确认的三件事（我会查）

1. **真实 JSON 里卡牌有没有 `id` 字段** — C++ 靠它反查 CardId，缺了会抛异常并回退规则。
   L1 日志会存下每条原始状态，可以直接验证。
2. **`degraded` 是否为 false** — 若大量为 true，说明模拟器灌入失败（`sim_error` 事件会带异常信息）。
3. **战斗 `PLAY i j` 是否被接受** — 历史失败就死在这里（`PLAY n` 被拒后无限循环）。

---

## 六、失败预案

| 情况 | 处置 |
|---|---|
| 游戏卡住不动 | 我停 agent → 你关游戏重开（L1 日志已存全部状态，可事后复现） |
| 想切回 v2 agent | 恢复 `config.properties.bak-v2-0910_1640` 覆盖 `config.properties` 即可 |
| agent 进程反复重启 | 把配置里 `runAtGameStart` 改成 `false`，改为手动启动 |
| 想手动启动 agent（不走 Mod 自动拉起） | `.venv\Scripts\python.exe main.py` （在项目根目录，直接跑即可，它会打印 ready 并等 stdin） |

---

## 七、当前配置内容（备查）

`%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`：

```
runAtLauncher=true
command=C\:/Youqing-Vault/Embodied-AI/07-projects/sts-agent/.venv/Scripts/python.exe \
        C\:/Youqing-Vault/Embodied-AI/07-projects/sts-agent/main.py \
        --checkpoint C\:/Youqing-Vault/Embodied-AI/07-projects/sts-agent/checkpoints/ppo_v9_10000.pt
runAtGameStart=true
```

> 注：`ppo_v9_10000.pt` 每 400 局被训练进程覆写一次（写入是整体覆盖，不会读到半个文件）。
> agent 只在**启动时加载一次**，所以中途更新权重不会影响正在玩的那一局。

---

# 八、2026-10-16 更新：接入 v14 权重 + Beam 战斗求解器

> ⚠️ 本章覆盖第一章的「配置内容」。**第一~七章的历史信息仍然有效**（Mod 安装、三步操作、
> 监控口径、失败预案都不变）。

## 8.1 变更内容

| 项 | 旧（9 月） | 新（2026-10-16） |
|---|---|---|
| **checkpoint** | `ppo_v9_10000.pt` | **`v14_defbeam_2k.pt`**（2000 局训练，通关率 **53.5%**） |
| **战斗求解器** | ISMCTS（`CombatBrain`） | **Beam Search**（`BeamBrain`，`preset defensive`） |
| 宏观决策 | Scorer 打分 argmax | 不变 |

新配置行（`%LOCALAPPDATA%\ModTheSpire\CommunicationMod\config.properties`）：

```
command=C\:/.../.venv/Scripts/python.exe C\:/.../main.py
        --checkpoint C\:/.../checkpoints/v14_defbeam_2k.pt
        --log-dir    C\:/.../logs/realtime
        --combat-solver beam --beam-weights-preset defensive
```

**备份**：`config.properties.bak-v9-1016_2011`（回退即覆盖回来）。

## 8.2 为什么换

| 依据 | 数据 |
|---|---|
| Beam >> MCTS（同 500 局训练） | 通关率 **24.0% vs 8.0%**（z≈3.09, p≈0.002） |
| v14（2000 局）vs v12/v13（500 局） | 通关率 **53.5%** vs 27–31%；且**通关局牌组更精简**（31.1 vs 40.4 张）|
| `defensive` 预设 | 200 局配对实测 **+9.5 pp**（两个独立模型均复现）|

> 详细报告：`logs/v12-beam-vs-mcts-report.md`、`logs/v14-training-report.md`、
> `logs/weight-tuning-report.md`

## 8.3 上线前已验证（**无需游戏**）

| 验证 | 结果 |
|---|---|
| **真机状态可用性** —— 150 帧 L1 真实战斗日志重放，按**真机路径**构造 bc<br>（`init_from_json` + `_sanitize` + `alias_power_ids`） | **150 / 150 成功**，单步 **1.70 ms**（P95 5.56 ms、最大 8.81 ms） |
| 真机限速余量 | `--min-cmd-interval-ms` 300 ms → **约 34×** |
| **协议自检** `scripts/selfcheck_bridge.py` | 默认(mcts) 与 beam 路径**均 10/10 PASS** |
| 默认路径不受影响 | `SimBrain()` 仍是 `CombatBrain`（`combat_solver=mcts`）|

## 8.4 `main.py` 新增 CLI

```
--combat-solver {mcts,beam}   默认 mcts（= 完全不变的旧行为）
--beam-width N                默认 2    ⭐ 实测 W=4/W=8 通关率反而更低（宽度不单调）
--beam-depth N                默认 6
--beam-max-nodes N            默认 300
--beam-weights-preset {default,defensive,defensive_max}    ⭐ 推荐 defensive
```

自检脚本支持环境变量覆盖，便于复现两条路径：

```bash
STS_BRIDGE_CKPT=checkpoints/v14_defbeam_2k.pt \
STS_BRIDGE_EXTRA_ARGS="--combat-solver beam --beam-weights-preset defensive" \
  .venv/Scripts/python.exe scripts/selfcheck_bridge.py
```

## 8.5 回退

| 想回到 | 操作 |
|---|---|
| **v9 + ISMCTS**（9 月状态） | `cp config.properties.bak-v9-1016_2011 config.properties` |
| 保留 v14、只回退战斗层 | 配置里删掉 `--combat-solver beam --beam-weights-preset defensive` || **完全不用 agent** | 配置里把 `runAtGameStart` 改成 `false` |

---

# 九、2026-10-16 晚 更新：Beam 换用**终局估值**（`--beam-value-mode rollout`）

> ⚠️ 本章覆盖第八章的配置行。**第一~八章的历史信息仍然有效**。

## 9.1 变更内容

| 项 | 第八章（已过时） | **新** |
|---|---|---|
| 节点估值 | `static`（静态中间状态，**无终局信息**） | **`rollout`**（叶节点打到终局再打分） |
| `--beam-depth` | 6 | **1**（rollout 已覆盖未来，深度无用） |
| checkpoint / 预设 | v14 + `defensive` | **不变** |

新配置行：

```
command=C\:/.../.venv/Scripts/python.exe C\:/.../main.py
        --checkpoint C\:/.../checkpoints/v14_defbeam_2k.pt
        --log-dir    C\:/.../logs/realtime
        --combat-solver beam --beam-value-mode rollout --beam-depth 1
        --beam-max-nodes 300 --beam-weights-preset defensive
```

**备份**：`config.properties.bak-beamstatic-1016_2209`（回退即覆盖回来）。

## 9.2 为什么换（依据）

| 依据 | 数据 |
|---|---|
| **100 局配对**（同 checkpoint / 同宏观策略，**只换估值函数**） | 通关率 **45.0% → 86.0%**（McNemar **30/1**, p<1e-4）；楼层 **34.11 → 46.78**（配对 t=6.26） |
| 逐种子稳健性 | 39 例升 / 60 例平 / **仅 1 例降**（且最差只掉 1 层） |
| 根因 | `evaluate()` 打分的是静态中间状态、**没有终局信息** → 这正是"更宽/更深/更大预算全部变差"的原因 |

> 详情：`logs/beam-terminal-value-report.md`（含四组深度/宽度对照与成本分解）

## 9.3 上线前已验证（**无需游戏**）

| 验证 | 结果 |
|---|---|
| **真机单步耗时**（150 帧 L1 真实战斗日志重放） | 均值 **5.19 ms**、中位 1.69、**P95 22.18**、最大 **74.76** → 限速 300 ms 安全 |
| 构造 + 求解成功率 | **150 / 150** |
| **协议自检** `selfcheck_bridge.py`（rollout 配置） | **10 / 10 PASS** |
| 确定性复跑 | 同配置前 20 种子与已存数据**逐项相同** |
| 默认路径不受影响 | `SimBrain()` 仍是 `CombatBrain`（`combat_solver=mcts`，`value_mode=static`） |

⚠️ **不要用 `--beam-depth 6` 配 rollout**：实测 P95 195 ms、**最大 367 ms 会突破真机 300 ms 限速**。

## 9.4 `main.py` 新增 CLI

```
--beam-value-mode {static,rollout}   默认 static（= 完全不变的行为）；推荐 rollout
--beam-rollout-max-steps N           默认 400（rollout 步数上限）
```

## 9.5 回退

| 想回到 | 操作 |
|---|---|
| 第八章（static d=6） | `cp config.properties.bak-beamstatic-1016_2209 config.properties` |
| 只回退估值方式 | 配置里删掉 `--beam-value-mode rollout --beam-depth 1`（其余不动） |

