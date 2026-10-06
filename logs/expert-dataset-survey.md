# 专家数据集调研报告（只读，未改项目代码）

> 工作目录：`07-projects/sts-agent/`；克隆存放：`data/external/`（均 `--depth 1`）
> 目标：验证「不下载人类数据、改用模拟器+MCTS 当专家、现场生成演示」这一判断

## 1. 数据源一览表

| 数据源 | 内容类型 | 规模 | 与我们的适配度 | 阻塞点 |
|---|---|---|---|---|
| **MaT1g3R/Slay-the-Spire-data** | Run History Plus 版 `.run`（逐决策记录） | 749 局，**203 Ironclad**，全 A20，383 胜 | **高**（宏观决策 SFT） | 2022 版本数据；主播水平参差；少数遗物不记增删卡；无逐出牌战斗日志 |
| **lgerrets/sts_card_picker** | 从 vanilla `.run` 差分重建选牌 | 模型约 **55%** top-1（skip 32% / 随机 25%） | **中**（重建逻辑可借鉴） | 仅训练「选牌」一类决策；需自行扩展到选路/事件/商店；55% 属模仿非专家 |
| **HF t22000t 六数据集** | 卡牌元数据 + 1024 维语义 embedding（无决策轨迹） | STS1 360 卡 / STS2 576 卡 | **低-中**（不能替代 card_data.py） | 冻结通用文本向量，非「防战」定制；补不了 6 条表达力缺口 |
| **benmuth/AutoClad** | sts-lightspeed + baseline agent 生成战斗演示 → SFT | 仅 Jaw Worm 战斗（已跑通） | **高**（战斗层蓝图） | 只做了战斗；宏观层未实现；SimpleAgent 是弱教师 |

## 2. 逐条结论

**① MaT1g3R —— 能用（强烈建议下载，用于宏观决策 SFT 预热）。**
`.run` 远比「整局汇总」丰富：直接含 `card_choices{picked, not_picked, floor}`（选牌选项+人类选择）、`event_choices`、`shop_contents`+`items_purchased`、`campfire_choices`、`boss_relics{not_picked,picked}`、`path_taken`、以及逐层 HP/金币/遗物轨迹。**选牌选项无需差分**——已直接记录。决策点当时的状态（牌组/遗物）可通过重放选择日志重建（见 sts_card_picker `rebuild_deck` 算法）。即：对选牌/事件/商店/营火/选路四类宏观决策，都能得到「可选集 + 人类选择 + 可重建状态」，正好喂给我们的 PPO Scorer 做监督预热。

**② sts_card_picker —— 部分可用（取其重建算法，不取其模型）。**
它针对 *vanilla* `.run`（只有最终牌组，缺逐决策选项），用 `deck_history.py`（1414 行）做回合间差分推断选牌。若我们直接用 MaT1g3R 的 Run History Plus 数据，差分这步可省。其价值在于 `rebuild_deck` 这种「按路径重放日志重建决策点牌组/遗物」的算法——可移植到我们的状态重建。但模型本身只学「选牌」一种决策，要覆盖选路/事件/商店需我们自建决策目标。

**③ HF 六数据集 —— 不能用 / 部分可用（不能替代 card_data.py）。**
六个全是**卡牌语义资源**，非决策轨迹：`*-cards`（元数据+立绘）、`*-card-embeddings`（1024 维，Qwen3-Embedding-0.6B 冻结、按「机械相似度」指令编码）。它们表达「卡牌间机制相似」，但**不是为「防战/格挡体系」定制**，也修不了我们手工 42 维字段的 6 条表达力缺口（那是字段梳理问题，非缺向量）。把 1024 维不透明向量并到 42 维可解释特征旁，反引入域偏移/重训练风险。最多作辅助检索特征，优先级低。

**④ AutoClad —— 能用（作战斗层蓝图，并验证 team-lead 的判断）。**
直接内嵌 sts-lightspeed（C++ RNG 精确）。流水线：`randomize_scenarios`（随机化 HP/牌组/种子做数据增广）→ 跑 SimpleAgent 快照 → `data_parser` 抽 state-action → 训练 → 导出 TorchScript `.pt` 供 C++ 推理。`state_converter` 明确「only works for combat states」——**目前只覆盖战斗（Jaw Worm）**。README 原话「人类数据不完整且过时，改用树搜索+简单 agent 生成训练数据」——这对**战斗层**完全成立（`.run` 根本不记录逐出牌），但对宏观层不成立（见①）。

## 3. 推荐路线（修正原判断）

**原判断「完全不下载人类数据」只对战斗层成立，对宏观层过强——建议改为混合路线：**

- **宏观决策（Scorer 预热）：下载 MaT1g3R 人类数据做 SFT 预热。** 数据已结构化、直接含「选项+选择+可重建状态」，性价比远高于从零造一个宏观专家。过滤到胜局/高手局（如 panacea108）以降噪；用 `master_deck` 校验重建正确性（sts_card_picker 已有此校验）。
- **战斗决策：坚持模拟器 + MCTS/强 agent 当专家、现场生成演示。** 人类 `.run` 无逐出牌日志，AutoClad 已证明此路可行；照搬其「场景随机化 + 快照抽 state-action + TorchScript 导出」工程。
- **card_data.py 的 6 条缺口：靠字段梳理 + 决策数据学习补，不靠 HF 嵌入向量。** 嵌入仅作可选辅助检索，非主线。

**人类数据是否真「不完整/过时」？** 部分属实：战斗层确实不完整（无逐出牌），且为 2022 版本；但宏观层**完整且结构化**，Ironclad 基础卡池稳定，「过时」影响有限。故「不下载」应限定在战斗层。

**最大风险**：人类数据为主播混合水平、可能灌输次优宏观策略；状态重建在少数遗物（Astrolabe/Tiny House 不记增删卡）有边缘缺口。缓解：胜局/高手过滤 + master_deck 一致性校验。
