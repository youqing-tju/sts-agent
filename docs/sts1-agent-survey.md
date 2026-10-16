# STS1 Agent 生态调研（工程细节向）

> **状态**：🔄 **第一批（2026-10-17 凌晨）**，按用户指示"到网上好好学习一下 STS1，
> 不管别人是不是和我做的是同一个东西，很多技术处理手段都可以学习，工程细节相似"。
> 下一批待补清单见 §5。
> ⚠️ 本文只记**事实 + 出处 + 可落地判断**；未核实的猜测一律标注。

## 1. 生态盘点（分层，按与我们工程的相关度）

| 层 | 项目 | 关键事实 |
|---|---|---|
| **接入层** | `ForgottenArbiter/CommunicationMod`（146★，Java） | 提供"外部进程控制游戏"的协议：stdin 收 STATE、stdout 发命令 |
| | `ForgottenArbiter/spirecomm`（103★，Python，2019 停更） | CommunicationMod 的 **Python 封装 + 参考 AI**；`main.py`、`utilities/`（kivy GUI）|
| | `ModTheSpire` | ⭐ **本机是 Steam Workshop 版**：`steamapps/workshop/content/646570/1605060445/ModTheSpire.jar`（这解释了游戏目录里为什么没有 `ModTheSpire.jar`）|
| **纯规则机器人** | ⭐⭐ `xaved88` / `lazy-410/bottled_ai`（**725 commits**，最近仍在修 bug） | **A20 通关率 20–52%**（最强是 Watcher 52%）。战斗 = **图遍历 + 自建模拟**枚举"所有出牌方式"，再用**约 40 个加权值**挑最优；选牌/遗物/升级 = **优先级列表**；**完全不用 ML** |
| **学习型** | `MANGO1234/AlphaStS` | AlphaZero 风格自对弈，**只做战斗** |
| | `gamerpuppy/sts_lightspeed` | C++ 全引擎重写（我们 vendored 的就是它）；自带 `BattleScumSearcher2` MCTS |
| | `daniel-ziegler/sts_lightspeed` fork | upstream 改 expectimax MCTS + RL，公开 **A0 心脏击杀 ~83%** |
| **LLM 型** | `AdrianNO1/SlayTheSpireAI` | GPT-4 / Claude 直接读状态做决策（含 pathing/card reward/shop/event）|
| **STS2** | `Manuelbbl/Communication_Mod_STS2`、`brmarcosbr/sts2-mcp-bridge` | 有篇工程复盘特别提到：**把"环境说 no"变成状态、用它过滤下一个动作，softlock 就没了**（不重训）|

## 2. ⭐ 可以直接抄的工程细节（按对我们的价值排序）

### 2.1 手动启动外部进程（**最高优先，能立刻解决我们今晚的启动困惑**）
bottled_ai 官方 README 的启动方式**不是**改 config 里的 `runAt*`，而是：
> 游戏主菜单 → **Mods** → **Communication Mod** → **Config**（"Return" 旁边）→ **Start external process**

⇒ 这直接绕开 `runAtLauncher` / `runAtGameStart` 的语义歧义（我们为此折腾了两轮）。
**建议**：真机实跑改走这条路；config 里只留 `command=`。

### 2.2 "10 秒超时"作为诊断判据
> "The process has a timeout of 10s so if you simply see that delay but nothing's happening
> — something's not working. To debug, check the output in the **ModTheSpire console**,
> or the `communication_mod_errors.log` in the StS folder."

⇒ 以后"没反应"先等 10s 再用这条判据；**ModTheSpire 控制台**是官方第一现场（我们的 stderr 被重定向到 devnull，等于自断这条路——值得重新考虑）。

### 2.3 🔴 与我们 10-16 血案**同款**的死循环案例
bottled_ai 的 commit（2025-12-30）：
> **Fix: GRID multi-select screens (Augmenter event) cause infinite loop**
> Problem: Augmenter 事件的 GRID 屏 `for_transform=False, num_cards=2`，而既有
> `CommonTransformHandler` 只处理 `for_transform=True` → `DefaultChooseHandler`
> **在"反复选中/取消第一张牌"上死循环**。
> Solution: 新增 `CommonGridSelectHandler`（处理任意 `num_cards>0` 的 GRID，正确追踪
> `selected_cards` 数量，选满再 `confirm`）。
> **Includes tests based on actual game log data from the Augmenter event.**

⇒ 三条可直接照搬的经验：
1. **界面处理器必须覆盖"参数的组合空间"**，不能只处理见过的那一种取值（我们的
   `COMBAT_REWARD` 只考虑"奖励按序出现"，没考虑"只剩卡牌项"）；
2. **修完立刻补一条回归测试，fixture 用真实游戏日志**（我们已经这么做了 ✅）；
3. 死循环的经典形态就是"handler 反复发同一条命令/在两屏间来回"。

### 2.4 Handler 分发架构（vs 我们的长 if-else 链）
bottled_ai：`Handler` 有 `can_handle(state)`，**自上而下匹配，第一个命中的执行**；
Common handler 跨策略共享，策略可自带 handler；`rs/ai/<strategy>/` 是一个策略一个目录。
⇒ 我们的 `SimBrain.decide()` 已经长成一条**按屏兜底的长链**（含大量早期 `return`），
10.1 的死锁正是这种结构的典型产物（新分支绕过守卫、条件没覆盖全）。
**P1 建议**：把"每屏一个 handler + 统一出口守卫"重构，让守卫对**所有**分支生效。

### 2.5 运行控制：`run_controller.txt` + 按楼层暂停
> Pause the bot in `run_controller.txt`（还支持"在指定楼层暂停"）

⇒ 我们目前"卡住只能杀进程"。做一个**外部控制文件**（pause / resume / 停在某层）成本极低。

### 2.6 把"表现层速度"与决策逻辑分离
`presentation_config.py`：专门放"给命令加 wait / 调整动作速度"。
⇒ 我们对应的是 `--min-cmd-interval-ms`；动机相同，但没形成独立配置层。
值得学的是**它的命名与边界**——表现层参数不该混在决策参数里。

### 2.7 战斗计算的**上限与退化行为都要写下来**
> **Calculation limit:** 一次出牌最多评估 **11,000 条战斗路径**（`max_path_count`），
> 超了就会**直接出最右边那张牌** —— 这是**有意为之的取舍**（vs 等很久）。

⇒ 我们有 `max_nodes=300` 和"节点用不满"的实测，但**没有记录超限时的退化行为**。
**P1**：明确写出"预算用尽时我们退化成什么"，并纳入监控。

### 2.8 `coverage_stats.py`：卡牌/功能**覆盖率**度量
他们用 `docs/capabilities.md` 明确列出**每个角色缺失的卡**（如 Ironclad 缺 Infernal Blade /
True Grit）与缺失的无色牌，还有一个脚本统计覆盖率。
⇒ 我们只有"降级率"，**没有覆盖率**。这是一个便宜的、能直接指出下一步该做什么的度量。

### 2.9 skills/战斗权重：他们 **~40 个** vs 我们 9–11 个
⇒ 说明"手工线性加权"的表达力上限比我们当前假设高得多；但也提醒：权重的**数量**不是关键，
他们同时有"图遍历 + 完整模拟"这两块我们也有（Beam + rollout）。

## 3. 与我们的差异（**避免错误对比**）

| 维度 | bottled_ai | 我们（sts-agent） | 判断 |
|---|---|---|---|
| 难度 | **A20**（最高） | **A0** | 我们 86% 与它 52% **不可直接比**，A20 难得多 |
| 心脏 / 钥匙 | ❌ "The bot doesn't know how to find the keys" | ❌ `GameOutcome` 三态、不区分心脏 | **同一种口径缺口**，两边都没覆盖 Act4 |
| 信息可见性 | 只用**玩家可见**信息（"No, it can only see what the player does"） | ⭐ lightspeed 模拟器**能看到抽牌堆顺序**（确定性全知） | ⚠️ **这可能是"仿真 86% vs 真机死在 Act1 Boss"的根因之一**：我们的策略被训练成利用一个真机不存在的信息优势 → 建议做成**显式的对照实验**（关掉抽牌堆可见性再训/评） |
| 宏观决策 | 优先级列表（纯规则） | PPO + Scorer（学习） | 他们**不用 ML** 就到 A20 52% ⇒ 值得重新审视"我们的 RL 宏观层到底贡献了多少" |

## 4. 立即可落地的行动项

| 优先级 | 事项 | 理由 |
|---|---|---|
| **P0** | 真机改走 **Mods → Communication Mod → Config → Start external process** | 绕开 `runAt*` 歧义 |
| **P0** | 给 `main.py` 加**单实例锁**（抢不到锁就退出） | 无论第二个进程从哪来都不写同一根管道（10-17 未解之谜的止损） |
| **P0** | 把"**模拟器全知 vs 真机可见**"做成显式对照 | 可能是仿真/真机落差的根因 |
| **P1** | 守卫对所有分支生效（decide() 出口统一过守卫）/ handler 化重构 | 防止再出现"新分支绕过守卫"的死锁 |
| **P1** | 覆盖率度量（缺哪些卡/机制未覆盖） | 便宜且直接指出下一步 |
| **P1** | 外部控制文件（pause/停在某层） | 替代"卡住只能杀进程" |
| **P1** | 恢复 ModTheSpire 控制台输出 / 不再无脑重定向 stderr | 官方第一现场 |
| **P2** | 记录"预算用尽时的退化行为" | 与 11,000 路径上限同构 |

## 5. 下一批待补（**未完成**）

- [ ] bottled_ai 的 `rs/calculator/`（战斗模拟器）与 `common_general_comparator.py`（约 40 权重构成）
- [ ] bottled_ai 的 `rs/common/handlers/`（界面 handler 全集，特别是 reward/grid/event）
- [ ] `spirecomm` 主循环与 `Coordinator`：超时、命令合法性、`available_commands` 的用法
- [ ] `CommunicationMod` 官方 README：完整命令表、超时语义、known issues
- [ ] `MANGO1234/AlphaStS`、`daniel-ziegler` fork 的实现要点（搜索/估值/训练）
- [ ] STS2 那篇"rejection channel"复盘的完整做法（把"环境拒绝"变成状态）

### 出处
- https://github.com/lazy-410/bottled_ai （README / docs/how_to_make_your_own_bot.md /
  docs/capabilities.md / commit `7b47a87` GRID 死循环修复）
- https://github.com/ForgottenArbiter/spirecomm
- https://github.com/ForgottenArbiter/CommunicationMod
- 生态综述（LibHunt 汇总）与 dev.to 盘点帖子
