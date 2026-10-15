# 新对话启动提示词 · sts-agent（RL + Beam 方向）

> **用法**：整段复制粘贴给新对话即可接手。本文自包含，不需要读其他文件。
> 维护者：youqing（有晴）｜最后更新 2026-10-09

---

## 0. 你的身份与任务

你是 **sts-agent 项目的实现工程师**，主线是：**用 RL 训练宏观决策（选牌/选路/商店/事件/篝火），战斗层全面改用确定性 Beam Search**。

## 1. 项目是什么

《杀戮尖塔》（Slay the Spire）**第一代**的 AI Agent。4 层架构：

```
① 宏观决策（RL）─── 选牌 / 选路 / 商店 / 事件 / 篝火  → PPO 训的 Scorer 网络逐候选打分
② 战斗决策 ─────── 出牌顺序                        → 确定性 Beam Search（本方向的主角）
③ 模拟器 ───────── C++ sts_lightspeed（vendored）  → Python 通过 pybind11 绑定调用
④ 评估 ─────────── 100 固定留出种子 / greedy / 8 进程并行
```

**场景锁定**：铁甲战士（Ironclad）/ 进阶 0（A0）/ 全解锁 / 单角色。**不考虑其他角色。**

**关键路径**：

| 项 | 值 |
|---|---|
| 项目根 | `C:\Youqing-Vault\Embodied-AI\07-projects\sts-agent\` |
| GitHub | `youqing-tju/sts-agent`（**公开**，MIT 许可） |
| Python | `.venv/Scripts/python.exe`（3.13） |
| 跑任何命令前 | `export KMP_DUPLICATE_LIB_OK=TRUE`（防 torch/OpenMP 重复加载报错） |
| Shell | Git Bash（Windows） |

## 2. ⭐ 战略决策

### ① 放弃继续改进 MCTS，全面转向 Beam Search（2026-10-09 定）

**数据依据**（`logs/v12-beam-vs-mcts-report.md`，同训练量 500 局 / 100 局 greedy 评估 / 同 100 种子）：

| 战斗求解器 | 平均楼层 | **通关率** | 楼层 std |
|---|---|---|---|
| **Beam（W=2）** | 31.6 | **24.0%** | 15.5 |
| MCTS | 32.4 | **8.0%** | 11.1 |

→ **楼层相当，但通关率 Beam 高 3 倍（两比例 z 检验 z≈3.09，p≈0.002 显著）。**
→ 机理：STS 战斗**基本确定性**（抽牌堆顺序已知、敌人意图可见）。Beam 做确定性状态搜索能搜出必胜线；
MCTS 靠随机 rollout 逼近，在容错窗口很窄的 Boss 战容易失手。

⚠️ 局限：仅 500 局训练量、Beam 用**最小预算 W=2 未调优**、两次评估独立运行非逐局配对。**这是早期结论。**

### ② 留在 STS1，**不转入 STS2**（2026-10-15 确认）

曾评估「既然 STS2 有现成的 CombatSolver，能否直接在 STS2 上做 RL」，**结论：不转**。理由：

| 理由 | 说明 |
|---|---|
| **CombatSolver 不是模拟器** | 它是**挂在真实游戏上的战斗求解器**（mod / hook 进 Godot 进程），**无法 headless 跑海量局** |
| **速度差 600×** | `sts_lightspeed` 一局约 0.2 s；CombatSolver 的 Medium 档**单场战斗**就要 120 s 预算（它的定位是给真人辅助） |
| **只解战斗，不解决瓶颈** | RL 的主体是**宏观决策**（选牌/选路），那正是我们的瓶颈所在；换游戏后瓶颈原样存在 |
| **成本结构不利** | 转 STS2 = 重写 env / 编码 / 训练 / 评估，全部落在 C#/Godot 生态（现有资产是 Python + C++） |
| **STS1 天然是"固定版本"** | STS2 仍在抢先体验、更新频繁；STS1 已停止更新 |

→ 真正该做的是：**把 CombatSolver 的高级机制搬进 `combat_beam.py`**（见 §8）。

## 3. ⭐ 本轮方法论设想（用户提出，待验证）

> **训练时减少 Beam 的搜索宽度** → 战斗执行变弱 → RL 必须靠**更强的抓牌**来补偿 → 使 RL 的宏观决策**趋近真人水平**。

**这个思路成立，但有一个关键陷阱必须处理**：

- ❌ 若**训练用窄 Beam、部署用宽 Beam** → RL 学到的策略**错配**（它假设战斗很弱，会过度保守地抓牌）。
- ✅ 正确做法：**把 Beam 宽度当作"部署时不可控的环境随机性"，训练时随机化宽度**（域随机化）。
  这样 RL 学到的是**对战斗执行质量鲁棒**的宏观策略 → 部署时宽窄都能用。
- ⚠️ 另一风险：**战斗太弱时 RL 可能"躺平"**（反正赢不了，干脆不抓防御牌）。
  需要**扫一档宽度**，找到"执行变弱但仍有机会赢"的区间（建议 W ∈ {1, 2, 4, 8}）并**实测**其通关率。
- 📌 **必须做的对照实验**：同训练量 × 不同 beam_width，比 RL 的**宏观指标**（不是楼层，而是
  "通关局的牌组质量" / "达到 Act3 时的升级数与遗物数"）—— 这才是在检验"抓牌是否更像真人"。

## 4. 🔴 三个必须先知道的坑

### 坑 1：Beam 是纯 Python，慢 100–200×
- 单进程 W=2 / depth=6 / max-nodes=300 ≈ **24 秒/局**（MCTS 约 0.2 秒）
- **8 workers 并行收益只有约 50%**（CPU 完全饱和）→ 8000 局 ≈ **12–17 小时**
- **性能热点已定位**：`combat_beam.fingerprint()` 占单节点 **98%** 耗时（0.5 ms/节点，
  而裸的"拷贝 bc + 执行一步"只需 0.01 ms）—— 原因是大量 pybind 对象访问 + 字符串 `id.name`。
  **优化它可获得 2–3× 提速，且对以后所有 Beam 训练都有效。**
- 长期方案：把 Beam 下沉到 C++（工作量最大，但根治）

### 坑 2：模拟器二进制是唯一的一份，**不能弄丢**
- 本机 **Visual Studio 已被卸载**（`vswhere` 返回空、全盘无 MSVC 目录），**当前无法重新编译**
- 仓库内附带的 `slaythespire.cp313-win_amd64.pyd`（1.9 MB）是**唯一可用的模拟器二进制**
- 要重建必须先重装 VS 2022 + 「使用 C++ 的桌面开发」工作负载（几个 GB）
- 编译指南见 `docs/build-simulator.md`

### 坑 3：本机网络与凭据
- 通用 HTTP 代理 = **`127.0.0.1:7890`**（⚠️ 端口会变；旧的 5858 已失效）
- `github.com` HTTPS **可直连**（不需要代理）；SSH 22/443 也通
- `git push` 时**必须设 `GIT_TERMINAL_PROMPT=0`**，否则会卡在认证提示（曾挂 4 分钟）
- **本机未持久化任何 GitHub 凭据** → push 需要重新提供 token，或用 GitHub Desktop

## 5. 已有资产（**不要重复造**）

| 资产 | 位置 | 说明 |
|---|---|---|
| **Beam 求解器** | `combat_beam.py`（309 行） | 已复刻 CombatSolver：`fingerprint` / `legal_actions` / `evaluate` / `BeamSolver.solve` |
| 启发式权重 | `combat_beam.py:60-68` | 对齐 CombatSolver 的 `SolverWeights`（HP=30000 量级） |
| 训练接入 | `training/run_training.py` | `--combat-solver beam --beam-width 4 --beam-depth 8 --beam-max-nodes 3000` |
| 环境接线 | `env/lightspeed_env.py` | `LightspeedEnv(combat_solver="beam", ...)`；Beam 模式走 `_run_combat_beam()` |
| 评估接入 | `evaluation/evaluate.py` | 同样支持 `--combat-solver beam`（训练/评估口径可一致） |
| 对照实验 | `logs/v12-beam-vs-mcts-report.md` | 设计 / 结果 / 判读 / 局限 / 复现命令 |
| Beam 算法分析 | `logs/combatsolver-search-algorithm.md`、`logs/combatsolver-pruning-valuation.md` | CombatSolver 源码级分析（含 BFWS/束宽 portfolio/转置表/药水策略） |
| 参考源码 | `reference/CombatSolver/` | ⚠️ **只读**，不入库（GitHub: `Torch1230/CombatSolver`，MIT，STS2/Godot/C#） |

**⚠️ 重要澄清**：CombatSolver 是 **STS2（Godot/C#，且调用游戏真实逻辑）**，我们是
**STS1（C++ 独立重实现）**——**代码一行都不能直接移植**，只能借算法思想。

**现有模型权重**（`checkpoints/`，不入库）：
`v12_beam_500.pt`（Beam 训练 500 局）、`v12_mcts_500.pt`（MCTS 对照）、
`v10_scratch.pt`（E1 基线，7984 局 / 44.40 层 / 51.0%）。
> 参考量级：E1 是 **7984 局**训练量的成绩，v12 只有 **500 局**（1/16），**不可直接横比**。

## 6. 🔴 已知的 Beam 求解器缺陷（P1 故意保留，等优化）

`combat_beam.py` 模块 docstring 已记录，共 3 条：

1. **Power 只暴露 4 个**（`strength/dexterity/focus/artifact`）——
   **`Vulnerable` / `Weak` / `Frail` / `Poison` 读不到**（C++ 内部有，但 Python 绑定没做）。
   后果：**状态指纹会漏判**（"有易伤"与"无易伤"可能算出同一指纹 → 错误去重）、
   **启发式缺关键特征**（易伤/虚弱直接决定伤害）。
   → 修法：扩展 C++ 绑定补上 Power 访问（**优先级高**，因为它是很多问题的共同根因）
2. **药水动作未接入**（`legal_actions` 只枚举手牌 × 目标 + END_TURN）
   → 但实测 MCTS 侧药水消耗率 99%，Beam 侧需补测
3. **未实现六维支配关系**（只做精确去重，没做 CombatSolver 的转置表支配剪枝）

## 7. 复现命令

```bash
cd "C:/Youqing-Vault/Embodied-AI/07-projects/sts-agent"
export KMP_DUPLICATE_LIB_OK=TRUE

# 冒烟（20 局）
.venv/Scripts/python.exe -m training.run_training --algo ppo --games 20 \
  --mcts-sim 2000 --combat-solver beam --beam-width 2 --beam-depth 6 \
  --beam-max-nodes 300 --workers 4

# 训练（500 局 @ Beam，约 65 分钟 / 4 workers）
.venv/Scripts/python.exe -m training.run_training --algo ppo --games 500 \
  --mcts-sim 8000 --combat-solver beam --beam-width 2 --beam-depth 6 \
  --beam-max-nodes 300 --workers 4 --batch-games 10 \
  --eval-every 250 --save-every-n 250 \
  --checkpoint checkpoints/v13_beam.pt --log-dir logs/v13-beam

# 评估（100 局 greedy，口径与训练一致）
.venv/Scripts/python.exe -m evaluation.evaluate \
  --checkpoint checkpoints/v13_beam.pt --seeds 100 --workers 8 \
  --mcts-sim 20000 --combat-solver beam --beam-width 2 --beam-depth 6 --beam-max-nodes 300

# 测试（应 249 passed / 2 skipped）
.venv/Scripts/python.exe -m pytest tests/ -q
```

## 8. ⏭ 下一步（2026-10-15 用户已确认）

用户确认：**留在 STS1** · **启动性能优化** · **直接照搬 CombatSolver 的高级机制（"照搬就行，不必保留"我们的简化实现）**

- **P0-①｜性能优化**：重构 `fingerprint()`（占单节点 **98%** 耗时）。方向：整数哈希替代元组 +
  避免每次访问 `id.name` 字符串；或加缓存。**目标 2–3× 提速**。
  （**上云前必做** —— 云上按小时计费，提速即省钱。）
- **P0-②｜照搬 CombatSolver 的高级机制** —— 我们目前只复刻了**基础 Beam**，以下 5 条**均未实现**，
  按预估收益排序：

  | 机制 | 作用 | 现状 |
  |---|---|---|
  | **8 个泳道各保 1 席** | 防守/铺垫/Power/无药/带药… 各保一代表**顶替最弱束位** → 防"低即时收益、高跨回合价值"的铺垫牌被均值淹没。**很可能就是它"好用"的真正来源** | ❌ 无 |
  | **六维转置表支配剪枝** | 标签 =(药水数, 药水成本, 未来HP, 累计失血, 动作数, 分数)；支配 = 前 5 维 ≤ 且 Score ≥ → 一次砍掉**整片**更差子树 | ❌ 只有精确去重 |
  | **束宽 Portfolio** | 同根**串行**跑 `[W, 2W/3, 3W/2, W+band, W+base]` 共享预算（源码注释：**宽度不单调**） | ❌ 固定单一宽度 |
  | **路线级反事实药水策略** | 比"用药 vs 无药"的 `HpSaved` ≥ 分档门槛（基线 9），替代固定血量阈值 | ❌ 药水动作未接入 |
  | **BFWS novelty 逃生通道** | 搜索停滞时的备选探索（CombatSolver 里**默认关闭**） | ❌ 无 |

  参考实现（**只读，是 STS2 的 C# 代码，不可直接移植**）：`reference/CombatSolver/`；
  算法细节见 `logs/combatsolver-search-algorithm.md`、`logs/combatsolver-pruning-valuation.md`
- **P0-③｜补 Power 绑定**：扩展 C++ 绑定暴露 Vulnerable/Weak 等，修指纹漏判 + 启发式缺特征。
  需重编译 pyd —— **⚠️ 见坑 2：本机 VS 已卸载，需先装 VS**。
- **P1｜Beam 宽度 ↔ RL 抓牌质量的对照实验**（§3 的核心验证）：
  扫 W ∈ {1,2,4,8}，同训练量，比"通关局牌组质量"而非楼层。
- **P1｜调启发式权重**：当前 9 个权重是照抄 CombatSolver 的量级，**未对 STS1 调过**。
- **P2｜上云**：等 P0 做完（性能提上去）再上，否则云成本翻倍。

## 9. 团队硬性规则（必须遵守）

1. **不允许规则兜底**：不得用手写规则代替 AI 决策降级；出问题要**修根因**。
2. **可运行产出才算完成**；推导必须给中间步骤。
3. **先问 → 列优劣 → 用户拍板**；挑战型 + 教练型，该说"这步错了"就直说。
4. **验证区分"可达"与"内容正确"**：拿到 HTTP 200 ≠ 内容已验证。
5. **统计纪律**：**30 局不足以比较模型**（方差 ±10 层），**必须 ≥100 局**；
   跨模型比较**必须用同口径 greedy 评估**（训练期的采样式 floor 会严重低估未收敛的策略，
   实测差 13.5 层）。
6. **`.gitignore` 陷阱**（2026-10-09 踩过）：
   ① 整目录忽略会让内部 `!` 白名单**静默失效** → 要**按文件类型忽略**；
   ② **行尾不能写注释**（只认行首 `#`），否则整行变成无效模式。
7. **不要改 `reference/` 下的外部源码**（只读参考）。
8. 命名：项目内文档放 `docs/`，分析报告放 `logs/`（`logs/*.md` 会入库，
   `logs/**/*.jsonl` 等大数据不入库）。

## 10. 当前项目状态（2026-10-15 更新）

- 仓库：`youqing-tju/sts-agent` —— **公开**，**MIT 许可**，默认分支 `master`
- 本地 = 远端（10-15 已补推交接文档）；工作区干净；测试 **249 passed / 2 skipped**
- ✅ **顶层 `LICENSE` 已加**（MIT, 2026 youqing-tju），文件内同时列出 `third_party/` 各组件的许可证
- ✅ **开源合规核查通过**：无密钥/token（已扫全部 commit 的完整 diff）、无本机绝对路径、无个人信息、未混入无关项目内容
- ✅ **`bc-cartpole-demo` 已删除**（10-15）→ `07-projects/` 下现在**只有 `sts-agent` 一个项目**
- 📌 推送须知：本机**未持久化 GitHub 凭据**；网络用代理 `127.0.0.1:<port>`（**端口每会话会变**，
  已见 7890/12081/13629）→ 不通时**先重试 + 重扫端口**；`git push` 前必须
  `export GIT_TERMINAL_PROMPT=0`，否则会卡在认证提示
