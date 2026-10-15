# sts-agent · 杀戮尖塔 AI Agent

RL 方向第一个自主项目：分层决策架构玩《杀戮尖塔》（A0 Ironclad）。

- **宏观决策**（选路/选牌/商店/篝火）→ RL 策略网络（PPO / REINFORCE，Scorer 打分网络）
- **战斗决策** → 求解器（默认 C++ MCTS，可选 Python Beam Search，见下文）
- **Baseline** → 规则 bot（先建，RL 必须打过它）

设计文档：`01-plans/02-sts-agent-design.md`
实现计划：`01-plans/03-sts-agent-impl-plan.md`
调研归档：`05-research/03-sts-agent-survey.md`

## 项目结构

```
sts-agent/
├── env/                    模拟器封装：把 vendored C++ 模拟器包成 Gym 风格环境
│   ├── base_env.py         环境抽象基类
│   └── lightspeed_env.py   sts_lightspeed（C++）的 Python 侧封装 / 桥接
├── training/               训练流程
│   ├── ppo.py              PPO 实现
│   ├── reinforce.py        REINFORCE 实现（早期算法）
│   ├── run_training.py     训练入口（--algo ppo|reinforce，含 --combat-solver）
│   ├── bc_warmup.py        行为克隆热启动
│   ├── human_data.py       人类 .run 数据集转换（生成 BC 数据集）
│   └── self_play_data.py   自对弈数据生成
├── models/                 网络与编码器
│   ├── obs_encoder.py      全局观测编码
│   ├── candidate_encoder.py 候选动作编码
│   ├── scorer.py           Scorer 打分网络（策略核心）
│   ├── card_data.py / card_index.py   卡牌静态数据与索引
│   ├── candidate_kb.py / event_kb.py / potion_kb.py / relic_kb.py  候选/事件/药水/遗物知识库
│   ├── live_candidates.py  运行时候选生成
│   ├── power_alias.py      Power（增益）别名表
│   └── weight_surgery.py   权重迁移/维度适配
├── agents/                 策略接口与 baseline
│   ├── base_policy.py      策略统一接口
│   └── rule_policy.py      规则 baseline（RL 的对照）
├── evaluation/             评估
│   └── evaluate.py         留出种子多进程并行评估（CLI：--policy/--seeds/--workers）
├── scripts/                工具脚本
│   ├── forensics_realtime.py / selfcheck_bridge.py / selfcheck_shop.py  实时桥接、商店自检
│   └── probes/             历史探针脚本（一次性分析/探测，非生产代码，见其 README）
├── experiments/            消融实验（dim_ablation / win_bonus_ablation）
├── configs/                配置（default.yaml：角色/分数/预算/留出种子等）
├── logging/                run_logger.py：运行期 JSONL 日志
├── docs/                   设计、分析与复盘文档
├── tests/                  pytest 测试
├── third_party/            vendored 依赖
│   └── sts_lightspeed/     C++ 模拟器**源码**（vendored；编译产物见下方根目录的 .pyd）
├── reference/              外部参考源码（CombatSolver / StS2.RandomForeseer，只读，不入库）
├── slaythespire.cp313-win_amd64.pyd   C++ 模拟器**编译产物**（Python 直接 import，见 docs/build-simulator.md）
├── main.py                 完整 Agent 主程序（真机/干跑桥接）
├── main_rule.py            规则 bot 入口（无 RL 的对照运行）
├── combat_beam.py          Python Beam Search 战斗求解器
├── combat_brain.py         战斗层逻辑
├── shop_brain.py           商店决策逻辑
└── verify_env.py           环境冒烟验证
```

## 仓库内容边界（哪些数据入库、哪些不入库）

**入库**：全部 Python 源码 · `logs/*.md` 分析报告 · 报告直接引用的实验数据（`logs/*.json`）·
每个训练 run 的曲线（`logs/**/training_curve.jsonl`）· `docs/` · vendored 模拟器源码 + 其编译产物。

**不入库**（体积大且可重建，详见 `.gitignore`）：

| 路径 | 体积 | 为什么不入库 |
|---|---|---|
| `logs/realtime/` | 243 MB | 真机原始帧日志（L1 层，用于事后排障） |
| `checkpoints/` | 87 MB | 模型权重，可由训练重现 |
| `data/external/` | 76 MB | 人类 `.run` 数据集，可从上游仓库重新下载 |
| `reference/` | 65 MB | 外部参考源码（CombatSolver / RandomForeseer），GitHub 可再 clone |
| `logs/m1-baseline/`、`logs/m2-pilot/` | 8.9 MB | 逐局对局 JSONL（共 601 个） |

> ⚠️ `.gitignore` 采用**按文件类型忽略**，而非「整目录忽略」——
> 整目录被忽略时 git 不会遍历该目录，其中的 `!` 白名单规则会**静默失效**
> （曾经因此丢失全部 23 份分析报告与 11 个被报告引用的数据文件）。

## 战斗求解器（两种，可切换）

训练入口 `training/run_training.py` 支持 `--combat-solver` 选择战斗层实现：

- **`--combat-solver mcts`（默认）**：调用 C++ 内置 Agent（UCT + rollout）自动打完战斗。
- **`--combat-solver beam`**：由 Python 确定性 Beam Search 接管战斗，相关参数：
  - `--beam-width`（默认 4）：束宽
  - `--beam-depth`（默认 8）：搜索最大深度
  - `--beam-max-nodes`（默认 3000）：单次求解最大展开节点数

实现见 `combat_beam.py`（含与 C# CombatSolver 的组件对应表与已知限制）。
**Beam 比 MCTS 更慢，但能给出跨回合的完整路线**（确定性搜索，不做随机 rollout）。

## 快速开始

```bash
# 1) 建环境并装依赖
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt

# 2) 冒烟验证 —— 仓库已附带编译好的 slaythespire.cp313-win_amd64.pyd，无需编译
#    （仅当平台/ Python 版本不同、或改了 C++ 源码时才需要重新编译 → docs/build-simulator.md）
.venv/Scripts/python verify_env.py 12345

# 3) 跑测试
KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python -m pytest tests/ -q

# 4) 评估规则 baseline（100 留出种子，8 进程）
.venv/Scripts/python -m evaluation.evaluate --policy rule --seeds 100 --workers 8

# 5) 小规模训练示例（完整参数见 --help）
KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python training/run_training.py --help
KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python training/run_training.py \
    --algo ppo --games 500 --mcts-sim 8000 --combat-solver mcts
```

> Windows 上建议设 `KMP_DUPLICATE_LIB_OK=TRUE`，避免 torch/OpenMP 重复加载报错。

## Baseline 成绩（M1 ✅ 2026-08-31）

100 个留出种子（master=20260831），MCTS@5000，8 进程并行（实测 100 局约 45 秒）：

| 策略 | 平均楼层 | 标准差 | 通关率 |
|---|---|---|---|
| **RL（5500 局训练）** | **31.6** | 11.1 | **3.0%** |
| rule v1 | 28.3 | 10.1 | 1.0% |
| random | 19.8 | 7.1 | 0.0% |

RL 超过规则 baseline：+3.3 层，通关率 3 倍。

### M2 训练曲线（5000 局，从 500 局试点热启动）

```
局数         楼层    熵      通关率
1-500       28.2   0.488    0.6%
501-1000    27.3   0.489    0.4%
1001-1500   29.0   0.339    1.0%
1501-2000   29.3   0.348    1.8%
2001-2500   30.5   0.240    1.8%
2501-3000   31.4   0.214    1.6%
3001-3500   31.1   0.171    3.4%
3501-4000   30.8   0.089    3.4%
4001-4500   32.1   0.105    3.0%
4501-5000   31.4   0.110    3.4%
```

## License

本项目采用 **MIT License** —— 见 [`LICENSE`](LICENSE)。

`third_party/` 下 vendored 的第三方组件保留其各自许可证：
`sts_lightspeed`（MIT, © 2021 gamerpuppy）、`nlohmann/json`（MIT）、`pybind11`（BSD-style），
详见各子目录内的 LICENSE 文件。
