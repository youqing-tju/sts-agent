# sts-agent · 杀戮尖塔 AI Agent

RL 方向第一个自主项目：分层决策架构玩《杀戮尖塔》（A0 Ironclad）。

- **宏观决策**（选路/选牌/商店/篝火）→ REINFORCE 策略网络
- **战斗决策** → sts_lightspeed 内置 MCTS
- **Baseline** → 规则 bot（先建，RL 必须打过它）

设计文档：`01-plans/02-sts-agent-design.md`
实现计划：`01-plans/03-sts-agent-impl-plan.md`
调研归档：`05-research/03-sts-agent-survey.md`

## 快速开始

```bash
# 建环境
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
# 编译模拟器（见实现计划 Task 1）
# 冒烟测试
.venv/Scripts/python verify_env.py 12345
# 评估规则 baseline
.venv/Scripts/python -m evaluation.evaluate --policy rule --seeds 100 --workers 8
```

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
