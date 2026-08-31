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
| rule v1 | **28.3** | 10.1 | 1.0% |
| random | 19.8 | 7.1 | 0.0% |

RL 及格线：平均楼层 > 28.3 且通关率 > 1%。
