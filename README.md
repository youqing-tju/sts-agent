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

## Baseline 成绩（M1 验收后填写）

| 策略 | 平均楼层 | 通关率 | 备注 |
|---|---|---|---|
| rule v1 | — | — | 待实测 |
