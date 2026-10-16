# -*- coding: utf-8 -*-
"""Beam Brain —— 真机（Communication Mod）的 **Beam Search 战斗大脑**。

与 `combat_brain.CombatBrain`（ISMCTS）**接口完全一致**，可在 `main.py` 里互换：

    decide(state, gc) -> "PLAY i j" / "END" / None        # 单次决策
    decide_with_fallback(state, gc, fallback_fn) -> str   # 失败回退规则

## 为什么需要这个文件

10 月主线已从 MCTS 全面转向 **确定性 Beam Search**（见 `logs/v12-beam-vs-mcts-report.md`、
`logs/beam-advanced-mechanisms.md`），但真机入口 `main.py` 的战斗层仍是
`CombatBrain`（ISMCTS，9 月方案）。本文件补上真机侧的 Beam 路径。

## 真机 bc 构造（**与训练侧不同，勿混用**）

| | 构造方式 |
|---|---|
| **真机**（本文件 / `CombatBrain`） | `bc.init_from_json(gc, json.dumps(_sanitize(state)))` + `alias_power_ids` |
| 训练（`env/lightspeed_env.py`） | `bc.init_from_gc(gc)` |

⚠️ `alias_power_ids`（`models/power_alias.py`）必须调用 —— 否则 `Weakened→Weak` 之类的
别名会被静默丢弃，**伤害计算全部多算 25%**（`combat_brain.py:225` 的原始注释）。
本文件与 `CombatBrain` 共用同一套预处理与命令翻译辅助函数，保证两条路径行为一致。

## 实测性能（`scripts/probes/_tmp_beam_realtime_probe.py`）

用 L1 日志的 **150 帧真实战斗状态**按上述真机路径重放：
- 构造 bc + 求解 **150/150 成功**
- 单步耗时 **均值 1.70 ms**（P95 5.56 ms、最大 8.81 ms）
- 真机限速 `--min-cmd-interval-ms` 默认 300 ms → **余量约 34×**

## 已知限制（与训练侧同一份）

`combat_beam.BeamSolver` 的局限同样适用于真机：
- **药水动作未接入**（只枚举"出牌 + 结束回合"）—— 但 `main.py` 的**濒死保命**
  检查（`_desperate_potion_cmd`）在求解器之外独立生效，不受影响；
- Power 绑定只暴露 4 个（`Vulnerable`/`Weak` 读不到）→ 见 `combat_beam` 模块 docstring。
"""
from __future__ import annotations

import json
import logging

import slaythespire as sts

# 复用 CombatBrain 的预处理与命令翻译（同源，避免两套实现漂移）
from combat_brain import _card_needs_target, _sanitize, _translate_target
from models.power_alias import alias_power_ids

log = logging.getLogger(__name__)


def _to_comm(action: tuple, state: dict | None = None) -> str | None:
    """把 Beam 动作元组翻译成 Communication Mod 命令。

    Beam 的动作格式（`combat_beam.legal_actions`）：`('CARD', i, j)` / `('END_TURN', 0, 0)`，
    其中 `i` 是**手牌下标**、`j` 是**模拟器视角的存活怪物下标**。

    ⚠️ 三条与 `CombatBrain._to_comm` 完全一致的约定（踩过的坑，不要改）：
    1. Communication Mod 是 **1-based** → `PLAY {i+1}`；
    2. **仅当该牌需要目标**时才追加 target —— 否则整条命令被拒
       （"Selected card cannot be played with the selected target"）且游戏**不执行任何操作**
       → 死循环；
    3. 目标要**从模拟器的"活怪序列下标"映射回真机完整数组下标**
       （灌入时死怪被过滤；不映射会导致"Boss 进斩杀线还优先打爪牙"）。
    """
    if not action:
        return None
    kind = action[0]

    if kind == "CARD":
        i = int(action[1])
        cmd = "PLAY %d" % (i + 1)
        if _card_needs_target(state or {}, i):
            tgt = _translate_target(state or {}, int(action[2]))
            if tgt is None:
                log.warning("怪物列表里没有活着的目标，放弃该 CARD 动作（交调用方走规则）")
                return None
            if tgt == "unknown":
                # 没有战斗信息 → 保守沿用模拟器给的下标
                tgt = int(action[2])
                if tgt < 0:
                    tgt = 0
            cmd += " %d" % tgt
        return cmd

    if kind == "END_TURN":
        return "END"

    # Beam 目前不枚举 POTION（见模块 docstring 已知限制）
    return None


class BeamBrain:
    """Beam Search 战斗大脑（真机）。接口与 `CombatBrain` 一致，可互换。

    Args:
        beam_width / beam_depth / beam_max_nodes: 搜索预算。
            ⭐ 生产配置 = `W=2 / depth=6 / nodes=300`（见 `logs/beam-advanced-mechanisms.md`：
            更宽 / 更深 / 更大预算实测**全部变差**）。
        weights: 启发式权重覆盖（dict）；推荐 `{"enemy_hp": 6000, "intent": 12000}`
            （= 预设 `defensive`，200 局配对实测通关率 +9.5 pp）。
    """

    def __init__(self, beam_width: int = 2, beam_depth: int = 6,
                 beam_max_nodes: int = 300, weights: dict | None = None,
                 value_mode: str = "static", rollout_max_steps: int = 400):
        from combat_beam import BeamSolver, Weights  # 延迟 import：避免无关场景加载

        self.solver = BeamSolver(beam_width=beam_width, max_depth=beam_depth,
                                 max_nodes=beam_max_nodes,
                                 weights=Weights.from_overrides(weights),
                                 value_mode=value_mode,
                                 rollout_max_steps=rollout_max_steps)
        self.last_action: tuple | None = None
        self.last_stats: dict = {}

    # ---- 决策入口（与 CombatBrain.decide 同签名）----
    def decide(self, state: dict, gc):
        """从真实游戏战斗状态生成 PLAY/END 命令；失败返回 None。

        Args:
            state: Communication Mod 的完整 JSON 状态（含 combat_state）
            gc: 已 `init_from_json` 的 GameContext（由调用方维护）
        """
        try:
            clean = _sanitize(state)
            alias_power_ids(clean)
            bc = sts.BattleContext()
            bc.init_from_json(gc, json.dumps(clean))
        except Exception:
            log.exception("Beam 战斗状态灌入失败")
            return None

        if bc.is_terminal():
            return None
        try:
            plan = self.solver.solve(bc)
        except Exception:
            log.exception("Beam 求解失败")
            return None

        self.last_stats = dict(getattr(self.solver, "last_stats", {}) or {})
        if not plan:
            log.warning("Beam 未返回方案（不空转：交调用方走规则）")
            return None
        self.last_action = plan[0]
        return _to_comm(plan[0], state)

    def decide_with_fallback(self, state: dict, gc, fallback_fn) -> str:
        """带规则降级的 decide。`fallback_fn` 是 `main_rule.combat_action`。"""
        try:
            result = self.decide(state, gc)
            if result is not None:
                return result
        except Exception:
            log.exception("Beam 战斗决策异常，回退规则")
        return fallback_fn(state)
