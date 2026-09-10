"""Power id 别名表：Communication Mod ↔ 模拟器内部枚举名。

真机实测（2026-09-11，logs/realtime/0911_013551）：
两套命名不完全一致，不匹配的 id 会被模拟器的 playerStatusFromString /
monsterStatusFromString 判为 INVALID 并**静默丢弃**（continue）——
最严重的："Weakened"（Comm Mod）vs "Weak"（模拟器）：
  玩家虚弱（23 帧实测）从未被灌入 → MCTS 的全部伤害计算多算 25%。
这正是用户报告的"MCTS 不会计算自身虚弱的影响"。

单一事实来源：main.py（GameContext 灌入）与 combat_brain.py（BattleContext
灌入）都必须调用 alias_power_ids()，防止两处漂移。
"""
from __future__ import annotations

# Comm Mod id → 模拟器表内名（表见 include/constants/PlayerStatusEffects.h 与
# MonsterStatusEffects.h 的 playerStatusStrings / enemyStatusStrings）
POWER_ID_ALIASES = {
    "Weakened": "Weak",     # 玩家/怪物 虚弱（真机 23 帧实测；表内也叫 "Weak"）
    "Anger": "Angry",       # 怪物 激怒（Comm Mod 给 "Anger"，monster 表用 "Angry"）
}


def alias_power_ids(payload) -> None:
    """就地修改 payload 中 player/monster powers 的 id 字段（仅别名替换）。

    payload: Communication Mod 的完整状态对象（dict，通常已过 _sanitize）。
    绝不新增/删除字段——只把已知别名替换成模拟器能识别的名字。
    失败安全：任何异常都不抛出（别名失败不该影响主流程）。
    """
    try:
        gs = payload.get("game_state") if isinstance(payload, dict) else None
        cs = (gs or {}).get("combat_state") or {}
        arrays = [((cs.get("player") or {}).get("powers") or [])]
        for m in (cs.get("monsters") or []):
            if isinstance(m, dict):
                arrays.append(m.get("powers") or [])
        for arr in arrays:
            for p in arr:
                if isinstance(p, dict):
                    pid = p.get("id")
                    if pid in POWER_ID_ALIASES:
                        p["id"] = POWER_ID_ALIASES[pid]
    except Exception:
        pass
