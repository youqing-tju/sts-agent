"""MCTS 战斗决策：把真实游戏战斗状态灌入模拟器，用 MCTS 搜索最优出牌。

架构：
- BattleContext.init_from_json 把 Communication Mod 的战斗 JSON 灌入模拟器
- sts.BattleSearcher（MCTS 搜索器，C++ 绑定）搜索最优动作
- 把搜索结果翻译成 Communication Mod 命令（PLAY/END）

降级策略：
- sts.BattleSearcher 绑定尚不可用 / 搜索失败 / 动作无法翻译 → 返回 None
- decide_with_fallback 在 decide 返回 None 或抛异常时回退规则决策
  （fallback_fn 通常是 main_rule.combat_action）
"""

import json
import logging

import slaythespire as sts

log = logging.getLogger("sts-agent.combat")


def _sanitize(obj):
    """递归清洗 Communication Mod JSON 里的残缺 UTF-16 代理对。

    CJK 版游戏用 GBK 编码中文，转换时产生 lone surrogate，
    json.dumps 会因此抛异常。全部替换为 U+FFFD。
    （与 main.py._sanitize 保持一致的小拷贝，约束要求不动 main.py）
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj


class CombatBrain:
    """战斗大脑：BattleContext.init_from_json + BattleSearcher → PLAY 命令。"""

    def __init__(self, simulation_count=2000):
        self.sim_count = simulation_count
        self._bc = None

    def decide(self, state: dict, gc):
        """从真实游戏战斗状态生成 PLAY/END 命令。

        Args:
            state: Communication Mod 的完整 JSON 状态（含 combat_state）
            gc: 已 init_from_json 的 GameContext（由调用方维护）

        Returns:
            Communication Mod 命令字符串（PLAY/END），失败返回 None
        """
        try:
            # 0. MCTS 绑定可用性检查（C++ 侧可能还没编译好）
            searcher_cls = getattr(sts, "BattleSearcher", None)
            if searcher_cls is None:
                log.warning("sts.BattleSearcher 绑定不可用，回退规则")
                return None

            # 1. 构造 BattleContext
            bc = sts.BattleContext()
            bc.init_from_json(gc, json.dumps(_sanitize(state)))
            self._bc = bc

            # 2. MCTS 搜索
            searcher = searcher_cls(bc, self.sim_count)
            action = searcher.best_action
            if callable(action):
                action = action()
            if action is None:
                log.warning("MCTS 未返回动作，回退规则")
                return None

            # 3. 翻译成 comm 命令
            return self._to_comm(action, state)
        except Exception:
            log.exception("MCTS 战斗决策失败")
            return None

    @staticmethod
    def _to_comm(action, state=None):
        """把 MCTS 动作翻译成 comm 命令；无法翻译返回 None（走规则）。"""
        action_type = getattr(action, "action_type", None)

        if action_type == "CARD":
            src = action.source_idx
            cmd = f"PLAY {src + 1}"

            # 只有真实游戏里这张牌确实需要目标时才附带 target_idx。
            # 模拟器对不需要目标的牌也会编码 target=0，直接发会被真实游戏拒绝
            # （"Selected card cannot be played with the selected target"）。
            needs_target = False
            if state:
                combat = state.get("game_state", {}).get("combat_state", {})
                hand = combat.get("hand", [])
                if 0 <= src < len(hand):
                    needs_target = bool(hand[src].get("has_target", False))

            if needs_target:
                target_idx = getattr(action, "target_idx", -1)
                if target_idx is not None and target_idx >= 0:
                    cmd += f" {target_idx}"
            return cmd

        if action_type == "END_TURN":
            return "END"

        if action_type == "POTION":
            log.info("MCTS 选择了药水，v1 不支持，回退规则")
            return None

        log.warning("未知 MCTS 动作类型: %r，回退规则", action_type)
        return None

    def decide_with_fallback(self, state: dict, gc, fallback_fn) -> str:
        """带规则降级的 decide。fallback_fn 是 main_rule.combat_action。"""
        try:
            result = self.decide(state, gc)
            if result is not None:
                return result
        except Exception:
            log.exception("MCTS 战斗决策失败，回退规则")
        return fallback_fn(state)
