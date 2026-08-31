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


def _norm_card_id(s) -> str:
    """归一化卡牌 ID 用于跨端匹配：模拟器枚举名（STRIKE_R）与
    Communication Mod 的字符串 ID（Strike_R）只是大小写/下划线风格不同。"""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


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

            # 2. MCTS 搜索（记录耗时，方便观察"思考深度"）
            import time
            t0 = time.monotonic()
            searcher = searcher_cls(bc, self.sim_count)
            action = searcher.best_action
            log.info("MCTS %d sims 用时 %.2fs", self.sim_count,
                     time.monotonic() - t0)
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

    def _to_comm(self, action, state=None):
        """把 MCTS 动作翻译成 comm 命令；无法翻译返回 None（走规则）。

        关键防御：MCTS 的 source_idx 是模拟器手牌的下标，和真实游戏的手牌
        顺序/内容可能不一致（Communication Mod 不传 RNG 计数器，模拟器重建
        的状态必然有偏差）。因此不按位置出牌，而是取模拟器选中牌的卡牌 ID，
        到真实游戏手牌里按 ID 找一张可出的同名牌——身份匹配而非位置匹配。
        """
        action_type = getattr(action, "action_type", None)

        if action_type == "CARD":
            src = action.source_idx

            # 1. 模拟器侧：读出选中牌的 ID（归一化后用于跨端匹配）
            sim_card_id = None
            bc = self._bc
            if bc is not None:
                try:
                    sim_hand = bc.hand
                    if 0 <= src < len(sim_hand):
                        sim_card_id = _norm_card_id(sim_hand[src].id.name)
                except Exception:
                    log.exception("读取模拟器手牌失败")

            # 2. 真实游戏侧：手牌与能量
            hand = []
            energy = 0
            if state:
                combat = state.get("game_state", {}).get("combat_state", {})
                hand = combat.get("hand", [])
                energy = combat.get("player", {}).get("energy", 0)

            # 3. 按 ID 在真实手牌里找可出的同名牌
            real_idx = None
            if sim_card_id and hand:
                for i, card in enumerate(hand):
                    if _norm_card_id(card.get("id")) != sim_card_id:
                        continue
                    cost = card.get("cost", 99)
                    if cost < 0:
                        cost = 99  # 诅咒/特殊牌不可出
                    if cost <= energy and card.get("is_playable", False):
                        real_idx = i
                        break
                if real_idx is None:
                    log.warning("真实手牌里找不到可出的同名牌（sim id=%s），回退规则",
                                sim_card_id)
                    return None
            elif hand:
                # 模拟器手牌读不到时退化为位置匹配 + 可出性校验
                if not (0 <= src < len(hand)):
                    log.warning("MCTS source_idx %d 超出真实手牌数 %d", src, len(hand))
                    return None
                card = hand[src]
                cost = card.get("cost", 99)
                if cost < 0:
                    cost = 99
                if cost > energy or not card.get("is_playable", False):
                    log.warning("MCTS 选的牌在真实游戏中不可出（%s, cost=%s, energy=%s）",
                                card.get("name"), cost, energy)
                    return None
                real_idx = src
            else:
                real_idx = src  # 无真实手牌数据（测试场景），直接信任模拟器

            cmd = f"PLAY {real_idx + 1}"

            # 只有真实游戏里这张牌确实需要目标时才附带 target_idx。
            needs_target = False
            if hand and 0 <= real_idx < len(hand):
                needs_target = bool(hand[real_idx].get("has_target", False))

            if needs_target:
                # 不信任 MCTS 的 target_idx，从真实游戏状态选第一个活着的怪
                target_idx = self._first_alive_monster(state)
                if target_idx is None:
                    return "END"
                cmd += f" {target_idx}"
            return cmd

        if action_type == "END_TURN":
            return "END"

        if action_type == "POTION":
            # MCTS 选择了用药水——用真实游戏的药水栏校验后翻译
            slot = getattr(action, "source_idx", -1)
            potions = (state or {}).get("game_state", {}).get("potions", [])
            if not (0 <= slot < len(potions)):
                log.warning("MCTS 药水槽位 %d 越界（真实共 %d），回退规则",
                            slot, len(potions))
                return None
            p = potions[slot]
            if not p.get("can_use") or p.get("id") in (None, "Potion Slot"):
                log.warning("MCTS 选的药水在真实游戏不可用（%s），回退规则",
                            p.get("id"))
                return None
            cmd = f"POTION USE {slot}"
            if p.get("requires_target"):
                target_idx = self._first_alive_monster(state)
                if target_idx is None:
                    return None
                cmd += f" {target_idx}"
            return cmd

        log.warning("未知 MCTS 动作类型: %r，回退规则", action_type)
        return None

    @staticmethod
    def _first_alive_monster(state):
        """从真实游戏 JSON 找第一个活着的怪物下标；没有则 None。"""
        if not state:
            return 0
        monsters = (state.get("game_state", {})
                    .get("combat_state", {}).get("monsters", []))
        for i, m in enumerate(monsters):
            if m.get("current_hp", 0) > 0 and not m.get("is_gone", False):
                return i
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
