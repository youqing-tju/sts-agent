"""ISMCTS 战斗决策：把真实游戏战斗状态灌入模拟器，用信息集 MCTS 搜索最优出牌。

架构（v2，见 docs/plans/2026-08-31-ismcts-v2-design.md）：
- BattleContext.init_from_json 把 Communication Mod 的战斗 JSON 灌入模拟器（一次重建）
- K 个平行未来：拷贝 bc0 → randomize_rng 重洗抽牌堆 + 重设 RNG → 各自 BattleSearcher 搜索
- 按卡牌身份投票（不按手牌下标），得票最高者胜出，平票取先出现者
- 胜出动作翻译成 Communication Mod 命令（PLAY/END）

为什么投票：真实游戏的 RNG 未知，单次搜索是在"锁死抽牌顺序的虚构未来"里规划，
斩杀线会落空。K 份采样未来上取期望最优（投票众数），决策不依赖任何单一虚构未来。

可复现性：base_seed 由战斗状态派生（seed + floor*1000 + turn），
同一状态重复决策结果一致，防决策抖动/死循环。

降级链：
- sts.BattleSearcher 绑定不可用 → None（规则兜底）
- BattleContext 无 randomize_rng（C++ 绑定未编译）→ v1 单次搜索
- 任何异常 → None（规则兜底）
- decide_with_fallback 在 decide 返回 None 或抛异常时回退 fallback_fn
  （通常是 main_rule.combat_action）
"""

import json
import logging
from collections import OrderedDict

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


def _norm_card_id(name) -> str:
    """归一化卡牌 id 用于投票：大小写/空白不敏感。"""
    return str(name).strip().lower()


class CombatBrain:
    """战斗大脑：ISMCTS（K 个确定性化未来 + 按卡牌身份投票）→ PLAY 命令。"""

    def __init__(self, simulation_count=2000, determinizations=10):
        self.sim_count = simulation_count        # 每个平行未来的模拟次数
        self.determinizations = determinizations  # 平行未来数量 K
        self._bc = None
        self.last_votes = None  # 最近一次决策的得票分布（观测/调试用）

    def decide(self, state: dict, gc):
        """从真实游戏战斗状态生成 PLAY/END 命令。

        Args:
            state: Communication Mod 的完整 JSON 状态（含 combat_state）
            gc: 已 init_from_json 的 GameContext（由调用方维护）

        Returns:
            Communication Mod 命令字符串（PLAY/END），失败返回 None
        """
        try:
            # 0. MCTS 绑定可用性检查
            searcher_cls = getattr(sts, "BattleSearcher", None)
            if searcher_cls is None:
                log.warning("sts.BattleSearcher 绑定不可用，回退规则")
                return None

            # 1. 一次重建 bc0
            bc0 = sts.BattleContext()
            bc0.init_from_json(gc, json.dumps(_sanitize(state)))
            self._bc = bc0

            # 2. randomize_rng 不可用 → v1 单次搜索（绑定未编译时的降级）
            if not self._supports_determinization(bc0):
                log.info("randomize_rng 不可用，降级 v1 单次搜索")
                return self._v1_search(bc0, searcher_cls)

            # 3. ISMCTS：K 个平行未来搜索 + 投票
            return self._ismcts_decide(state, gc, bc0, searcher_cls)
        except Exception:
            log.exception("MCTS 战斗决策失败")
            return None

    # ---- ISMCTS 主循环 ----

    @staticmethod
    def _supports_determinization(bc) -> bool:
        return callable(getattr(bc, "randomize_rng", None))

    @staticmethod
    def _base_seed(state: dict, gc) -> int:
        """由战斗状态派生确定性种子：seed + floor*1000 + turn。"""
        game = state.get("game_state", {})
        combat = game.get("combat_state", {}) or {}
        turn = int(combat.get("turn", 0) or 0)
        seed = int(getattr(gc, "seed", 0) or 0)
        floor = int(getattr(gc, "floor_num", 0) or 0)
        return (seed + floor * 1000 + turn) & 0xFFFFFFFFFFFFFFFF

    def _ismcts_decide(self, state: dict, gc, bc0, searcher_cls):
        base_seed = self._base_seed(state, gc)
        votes = OrderedDict()   # key -> 得票数（保序，平票取先出现者）
        winners = {}            # key -> 该 key 首次出现的 action
        for k in range(self.determinizations):
            bc_k = sts.BattleContext(bc0)          # 拷贝构造
            bc_k.randomize_rng(base_seed + k)      # 重洗抽牌堆 + 重设 RNG
            action = searcher_cls(bc_k, self.sim_count).best_action
            if callable(action):
                action = action()
            if action is None:
                log.warning("未来 %d 搜索未返回动作，跳过", k)
                continue
            key = self._vote_key(action, bc0)
            votes[key] = votes.get(key, 0) + 1
            winners.setdefault(key, action)

        self.last_votes = dict(votes)
        log.info("ISMCTS votes: %s", dict(votes))
        if not votes:
            return None
        # max 对平票返回先出现的 key（OrderedDict 保序）
        winner_key = max(votes, key=lambda kk: votes[kk])
        return self._to_comm(winners[winner_key])

    @staticmethod
    def _vote_key(action, bc0):
        """投票键 = (action_type, 归一化卡 id)。END_TURN/POTION 用类型本身。"""
        action_type = getattr(action, "action_type", None)
        if action_type == "CARD":
            idx = getattr(action, "source_idx", -1)
            try:
                name = bc0.hand[idx].id.name
            except Exception:
                name = f"idx{idx}"
            return (action_type, _norm_card_id(name))
        return (action_type, action_type)

    # ---- v1 单次搜索（降级路径） ----

    def _v1_search(self, bc0, searcher_cls):
        action = searcher_cls(bc0, self.sim_count).best_action
        if callable(action):
            action = action()
        if action is None:
            log.warning("MCTS 未返回动作，回退规则")
            return None
        return self._to_comm(action)

    # ---- 动作翻译 ----

    @staticmethod
    def _to_comm(action):
        """把 MCTS 动作翻译成 comm 命令；无法翻译返回 None（走规则）。"""
        action_type = getattr(action, "action_type", None)

        if action_type == "CARD":
            cmd = f"PLAY {action.source_idx + 1}"
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
