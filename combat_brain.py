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
from models.power_alias import alias_power_ids
from models.card_data import get_card_effects

log = logging.getLogger("sts-agent.combat")

# 真实状态里可能携带的「是否需要目标」字段名（容错探测，不写死一个键）
_TARGET_FIELD_CANDIDATES = (
    "has_target", "requires_target", "target_type", "needs_target",
)


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


# （2026-09-11 备注：曾在此处加过"Stasis 怪 → 跳过 MCTS 回退规则"的临时补丁。
#  C++ 模拟器已根治该缺陷（BattleContext.cpp 的 stasisCards 重置顺序 +
#  MonsterSpecific.cpp 的 returnStasisCard 加固），补丁已移除——不允许规则兜底。）


def _first_alive_target(state: dict):
    """真实状态里第一个**还活着**的怪物下标（0-based）。

    返回值三种：整数=可选目标；None=有怪物列表但全死了（无合法目标）；
    "unknown"=这帧没有 monsters 信息（退回保守行为）。

    真机 2026-09-11 血案：打死一只怪后，`monsters[0].is_gone` 恒为 True，
    而我们把 `target_idx` 直接当目标发出去 → 目标恒为 0 → 指向尸体 →
    游戏回 "Selected card cannot be played with the selected target." 拒绝，
    翻译失败后退化成 END。实测 79/105 个战斗状态都在这个坑里，
    表现为"连续好几次什么牌都不打就结束回合"。
    """
    gs = state.get("game_state") or {}
    cs = gs.get("combat_state") or {}
    monsters = cs.get("monsters")
    if not monsters:
        # 列表缺失：说明这帧没给战斗信息，退回保守行为（由调用方决定），
        # 不能用"全死"来判定——否则非战斗帧会被误判成无目标。
        return "unknown"
    for i, m in enumerate(monsters):
        if isinstance(m, dict) and m.get("is_gone"):
            continue
        return i
    return None            # 列表存在但全死了 → 确实没有合法目标


def _translate_target(state: dict, sim_target):
    """把 MCTS 的 target_idx（**活怪序列**下标）翻译成真机完整数组下标。

    实测（2026-09-11）：initFromJson 只灌**活着**的怪——真机 4 只（3 具尸体）
    在模拟器里只剩 1 只，所以模拟器的 target 是"活怪序列"下标，与真机
    完整数组（含尸体）不同空间。旧版直接丢弃模拟器的目标、总发"真机第一个
    活怪"——用户观察："总是从左边打起，Boss 进斩杀线也优先打爪牙"。

    返回：真机数组下标（int）；None=没有活怪（无合法目标）；"unknown"=帧里
    没有 monsters 信息（退回保守行为，交调用方处理）。
    """
    gs = state.get("game_state") or {}
    monsters = (gs.get("combat_state") or {}).get("monsters")
    if not monsters:
        return "unknown"
    alive = [i for i, m in enumerate(monsters)
             if isinstance(m, dict) and not m.get("is_gone")]
    if not alive:
        return None
    if isinstance(sim_target, int) and 0 <= sim_target < len(alive):
        return alive[sim_target]
    return alive[0]              # 无效/越界目标 → 回退第一个活怪


def _card_needs_target(state: dict, source_idx: int) -> bool:
    """判断 hand[source_idx] 这张牌打出时是否需要带目标下标。

    命令约定：PLAY {source_idx+1} [target_idx]，仅当牌需要目标才追加 target_idx。
    手牌列表是 0-based，命令里的卡牌下标是 1-based（source_idx 已是 0-based）。

    优先级（高→低）：
      (a) 真实状态里的字段（容错探测 has_target / target_type / targets 等）；
      (b) 本地卡牌知识（models.card_data）：ATTACK 且非 AOE → 需要；
          SKILL / POWER / STATUS / CURSE → 不需要；
      (c) 判定不出 → 默认需要目标（保持现状，靠 main.py 的 error 帧纠错兜底）。
    """
    game = state.get("game_state", {}) or {}
    cs = game.get("combat_state") or game
    hand = cs.get("hand") or []
    if not isinstance(hand, list) or source_idx < 0 or source_idx >= len(hand):
        return True  # (c) 拿不到手牌 → 默认需要（靠纠错）

    card = hand[source_idx]
    if not isinstance(card, dict):
        return True  # (c)

    # (a) 真实状态字段容错探测
    for field in _TARGET_FIELD_CANDIDATES:
        if field in card:
            val = card[field]
            if val is None or val == "":
                continue
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            s = str(val).strip().upper()
            if s in ("", "NONE", "SELF"):
                return False
            if s in ("ENEMY", "MONSTER"):
                return True           # 单体敌人 → 需要目标
            if s in ("ALL_ENEMY", "ALL", "AOES", "AOE"):
                return False          # 群体 → 不需要单体目标
            return True               # 其它非空字符串按「需要」处理
    # targets 字段：敌人列表非空 → 需要目标
    if "targets" in card:
        tg = card.get("targets")
        if isinstance(tg, list):
            return len(tg) > 0
        return bool(tg)

    # (b) 本地卡牌知识
    name = card.get("name") or card.get("id")
    ctype = str(card.get("type") or "").upper()
    if ctype in ("SKILL", "POWER", "STATUS", "CURSE"):
        return False                 # 群体内/自身效应，不需要单体目标
    if ctype == "ATTACK":
        eff = get_card_effects(name) if name else None
        if eff is not None and eff.get("is_aoe"):
            return False             # 群体攻击不需要单体目标
        return True                  # 单体攻击（含未知攻击）→ 需要目标
    if name:
        eff = get_card_effects(name)
        if eff is not None:
            if eff.get("is_attack") and not eff.get("is_aoe"):
                return True
            return False

    # (c) 判定不出 → 默认需要目标（靠 main.py error 帧纠错）
    return True


class CombatBrain:
    """战斗大脑：ISMCTS（K 个确定性化未来 + 按卡牌身份投票）→ PLAY 命令。"""

    def __init__(self, simulation_count=2000, determinizations=10):
        self.sim_count = simulation_count        # 每个平行未来的模拟次数
        self.determinizations = determinizations  # 平行未来数量 K
        self._bc = None
        self.last_votes = None  # 最近一次决策的得票分布（观测/调试用）
        self._hand_fields_logged = False  # 一次性诊断：手牌字段究竟叫什么

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

            # 6. 一次性诊断：首次见到含 hand 的真实战斗状态，打印手牌字段名，
            #    以便下次真机确认 (a) 那条「真实状态字段判定」到底能不能用。
            if not self._hand_fields_logged:
                cs = (state.get("game_state", {}) or {}).get("combat_state") or {}
                hand = cs.get("hand") or []
                if hand and isinstance(hand[0], dict):
                    self._hand_fields_logged = True
                    log.info("[diag] 手牌卡牌字段: %s", list(hand[0].keys()))
                    log.info("[diag] 首张手牌完整内容: %s", hand[0])

            # 1. 一次重建 bc0
            bc0 = sts.BattleContext()
            _clean = _sanitize(state)
            # F1（2026-09-11）：Weakened→Weak 等——否则玩家虚弱被静默丢弃，
            # MCTS 的伤害计算全部多算 25%（见 models/power_alias.py）。
            alias_power_ids(_clean)
            bc0.init_from_json(gc, json.dumps(_clean))
            self._bc = bc0

            # 2. randomize_rng 不可用 → v1 单次搜索（绑定未编译时的降级）
            if not self._supports_determinization(bc0):
                log.info("randomize_rng 不可用，降级 v1 单次搜索")
                return self._v1_search(bc0, searcher_cls, state)

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

    def _ismcts_decide(self, state: dict, gc, bc0, searcher_cls, sims=None):
        base_seed = self._base_seed(state, gc)
        _sims = sims or self.sim_count
        votes = OrderedDict()   # key -> 得票数（保序，平票取先出现者）
        winners = {}            # key -> 该 key 首次出现的 action
        for k in range(self.determinizations):
            bc_k = sts.BattleContext(bc0)          # 拷贝构造
            bc_k.randomize_rng(base_seed + k)      # 重洗抽牌堆 + 重设 RNG
            action = searcher_cls(bc_k, _sims).best_action
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
        return self._to_comm(winners[winner_key], state)

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

    def _v1_search(self, bc0, searcher_cls, state=None, sims=None):
        action = searcher_cls(bc0, sims or self.sim_count).best_action
        if callable(action):
            action = action()
        if action is None:
            log.warning("MCTS 未返回动作，回退规则")
            return None
        return self._to_comm(action, state)

    # ---- 动作翻译 ----

    @staticmethod
    def _to_comm(action, state=None):
        """把 MCTS 动作翻译成 comm 命令；无法翻译返回 None（走规则）。

        CARD 动作：仅当该牌确实需要目标时才追加 target_idx，否则整条命令会被
        Communication Mod 拒绝（"Selected card cannot be played with the selected
        target"）并导致游戏不执行任何操作 → 死循环。目标判定见 _card_needs_target。
        """
        action_type = getattr(action, "action_type", None)

        if action_type == "CARD":
            cmd = f"PLAY {action.source_idx + 1}"
            if _card_needs_target(state or {}, action.source_idx):
                # 翻译 MCTS 选定的目标：模拟器的 target 是"活怪序列"下标
                # （死怪在灌入时被过滤），必须映射回真机完整数组下标。
                # 旧版丢弃模拟器的目标、永远发"真机第一个活怪"——导致所有
                # 攻击都打最左边的怪（用户观察："Boss 进斩杀线也优先打爪牙"）。
                target_idx = _translate_target(state or {},
                                               getattr(action, "target_idx", None))
                if target_idx is None:
                    log.warning("怪物列表里没有活着的目标，放弃该 CARD 动作（交调用方走规则）")
                    return None
                if target_idx == "unknown":
                    # 没有战斗信息：退回保守行为（沿用模拟器给的 target_idx）
                    target_idx = getattr(action, "target_idx", 0)
                    if target_idx is None or target_idx < 0:
                        target_idx = 0
                cmd += f" {target_idx}"
            return cmd

        if action_type == "END_TURN":
            return "END"

        if action_type == "POTION":
            # F3（2026-09-11）：MCTS 经常给药水投票（2-3 票）却从未被执行。
            # Comm Mod 官方语法：POTION Use|Discard PotionSlot [TargetIndex]
            # 插槽按 0-based（与游戏 potions 数组一致）；用真实状态校验 can_use，
            # 避免喝空槽/错槽。
            # 2026-09-12：补上 requires_target 药水的翻译——此前直接放弃，
            # 导致真机上"MCTS 投票最高的药水"永远喝不到（只能回退规则）。
            slot = getattr(action, "source_idx", None)
            if slot is None or int(slot) < 0:
                log.warning("POTION 动作缺 source_idx，回退规则")
                return None
            slot = int(slot)
            p = None
            try:
                pots = ((state or {}).get("game_state") or {}).get("potions") or []
                if slot < len(pots):
                    p = pots[slot] or {}
            except Exception:
                p = None
            if isinstance(p, dict):
                if not p.get("can_use"):
                    log.warning("POTION 槽 %s 不可用（can_use=false），回退规则", slot)
                    return None
                if p.get("requires_target"):
                    # 模拟器的 target_idx 是"活怪序列"下标，须映射回真机完整
                    # 数组下标（与 CARD 动作同一套翻译）。
                    target_idx = _translate_target(
                        state or {}, getattr(action, "target_idx", None))
                    if target_idx is None:
                        log.warning("POTION %s 需要目标但场上无活怪，回退规则",
                                    p.get("id"))
                        return None
                    if target_idx == "unknown":
                        target_idx = getattr(action, "target_idx", 0)
                        if not isinstance(target_idx, int) or target_idx < 0:
                            target_idx = 0
                    log.info("MCTS 选择药水：POTION Use %d %d（%s，需目标）",
                             slot, target_idx, p.get("id"))
                    return f"POTION Use {slot} {target_idx}"
            log.info("MCTS 选择药水：POTION Use %d", slot)
            return f"POTION Use {slot}"

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
