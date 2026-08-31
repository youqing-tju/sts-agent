"""规则 baseline policy v1（对标 bottled_ai 思路的最小规则集）。

label 是 lightspeed GameAction.describe() 的可读文本（如 "Monster" / "Elite" /
"Rest Site" / "Strike"），事件类动作可能回退为 action_0x<bits>，匹配一律小写
关键词子串，尽量鲁棒。
"""
from .base_policy import Policy

# 血量阈值
_HP_HIGH = 0.7   # 高于此比例才考虑打精英
_HP_LOW = 0.4    # 低于此比例走生存向（篝火/睡觉）

# 卡组臃肿阈值：达到即跳过卡牌奖励
_DECK_MAX = 15

# 拿牌白名单关键词（攻击牌/优质能力牌；v1 粗粒度，靠日志迭代）
_CARD_TAKE_HINTS = ("attack", "strike", "inflame", "metallicize", "demon form",
                    "barricade", "corruption", "offering", "battle trance",
                    "shrug it off", "pommel", "twin strike", "clothesline",
                    "iron wave", "anger", "carnage", "uppercut", "whirlwind")


def _hp_ratio(obs) -> float:
    return obs.hp / obs.max_hp if obs.max_hp else 1.0


def _find(candidates: list, *keywords: str):
    """返回第一个 label 含任一关键词（小写）的候选，无则 None。"""
    for c in candidates:
        label = c.label.lower()
        if any(k in label for k in keywords):
            return c
    return None


def _skip(candidates: list):
    """返回 SKIP 候选（kind=="SKIP" 或 label 为 SKIP），无则 None。"""
    for c in candidates:
        if c.kind == "SKIP" or c.label.strip().upper() == "SKIP":
            return c
    return None


class RulePolicy(Policy):
    def choose(self, obs, candidates: list) -> int:
        if not candidates:
            raise ValueError("RulePolicy.choose: candidates 为空")
        kinds = {c.kind for c in candidates}
        if "MAP_NODE" in kinds:
            chosen = self._choose_map(obs, candidates)
        elif "REST_OPTION" in kinds:
            chosen = self._choose_rest(obs, candidates)
        elif "SHOP_ITEM" in kinds:
            chosen = _skip(candidates)          # v1 保守：商店直接跳过
        elif "CARD_REWARD" in kinds or "REWARD" in kinds:
            chosen = self._choose_card_reward(obs, candidates)
        elif "EVENT_OPTION" in kinds:
            chosen = candidates[0]              # v1 选第一个
        else:
            chosen = candidates[0]              # 无法判断 → idx=0 方向
        return chosen.idx

    # ---- 各场景规则 ----

    def _choose_map(self, obs, candidates: list):
        hp = _hp_ratio(obs)
        if hp < _HP_LOW:
            rest = _find(candidates, "rest", "bonfire", "fire")
            if rest is not None:
                return rest                     # 低血且有篝火方向 → 生存向
        if hp > _HP_HIGH:
            elite = _find(candidates, "elite")
            if elite is not None:
                return elite                    # 高血优先精英
        unknown = _find(candidates, "unknown", "?")
        if unknown is not None:
            return unknown                      # 否则优先问号
        monster = _find(candidates, "monster", "enemy")
        if monster is not None:
            return monster                      # 再小怪
        return candidates[0]

    def _choose_card_reward(self, obs, candidates: list):
        if len(obs.deck) >= _DECK_MAX:
            skip = _skip(candidates)
            if skip is not None:
                return skip                     # 卡组臃肿 → 跳过
            return candidates[0]
        non_skip = [c for c in candidates
                    if c.kind != "SKIP" and c.label.strip().upper() != "SKIP"]
        for c in non_skip:
            if any(k in c.label.lower() for k in _CARD_TAKE_HINTS):
                return c                        # 白名单攻击牌/能力牌
        return non_skip[0] if non_skip else candidates[0]

    def _choose_rest(self, obs, candidates: list):
        if _hp_ratio(obs) < _HP_LOW:
            rest = _find(candidates, "rest")
            if rest is not None:
                return rest                     # 低血睡觉
        smith = _find(candidates, "smith", "upgrade")
        if smith is not None:
            return smith                        # 否则升牌
        return candidates[0]
