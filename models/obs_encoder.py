"""观测增强：把 C++ 模拟器给的 412 维观测扩展到 468 维。

412 维布局（来自 bindings-util.cpp:40-67）：
    0          当前 HP
    1          最大 HP
    2          金币
    3          楼层号
    4-13       10  Boss one-hot
    14-233     220 牌组卡牌计数（110 种卡 × {未升级, 已升级}，每维张数）
    234-411    178 遗物 one-hot（按 RelicId 序号）

在本文件之后追加 56 维：
    段1  24  持有药水种类 one-hot（24 种常见药水）
    段2   1  药水数量 /5
    段3   4  地图摘要：前方 6 层 ELITE/REST/SHOP/OTHER 计数（各 /6）  —— 当前 API 拿不到，永久退化全零
    段4   5  牌组流派强度（力量/格挡/消耗/自伤/状态，raw/total 归一化到 [0,1]）
    段5   8  牌组派生统计（总牌数/40、攻击占比、技能占比、能力数/10、平均费用/3、消耗牌数/10、状态牌数/10、升级数/20）
    段6   4  Act one-hot（Act 1-4）
    段7  10  当前房间类型 one-hot（MONSTER/ELITE/REST/SHOP/TREASURE/EVENT/BOSS/BOSS_TREASURE/NONE/INVALID）

依赖注入：card_index / arch_table / card_effects 都是可选参数，由别的 worker
并行产出。缺失时对应段安静填 0，绝不抛异常。

CARD_INDEX（长度 110）的确定方式见文件底部注释：从 Cards.h 的 cardColors /
cardEnumStrings 解析出 RED 卡（75 张，按 CardId 枚举序），再追加
srcColorlessCardPool（35 张，按枚举名排序），并用运行时观测实证核对
（STRIKE_RED→66、DEFEND_RED→19、BASH→3）。
"""

from typing import Dict, List, Optional, Sequence

# --------------------------------------------------------------------------
# 维度常量
# --------------------------------------------------------------------------
OBS_BASE_DIM = 412
OBS_EXTRA_DIM = 56
OBS_DIM = 468

# encode_context 维度分解：24 药水 + 1 数量 + 4 地图 + 4 Act + 10 房间 = 43
CTX_POTION_DIM = 24
CTX_COUNT_DIM = 1
CTX_MAP_DIM = 4
CTX_ACT_DIM = 4
CTX_ROOM_DIM = 10
CTX_DIM = CTX_POTION_DIM + CTX_COUNT_DIM + CTX_MAP_DIM + CTX_ACT_DIM + CTX_ROOM_DIM  # 43

# encode_deck_stats 维度分解：5 流派强度 + 8 牌组统计 = 13
DECK_ARCH_DIM = 5
DECK_STATS_DIM = 8
DECK_DIM = DECK_ARCH_DIM + DECK_STATS_DIM  # 13

assert CTX_DIM + DECK_DIM == OBS_EXTRA_DIM


# --------------------------------------------------------------------------
# 卡序（obs[14 + i*2] / obs[14 + i*2 + 1]）
# --------------------------------------------------------------------------
# 与 C++ createOneHotCardEncodingMap（bindings-util.cpp:91）的卡序严格一致：
#   1) 从 CardId::INVALID 到 CardId::ZAP 中 color==RED 的卡，按枚举序
#   2) srcColorlessCardPool（35 张），按 getCardEnumName 字符串升序
CARD_INDEX: List[str] = [
    # --- 75 张 RED 卡（按 CardId 枚举序） ---
    "ANGER", "ARMAMENTS", "BARRICADE", "BASH", "BATTLE_TRANCE", "BERSERK",
    "BLOODLETTING", "BLOOD_FOR_BLOOD", "BLUDGEON", "BODY_SLAM", "BULLET_TIME",
    "BURNING_PACT", "CARNAGE", "CLASH", "CLEAVE", "CLOTHESLINE", "CONCENTRATE",
    "CORRUPTION", "DARK_EMBRACE", "DEFEND_RED", "DEMON_FORM", "DISARM",
    "DOUBLE_TAP", "DROPKICK", "DUAL_WIELD", "ENTRENCH", "EVOLVE", "EXHUME",
    "FEED", "FEEL_NO_PAIN", "FIEND_FIRE", "FIRE_BREATHING", "FLAME_BARRIER",
    "FLEX", "GHOSTLY_ARMOR", "HAVOC", "HEADBUTT", "HEAVY_BLADE", "HEMOKINESIS",
    "IMMOLATE", "IMPERVIOUS", "INFERNAL_BLADE", "INFLAME", "INTIMIDATE",
    "IRON_WAVE", "JUGGERNAUT", "LIMIT_BREAK", "METALLICIZE", "OFFERING",
    "PERFECTED_STRIKE", "POMMEL_STRIKE", "POWER_THROUGH", "PUMMEL", "RAGE",
    "RAMPAGE", "REAPER", "RECKLESS_CHARGE", "RUPTURE", "SEARING_BLOW",
    "SECOND_WIND", "SEEING_RED", "SENTINEL", "SEVER_SOUL", "SHOCKWAVE",
    "SHRUG_IT_OFF", "SPOT_WEAKNESS", "STRIKE_RED", "SWORD_BOOMERANG",
    "THUNDERCLAP", "TRUE_GRIT", "TWIN_STRIKE", "UPPERCUT", "WARCRY",
    "WHIRLWIND", "WILD_STRIKE",
    # --- 35 张 colorless（按枚举名升序） ---
    "APOTHEOSIS", "BANDAGE_UP", "BLIND", "CHRYSALIS", "DARK_SHACKLES",
    "DEEP_BREATH", "DISCOVERY", "DRAMATIC_ENTRANCE", "ENLIGHTENMENT", "FINESSE",
    "FLASH_OF_STEEL", "FORETHOUGHT", "GOOD_INSTINCTS", "HAND_OF_GREED",
    "IMPATIENCE", "JACK_OF_ALL_TRADES", "MADNESS", "MAGNETISM", "MASTER_OF_STRATEGY",
    "MAYHEM", "METAMORPHOSIS", "MIND_BLAST", "PANACEA", "PANACHE", "PANIC_BUTTON",
    "PURITY", "SADISTIC_NATURE", "SECRET_TECHNIQUE", "SECRET_WEAPON",
    "SWIFT_STRIKE", "THE_BOMB", "THINKING_AHEAD", "TRANSMUTATION", "TRIP",
    "VIOLENCE",
]

NUM_CARDS = len(CARD_INDEX)  # 110
assert NUM_CARDS == 110, NUM_CARDS
assert len(set(CARD_INDEX)) == NUM_CARDS, "CARD_INDEX 存在重复"

# 牌组计数段在 obs 中的基址与步长
DECK_OFFSET = 14
DECK_STRIDE = 2


def _deck_count(obs: Sequence, i: int) -> int:
    """第 i 张卡（CARD_INDEX[i]）在 obs 中的总张数（未升级 + 已升级）。"""
    base = DECK_OFFSET + i * DECK_STRIDE
    if base + 1 >= len(obs):
        return 0
    return int(obs[base]) + int(obs[base + 1])


# --------------------------------------------------------------------------
# 药水
# --------------------------------------------------------------------------
# 完整的 Potion 枚举名（来自 Potions.h 的 potionEnumNames），下标即
# gc.potion_slots 返回的整数值。
POTION_ENUM_NAMES: List[str] = [
    "INVALID", "EMPTY_POTION_SLOT", "AMBROSIA", "ANCIENT_POTION", "ATTACK_POTION",
    "BLESSING_OF_THE_FORGE", "BLOCK_POTION", "BLOOD_POTION", "BOTTLED_MIRACLE",
    "COLORLESS_POTION", "CULTIST_POTION", "CUNNING_POTION", "DEXTERITY_POTION",
    "DISTILLED_CHAOS", "DUPLICATION_POTION", "ELIXIR_POTION", "ENERGY_POTION",
    "ENTROPIC_BREW", "ESSENCE_OF_DARKNESS", "ESSENCE_OF_STEEL", "EXPLOSIVE_POTION",
    "FAIRY_POTION", "FEAR_POTION", "FIRE_POTION", "FLEX_POTION", "FOCUS_POTION",
    "FRUIT_JUICE", "GAMBLERS_BREW", "GHOST_IN_A_JAR", "HEART_OF_IRON",
    "LIQUID_BRONZE", "LIQUID_MEMORIES", "POISON_POTION", "POTION_OF_CAPACITY",
    "POWER_POTION", "REGEN_POTION", "SKILL_POTION", "SMOKE_BOMB", "SNECKO_OIL",
    "SPEED_POTION", "STANCE_POTION", "STRENGTH_POTION", "SWIFT_POTION", "WEAK_POTION",
]

# 24 种常见药水（one-hot 固定顺序）
POTION_INDEX: List[str] = [
    "ATTACK_POTION", "BLOCK_POTION", "BLOOD_POTION", "STRENGTH_POTION",
    "DEXTERITY_POTION", "ENERGY_POTION", "FIRE_POTION", "POISON_POTION",
    "WEAK_POTION", "SPEED_POTION", "FLEX_POTION", "FOCUS_POTION",
    "SKILL_POTION", "POWER_POTION", "REGEN_POTION", "SWIFT_POTION",
    "EXPLOSIVE_POTION", "CUNNING_POTION", "DUPLICATION_POTION", "FRUIT_JUICE",
    "LIQUID_MEMORIES", "LIQUID_BRONZE", "SMOKE_BOMB", "ANCIENT_POTION",
]
assert len(POTION_INDEX) == 24
assert len(set(POTION_INDEX)) == 24
_POTION_SET = set(POTION_INDEX)


# --------------------------------------------------------------------------
# 房间 / 流派
# --------------------------------------------------------------------------
# 顺序与任务规定一致
ROOM_INDEX: List[str] = [
    "MONSTER", "ELITE", "REST", "SHOP", "TREASURE",
    "EVENT", "BOSS", "BOSS_TREASURE", "NONE", "INVALID",
]
_ROOM_SET = set(ROOM_INDEX)

# 五流派顺序：力量 / 格挡 / 消耗 / 自伤 / 状态
ARCH_NAMES: List[str] = ["strength", "block", "exhaust", "self_damage", "status"]


# --------------------------------------------------------------------------
# encode_context：返回 43 维（药水24 + 数量1 + 地图4 + Act4 + 房间10）
# --------------------------------------------------------------------------
def _encode_potions(gc) -> List[float]:
    vec = [0.0] * CTX_POTION_DIM
    slots = getattr(gc, "potion_slots", None)
    if not slots:
        return vec
    for v in slots:
        try:
            name = POTION_ENUM_NAMES[int(v)]
        except (IndexError, TypeError, ValueError):
            continue
        if name in _POTION_SET:
            vec[POTION_INDEX.index(name)] = 1.0
    return vec


def _encode_potion_count(gc) -> List[float]:
    slots = getattr(gc, "potion_slots", None)
    n = len(slots) if slots else 0
    return [max(0.0, min(1.0, n / 5.0))]


def _encode_map_summary(gc) -> List[float]:
    # 当前 bindings（slaythespire.cpp）未暴露 gc.map，无法取得“前方 6 层”的
    # 节点类型，因此该段永久为全零。若未来暴露 map 对象，可在此实现
    # ELITE/REST/SHOP/OTHER 计数（各 /6）。详见发送给 team-lead 的报告。
    return [0.0] * CTX_MAP_DIM


def _encode_act(gc) -> List[float]:
    vec = [0.0] * CTX_ACT_DIM
    act = getattr(gc, "act", None)
    if isinstance(act, int) and act > 0:
        idx = 3 if act >= 4 else act - 1
        vec[idx] = 1.0
    return vec


def _encode_room(gc) -> List[float]:
    vec = [0.0] * CTX_ROOM_DIM
    room = getattr(gc, "cur_room", None)
    if room is not None:
        name = getattr(room, "name", None)
        if name in _ROOM_SET:
            vec[ROOM_INDEX.index(name)] = 1.0
    return vec


def encode_context(gc) -> List[float]:
    """从 GameContext 提取上下文特征，返回 43 维。

    gc 为 None 或取不到信息时对应段全零，不抛异常。
    """
    if gc is None:
        return [0.0] * CTX_DIM
    try:
        out = (
            _encode_potions(gc)
            + _encode_potion_count(gc)
            + _encode_map_summary(gc)
            + _encode_act(gc)
            + _encode_room(gc)
        )
        return out
    except Exception:
        # 任何意外都安静降级为全零
        return [0.0] * CTX_DIM


# --------------------------------------------------------------------------
# encode_deck_stats：返回 13 维（流派强度5 + 牌组统计8）
# --------------------------------------------------------------------------
def encode_deck_stats(
    obs412: Sequence,
    card_index: Optional[Sequence[str]] = None,
    arch_table: Optional[Dict[str, Sequence[float]]] = None,
    card_effects: Optional[Dict[str, dict]] = None,
) -> List[float]:
    """从 412 维观测的牌组计数段 [14:234] 计算 13 维。

    card_index:   长度 110 的卡名列表，下标对应 obs 偏移 (14 + i*2)；
                  None 则使用本模块硬编码的 CARD_INDEX。
    arch_table:   卡名 -> [5 维流派权重]；None（或 card_index 为 None）则流派段全零。
    card_effects: 卡名 -> 效果 dict（含 base_cost / is_attack / is_skill /
                  is_power / exhausts_self / is_status）；None 则统计段全零。
    依赖缺失时对应段安静地填 0，绝不抛异常。
    """
    ci: Sequence[str] = list(card_index) if card_index is not None else CARD_INDEX

    # ---- 流派强度（5 维）：raw/total 归一化到 [0,1] ----
    arch = [0.0] * DECK_ARCH_DIM
    if arch_table is not None:
        raw = [0.0] * DECK_ARCH_DIM
        total = 0
        for i in range(min(NUM_CARDS, len(ci))):
            name = ci[i]
            c = _deck_count(obs412, i)
            if c == 0:
                continue
            total += c
            w = arch_table.get(name)
            if not w:
                continue
            for a in range(DECK_ARCH_DIM):
                raw[a] += c * float(w[a])
        if total > 0:
            for a in range(DECK_ARCH_DIM):
                arch[a] = max(0.0, min(1.0, raw[a] / total))

    # ---- 牌组统计（8 维）----
    stats = [0.0] * DECK_STATS_DIM
    if card_effects is not None:
        total = 0
        attack = skill = power = exhaust = status = 0
        cost_sum = 0.0
        upgraded = 0
        for i in range(min(NUM_CARDS, len(ci))):
            name = ci[i]
            c = _deck_count(obs412, i)
            total += c
            upgraded += _upgraded_count(obs412, i)
            eff = card_effects.get(name)
            if eff:
                if eff.get("is_attack"):
                    attack += c
                if eff.get("is_skill"):
                    skill += c
                if eff.get("is_power"):
                    power += c
                if eff.get("exhausts_self"):
                    exhaust += c
                if eff.get("is_status"):
                    status += c
                bc = eff.get("base_cost") or 0
                cost_sum += c * float(bc)
        stats[0] = total / 40.0 if total > 0 else 0.0
        stats[1] = attack / total if total > 0 else 0.0
        stats[2] = skill / total if total > 0 else 0.0
        stats[3] = power / 10.0
        stats[4] = (cost_sum / total) / 3.0 if total > 0 else 0.0
        stats[5] = exhaust / 10.0
        stats[6] = status / 10.0
        stats[7] = upgraded / 20.0
        for k in range(DECK_STATS_DIM):
            stats[k] = max(0.0, min(1.0, stats[k]))

    return arch + stats


def _upgraded_count(obs: Sequence, i: int) -> int:
    base = DECK_OFFSET + i * DECK_STRIDE + 1  # 已升级那一维
    if base >= len(obs):
        return 0
    return int(obs[base])


# --------------------------------------------------------------------------
# enhance_observation：412 -> 468
# --------------------------------------------------------------------------
def enhance_observation(
    obs412: Sequence,
    gc=None,
    card_index: Optional[Sequence[str]] = None,
    arch_table: Optional[Dict[str, Sequence[float]]] = None,
    card_effects: Optional[Dict[str, dict]] = None,
) -> List[float]:
    """返回 468 维：list(obs412) + encode_context(gc) + encode_deck_stats(...)。"""
    base = list(obs412)
    ctx = encode_context(gc)
    deck = encode_deck_stats(
        base,
        card_index=card_index,
        arch_table=arch_table,
        card_effects=card_effects,
    )
    return base + ctx + deck


# --------------------------------------------------------------------------
# CARD_INDEX 确定方式与实证验证（说明用，不执行）
# --------------------------------------------------------------------------
# 方法：读 C++ 头文件 third_party/sts_lightspeed/include/constants/Cards.h
#   - cardEnumStrings[]（下标 == CardId 整数值）给出卡名
#   - cardColors[]       给出每个 CardId 的颜色
#   RED 卡 = cardColors[i]==CardColor::RED 的那些，按 i 升序（即枚举序）。
#   colorless 段 = srcColorlessCardPool（35 张，CardPools.h），按
#   getCardEnumName 字符串升序排序（等价于 Python 枚举成员名排序）。
# 验证：用 .venv 跑运行时 obs，铁甲战士起始牌组为 5 Strike / 4 Defend / 1 Bash，
#   核对 obs[14:234] 的非零位与 CARD_INDEX 下标：
#       STRIKE_RED -> idx 66（obs 偏移 14+66*2=146，计数 5）
#       DEFEND_RED -> idx 19（obs 偏移 52，计数 4）
#       BASH       -> idx 3 （obs 偏移 20，计数 1）
#   三项全部吻合（见 tmp_gen_card_index.py）。
