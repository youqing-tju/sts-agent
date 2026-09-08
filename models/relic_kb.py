"""遗物知识库（Relic Knowledge Base）。

把"遗物候选"编码成 **32 维**语义特征向量，供 candidate_encoder 拼接进
367 维候选描述向量（team-lead 稍后接入；本文件只提供纯查询接口）。

数据来源：《杀戮尖塔-铁甲战士-深度调研报告》第五章"铁甲战士专属遗物（11 件）"
+ 通用/Boss 遗物（报告 1.3 / 6.x 提及）+ Slay the Spire Wiki。

======================================================================
⚠️ 重要接入说明（probe 结论，见 commit message / 项目报告）
----------------------------------------------------------------------
本作 C++ 绑定的 `GameAction.describe(gc)` 对**所有**候选都返回空串，遗物
候选的 label 是 `action_0x<bits>`（只编码 reward_type + 槽位序号，**不含
遗物名字**）。已持有遗物可用 `gc.relics` 拿到（RelicId 枚举名，如
"BURNING_BLOOD"），但**当前奖励/商店里的具体遗物名绑定未暴露**。

调用方必须自行解析出遗物名再传给 `get_relic_features(name)`。key 经
`_norm_relic` 归一化（小写 / 空格与连字符→下划线 / 去非字母数字），同时
兼容枚举名（"BURNING_BLOOD"）与显示名（"Burning Blood"）两种写法。
本文件内置 `ALL_RELIC_IDS`（来自 `sts.RelicId` 全量 180 个有效枚举名），
对所有有效枚举名都给出至少 baseline 画像，故"未收录"仅指完全不在枚举里的
脏名（返回 32 个 0）。
======================================================================

32 维布局（顺序敏感，勿改）：
  [0,  5)  五流派贡献 [力量, 格挡, 消耗, 自伤, 状态]（0~1）
            ——顺序与 models.card_data.CARD_ARCHETYPE 完全一致：
              [strength, block, exhaust, self_damage, status]
  [5, 11)  稀有度/来源 one-hot：
            5 COMMON / 6 UNCOMMON / 7 RARE / 8 BOSS / 9 SHOP / 10 EVENT
  [11,12)  是否铁甲战士专属（ironclad_exclusive，0/1）
  [12,13)  是否有负面代价（has_drawback，0/1）
  [13,18)  触发时机 one-hot：
            13 COMBAT_START 战斗开始 / 14 PER_TURN 每回合 /
            15 ON_HP_LOSS 失血时 / 16 ON_EXHAUST 消耗时 / 17 IMMEDIATE 立即/被动
  [18,32)  细分数值分档（均 0~1）：
            18 power       综合数值强度
            19 strength    力量增益      20 block      格挡/板甲
            21 energy      能量增益      22 draw       抽牌
            23 heal        治疗          24 damage     直接伤害
            25 vuln_weak   易伤/虚弱施加 26 exhaust    消耗流支持
            27 status      状态流支持    28 hp_loss    自伤量
            29 rarity_score 稀有度标量（COMMON0/UNCOMMON0.5/RARE1/BOSS1/SHOP0.5/EVENT0.3）
            30 is_boss     是否 Boss 遗物（0/1）
            31 ironclad_synergy 与该角色构筑契合度（0~1，来自报告）
"""
import re

RELIC_FEATURE_DIM = 32

# 流派顺序——与 models.card_data.CARD_ARCHETYPE 严格一致（锁死测试见 test_relic_kb.py）
RELIC_ARCHETYPE_ORDER = ["strength", "block", "exhaust", "self_damage", "status"]

RELIC_SOURCE = ["COMMON", "UNCOMMON", "RARE", "BOSS", "SHOP", "EVENT"]
RELIC_TRIGGER = ["COMBAT_START", "PER_TURN", "ON_HP_LOSS", "ON_EXHAUST", "IMMEDIATE"]

# 细分数值字段名（顺序即 [18,32) 的下标偏移）
_RELIC_NUMERIC_FIELDS = [
    "strength", "block", "energy", "draw", "heal", "damage",
    "vuln_weak", "exhaust", "status", "hp_loss",
]

# 全量有效 RelicId 枚举名（来自 sts.RelicId，已剔除末尾的 INVALID 哨兵）。
# 用于对所有合法遗物名兜底 baseline，确保"覆盖全表"。
ALL_RELIC_IDS = [
    "AKABEKO", "ART_OF_WAR", "BIRD_FACED_URN", "BLOODY_IDOL", "BLUE_CANDLE",
    "BRIMSTONE", "CALIPERS", "CAPTAINS_WHEEL", "CENTENNIAL_PUZZLE", "CERAMIC_FISH",
    "CHAMPION_BELT", "CHARONS_ASHES", "CHEMICAL_X", "CLOAK_CLASP", "DARKSTONE_PERIAPT",
    "DEAD_BRANCH", "DUALITY", "ECTOPLASM", "EMOTION_CHIP", "FROZEN_CORE", "FROZEN_EYE",
    "GAMBLING_CHIP", "GINGER", "GOLDEN_EYE", "GREMLIN_HORN", "HAND_DRILL", "HAPPY_FLOWER",
    "HORN_CLEAT", "HOVERING_KITE", "ICE_CREAM", "INCENSE_BURNER", "INK_BOTTLE", "INSERTER",
    "KUNAI", "LETTER_OPENER", "LIZARD_TAIL", "MAGIC_FLOWER", "MARK_OF_THE_BLOOM", "MEDICAL_KIT",
    "MELANGE", "MERCURY_HOURGLASS", "MUMMIFIED_HAND", "NECRONOMICON", "NILRYS_CODEX", "NUNCHAKU",
    "ODD_MUSHROOM", "OMAMORI", "ORANGE_PELLETS", "ORICHALCUM", "ORNAMENTAL_FAN", "PAPER_KRANE",
    "PAPER_PHROG", "PEN_NIB", "PHILOSOPHERS_STONE", "POCKETWATCH", "RED_SKULL", "RUNIC_CUBE",
    "RUNIC_DOME", "RUNIC_PYRAMID", "SACRED_BARK", "SELF_FORMING_CLAY", "SHURIKEN", "SNECKO_EYE",
    "SNECKO_SKULL", "SOZU", "STONE_CALENDAR", "STRANGE_SPOON", "STRIKE_DUMMY", "SUNDIAL",
    "THE_ABACUS", "THE_BOOT", "THE_SPECIMEN", "TINGSHA", "TOOLBOX", "TORII", "TOUGH_BANDAGES",
    "TOY_ORNITHOPTER", "TUNGSTEN_ROD", "TURNIP", "TWISTED_FUNNEL", "UNCEASING_TOP", "VELVET_CHOKER",
    "VIOLET_LOTUS", "WARPED_TONGS", "WRIST_BLADE", "BLACK_BLOOD", "BURNING_BLOOD", "MEAT_ON_THE_BONE",
    "FACE_OF_CLERIC", "ANCHOR", "ANCIENT_TEA_SET", "BAG_OF_MARBLES", "BAG_OF_PREPARATION",
    "BLOOD_VIAL", "BOTTLED_FLAME", "BOTTLED_LIGHTNING", "BOTTLED_TORNADO", "BRONZE_SCALES",
    "BUSTED_CROWN", "CLOCKWORK_SOUVENIR", "COFFEE_DRIPPER", "CRACKED_CORE", "CURSED_KEY", "DAMARU",
    "DATA_DISK", "DU_VU_DOLL", "ENCHIRIDION", "FOSSILIZED_HELIX", "FUSION_HAMMER", "GIRYA",
    "GOLD_PLATED_CABLES", "GREMLIN_VISAGE", "HOLY_WATER", "LANTERN", "MARK_OF_PAIN", "MUTAGENIC_STRENGTH",
    "NEOWS_LAMENT", "NINJA_SCROLL", "NUCLEAR_BATTERY", "ODDLY_SMOOTH_STONE", "PANTOGRAPH", "PRESERVED_INSECT",
    "PURE_WATER", "RED_MASK", "RING_OF_THE_SERPENT", "RING_OF_THE_SNAKE", "RUNIC_CAPACITOR",
    "SLAVERS_COLLAR", "SLING_OF_COURAGE", "SYMBIOTIC_VIRUS", "TEARDROP_LOCKET", "THREAD_AND_NEEDLE",
    "VAJRA", "ASTROLABE", "BLACK_STAR", "CALLING_BELL", "CAULDRON", "CULTIST_HEADPIECE", "DOLLYS_MIRROR",
    "DREAM_CATCHER", "EMPTY_CAGE", "ETERNAL_FEATHER", "FROZEN_EGG", "GOLDEN_IDOL", "JUZU_BRACELET",
    "LEES_WAFFLE", "MANGO", "MATRYOSHKA", "MAW_BANK", "MEAL_TICKET", "MEMBERSHIP_CARD", "MOLTEN_EGG",
    "NLOTHS_GIFT", "NLOTHS_HUNGRY_FACE", "OLD_COIN", "ORRERY", "PANDORAS_BOX", "PEACE_PIPE", "PEAR",
    "POTION_BELT", "PRAYER_WHEEL", "PRISMATIC_SHARD", "QUESTION_CARD", "REGAL_PILLOW", "SSSERPENT_HEAD",
    "SHOVEL", "SINGING_BOWL", "SMILING_MASK", "SPIRIT_POOP", "STRAWBERRY", "THE_COURIER", "TINY_CHEST",
    "TINY_HOUSE", "TOXIC_EGG", "WAR_PAINT", "WHETSTONE", "WHITE_BEAST_STATUE", "WING_BOOTS", "CIRCLET",
    "RED_CIRCLET",
]


def _norm_relic(name: str) -> str:
    """遗物名 → 归一化 key：小写；撇号直接删除；其余非字母数字（空格/连字符/下划线）
    统一折叠为单个下划线。

    同时兼容枚举名（"BURNING_BLOOD"）与显示名（"Burning Blood"）→ "burning_blood"，
    对大小写、空格、连字符、下划线变体都鲁棒。
    """
    n = (name or "").strip().lower()
    n = n.replace("'", "")
    n = re.sub(r"[^a-z0-9]+", "_", n)
    return n.strip("_")


# 策划条目（高置信度）。字段：arch(5) / rs(来源) / ironclad / drawback / trigger /
# power / 10 个细分数值 / rarity_score / is_boss / synergy。
# arch 与 RELIC_ARCHETYPE_ORDER 对齐：[力量,格挡,消耗,自伤,状态]
_RELIC_CURATED = {
    # ===== 铁甲战士专属 11 件（报告 5，高置信）=====
    "Burning Blood": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=1, drawback=0, trigger=None,
        power=0.30, heal=0.30, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Black Blood": dict(arch=[0.8,0,0,0,0], rs="BOSS", ironclad=1, drawback=0, trigger="COMBAT_START",
        power=0.80, strength=0.80, heal=0.60, rarity_score=1.0, is_boss=1, synergy=0.90),
    "Red Skull": dict(arch=[0.7,0,0,0,0], rs="COMMON", ironclad=1, drawback=0, trigger="PER_TURN",
        power=0.60, strength=0.70, rarity_score=0.0, is_boss=0, synergy=0.80),
    "Paper Phrog": dict(arch=[0,0,0,0,0], rs="UNCOMMON", ironclad=1, drawback=0, trigger=None,
        power=0.40, vuln_weak=0.60, rarity_score=0.5, is_boss=0, synergy=0.70),
    "Self-Forming Clay": dict(arch=[0,0.4,0,0.7,0], rs="UNCOMMON", ironclad=1, drawback=0, trigger="ON_HP_LOSS",
        power=0.60, block=0.50, hp_loss=0.50, rarity_score=0.5, is_boss=0, synergy=0.80),
    "Champion Belt": dict(arch=[0,0,0,0,0], rs="RARE", ironclad=1, drawback=0, trigger="IMMEDIATE",
        power=0.35, vuln_weak=0.50, rarity_score=1.0, is_boss=0, synergy=0.60),
    "Charon's Ashes": dict(arch=[0,0,0.9,0,0], rs="RARE", ironclad=1, drawback=0, trigger="ON_EXHAUST",
        power=0.80, damage=0.50, exhaust=0.90, rarity_score=1.0, is_boss=0, synergy=0.85),
    "Magic Flower": dict(arch=[0,0,0,0,0], rs="RARE", ironclad=1, drawback=0, trigger=None,
        power=0.30, heal=0.40, rarity_score=1.0, is_boss=0, synergy=0.60),
    "Mark of Pain": dict(arch=[0,0,0,0,0.8], rs="BOSS", ironclad=1, drawback=0, trigger="PER_TURN",
        power=0.70, status=0.70, energy=0.50, rarity_score=1.0, is_boss=1, synergy=0.85),
    "Runic Cube": dict(arch=[0,0,0,0.8,0], rs="BOSS", ironclad=1, drawback=0, trigger="ON_HP_LOSS",
        power=0.80, draw=0.60, hp_loss=0.60, rarity_score=1.0, is_boss=1, synergy=0.85),
    "Brimstone": dict(arch=[0.8,0,0,0,0], rs="SHOP", ironclad=1, drawback=1, trigger="PER_TURN",
        power=0.70, strength=0.80, rarity_score=0.5, is_boss=0, synergy=0.85),

    # ===== 通用 / Boss 遗物（铁甲常用，报告 1.3/6.x 提及，高-中置信）=====
    "Anchor": dict(arch=[0,0.6,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.60, block=0.60, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Bag of Marbles": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.40, vuln_weak=0.50, rarity_score=0.0, is_boss=0, synergy=0.60),
    "Potion Belt": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.30, rarity_score=0.0, is_boss=0, synergy=0.50),
    "White Beast Statue": dict(arch=[0,0,0,0,0], rs="SHOP", ironclad=0, drawback=0, trigger=None,
        power=0.30, heal=0.30, rarity_score=0.5, is_boss=0, synergy=0.50),
    "Girya": dict(arch=[0.5,0,0,0,0], rs="SHOP", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.60, strength=0.50, rarity_score=0.5, is_boss=0, synergy=0.80),
    "Bronze Scales": dict(arch=[0,0.4,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="ON_HP_LOSS",
        power=0.40, block=0.40, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Vajra": dict(arch=[0.4,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.50, strength=0.40, rarity_score=0.0, is_boss=0, synergy=0.70),
    "Pen Nib": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.30, damage=0.30, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Kunai": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.40, energy=0.40, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Happy Flower": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="PER_TURN",
        power=0.40, energy=0.40, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Odd Mushroom": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.30, vuln_weak=0.20, rarity_score=0.0, is_boss=0, synergy=0.40),
    "Bag of Preparation": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.30, draw=0.30, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Blood Vial": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.30, heal=0.20, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Bottled Flame": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.40, strength=0.30, rarity_score=0.0, is_boss=0, synergy=0.60),
    "Bottled Lightning": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.40, energy=0.40, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Bottled Tornado": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.30, block=0.30, rarity_score=0.0, is_boss=0, synergy=0.40),
    "Violet Lotus": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.40, energy=0.40, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Ancient Tea Set": dict(arch=[0,0,0,0,0], rs="SHOP", ironclad=0, drawback=0, trigger=None,
        power=0.30, heal=0.30, rarity_score=0.5, is_boss=0, synergy=0.40),
    "Art of War": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.40, energy=0.40, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Akabeko": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.30, damage=0.30, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Bird-Faced Urn": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="PER_TURN",
        power=0.40, strength=0.40, rarity_score=0.0, is_boss=0, synergy=0.60),
    "Centennial Puzzle": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.30, energy=0.30, rarity_score=0.0, is_boss=0, synergy=0.40),
    "Ceramic Fish": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.40, draw=0.30, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Meat on the Bone": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="ON_HP_LOSS",
        power=0.50, heal=0.50, rarity_score=0.0, is_boss=0, synergy=0.50),
    "Strawberry": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
        power=0.30, heal=0.20, rarity_score=0.0, is_boss=0, synergy=0.40),
    "Matryoshka": dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger="COMBAT_START",
        power=0.30, heal=0.20, rarity_score=0.0, is_boss=0, synergy=0.40),

    # ===== Boss 遗物（BOSS_RELIC 候选常见；能量类多数带副作用，报告 1.3）=====
    "Coffee Dripper": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.50, energy=0.60, rarity_score=1.0, is_boss=1, synergy=0.40),
    "Cursed Key": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.15, rarity_score=1.0, is_boss=1, synergy=0.20),
    "Philosopher's Stone": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.40, energy=0.60, rarity_score=1.0, is_boss=1, synergy=0.40),
    "Ectoplasm": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.40, energy=0.60, rarity_score=1.0, is_boss=1, synergy=0.40),
    "Sozu": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.40, energy=0.50, rarity_score=1.0, is_boss=1, synergy=0.35),
    "Velvet Choker": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.40, energy=0.50, rarity_score=1.0, is_boss=1, synergy=0.40),
    "Runic Dome": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.40, energy=0.50, rarity_score=1.0, is_boss=1, synergy=0.40),
    "Fusion Hammer": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.30, rarity_score=1.0, is_boss=1, synergy=0.35),
    "Cultist Headpiece": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=0, trigger="PER_TURN",
        power=0.60, strength=0.50, rarity_score=1.0, is_boss=1, synergy=0.70),
    "Calling Bell": dict(arch=[0,0,0,0,0], rs="BOSS", ironclad=0, drawback=1, trigger=None,
        power=0.25, rarity_score=1.0, is_boss=1, synergy=0.30),
}


def _baseline_relic() -> dict:
    """长尾遗物的保守基准画像：低信息量、非负但几乎全 0（仅 power=0.2 表示"已收录"）。"""
    return dict(arch=[0,0,0,0,0], rs="COMMON", ironclad=0, drawback=0, trigger=None,
                power=0.20, rarity_score=0.0, is_boss=0, synergy=0.20)


def _build_relic_kb():
    kb = {}
    # 先放策划条目（用显示名/枚举名两种写法都索引，提升鲁棒性）
    for disp_name, raw in _RELIC_CURATED.items():
        kb[_norm_relic(disp_name)] = raw
        kb[_norm_relic(disp_name.upper().replace(" ", "_"))] = raw  # 枚举名别名
    # 全量枚举名兜底 baseline（已收录的会被策划条目覆盖）
    for rid in ALL_RELIC_IDS:
        key = _norm_relic(rid)
        if key not in kb:
            kb[key] = _baseline_relic()
    return kb


# 归一化 key → 条目；调用方可读 RELIC_KB 看收录规模
RELIC_KB = _build_relic_kb()


def get_relic_features(name: str) -> list:
    """返回 32 维遗物特征向量；完全不在 RelicId 枚举里的脏名返回 32 个 0.0。"""
    entry = RELIC_KB.get(_norm_relic(name))
    if entry is None:
        return [0.0] * RELIC_FEATURE_DIM

    vec = [0.0] * RELIC_FEATURE_DIM
    # 0-4 五流派贡献
    arch = entry.get("arch") or [0,0,0,0,0]
    for i in range(5):
        vec[i] = float(arch[i]) if i < len(arch) else 0.0
    # 5-10 稀有度/来源 one-hot
    rs = entry.get("rs")
    if rs in RELIC_SOURCE:
        vec[5 + RELIC_SOURCE.index(rs)] = 1.0
    # 11 铁甲专属 / 12 负面代价
    vec[11] = 1.0 if entry.get("ironclad") else 0.0
    vec[12] = 1.0 if entry.get("drawback") else 0.0
    # 13-17 触发时机 one-hot
    tr = entry.get("trigger")
    if tr in RELIC_TRIGGER:
        vec[13 + RELIC_TRIGGER.index(tr)] = 1.0
    # 18 综合强度
    vec[18] = float(entry.get("power", 0.0))
    # 19-28 细分数值
    for off, fld in enumerate(_RELIC_NUMERIC_FIELDS):
        v = entry.get(fld)
        if v is not None:
            vec[19 + off] = float(v)
    # 29 稀有度标量 / 30 is_boss / 31 铁甲契合度
    vec[29] = float(entry.get("rarity_score", 0.0))
    vec[30] = 1.0 if entry.get("is_boss") else 0.0
    vec[31] = float(entry.get("synergy", 0.0))
    return vec
