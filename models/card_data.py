"""铁甲战士（IRONCLAD）卡牌效果字典（基础值，未升级）。

key 为归一化卡名（normalize_card_name：小写 + 去 _red 等颜色后缀），
value 为效果 dict，字段见 EFFECT_KEYS（42 维，与 candidate_encoder 的
card-effects 段一一对应）。数值来自《杀戮尖塔-铁甲战士-深度调研报告》
第三章（基础值，未升级）。

编码归一化（见 candidate_encoder）约定：
- base_damage / base_block 除以 20，base_cost 除以 3（is_x_cost 时 cost 为 None）
- num_hits / draw_count 除以 5
- 布尔字段取 0/1，rarity_score 取 0.0/0.5/1.0（普通/罕见/稀有）
- 缺失字段不写出（稀疏），编码时按 0 处理

另提供五流派归属表 CARD_ARCHETYPE 与查询函数 get_card_archetype：
维度顺序固定为 [力量, 格挡, 消耗, 自伤, 状态]，各维 0.0~1.0。
"""

# 42 维效果字段顺序（与 candidate_encoder 对齐，顺序敏感，勿改）
EFFECT_KEYS = [
    # 基础数值 (6)
    "base_damage", "base_block", "base_cost", "num_hits", "draw_count", "is_x_cost",
    # 铁甲核心资源 (8)
    "energy_gain", "strength_gain", "strength_multiplier", "dexterity_gain",
    "self_damage", "self_damage_payoff", "hp_gain", "exhausts_self",
    # 消耗流 (5)
    "exhausts_others", "on_exhaust_trigger", "exhaust_payoff_scaling",
    "cost_reduction", "recover_from_exhaust",
    # 格挡流 (6)
    "block_multiplier", "block_retain", "block_to_damage",
    "block_on_gain_trigger", "block_per_turn", "block_efficiency",
    # 状态流 (6)
    "injects_status_enemy", "injects_status_self", "status_payoff_draw",
    "status_payoff_damage", "applies_vulnerable", "applies_weak",
    # 牌库操控 (6)
    "discard", "draw_specific", "deck_manipulation", "retain", "ethereal",
    "conditional_play",
    # 结构标记 (5)
    "is_power", "is_aoe", "is_attack", "is_skill", "rarity_score",
]

_COLOR_SUFFIXES = ("_red", "_green", "_blue", "_purple", "_colorless")


def normalize_card_name(name: str) -> str:
    """卡牌 id 名 → 字典 key（小写、去颜色后缀、下划线保留）。"""
    n = (name or "").strip().lower().replace("-", "_").replace(" ", "_")
    for suf in _COLOR_SUFFIXES:
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    return n


# 数值来自调研报告第三章（基础值，未升级）。仅写有值的字段（稀疏）。
CARD_EFFECTS = {
    # ---- 初始卡 (BASIC, rarity 0.0) ----
    "strike": {"base_damage": 6, "base_cost": 1, "is_attack": 1, "rarity_score": 0.0},
    "defend": {"base_block": 5, "base_cost": 1, "is_skill": 1, "rarity_score": 0.0},
    "bash": {"base_damage": 8, "base_cost": 2, "applies_vulnerable": 1,
             "is_attack": 1, "rarity_score": 0.0},

    # ---- 普通 (COMMON, 0.0) ----
    "anger": {"base_damage": 6, "base_cost": 0, "is_attack": 1, "rarity_score": 0.0},
    "armaments": {"base_block": 5, "base_cost": 1, "is_skill": 1, "rarity_score": 0.0},
    "body_slam": {"base_cost": 1, "block_to_damage": 1, "conditional_play": 1,
                  "is_attack": 1, "rarity_score": 0.0},
    "clash": {"base_damage": 14, "base_cost": 0, "conditional_play": 1,
              "is_attack": 1, "rarity_score": 0.0},
    "cleave": {"base_damage": 8, "base_cost": 1, "is_aoe": 1,
               "is_attack": 1, "rarity_score": 0.0},
    "clothesline": {"base_damage": 12, "base_cost": 2, "applies_weak": 1,
                    "is_attack": 1, "rarity_score": 0.0},
    "flex": {"base_cost": 0, "strength_gain": 2, "is_skill": 1, "rarity_score": 0.0},
    "havoc": {"base_cost": 1, "exhausts_others": 1, "is_skill": 1, "rarity_score": 0.0},
    "headbutt": {"base_damage": 9, "base_cost": 1, "deck_manipulation": 1,
                 "is_attack": 1, "rarity_score": 0.0},
    "heavy_blade": {"base_damage": 14, "base_cost": 2, "is_attack": 1, "rarity_score": 0.0},
    "iron_wave": {"base_damage": 5, "base_block": 5, "base_cost": 1,
                  "is_attack": 1, "rarity_score": 0.0},
    "perfected_strike": {"base_damage": 6, "base_cost": 2, "is_attack": 1, "rarity_score": 0.0},
    "pommel_strike": {"base_damage": 9, "base_cost": 1, "draw_count": 1,
                      "is_attack": 1, "rarity_score": 0.0},
    "shrug_it_off": {"base_block": 8, "base_cost": 1, "draw_count": 1,
                     "is_skill": 1, "rarity_score": 0.0},
    "sword_boomerang": {"base_damage": 3, "base_cost": 1, "num_hits": 3,
                        "is_attack": 1, "rarity_score": 0.0},
    "thunderclap": {"base_damage": 4, "base_cost": 1, "is_aoe": 1,
                    "applies_vulnerable": 1, "is_attack": 1, "rarity_score": 0.0},
    "true_grit": {"base_block": 7, "base_cost": 1, "exhausts_others": 1,
                  "is_skill": 1, "rarity_score": 0.0},
    "twin_strike": {"base_damage": 5, "base_cost": 1, "num_hits": 2,
                    "is_attack": 1, "rarity_score": 0.0},
    "warcry": {"base_cost": 0, "draw_count": 1, "deck_manipulation": 1,
               "exhausts_self": 1, "is_skill": 1, "rarity_score": 0.0},
    "wild_strike": {"base_damage": 12, "base_cost": 1, "injects_status_enemy": 1,
                    "is_attack": 1, "rarity_score": 0.0},

    # ---- 罕见 (UNCOMMON, 0.5) ----
    "battle_trance": {"base_cost": 0, "draw_count": 3, "is_skill": 1, "rarity_score": 0.5},
    "blood_for_blood": {"base_damage": 18, "base_cost": 4, "is_attack": 1, "rarity_score": 0.5},
    "bloodletting": {"base_cost": 0, "energy_gain": 2, "self_damage": 1,
                     "is_skill": 1, "rarity_score": 0.5},
    "burning_pact": {"base_cost": 1, "draw_count": 2, "exhausts_others": 1,
                     "is_skill": 1, "rarity_score": 0.5},
    "carnage": {"base_damage": 20, "base_cost": 2, "ethereal": 1,
                "is_attack": 1, "rarity_score": 0.5},
    "combust": {"base_cost": 1, "self_damage": 1, "is_aoe": 1,
                "is_power": 1, "rarity_score": 0.5},
    "dark_embrace": {"base_cost": 2, "on_exhaust_trigger": 1,
                     "is_power": 1, "rarity_score": 0.5},
    "disarm": {"base_cost": 1, "exhausts_self": 1, "is_skill": 1, "rarity_score": 0.5},
    "dropkick": {"base_damage": 5, "base_cost": 1, "is_attack": 1, "rarity_score": 0.5},
    "dual_wield": {"base_cost": 1, "is_skill": 1, "rarity_score": 0.5},
    "entrench": {"base_cost": 2, "block_multiplier": 1, "is_skill": 1, "rarity_score": 0.5},
    "evolve": {"base_cost": 1, "status_payoff_draw": 1, "is_power": 1, "rarity_score": 0.5},
    "feel_no_pain": {"base_cost": 1, "on_exhaust_trigger": 1,
                     "is_power": 1, "rarity_score": 0.5},
    "fire_breathing": {"base_cost": 1, "status_payoff_damage": 1, "is_aoe": 1,
                       "is_power": 1, "rarity_score": 0.5},
    "flame_barrier": {"base_block": 12, "base_cost": 2, "is_skill": 1, "rarity_score": 0.5},
    "ghostly_armor": {"base_block": 10, "base_cost": 1, "ethereal": 1,
                      "is_skill": 1, "rarity_score": 0.5},
    "hemokinesis": {"base_damage": 15, "base_cost": 1, "self_damage": 1,
                    "is_attack": 1, "rarity_score": 0.5},
    "infernal_blade": {"base_cost": 1, "draw_specific": 1, "exhausts_self": 1,
                       "is_skill": 1, "rarity_score": 0.5},
    "inflame": {"base_cost": 1, "strength_gain": 2, "is_power": 1, "rarity_score": 0.5},
    "intimidate": {"base_cost": 0, "applies_weak": 1, "is_aoe": 1, "exhausts_self": 1,
                   "is_skill": 1, "rarity_score": 0.5},
    "metallicize": {"base_cost": 1, "block_per_turn": 1, "is_power": 1, "rarity_score": 0.5},
    "power_through": {"base_block": 15, "base_cost": 1, "injects_status_self": 1,
                      "is_skill": 1, "rarity_score": 0.5},
    "pummel": {"base_damage": 2, "base_cost": 1, "num_hits": 4, "exhausts_self": 1,
               "is_attack": 1, "rarity_score": 0.5},
    "rage": {"base_cost": 0, "exhausts_self": 1, "is_skill": 1, "rarity_score": 0.5},
    "rampage": {"base_damage": 8, "base_cost": 1, "is_attack": 1, "rarity_score": 0.5},
    "reckless_charge": {"base_damage": 7, "base_cost": 0, "injects_status_enemy": 1,
                        "is_attack": 1, "rarity_score": 0.5},
    "rupture": {"base_cost": 1, "self_damage_payoff": 1, "is_power": 1, "rarity_score": 0.5},
    "searing_blow": {"base_damage": 12, "base_cost": 2, "is_attack": 1, "rarity_score": 0.5},
    "second_wind": {"base_cost": 1, "exhausts_others": 1, "exhaust_payoff_scaling": 1,
                    "is_skill": 1, "rarity_score": 0.5},
    "seeing_red": {"base_cost": 1, "energy_gain": 2, "exhausts_self": 1,
                   "is_skill": 1, "rarity_score": 0.5},
    "sentinel": {"base_block": 5, "base_cost": 1, "on_exhaust_trigger": 1,
                 "is_skill": 1, "rarity_score": 0.5},
    "sever_soul": {"base_damage": 16, "base_cost": 2, "exhausts_others": 1,
                   "is_attack": 1, "rarity_score": 0.5},
    "shockwave": {"base_cost": 2, "applies_weak": 1, "applies_vulnerable": 1,
                  "is_aoe": 1, "exhausts_self": 1, "is_skill": 1, "rarity_score": 0.5},
    "spot_weakness": {"base_cost": 1, "strength_gain": 3, "is_skill": 1, "rarity_score": 0.5},
    "uppercut": {"base_damage": 13, "base_cost": 2, "applies_weak": 1,
                 "applies_vulnerable": 1, "is_attack": 1, "rarity_score": 0.5},
    "whirlwind": {"base_damage": 5, "is_x_cost": 1, "is_aoe": 1,
                  "is_attack": 1, "rarity_score": 0.5},

    # ---- 稀有 (RARE, 1.0) ----
    "barricade": {"base_cost": 3, "block_retain": 1, "is_power": 1, "rarity_score": 1.0},
    "berserk": {"base_cost": 0, "applies_vulnerable": 1, "is_power": 1, "rarity_score": 1.0},
    "bludgeon": {"base_damage": 32, "base_cost": 3, "is_attack": 1, "rarity_score": 1.0},
    "brutality": {"base_cost": 0, "self_damage": 1, "draw_count": 1,
                  "is_power": 1, "rarity_score": 1.0},
    "corruption": {"base_cost": 3, "cost_reduction": 1, "is_power": 1, "rarity_score": 1.0},
    "demon_form": {"base_cost": 3, "strength_gain": 2, "is_power": 1, "rarity_score": 1.0},
    "double_tap": {"base_cost": 1, "is_skill": 1, "rarity_score": 1.0},
    "exhume": {"base_cost": 1, "recover_from_exhaust": 1, "exhausts_self": 1,
               "is_skill": 1, "rarity_score": 1.0},
    "feed": {"base_damage": 10, "base_cost": 1, "exhausts_self": 1,
             "is_attack": 1, "rarity_score": 1.0},
    "fiend_fire": {"base_damage": 7, "base_cost": 2, "exhausts_self": 1,
                   "exhaust_payoff_scaling": 1, "is_attack": 1, "rarity_score": 1.0},
    "immolate": {"base_damage": 21, "base_cost": 2, "is_aoe": 1,
                 "injects_status_self": 1, "is_attack": 1, "rarity_score": 1.0},
    "impervious": {"base_block": 30, "base_cost": 2, "exhausts_self": 1,
                   "is_skill": 1, "rarity_score": 1.0},
    "juggernaut": {"base_cost": 2, "block_on_gain_trigger": 1, "is_power": 1, "rarity_score": 1.0},
    "limit_break": {"base_cost": 1, "strength_multiplier": 1, "exhausts_self": 1,
                    "is_skill": 1, "rarity_score": 1.0},
    "offering": {"base_cost": 0, "energy_gain": 2, "self_damage": 1, "draw_count": 3,
                 "exhausts_self": 1, "is_skill": 1, "rarity_score": 1.0},
    "reaper": {"base_damage": 4, "base_cost": 2, "is_aoe": 1, "exhausts_self": 1,
               "hp_gain": 1, "is_attack": 1, "rarity_score": 1.0},
}


def get_card_effects(name: str) -> dict:
    """按卡名查效果 dict；未知返回 None，已知则补全 42 字段（缺省 None）。"""
    e = CARD_EFFECTS.get(normalize_card_name(name))
    if e is None:
        return None
    return {k: e.get(k) for k in EFFECT_KEYS}


def get_base_cost(name: str) -> float:
    """查卡牌基础费用；未知/无费用返回 0.0。"""
    e = get_card_effects(name)
    if not e:
        return 0.0
    return float(e.get("base_cost") or 0.0)


# 五流派归属：[力量, 格挡, 消耗, 自伤, 状态]，各维 0.0~1.0。
# 引擎卡（流派启动器）→ 1.0；放大器/核心配合 → 0.6~0.9；沾边 → 0.2~0.5；无关不写。
# 一张卡可属多流派。未收录卡查询返回全零（见 get_card_archetype）。
CARD_ARCHETYPE = {
    # 力量流
    "inflame": [1.0, 0.0, 0.0, 0.0, 0.0],
    "demon_form": [1.0, 0.0, 0.0, 0.0, 0.0],
    "limit_break": [1.0, 0.0, 0.3, 0.0, 0.0],
    "rupture": [0.5, 0.0, 0.0, 1.0, 0.0],
    "spot_weakness": [0.7, 0.0, 0.0, 0.0, 0.0],
    "flex": [0.3, 0.0, 0.0, 0.0, 0.0],
    "heavy_blade": [0.9, 0.0, 0.0, 0.0, 0.0],
    "whirlwind": [0.8, 0.0, 0.0, 0.0, 0.0],
    "twin_strike": [0.5, 0.0, 0.0, 0.0, 0.0],
    "sword_boomerang": [0.6, 0.0, 0.0, 0.0, 0.0],
    "pummel": [0.6, 0.0, 0.3, 0.0, 0.0],
    "rampage": [0.5, 0.0, 0.0, 0.0, 0.0],
    "searing_blow": [0.4, 0.0, 0.0, 0.0, 0.0],
    "cleave": [0.3, 0.0, 0.0, 0.0, 0.0],
    "anger": [0.3, 0.0, 0.0, 0.0, 0.0],
    "iron_wave": [0.3, 0.4, 0.0, 0.0, 0.0],
    # 格挡流
    "barricade": [0.0, 1.0, 0.0, 0.0, 0.0],
    "metallicize": [0.0, 1.0, 0.0, 0.0, 0.0],
    "entrench": [0.0, 1.0, 0.0, 0.0, 0.0],
    "body_slam": [0.0, 1.0, 0.0, 0.0, 0.0],
    "juggernaut": [0.0, 1.0, 0.0, 0.0, 0.0],
    "impervious": [0.0, 0.8, 0.4, 0.0, 0.0],
    "shrug_it_off": [0.0, 0.6, 0.0, 0.0, 0.0],
    "defend": [0.0, 0.6, 0.0, 0.0, 0.0],
    "armaments": [0.0, 0.6, 0.0, 0.0, 0.0],
    "flame_barrier": [0.0, 0.6, 0.0, 0.0, 0.0],
    "sentinel": [0.0, 0.6, 0.8, 0.0, 0.0],
    "ghostly_armor": [0.0, 0.6, 0.0, 0.0, 0.0],
    "true_grit": [0.0, 0.5, 0.5, 0.0, 0.0],
    "power_through": [0.0, 0.5, 0.0, 0.0, 1.0],
    # 消耗流
    "corruption": [0.0, 0.0, 1.0, 0.0, 0.0],
    "dark_embrace": [0.0, 0.0, 1.0, 0.0, 0.0],
    "feel_no_pain": [0.0, 0.0, 1.0, 0.0, 0.0],
    "second_wind": [0.0, 0.0, 0.7, 0.0, 0.0],
    "sever_soul": [0.0, 0.0, 0.6, 0.0, 0.0],
    "fiend_fire": [0.0, 0.0, 0.7, 0.0, 0.0],
    "exhume": [0.0, 0.0, 0.6, 0.0, 0.0],
    "infernal_blade": [0.0, 0.0, 0.4, 0.0, 0.0],
    "warcry": [0.0, 0.0, 0.4, 0.0, 0.0],
    "seeing_red": [0.0, 0.0, 0.5, 0.0, 0.0],
    "shockwave": [0.0, 0.0, 0.3, 0.0, 0.0],
    "reaper": [0.0, 0.0, 0.4, 0.0, 0.0],
    "intimidate": [0.0, 0.0, 0.2, 0.0, 0.0],
    "rage": [0.0, 0.0, 0.3, 0.0, 0.0],
    "feed": [0.0, 0.0, 0.4, 0.0, 0.0],
    # 自伤流
    "bloodletting": [0.0, 0.0, 0.5, 0.8, 0.0],
    "offering": [0.0, 0.0, 0.5, 0.8, 0.0],
    "hemokinesis": [0.0, 0.0, 0.0, 0.8, 0.0],
    "combust": [0.0, 0.0, 0.0, 0.7, 0.0],
    "brutality": [0.0, 0.0, 0.0, 0.8, 0.0],
    # 状态流
    "evolve": [0.0, 0.0, 0.0, 0.0, 1.0],
    "fire_breathing": [0.0, 0.0, 0.0, 0.0, 1.0],
    "immolate": [0.0, 0.0, 0.0, 0.0, 0.8],
    "wild_strike": [0.0, 0.0, 0.0, 0.0, 0.7],
    "reckless_charge": [0.0, 0.0, 0.0, 0.0, 0.7],
}


def get_card_archetype(name: str) -> list:
    """查卡牌五流派归属向量 [力量, 格挡, 消耗, 自伤, 状态]；未收录返回全零。"""
    v = CARD_ARCHETYPE.get(normalize_card_name(name))
    if v is None:
        return [0.0] * 5
    return list(v)
