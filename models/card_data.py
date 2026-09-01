"""铁甲战士（IRONCLAD）卡牌效果字典（基础值，未升级）。

key 为归一化卡名（normalize_card_name：小写 + 去 _red 等颜色后缀），
value 为效果 dict，字段见 EFFECT_KEYS（16 维，与 candidate_encoder 的
card-effects 段一一对应）。数值为基础值；不确定的置 None（编码时填 0）。

归一化：base_damage/base_block/base_cost 除以 20，num_hits/draw_count 除以 5，
布尔取 0/1（见 candidate_encoder._encode_card_effects）。
"""

# 16 维效果字段顺序（与 candidate_encoder 对齐，顺序敏感）
EFFECT_KEYS = [
    "base_damage", "base_block", "base_cost", "is_power", "is_aoe",
    "num_hits", "draw_count", "exhausts", "self_damage", "retain",
    "scales_strength", "applies_vulnerable", "applies_weak",
    "gains_block_scaling", "is_status", "is_curse",
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


CARD_EFFECTS = {
    # ---- 起始牌 ----
    "strike": {"base_damage": 6, "base_cost": 1},
    "defend": {"base_block": 5, "base_cost": 1},
    "bash": {"base_damage": 8, "base_cost": 2, "applies_vulnerable": 1},

    # ---- 普通攻击 ----
    "anger": {"base_damage": 6, "base_cost": 0},
    "body_slam": {"base_cost": 1},
    "clash": {"base_damage": 14, "base_cost": 0},
    "cleave": {"base_damage": 8, "base_cost": 1, "is_aoe": 1},
    "headbutt": {"base_damage": 9, "base_cost": 1},
    "heavy_blade": {"base_damage": 14, "base_cost": 2, "scales_strength": 1},
    "iron_wave": {"base_damage": 5, "base_block": 5, "base_cost": 1},
    "perfected_strike": {"base_damage": 6, "base_cost": 2},
    "pommel_strike": {"base_damage": 9, "base_cost": 1, "draw_count": 1},
    "sword_boomerang": {"base_damage": 3, "base_cost": 1, "num_hits": 3},
    "thunderclap": {"base_damage": 4, "base_cost": 1, "is_aoe": 1,
                    "applies_vulnerable": 1},
    "twin_strike": {"base_damage": 5, "base_cost": 1, "num_hits": 2},
    "wild_strike": {"base_damage": 12, "base_cost": 1},

    # ---- 普通技能 ----
    "armaments": {"base_block": 5, "base_cost": 1},
    "flex": {"base_cost": 0},
    "shrug_it_off": {"base_block": 8, "base_cost": 1, "draw_count": 1},
    "true_grit": {"base_block": 7, "base_cost": 1, "exhausts": 1},
    "warcry": {"base_cost": 0, "draw_count": 1, "exhausts": 1},

    # ---- 罕见 ----
    "battle_trance": {"base_cost": 0, "draw_count": 3},
    "blood_for_blood": {"base_damage": 18, "base_cost": 4},
    "bloodletting": {"base_cost": 0, "self_damage": 1},
    "burning_pact": {"base_cost": 1, "draw_count": 2},
    "carnage": {"base_damage": 20, "base_cost": 2, "exhausts": 1},
    "combust": {"base_cost": 1, "is_power": 1, "self_damage": 1},
    "dark_embrace": {"base_cost": 2, "is_power": 1},
    "disarm": {"base_cost": 1},
    "double_tap": {"base_cost": 1},
    "dropkick": {"base_damage": 5, "base_cost": 1},
    "dual_wield": {"base_cost": 1},
    "entrench": {"base_cost": 2},
    "evolve": {"base_cost": 1, "is_power": 1},
    "feel_no_pain": {"base_cost": 1, "is_power": 1},
    "fire_breathing": {"base_cost": 1, "is_power": 1},
    "flame_barrier": {"base_block": 12, "base_cost": 2},
    "ghostly_armor": {"base_block": 10, "base_cost": 1, "exhausts": 1},
    "havoc": {"base_cost": 1},
    "hemokinesis": {"base_damage": 15, "base_cost": 1, "self_damage": 1},
    "infernal_blade": {"base_cost": 1},
    "inflame": {"base_cost": 1, "is_power": 1},
    "intimidate": {"base_cost": 0, "is_aoe": 1, "applies_weak": 1,
                   "exhausts": 1},
    "metallicize": {"base_cost": 1, "is_power": 1},
    "power_through": {"base_cost": 1},
    "pummel": {"base_damage": 2, "base_cost": 1, "num_hits": 4, "exhausts": 1},
    "rage": {"base_cost": 0, "exhausts": 1},
    "rampage": {"base_damage": 8, "base_cost": 1},
    "reckless_charge": {"base_damage": 7, "base_cost": 0},
    "rupture": {"base_cost": 1, "is_power": 1},
    "searing_blow": {"base_damage": 12, "base_cost": 2},
    "second_wind": {"base_cost": 1, "exhausts": 1},
    "seeing_red": {"base_cost": 1, "exhausts": 1},
    "sentinel": {"base_block": 5, "base_cost": 1, "exhausts": 1},
    "sever_soul": {"base_damage": 16, "base_cost": 2, "exhausts": 1},
    "shockwave": {"base_cost": 2, "is_aoe": 1, "applies_weak": 1,
                  "applies_vulnerable": 1, "exhausts": 1},
    "spot_weakness": {"base_cost": 1},
    "uppercut": {"base_damage": 13, "base_cost": 2, "applies_weak": 1,
                 "applies_vulnerable": 1},
    "whirlwind": {"base_damage": 5, "base_cost": None, "is_aoe": 1},

    # ---- 稀有 ----
    "barricade": {"base_cost": 3, "is_power": 1},
    "berserk": {"base_cost": 0, "is_power": 1, "self_damage": 1},
    "bludgeon": {"base_damage": 32, "base_cost": 3},
    "brutality": {"base_cost": 1, "is_power": 1, "self_damage": 1},
    "corruption": {"base_cost": 3, "is_power": 1},
    "demon_form": {"base_cost": 3, "is_power": 1, "scales_strength": 1},
    "exhume": {"base_cost": 1, "exhausts": 1},
    "feed": {"base_damage": 10, "base_cost": 1, "exhausts": 1},
    "fiend_fire": {"base_cost": 2, "exhausts": 1},
    "immolate": {"base_damage": 21, "base_cost": 2, "is_aoe": 1},
    "impervious": {"base_block": 30, "base_cost": 2, "exhausts": 1},
    "juggernaut": {"base_cost": 2, "is_power": 1},
    "limit_break": {"base_cost": 1, "exhausts": 1},
    "offering": {"base_cost": 0, "self_damage": 1, "draw_count": 3},
    "reaper": {"base_damage": 4, "base_cost": 2, "is_aoe": 1},
}


def get_card_effects(name: str) -> dict:
    """按卡名查效果 dict；未知返回 None，已知则补全 16 字段（缺省 None）。"""
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
