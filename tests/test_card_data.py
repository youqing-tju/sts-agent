"""models/card_data.py 单元测试：卡牌效果字典查询、归一化、五流派归属。"""
from models.card_data import (
    CARD_ARCHETYPE,
    CARD_EFFECTS,
    EFFECT_KEYS,
    get_base_cost,
    get_card_archetype,
    get_card_effects,
    normalize_card_name,
)


def test_normalize_card_name():
    assert normalize_card_name("STRIKE_RED") == "strike"
    assert normalize_card_name("DEFEND_RED") == "defend"
    assert normalize_card_name("BASH") == "bash"
    assert normalize_card_name("HEAVY_BLADE") == "heavy_blade"
    assert normalize_card_name("bash") == "bash"
    assert normalize_card_name("  BASH  ") == "bash"


def test_bash_effects():
    e = get_card_effects("BASH")
    assert e is not None
    assert e["base_damage"] == 8
    assert e["base_cost"] == 2
    assert e["applies_vulnerable"] == 1
    # 42 字段版：缺失字段补全为 None
    assert e["base_block"] is None
    assert "scales_strength" not in e      # 旧字段已移除
    assert "exhausts" not in e            # 旧字段已拆分为 exhausts_self/others


def test_bludgeon_effects():
    """重锤：32 伤害 3 费。"""
    e = get_card_effects("BLUDGEON")
    assert e is not None
    assert e["base_damage"] == 32
    assert e["base_cost"] == 3


def test_heavy_blade_strength_amplifier():
    """重刃：14 伤、攻击牌；力量通过多段放大（无独立字段）。"""
    e = get_card_effects("HEAVY_BLADE")
    assert e is not None
    assert e["base_damage"] == 14
    assert e["base_cost"] == 2
    assert e["is_attack"] == 1


def test_strike_red():
    e = get_card_effects("STRIKE_RED")
    assert e is not None
    assert e["base_damage"] == 6
    assert e["base_cost"] == 1


def test_unknown_card_none():
    assert get_card_effects("NOT_A_REAL_CARD") is None


def test_get_base_cost():
    assert get_base_cost("BASH") == 2.0
    assert get_base_cost("STRIKE_RED") == 1.0
    assert get_base_cost("NOT_A_REAL_CARD") == 0.0


def test_coverage_and_keys():
    assert len(CARD_EFFECTS) == 75          # 全部 75 张卡收录（含 clothesline）
    assert len(EFFECT_KEYS) == 48           # 效果字段 16 → 42 → 48（V10 增 6 防战维度）
    # 所有字典值字段必须是 EFFECT_KEYS 子集
    for name, e in CARD_EFFECTS.items():
        assert set(e.keys()) <= set(EFFECT_KEYS), f"{name} 含未知字段"
    # 每张卡必须有类型标记 + 稀有度分数
    for name, e in CARD_EFFECTS.items():
        assert e.get("rarity_score") in (0.0, 0.5, 1.0), f"{name} 缺 rarity_score"
        assert (e.get("is_attack") or e.get("is_skill") or e.get("is_power")) == 1, \
            f"{name} 缺类型标记"


# ---- 数据错漏修正（13 处）----

def test_clothesline_added():
    """缺失卡 clothesline 已收录。"""
    e = get_card_effects("CLOTHESLINE")
    assert e is not None
    assert e["base_damage"] == 12
    assert e["base_cost"] == 2
    assert e["applies_weak"] == 1


def test_exhaust_self_fixes():
    """9 处 exhaust 修正：offering/reaper/disarm/infernal_blade 改标 exhausts_self。"""
    for name in ["offering", "reaper", "disarm", "infernal_blade"]:
        e = get_card_effects(name)
        assert e["exhausts_self"] == 1, f"{name} 应为 exhausts_self"


def test_sentinel_not_self_exhaust():
    """哨兵：自身不消耗，改为 on_exhaust_trigger。"""
    e = get_card_effects("SENTINEL")
    assert e.get("exhausts_self") is None
    assert e["on_exhaust_trigger"] == 1


def test_second_wind_and_sever_soul_exhaust_others():
    """回风/斩断灵魂：消耗其他牌，自身不消耗。"""
    for name in ["second_wind", "sever_soul"]:
        e = get_card_effects(name)
        assert e.get("exhausts_self") is None
        assert e["exhausts_others"] == 1


def test_carnage_and_ghostly_armor_ethereal():
    """屠戮/幽魂护甲：是虚无（ethereal）而非消耗。"""
    for name in ["carnage", "ghostly_armor"]:
        e = get_card_effects(name)
        assert e.get("exhausts_self") is None
        assert e["ethereal"] == 1


def test_berserk_applies_vulnerable_not_self_damage():
    """狂暴：不掉血，改为对自己施加易伤。"""
    e = get_card_effects("BERSERK")
    assert e.get("self_damage") is None
    assert e["applies_vulnerable"] == 1


def test_brutality_draw_count():
    """残暴：补 draw_count。"""
    e = get_card_effects("BRUTALITY")
    assert e["draw_count"] == 1
    assert e["self_damage"] == 1


def test_power_through_base_block():
    """强行通过：补 base_block=15。"""
    e = get_card_effects("POWER_THROUGH")
    assert e["base_block"] == 15


def test_true_grit_exhausts_others_not_self():
    """（重构附带修正）坚韧消耗手牌中的牌，自身不消耗。"""
    e = get_card_effects("TRUE_GRIT")
    assert e.get("exhausts_self") is None
    assert e["exhausts_others"] == 1


# ---- 五流派归属表 ----

def test_archetype_unknown_returns_zero():
    assert get_card_archetype("NOT_A_CARD") == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert get_card_archetype("STRIKE") == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_archetype_engine_cards():
    """代表引擎卡的流派向量正确。"""
    # 燃心 = 纯力量引擎
    assert get_card_archetype("inflame") == [1.0, 0.0, 0.0, 0.0, 0.0]
    # 壁垒 = 纯格挡引擎
    assert get_card_archetype("barricade") == [0.0, 1.0, 0.0, 0.0, 0.0]
    # 腐化 = 纯消耗引擎
    assert get_card_archetype("corruption") == [0.0, 0.0, 1.0, 0.0, 0.0]
    # 破裂 = 自伤流引擎（同时给力量）
    assert get_card_archetype("rupture") == [0.5, 0.0, 0.0, 1.0, 0.0]
    # 进化 = 纯状态流引擎
    assert get_card_archetype("evolve") == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_archetype_normalized_name():
    """流派查询同样按归一化名匹配（带颜色后缀 id）。"""
    assert get_card_archetype("INFLAME_RED") == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_archetype_table_dimensions():
    """表内每个向量都是 5 维、取值 0.0~1.0。"""
    for name, v in CARD_ARCHETYPE.items():
        assert len(v) == 5
        assert all(0.0 <= x <= 1.0 for x in v)
