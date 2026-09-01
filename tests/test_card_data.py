"""models/card_data.py 单元测试：卡牌效果字典查询与归一化。"""
from models.card_data import (
    CARD_EFFECTS,
    EFFECT_KEYS,
    get_base_cost,
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
    assert e["base_block"] is None


def test_bludgeon_effects():
    """重锤：32 伤害 3 费。"""
    e = get_card_effects("BLUDGEON")
    assert e is not None
    assert e["base_damage"] == 32
    assert e["base_cost"] == 3


def test_heavy_blade_scales_strength():
    e = get_card_effects("HEAVY_BLADE")
    assert e is not None
    assert e["base_damage"] == 14
    assert e["scales_strength"] == 1


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
    assert len(CARD_EFFECTS) >= 60
    assert len(EFFECT_KEYS) == 16
    # 所有字典值字段必须是 EFFECT_KEYS 子集
    for name, e in CARD_EFFECTS.items():
        assert set(e.keys()) <= set(EFFECT_KEYS), f"{name} 含未知字段"
