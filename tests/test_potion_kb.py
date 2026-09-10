"""药水知识库测试（严格 TDD）。

覆盖：向量长度、未收录全零不抛异常、归一化鲁棒（大小写/空格/连字符）、
≥3 个手算期望值、类别 one-hot 互斥、收录规模。
"""
import pytest

from models.potion_kb import (
    POTION_FEATURE_DIM,
    POTION_KB,
    get_potion_features,
)


def test_dim_is_24():
    assert POTION_FEATURE_DIM == 24
    assert len(get_potion_features("Blood Potion")) == 24
    assert len(get_potion_features("")) == 24


def test_unknown_returns_zeros_no_raise():
    assert get_potion_features("这不是药水 whatsoever") == [0.0] * 24
    assert get_potion_features("zzz_nope") == [0.0] * 24
    # None / 空串也不应抛异常
    assert get_potion_features("") == [0.0] * 24
    assert get_potion_features(None) == [0.0] * 24


def test_normalization_robust():
    base = get_potion_features("Blood Potion")
    variants = [
        "blood potion",
        "BLOOD POTION",
        "Blood-Potion",
        "blood_potion",
        "  Blood   Potion  ",
    ]
    for v in variants:
        assert get_potion_features(v) == base, f"变体未归一化到同一 key: {v!r}"
    # 撇号 / 显示名变体
    gb = get_potion_features("Gambler's Brew")
    assert get_potion_features("gamblers brew") == gb
    assert get_potion_features("GAMBLERS_BREW") == gb
    assert get_potion_features("Gambler's-Brew") == gb
    # 多词显示名
    fb = get_potion_features("Fairy in a Bottle")
    assert get_potion_features("fairy_in_a_bottle") == fb
    assert get_potion_features("FAIRY IN A BOTTLE") == fb


def test_cultist_potion_hand_computed():
    # 报告：力量流神级（仪式匕首每回合 +1 力量）→ STRENGTH 类别 + 高力量/强度分
    v = get_potion_features("Cultist Potion")
    # 类别 one-hot：STRENGTH 在第 5 维
    assert v[5] == 1.0
    assert sum(v[0:8]) == 1.0  # 类别互斥
    # 细分数值：strength 在 [19]，power 在 [13]
    assert v[19] == pytest.approx(0.80)
    assert v[13] == pytest.approx(0.85)
    assert v[10 + 2] == 1.0  # RARE one-hot（10 COMMON/11 UNCOMMON/12 RARE）
    assert v[8] == 1.0 and v[9] == 1.0  # 战斗内/外均可用


def test_block_potion_hand_computed():
    v = get_potion_features("Block Potion")
    assert v[1] == 1.0  # DEFENSE 类别
    assert v[15] == pytest.approx(0.60)  # block 数值
    assert v[10] == 1.0  # COMMON
    assert sum(v[10:13]) == 1.0  # 稀有度互斥


def test_blood_potion_hand_computed():
    v = get_potion_features("Blood Potion")
    assert v[4] == 1.0  # HEAL 类别
    assert v[14] == pytest.approx(0.50)  # heal 数值
    assert v[8] == 1.0 and v[9] == 1.0


def test_category_onehot_mutually_exclusive():
    for name in POTION_KB:
        disp = name.replace("_", " ")
        # 仅检查能映射到显示名的少量样本，避免别名歧义
        v = get_potion_features(name)
        assert sum(v[0:8]) == pytest.approx(1.0), f"{name}: 类别 one-hot 应恰好 1 个"


def test_coverage_count():
    assert len(POTION_KB) == 33


def test_cpp_display_name_aliases_resolve():
    # 任务 A：C++ 绑定返回的显示名应能命中同一条目（别名映射）
    # Elixir Potion → Elixir（铁甲专属药水）
    v_cpp = get_potion_features("Elixir Potion")
    v_kb = get_potion_features("Elixir")
    assert v_cpp == v_kb, "Elixir Potion 与 Elixir 应逐元素相等"
    assert any(v_cpp), "Elixir Potion 查表不应全零"
    assert v_cpp[7] == 1.0  # OTHER 类别（Elixir 原标 OTHER）
    # Fairy Potion → Fairy in a Bottle（稀有复活药）
    v_cpp2 = get_potion_features("Fairy Potion")
    v_kb2 = get_potion_features("Fairy in a Bottle")
    assert v_cpp2 == v_kb2, "Fairy Potion 与 Fairy in a Bottle 应逐元素相等"
    assert any(v_cpp2), "Fairy Potion 查表不应全零"
    assert v_cpp2[4] == 1.0  # HEAL 类别
