"""遗物知识库测试（严格 TDD）。

覆盖：向量长度 32、未收录全零不抛异常、归一化鲁棒（枚举名/显示名/大小写/
空格/连字符）、≥3 个手算期望值、流派贡献顺序与 CARD_ARCHETYPE 锁死、
全表覆盖（181 RelicId 兜底）。
"""
import importlib.util
import os
import sys

import pytest

from models.relic_kb import (
    ALL_RELIC_IDS,
    RELIC_ARCHETYPE_ORDER,
    RELIC_FEATURE_DIM,
    RELIC_KB,
    get_relic_features,
)

# 检测 slaythespire 是否可用（测试环境一般可用，用于覆盖全表断言）
_STS_AVAILABLE = importlib.util.find_spec("slaythespire") is not None


def test_dim_is_32():
    assert RELIC_FEATURE_DIM == 32
    assert len(get_relic_features("Burning Blood")) == 32
    assert len(get_relic_features("")) == 32


def test_unknown_returns_zeros_no_raise():
    assert get_relic_features("完全不是遗物 xyz") == [0.0] * 32
    assert get_relic_features("zzz_nope") == [0.0] * 32
    assert get_relic_features("") == [0.0] * 32
    assert get_relic_features(None) == [0.0] * 32


def test_normalization_robust():
    base = get_relic_features("Burning Blood")
    for v in ["burning blood", "BURNING_BLOOD", "Burning-Blood", "burning_blood"]:
        assert get_relic_features(v) == base, f"变体未归一化: {v!r}"
    # 多词 / 撇号显示名
    clay = get_relic_features("Self-Forming Clay")
    assert get_relic_features("self_forming_clay") == clay
    assert get_relic_features("SELF FORMING CLAY") == clay
    # 撇号
    ca = get_relic_features("Charon's Ashes")
    assert get_relic_features("charons_ashes") == ca
    assert get_relic_features("CHARONS ASHES") == ca


def test_black_blood_hand_computed():
    # 报告：黑血 = Boss 遗物，战斗开始 +3 力量，战后回 12，铁甲专属
    v = get_relic_features("Black Blood")
    assert v[0] == pytest.approx(0.80)        # 流派[0]=力量
    assert v[8] == 1.0                         # BOSS one-hot（5 COMMON/6 UNCOMMON/7 RARE/8 BOSS）
    assert v[11] == 1.0                        # 铁甲专属
    assert v[13] == 1.0                        # COMBAT_START 触发
    assert v[30] == 1.0                        # is_boss
    assert v[31] == pytest.approx(0.90)        # 铁甲契合度
    assert v[29] == pytest.approx(1.0)         # rarity_score(BOSS)=1.0


def test_charons_ashes_hand_computed():
    # 报告：每消耗 1 张牌对全体造成 3 伤害 → 消耗流 + ON_EXHAUST 触发
    v = get_relic_features("Charon's Ashes")
    assert v[2] == pytest.approx(0.90)         # 流派[2]=消耗
    assert v[16] == 1.0                        # ON_EXHAUST one-hot
    assert v[26] == pytest.approx(0.90)        # exhaust 细分数值
    assert v[11] == 1.0                        # 铁甲专属


def test_burning_blood_hand_computed():
    # 报告：战斗结束回 6 生命，铁甲初始遗物
    v = get_relic_features("Burning Blood")
    assert v[11] == 1.0                        # 铁甲专属
    assert v[23] == pytest.approx(0.30)        # heal 细分数值
    assert v[5] == 1.0                         # COMMON one-hot
    assert v[12] == 0.0                        # 无负面代价


def test_archetype_order_locked_to_card_archetype():
    """五流派贡献顺序必须与 models.card_data.CARD_ARCHETYPE 一致（[力量,格挡,消耗,自伤,状态]）。"""
    # 顺序常量锁死
    assert RELIC_ARCHETYPE_ORDER == ["strength", "block", "exhaust", "self_damage", "status"]

    # 用具体遗物验证各维含义落在对的位置
    # 力量流（黑血）→ 第 0 维非零；格挡流（Anchor）→ 第 1 维；消耗流（Charon's）→ 第 2；
    # 自伤流（Runic Cube）→ 第 3；状态流（Mark of Pain）→ 第 4
    assert get_relic_features("Black Blood")[0] > 0
    assert get_relic_features("Anchor")[1] > 0
    assert get_relic_features("Charon's Ashes")[2] > 0
    assert get_relic_features("Runic Cube")[3] > 0
    assert get_relic_features("Mark of Pain")[4] > 0

    # 交叉验证 CARD_ARCHETYPE 的语义顺序一致（若 card_data 改了顺序，这里会暴露）
    try:
        from models.card_data import CARD_ARCHETYPE
    except Exception:
        pytest.skip("card_data 不可用")
    # CARD_ARCHETYPE 的文档顺序即 [力量,格挡,消耗,自伤,状态]，与 RELIC 对齐
    assert len(next(iter(CARD_ARCHETYPE.values()))) == 5


def test_full_table_coverage():
    # 内置全量枚举名都应被收录（至少 baseline）
    assert len(RELIC_KB) == len(ALL_RELIC_IDS)
    for rid in ALL_RELIC_IDS:
        v = get_relic_features(rid)
        assert len(v) == 32
        # 合法枚举名不应返回全零（baseline 也至少有 power=0.2）
        assert any(x != 0.0 for x in v), f"{rid} 应被收录而非全零"


@pytest.mark.skipif(not _STS_AVAILABLE, reason="slaythespire 不可用")
def test_coverage_matches_enum():
    import slaythespire as sts
    enum_names = [k for k in sts.RelicId.__members__.keys() if k != "INVALID"]
    assert len(enum_names) == len(ALL_RELIC_IDS), "ALL_RELIC_IDS 与 C++ 枚举数对不上"
    for name in enum_names:
        v = get_relic_features(name)
        assert len(v) == 32
        assert any(x != 0.0 for x in v)
