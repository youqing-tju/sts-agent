"""tests/test_event_kb.py — 事件选项 20 维知识库（严格 TDD）。

覆盖：
1. 向量长度恰好 20
2. 未收录 / 越界返回全零且不抛异常
3. 至少 3 个具体事件选项的手算期望值（非「跑一遍看输出」式断言）
4. 类别 one-hot 与代价段位置正确、互不串位
5. 归一化健壮性（大小写 / 空格 / 标点变体都能查到）
"""
import pytest

from models.event_kb import (
    EVENT_KB_DIM,
    CATEGORY_LIST,
    _NCAT,
    get_event_option_features,
    get_event_option_rating,
    normalize_event_key,
    event_count,
    option_count,
)

# 手算期望值（与实现解耦，依据 event_kb.py 文档里的归一化分母）：
#   _HP_NORM=30, _GOLD_NORM=300, _CARD_NORM=3
#   hp_cost = loss/30 ; gold = Δ/300(有符号,截断±1) ; remove/upgrade = n/3
#   hp_cost=7  -> 0.23333 ; remove=1 -> 0.33333 ; gold=275 -> 0.91667


def test_dim_is_20():
    assert EVENT_KB_DIM == 20
    assert _NCAT == 12
    v = get_event_option_features("The Cleric", 1)
    assert len(v) == 20


def test_unknown_event_returns_zeros_no_raise():
    # 完全没收录的事件名
    v = get_event_option_features("Nonexistent Event XYZ", 0)
    assert v == [0.0] * 20
    # 空串 / None 也不抛
    assert get_event_option_features("", 0) == [0.0] * 20
    assert get_event_option_features(None, 0) == [0.0] * 20


def test_option_idx_none_or_out_of_range_returns_zeros():
    # option_idx=None → 全零
    assert get_event_option_features("The Cleric", None) == [0.0] * 20
    # 越界（事件只有 3 个选项，传 5）→ 全零且不抛
    assert get_event_option_features("The Cleric", 5) == [0.0] * 20
    assert get_event_option_features("The Cleric", -1) == [0.0] * 20


def test_wing_statue_pray_hand_computed():
    # Wing Statue 祈祷：删卡(REMOVE=idx0) + 失去 7 HP + 删 1 张
    # 期望：A[0]=1，B[12]=7/30≈0.2333，B[16]=1/3≈0.3333，其余 0
    v = get_event_option_features("Wing Statue", 0)
    assert v == pytest.approx(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        # A 段 one-hot
         0.233333, 0.0, 0.0, 0.0, 0.333333, 0.0, 0.0, 0.0],  # B 段
        abs=1e-4,
    )


def test_golden_shrine_defile_hand_computed():
    # Golden Shrine 亵渎：金币(GOLD=idx2) + 275 金币 + 诅咒
    # 期望：A[2]=1，B[14]=275/300≈0.9167，B[15]=1(诅咒)，其余 0
    v = get_event_option_features("Golden Shrine", 1)
    assert v == pytest.approx(
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0.0, 0.0, 0.916667, 1.0, 0.0, 0.0, 0.0, 0.0],
        abs=1e-4,
    )


def test_purifier_pray_hand_computed():
    # Purifier 祈祷：删卡(REMOVE=idx0) + 删 1 张
    v = get_event_option_features("Purifier", 0)
    assert v == pytest.approx(
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0.0, 0.0, 0.0, 0.0, 0.333333, 0.0, 0.0, 0.0],
        abs=1e-4,
    )


def test_category_onehot_exclusive_and_in_range():
    # 任一收录选项，A 段恰好一个 1，且位于 [0,12)
    for ek, rec in _iter_all_options():
        v = get_event_option_features(ek, rec[1])
        a = v[:_NCAT]
        assert sum(a) == pytest.approx(1.0), f"{ek}#{rec[1]} one-hot 和不为1: {a}"
        assert all(x in (0.0, 1.0) for x in a)
        # 类别 one-hot 下标与 CATEGORY_LIST 对齐
        ci = CATEGORY_LIST.index(rec[0]["cat"])
        assert a[ci] == 1.0


def test_cost_segment_no_cross_talk():
    # 以 Golden Shrine 亵渎为例：GOLD 类别置位，RELIC/POTION/HEAL 等类别必须 0；
    # 代价段只有 gold/curse 非零，remove/upgrade/relic/potion 必须 0。
    v = get_event_option_features("Golden Shrine", 1)
    # 类别段：仅 idx2(GOLD)=1
    assert v[2] == 1.0
    for i in (0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11):
        assert v[i] == 0.0, f"类别段串位 idx{i}"
    # 代价段：idx14(gold)/15(curse) 非零；idx12,13,16,17,18,19 必须 0
    assert v[14] == pytest.approx(0.916667, abs=1e-4)
    assert v[15] == 1.0
    for i in (12, 13, 16, 17, 18, 19):
        assert v[i] == 0.0, f"代价段串位 idx{i}"


def test_normalization_robustness():
    # 同一事件的不同写法都应解析到同一选项（大小写/空格/标点变体）
    variants = ["The Cleric", "the cleric", "THE_CLERIC", "  The  Cleric! ", "the.cleric"]
    for name in variants:
        assert normalize_event_key(name) == "thecleric"
        # 删除一张卡（idx1）类别应为 REMOVE(idx0)=1
        v = get_event_option_features(name, 1)
        assert v[0] == 1.0, f"变体 {name!r} 未解析到 The Cleric"
        assert get_event_option_rating(name, 1) == 5
    # 带感叹号的神龛名
    assert normalize_event_key("We Meet Again!") == "wemeetagain"
    assert get_event_option_features("we meet again!", 0)[6] == 1.0  # RELIC


def test_rating_out_of_vector_and_separate():
    # 推荐度不进向量，单独由 get_event_option_rating 提供
    v = get_event_option_features("The Cleric", 1)
    assert 5.0 not in v, "推荐度不应出现在 20 维向量里"
    assert get_event_option_rating("The Cleric", 1) == 5
    assert get_event_option_rating("The Cleric", 0) == 3
    assert get_event_option_rating("Unknown Event", 0) == 0
    assert get_event_option_rating("The Cleric", 99) == 0


def test_kb_coverage_counts():
    # 报告第七章录入：24 个事件、55 个选项（作为回归基线，改动录入需同步更新）
    assert event_count() == 24
    assert option_count() == 55
    assert len(CATEGORY_LIST) == 12


# ---- 测试内部工具 ----
def _iter_all_options():
    """yield (event_key, (opt_record, option_idx)) 遍历 KB 全部选项。"""
    from models.event_kb import EVENT_KB
    for ek, rec in EVENT_KB.items():
        for idx, opt in enumerate(rec["options"]):
            yield ek, (opt, idx)
