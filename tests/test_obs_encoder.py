"""models.obs_encoder 单元测试（不依赖 slaythespire pyd）。

覆盖：
- encode_context(None) -> 43 维全零
- encode_context(fake_gc) -> 各段按规则填对（药水/数量/Act/房间，地图段退化全零）
- encode_deck_stats 所有可选依赖为 None -> 13 维全零
- encode_deck_stats 用假数据（已知 card_index + arch_table + card_effects）-> 手算期望值
- enhance_observation -> 468 维且前 412 维逐元素相等
- CARD_INDEX 长度 110、无重复、关键卡位置与运行时观测一致
- POTION_INDEX 长度 24、无重复
"""
import math

import pytest

from models.obs_encoder import (
    OBS_BASE_DIM,
    OBS_EXTRA_DIM,
    OBS_DIM,
    CARD_INDEX,
    POTION_INDEX,
    ROOM_INDEX,
    ARCH_NAMES,
    encode_context,
    encode_deck_stats,
    enhance_observation,
)


# --------------------------------------------------------------------------
# 基础常量
# --------------------------------------------------------------------------
def test_dim_constants():
    assert OBS_BASE_DIM == 412
    assert OBS_EXTRA_DIM == 56
    assert OBS_DIM == 468
    assert OBS_BASE_DIM + OBS_EXTRA_DIM == OBS_DIM


def test_card_index_basic():
    assert len(CARD_INDEX) == 110
    assert len(set(CARD_INDEX)) == 110  # 无重复
    assert all(isinstance(n, str) for n in CARD_INDEX)
    # 与运行时观测核对（见 tmp_gen_card_index.py 的实证验证）
    assert CARD_INDEX.index("STRIKE_RED") == 66
    assert CARD_INDEX.index("DEFEND_RED") == 19
    assert CARD_INDEX.index("BASH") == 3


def test_potion_index_basic():
    assert len(POTION_INDEX) == 24
    assert len(set(POTION_INDEX)) == 24
    # 房间 one-hot 顺序与任务规定一致
    assert ROOM_INDEX == [
        "MONSTER", "ELITE", "REST", "SHOP", "TREASURE",
        "EVENT", "BOSS", "BOSS_TREASURE", "NONE", "INVALID",
    ]
    # 五流派顺序
    assert ARCH_NAMES == ["strength", "block", "exhaust", "self_damage", "status"]


# --------------------------------------------------------------------------
# encode_context
# --------------------------------------------------------------------------
def test_encode_context_none_is_all_zero():
    out = encode_context(None)
    assert len(out) == 43
    assert all(v == 0 for v in out)


class _FakeRoom:
    def __init__(self, name):
        self.name = name


class _FakeGC:
    """模拟真实 gc 的属性访问方式（potion_slots 为 int 列表、act 为 int、
    cur_room 为带 .name 的对象），以便在不依赖 pyd 的情况下测试逻辑。"""

    def __init__(self, potion_slots, act, room_name):
        self.potion_slots = potion_slots
        self.act = act
        self.cur_room = _FakeRoom(room_name)


def test_encode_context_fake_gc():
    # POTION_ENUM_NAMES[4] == "ATTACK_POTION"，它在 POTION_INDEX 中位于 0
    # MONSTER 房间在 ROOM_INDEX 中位于 0；act=1 -> Act1 独热
    gc = _FakeGC(potion_slots=[4], act=1, room_name="MONSTER")
    out = encode_context(gc)
    assert len(out) == 43

    potions = out[0:24]
    count = out[24]
    map_summary = out[25:29]
    act_onehot = out[29:33]
    room_onehot = out[33:43]

    # 药水 one-hot：ATTACK_POTION 在 POTION_INDEX 的第 0 位
    assert potions[0] == 1
    assert sum(potions) == 1
    # 数量 /5：持有 1 瓶
    assert count == pytest.approx(1 / 5)
    # 地图段退化全零（当前 API 拿不到 map）
    assert map_summary == [0, 0, 0, 0]
    # Act 1 -> [1,0,0,0]
    assert act_onehot == [1, 0, 0, 0]
    # 房间 MONSTER -> [1,0,0,0,0,0,0,0,0,0]
    assert room_onehot[0] == 1 and sum(room_onehot) == 1


def test_encode_context_act_clamped_to_four():
    # act=4（心脏）和 act=99 都应映射到 Act4 独热（index 3）
    for act in (4, 99):
        gc = _FakeGC(potion_slots=[], act=act, room_name="INVALID")
        out = encode_context(gc)
        assert out[29:33] == [0, 0, 0, 1]


def test_encode_context_map_zero_without_seed():
    # 没有 seed 属性 -> 地图段全零（不重建 SpireMap）
    gc = _FakeGC(potion_slots=[], act=1, room_name="MONSTER")
    out = encode_context(gc)
    assert out[25:29] == [0, 0, 0, 0]


def test_encode_context_map_zero_when_act_ge_4():
    # act>=4（心脏）-> 地图段全零，即使在 fake gc 上挂了 seed
    gc = _FakeGC(potion_slots=[], act=4, room_name="MONSTER")
    gc.seed = 12345
    out = encode_context(gc)
    assert out[25:29] == [0, 0, 0, 0]


def test_encode_context_map_computed_with_seed():
    # 有 seed 且 act<4 -> 真实重建 SpireMap，地图段非全零（依赖 slaythespire pyd）
    slaythespire = pytest.importorskip("slaythespire")
    gc = _FakeGC(potion_slots=[], act=1, room_name="MONSTER")
    gc.seed = 12345
    gc.cur_map_node_y = -1  # 构造时未进入地图
    out = encode_context(gc)
    map_summary = out[25:29]
    assert len(map_summary) == 4
    # 维度在 [0,1]，且至少一层有 ELITE/REST/SHOP/OTHER -> 非全零
    assert all(0.0 <= v <= 1.0 for v in map_summary)
    assert sum(map_summary) > 0


# --------------------------------------------------------------------------
# encode_deck_stats
# --------------------------------------------------------------------------
def _make_obs412(counts=None):
    """构造 412 维观测，counts: {deck_index: (unupgraded, upgraded)}。"""
    obs = [0] * 412
    if counts:
        for i, (u, up) in counts.items():
            obs[14 + i * 2] = u
            obs[14 + i * 2 + 1] = up
    return obs


def test_encode_deck_stats_all_none_returns_13_zeros():
    obs = _make_obs412({0: (5, 2), 1: (3, 0)})
    out = encode_deck_stats(obs, card_index=None, arch_table=None, card_effects=None)
    assert len(out) == 13
    assert all(v == 0 for v in out)


def test_encode_deck_stats_card_effects_none_stats_zero():
    # 给了 arch_table 但没给 card_effects -> 流派段可算（card_index 给了）而统计段全零
    card_index = [f"CARD_{i}" for i in range(110)]
    arch_table = {"card_0": [1, 0, 0, 0, 0]}
    obs = _make_obs412({0: (5, 1)})
    out = encode_deck_stats(obs, card_index=card_index, arch_table=arch_table, card_effects=None)
    assert len(out) == 13
    assert out[0:5] == [pytest.approx(1.0), 0, 0, 0, 0]  # 全部是 CARD_0，strength 占满
    assert out[5:13] == [0] * 8  # 统计段因缺 card_effects 全零


def test_encode_deck_stats_hand_computed():
    # 关键：期望值全部手算，不靠“跑一遍看输出”
    card_index = [f"CARD_{i}" for i in range(110)]
    # CARD_0: 未升级2 + 升级1 = 总数3，力量权重1
    # CARD_1: 未升级3 + 升级0 = 总数3，格挡权重1
    arch_table = {
        "card_0": [1, 0, 0, 0, 0],
        "card_1": [0, 1, 0, 0, 0],
    }
    card_effects = {
        "card_0": {"base_cost": 2, "is_attack": 1, "is_skill": 0,
                   "is_power": 0, "exhausts_self": 0, "is_status": 0},
        "card_1": {"base_cost": 1, "is_attack": 0, "is_skill": 1,
                   "is_power": 0, "exhausts_self": 1, "is_status": 0},
    }
    obs = _make_obs412({0: (2, 1), 1: (3, 0)})
    out = encode_deck_stats(obs, card_index=card_index,
                            arch_table=arch_table, card_effects=card_effects)

    assert len(out) == 13
    total = 6  # 3 + 3

    # --- 流派强度（5 维）：raw/total 归一化 ---
    # strength: 3*1=3 -> 3/6=0.5 ; block: 3*1=3 -> 3/6=0.5
    arch = out[0:5]
    assert arch[0] == pytest.approx(0.5)
    assert arch[1] == pytest.approx(0.5)
    assert arch[2] == pytest.approx(0.0)
    assert arch[3] == pytest.approx(0.0)
    assert arch[4] == pytest.approx(0.0)

    # --- 牌组统计（8 维） ---
    stats = out[5:13]
    # 0: 总牌数/40 = 6/40
    assert stats[0] == pytest.approx(6 / 40)
    # 1: 攻击占比 = 3/6
    assert stats[1] == pytest.approx(3 / 6)
    # 2: 技能占比 = 3/6
    assert stats[2] == pytest.approx(3 / 6)
    # 3: 能力数/10 = 0
    assert stats[3] == pytest.approx(0.0)
    # 4: 平均费用/3 = (3*2 + 3*1)/6 /3 = 9/6/3 = 0.5
    assert stats[4] == pytest.approx(0.5)
    # 5: 消耗牌数/10 = 3/10
    assert stats[5] == pytest.approx(3 / 10)
    # 6: 状态牌数/10 = 0
    assert stats[6] == pytest.approx(0.0)
    # 7: 升级数/20 = 升级计数 1 /20
    assert stats[7] == pytest.approx(1 / 20)


# --------------------------------------------------------------------------
# enhance_observation
# --------------------------------------------------------------------------
def test_enhance_observation_shape_and_prefix():
    obs412 = [i % 7 for i in range(412)]
    out = enhance_observation(obs412)  # 所有可选依赖缺省 None
    assert len(out) == 468
    assert out[0:412] == obs412  # 前 412 维逐元素相等
    assert out[412:468] == [0] * 56  # 上下文+牌组段全零


def test_enhance_observation_with_deps():
    obs412 = [i % 5 for i in range(412)]
    card_index = [f"CARD_{i}" for i in range(110)]
    arch_table = {"card_0": [1, 0, 0, 0, 0]}
    card_effects = {
        "card_0": {"base_cost": 1, "is_attack": 1, "is_skill": 0,
                   "is_power": 0, "exhausts_self": 0, "is_status": 0},
    }
    out = enhance_observation(obs412, gc=None,
                              card_index=card_index,
                              arch_table=arch_table,
                              card_effects=card_effects)
    assert len(out) == 468
    assert out[0:412] == obs412
    # 上下文（gc=None）全零
    assert out[412:412 + 43] == [0] * 43
    # 牌组段等于 encode_deck_stats 输出
    expected_deck = encode_deck_stats(obs412, card_index=card_index,
                                      arch_table=arch_table,
                                      card_effects=card_effects)
    assert out[412 + 43:468] == expected_deck


def test_encode_deck_stats_real_starting_deck():
    """用真实 CARD_INDEX + CARD_EFFECTS 验证归一化查表。

    回归守卫：曾因 CARD_INDEX 的枚举名（"STRIKE_RED"）未归一化就去查
    CARD_EFFECTS（键为 "strike"），导致 8 维牌组统计里 7 维静默全零。
    仅用假卡名的测试发现不了这个 bug，必须用真实数据。
    """
    import slaythespire as sts
    from models.card_data import CARD_ARCHETYPE, CARD_EFFECTS
    from models.card_index import CARD_INDEX

    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
    raw = list(sts.getNNInterface().getObservation(gc))
    out = encode_deck_stats(raw, CARD_INDEX, CARD_ARCHETYPE, CARD_EFFECTS)
    assert len(out) == 13

    # 铁甲起始牌组 10 张 = 5 打击(攻,1费) + 4 防御(技,1费) + 1 重击(攻,2费)
    stats = out[5:13]
    assert stats[0] == pytest.approx(10 / 40)                        # 总牌数/40
    assert stats[1] == pytest.approx(6 / 10)                         # 攻击占比
    assert stats[2] == pytest.approx(4 / 10)                         # 技能占比
    assert stats[4] == pytest.approx((5 * 1 + 4 * 1 + 1 * 2) / 10 / 3)  # 平均费用/3

    # 防御带格挡权重，流派强度不应全零
    assert any(v > 0 for v in out[0:5])
