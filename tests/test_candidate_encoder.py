"""候选动作 367 维描述向量编码的单元测试（不依赖 slaythespire pyd）。

覆盖：维度、确定性、kind/room one-hot、卡牌精确 one-hot、语义段（手算）、
流派交叉项（手算含 i≠j 外积）、label 哈希兜底、地图前瞻（12 层）、bits 解码。

布局偏移与 candidate_encoder 内部一致，测试里显式写死以便发现布局漂移。
"""
import pytest

from env.base_env import Candidate
from models.candidate_encoder import (
    ARCHETYPE_DIM,
    CardInfo,
    CARD_INDEX_MAP,
    DESC_VEC_DIM,
    KIND_LIST,
    ROOM_LIST,
    SEMANTIC_KEYS,
    SEMANTIC_DIM,
    compute_map_lookahead,
    decode_action_bits,
    encode_candidate,
)

# 各段偏移（与 candidate_encoder 内部一致）
_OFF_KIND = 0
_OFF_CARD = 10
_OFF_CARD_ONEHOT = 22
_OFF_SEMANTIC = 132
_OFF_CROSS = 170
_OFF_ROOM = 205
_OFF_NUM = 215
_OFF_LOOKAHEAD = 223
_OFF_LABEL = 271

# 卡牌精确 one-hot 维度（= CARD_INDEX_SIZE）
_CARD_ONEHOT_DIM = 110
# 流派交叉项维度 = 5 + 5 + 25
_CROSS_DIM = ARCHETYPE_DIM * ARCHETYPE_DIM + 2 * ARCHETYPE_DIM  # 35
_LOOKAHEAD_LAYERS = 12


def _cand(kind="ACTION", label="action_0x00000000", idx=0):
    return Candidate(idx=idx, kind=kind, desc_vec=[], label=label)


# ---- 维度 ----

def test_dim_is_367():
    vec = encode_candidate(_cand())
    assert len(vec) == 367
    assert len(vec) == DESC_VEC_DIM
    # 后端：语义段确实是 38 维、交叉项 35 维
    assert SEMANTIC_DIM == 38
    assert _CROSS_DIM == 35


def test_layout_offsets_sum_to_dim():
    # 段 1..9 偏移与维度自检，确保总和为 367，无重叠/空洞
    assert _OFF_SEMANTIC - _OFF_CARD_ONEHOT == _CARD_ONEHOT_DIM   # 110
    assert _OFF_CROSS - _OFF_SEMANTIC == SEMANTIC_DIM            # 38
    assert _OFF_ROOM - _OFF_CROSS == _CROSS_DIM                 # 35
    assert _OFF_NUM - _OFF_ROOM == len(ROOM_LIST)               # 10
    assert _OFF_LOOKAHEAD - _OFF_NUM == 8                       # 8
    assert DESC_VEC_DIM - _OFF_LABEL == 96                      # 96
    assert _OFF_LABEL == 271
    assert DESC_VEC_DIM == 367


# ---- 确定性 ----

def test_deterministic():
    card = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC",
                    cost=1.0, upgraded=False)
    kwargs = dict(card=card, room="MONSTER", bits=0x12345678,
                  lookahead_rooms=["MONSTER", "ELITE", "REST"],
                  deck_arch=[0.5, 0.2, 0.0, 0.3, 0.1])
    v1 = encode_candidate(_cand("CARD_REWARD", "Take Strike"), **kwargs)
    v2 = encode_candidate(_cand("CARD_REWARD", "Take Strike"), **kwargs)
    assert v1 == v2


# ---- kind one-hot ----

def test_different_kinds_differ():
    vecs = {k: encode_candidate(_cand(k)) for k in KIND_LIST}
    keys = list(vecs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert vecs[keys[i]] != vecs[keys[j]]
    for k, v in vecs.items():
        assert sum(v[:10]) == 1.0
        assert v[KIND_LIST.index(k)] == 1.0


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        encode_candidate(_cand("NOPE"))


# ---- 全零输入不崩 ----

def test_zero_input_no_crash():
    vec = encode_candidate(_cand("ACTION", label=""), bits=None)
    assert len(vec) == 367
    assert sum(vec[:10]) == 1.0            # kind 仍有 one-hot
    assert all(x == 0.0 for x in vec[10:])  # 其余全零


# ---- 卡牌属性段 [10,22) ----

def test_card_attrs_set():
    card = CardInfo(name="BASH", card_type="ATTACK", rarity="BASIC",
                    cost=2.0, upgraded=True)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    seg = vec[_OFF_CARD:_OFF_CARD + 12]
    assert sum(seg[0:5]) == 1.0     # CardType one-hot
    assert seg[0] == 1.0            # ATTACK 词表第一个
    assert sum(seg[5:10]) == 1.0    # CardRarity one-hot
    assert seg[5] == 1.0            # BASIC 词表第一个
    assert seg[10] == pytest.approx(2.0 / 3.0)  # 费用归一化
    assert seg[11] == 1.0           # upgraded


def test_cost_falls_back_to_card_data():
    """cost=0（绑定层拿不到）时回退查字典 base_cost（Bash=2）。"""
    card = CardInfo(name="BASH", card_type="ATTACK", rarity="BASIC",
                    cost=0.0, upgraded=False)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    assert vec[_OFF_CARD + 10] == pytest.approx(2.0 / 3.0)


def test_no_card_block_all_zero():
    vec = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec[_OFF_CARD:_OFF_CARD + 12])


def test_different_cards_differ():
    c1 = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC")
    c2 = CardInfo(name="DEFEND_RED", card_type="SKILL", rarity="BASIC")
    v1 = encode_candidate(_cand(), card=c1)
    v2 = encode_candidate(_cand(), card=c2)
    assert v1[_OFF_CARD:_OFF_CARD + 12] != v2[_OFF_CARD:_OFF_CARD + 12]


# ---- 卡牌精确 one-hot 段 [22,132) ----

def test_card_precise_onehot_single_bit():
    """STRIKE_RED 只在 22 + CARD_INDEX_MAP['STRIKE_RED'] 置位，其余段不受污染。"""
    card = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC",
                    cost=1.0)
    vec = encode_candidate(_cand("CARD_REWARD", "Take Strike"), card=card)
    seg = vec[_OFF_CARD_ONEHOT:_OFF_CARD_ONEHOT + _CARD_ONEHOT_DIM]
    expected_idx = CARD_INDEX_MAP["STRIKE_RED"]
    # 整段只有一个 1.0，且位置严格为 22 + map 下标（落在该段内，不外溢）
    single = [0.0] * _CARD_ONEHOT_DIM
    single[expected_idx] = 1.0
    assert seg == single
    assert vec[_OFF_CARD_ONEHOT + expected_idx] == 1.0
    assert _OFF_CARD_ONEHOT <= _OFF_CARD_ONEHOT + expected_idx < _OFF_SEMANTIC


def test_different_cards_different_onehot():
    c1 = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC")
    c2 = CardInfo(name="DEFEND_RED", card_type="SKILL", rarity="BASIC")
    v1 = encode_candidate(_cand(), card=c1)
    v2 = encode_candidate(_cand(), card=c2)
    assert v1[_OFF_CARD_ONEHOT:_OFF_CARD_ONEHOT + _CARD_ONEHOT_DIM] != \
           v2[_OFF_CARD_ONEHOT:_OFF_CARD_ONEHOT + _CARD_ONEHOT_DIM]


def test_unknown_card_onehot_zero():
    """不在 110 词表内的卡，精确 one-hot 段整段保持全零（不崩）。"""
    card = CardInfo(name="SOME_EVENT_SPECIAL_CARD", card_type="ATTACK",
                    rarity="SPECIAL")
    vec = encode_candidate(_cand(), card=card)
    seg = vec[_OFF_CARD_ONEHOT:_OFF_CARD_ONEHOT + _CARD_ONEHOT_DIM]
    assert all(x == 0.0 for x in seg)


def test_no_card_onehot_zero():
    vec = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec[_OFF_CARD_ONEHOT:_OFF_CARD_ONEHOT + _CARD_ONEHOT_DIM])


# ---- 卡牌语义段 [132,170)（手算期望值）----

def test_card_semantic_body_slam():
    """全身撞击：只有 block_to_damage / conditional_play / is_attack / base_cost 有值。"""
    card = CardInfo(name="BODY_SLAM", card_type="ATTACK", rarity="COMMON",
                    cost=1.0)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    seg = vec[_OFF_SEMANTIC:_OFF_SEMANTIC + SEMANTIC_DIM]
    # 手算：除以下 4 位外全零
    assert seg[2] == pytest.approx(1.0 / 3.0)   # base_cost=1 → /3
    assert seg[20] == 1.0                        # block_to_damage
    assert seg[32] == 1.0                        # conditional_play
    assert seg[35] == 1.0                        # is_attack
    # 仅这 4 个非零
    nz = [(i, v) for i, v in enumerate(seg) if v != 0.0]
    assert len(nz) == 4
    assert sum(seg) == pytest.approx(1.0 / 3.0 + 1.0 + 1.0 + 1.0)


def test_card_semantic_immolate_numeric_and_rarity():
    """火祭：数值归一化 + rarity_score 直接写值（1.0）。"""
    card = CardInfo(name="IMMOLATE", card_type="ATTACK", rarity="RARE",
                    cost=2.0)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    seg = vec[_OFF_SEMANTIC:_OFF_SEMANTIC + SEMANTIC_DIM]
    assert seg[0] == pytest.approx(21.0 / 20.0)  # base_damage=21 → /20
    assert seg[2] == pytest.approx(2.0 / 3.0)    # base_cost=2 → /3
    assert seg[24] == 1.0                         # injects_status_self
    assert seg[34] == 1.0                         # is_aoe
    assert seg[35] == 1.0                         # is_attack
    assert seg[37] == 1.0                         # rarity_score = 1.0（直接写值）


def test_card_semantic_non_card_zero():
    vec = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec[_OFF_SEMANTIC:_OFF_SEMANTIC + SEMANTIC_DIM])


# ---- 流派交叉项段 [170,205)（手算期望值，含 i≠j 外积）----

def test_cross_one_stream_inferno():
    """INFLAME 流派归属 [1,0,0,0,0] × 牌组 [0.5,0.2,0,0.3,0.1]。"""
    card = CardInfo(name="INFLAME", card_type="POWER", rarity="UNCOMMON",
                    cost=1.0)
    deck_arch = [0.5, 0.2, 0.0, 0.3, 0.1]
    vec = encode_candidate(_cand("CARD_REWARD"), card=card, deck_arch=deck_arch)
    seg = vec[_OFF_CROSS:_OFF_CROSS + _CROSS_DIM]
    # a(5): [1,0,0,0,0]; d(5): [0.5,0.2,0,0.3,0.1]
    # outer[0][j] = 1*d[j]（i=0 行），其余行全 0
    expected = ([1.0, 0.0, 0.0, 0.0, 0.0]      # a(5)
                + [0.5, 0.2, 0.0, 0.3, 0.1]    # d(5)
                + [0.5, 0.2, 0.0, 0.3, 0.1]    # outer i=0 行：a[0]=1 → = d
                + [0.0] * 20)                  # outer i=1..4 行：a=0
    assert seg == pytest.approx(expected)


def test_cross_multi_stream_outer_offdiag():
    """BLOODLETTING [0,0,0.5,0.8,0] × 牌组 [0.5,0.2,0,0.3,0.1]：
    验证 i≠j 外积项（如 a[2]*d[3]=0.5*0.3=0.15）被写入。"""
    card = CardInfo(name="BLOODLETTING", card_type="SKILL", rarity="UNCOMMON",
                    cost=0.0)
    deck_arch = [0.5, 0.2, 0.0, 0.3, 0.1]
    vec = encode_candidate(_cand("CARD_REWARD"), card=card, deck_arch=deck_arch)
    seg = vec[_OFF_CROSS:_OFF_CROSS + _CROSS_DIM]
    expected = ([0.0, 0.0, 0.5, 0.8, 0.0]          # a
                + [0.5, 0.2, 0.0, 0.3, 0.1]        # d
                + [0.0] * 10                        # i=0,1 行（a=0）
                + [0.25, 0.1, 0.0, 0.15, 0.05]     # i=2 行：a[2]=0.5
                + [0.4, 0.16, 0.0, 0.24, 0.08]     # i=3 行：a[3]=0.8
                + [0.0] * 5)                        # i=4 行（a=0）
    assert seg == pytest.approx(expected)
    # i≠j 项确实非零（a[2]*d[3] = 0.15，位于 outer 第 2 行第 3 列 = idx 10+2*5+3）
    assert seg[10 + 2 * 5 + 3] == pytest.approx(0.15)


def test_cross_none_is_zero():
    """deck_arch=None 时交叉项 35 维全零（即使有卡）。"""
    card = CardInfo(name="INFLAME", card_type="POWER", rarity="UNCOMMON",
                    cost=1.0)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card, deck_arch=None)
    seg = vec[_OFF_CROSS:_OFF_CROSS + _CROSS_DIM]
    assert all(x == 0.0 for x in seg)


def test_cross_non_card_is_zero():
    """非卡牌候选（无 card）交叉项全零，即使传了 deck_arch。"""
    deck_arch = [0.5, 0.2, 0.0, 0.3, 0.1]
    vec = encode_candidate(_cand("MAP_NODE"), deck_arch=deck_arch)
    seg = vec[_OFF_CROSS:_OFF_CROSS + _CROSS_DIM]
    assert all(x == 0.0 for x in seg)


# ---- 房间特征段 [205,215) ----

def test_room_one_hot():
    for room in ROOM_LIST:
        vec = encode_candidate(_cand("MAP_NODE"), room=room)
        seg = vec[_OFF_ROOM:_OFF_ROOM + len(ROOM_LIST)]
        assert sum(seg) == 1.0
        assert seg[ROOM_LIST.index(room)] == 1.0


def test_no_room_all_zero():
    vec = encode_candidate(_cand("EVENT_OPTION"))
    assert all(x == 0.0 for x in vec[_OFF_ROOM:_OFF_ROOM + len(ROOM_LIST)])


# ---- 数值段 [215,223) ----

def test_bits_numeric_decode():
    bits = 0x00123456
    vec = encode_candidate(_cand("CARD_REWARD"), bits=bits)
    seg = vec[_OFF_NUM:_OFF_NUM + 8]
    assert seg[0] == pytest.approx(0x56 / 255)
    assert seg[1] == pytest.approx(0x34 / 255)
    assert seg[2] == pytest.approx(0x12 / 255)
    assert seg[3] == pytest.approx(0 / 6)
    assert seg[4] == 0.0
    assert seg[5] == 0.0


def test_bits_potion_flags():
    bits = 0xC0000001
    d = decode_action_bits(bits)
    assert d["is_potion"] and d["is_potion_discard"] and d["idx1"] == 1
    vec = encode_candidate(_cand(), bits=bits)
    assert vec[_OFF_NUM + 4] == 1.0 and vec[_OFF_NUM + 5] == 1.0


def test_no_bits_numeric_zero():
    vec = encode_candidate(_cand())
    assert all(x == 0.0 for x in vec[_OFF_NUM:_OFF_NUM + 8])


# ---- label 哈希段 [271,367) ----

def test_label_hash_96_buckets():
    vec = encode_candidate(_cand(label="Take Strike upgraded"))
    seg = vec[_OFF_LABEL:DESC_VEC_DIM]
    assert len(seg) == 96                      # 96 桶兜底哈希
    assert _OFF_LABEL >= 271 and DESC_VEC_DIM == 367
    assert sum(seg) >= 1.0                     # 至少命中一个桶
    assert all(x in (0.0, 1.0) for x in seg)   # multi-hot


def test_different_labels_differ():
    v1 = encode_candidate(_cand(label="Take Strike"))
    v2 = encode_candidate(_cand(label="Take Bash"))
    assert v1[_OFF_LABEL:DESC_VEC_DIM] != v2[_OFF_LABEL:DESC_VEC_DIM]


def test_same_label_same_hash():
    v1 = encode_candidate(_cand("REWARD", "hello world"))
    v2 = encode_candidate(_cand("SHOP_ITEM", "hello world"))
    assert v1[_OFF_LABEL:DESC_VEC_DIM] == v2[_OFF_LABEL:DESC_VEC_DIM]


# ---- 地图前瞻段 [223,271)（12 层 × 4）----

def test_map_lookahead_onehot_12_layers():
    rooms = ["ELITE", "REST", "SHOP", "MONSTER", "BOSS", "EVENT",
             "ELITE", "SHOP", "REST", "MONSTER", "EVENT", "BOSS"]
    vec = encode_candidate(_cand("MAP_NODE"), lookahead_rooms=rooms)
    seg = vec[_OFF_LOOKAHEAD:_OFF_LOOKAHEAD + _LOOKAHEAD_LAYERS * 4]
    assert len(seg) == 48                      # 12 层 × 4
    for i, room in enumerate(rooms):
        off = i * 4
        idx = {"ELITE": 0, "REST": 1, "SHOP": 2}.get(room, 3)
        assert seg[off:off + 4] == [1.0 if j == idx else 0.0 for j in range(4)]


def test_map_lookahead_partial_and_none():
    vec = encode_candidate(_cand("MAP_NODE"), lookahead_rooms=["ELITE"])
    assert vec[_OFF_LOOKAHEAD:_OFF_LOOKAHEAD + 4] == [1.0, 0.0, 0.0, 0.0]
    assert all(x == 0.0 for x in vec[_OFF_LOOKAHEAD + 4:_OFF_LOOKAHEAD + 48])

    vec0 = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec0[_OFF_LOOKAHEAD:_OFF_LOOKAHEAD + 48])


def test_compute_map_lookahead_greedy():
    """纯函数前瞻：每层取离当前 x 最近的子节点。"""
    edges = {(0, 0): [1, 2], (1, 1): [0, 1, 2], (2, 2): [2]}
    rooms = {(1, 1): "ELITE", (2, 1): "REST", (0, 2): "MONSTER",
             (1, 2): "SHOP", (2, 2): "EVENT", (2, 3): "BOSS"}

    def edge_fn(x, y, x2):
        return x2 in edges.get((x, y), [])

    def room_fn(x, y):
        return rooms.get((x, y))

    out = compute_map_lookahead(0, 0, room_fn, edge_fn)
    assert out == ["ELITE", "SHOP"]
