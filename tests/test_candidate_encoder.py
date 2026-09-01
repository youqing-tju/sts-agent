"""候选动作 408 维描述向量编码的单元测试（不依赖 slaythespire pyd）。"""
import pytest

from env.base_env import Candidate
from models.candidate_encoder import (
    CardInfo,
    DESC_VEC_DIM,
    KIND_LIST,
    ROOM_LIST,
    compute_map_lookahead,
    decode_action_bits,
    encode_candidate,
)

# 各段偏移（与 candidate_encoder 内部一致，测试里显式写死以便发现布局漂移）
_OFF_CARD = 10
_OFF_ROOM = 110
_OFF_NUM = 120
_OFF_LABEL = 128
_OFF_LOOKAHEAD = 368
_OFF_EFFECTS = 392


def _cand(kind="ACTION", label="action_0x00000000", idx=0):
    return Candidate(idx=idx, kind=kind, desc_vec=[], label=label)


# ---- 维度 ----

def test_dim_is_408():
    vec = encode_candidate(_cand())
    assert len(vec) == 408
    assert len(vec) == DESC_VEC_DIM


def test_dim_matches_env_constant():
    from env.lightspeed_env import _DESC_VEC_DIM
    assert DESC_VEC_DIM == _DESC_VEC_DIM == 408


# ---- 确定性 ----

def test_deterministic():
    card = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC",
                    cost=1.0, upgraded=False)
    kwargs = dict(card=card, room="MONSTER", bits=0x12345678,
                  lookahead_rooms=["MONSTER", "ELITE", "REST"])
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
    assert len(vec) == 408
    assert sum(vec[:10]) == 1.0            # kind 仍有 one-hot
    assert all(x == 0.0 for x in vec[10:])  # 其余全零


# ---- 卡牌特征段 [10,110) ----

def test_card_features_set():
    card = CardInfo(name="BASH", card_type="ATTACK", rarity="BASIC",
                    cost=2.0, upgraded=True)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    seg = vec[_OFF_CARD:110]
    assert sum(seg[0:5]) == 1.0     # CardType one-hot
    assert seg[0] == 1.0            # ATTACK 是词表第一个
    assert sum(seg[5:10]) == 1.0    # CardRarity one-hot
    assert seg[5] == 1.0            # BASIC 是词表第一个
    assert seg[10] == pytest.approx(2.0 / 3.0)  # 费用归一化
    assert seg[11] == 1.0           # upgraded
    assert sum(seg[12:100]) == 1.0  # id 哈希桶 one-hot


def test_cost_falls_back_to_card_data():
    """cost=0（绑定层拿不到）时回退查字典 base_cost（Bash=2）。"""
    card = CardInfo(name="BASH", card_type="ATTACK", rarity="BASIC",
                    cost=0.0, upgraded=False)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    assert vec[_OFF_CARD + 10] == pytest.approx(2.0 / 3.0)


def test_no_card_block_all_zero():
    vec = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec[_OFF_CARD:110])


def test_different_cards_differ():
    c1 = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC")
    c2 = CardInfo(name="DEFEND_RED", card_type="SKILL", rarity="BASIC")
    v1 = encode_candidate(_cand(), card=c1)
    v2 = encode_candidate(_cand(), card=c2)
    assert v1[_OFF_CARD:110] != v2[_OFF_CARD:110]


# ---- 房间特征段 [110,120) ----

def test_room_one_hot():
    for room in ROOM_LIST:
        vec = encode_candidate(_cand("MAP_NODE"), room=room)
        seg = vec[_OFF_ROOM:120]
        assert sum(seg) == 1.0
        assert seg[ROOM_LIST.index(room)] == 1.0


def test_no_room_all_zero():
    vec = encode_candidate(_cand("EVENT_OPTION"))
    assert all(x == 0.0 for x in vec[_OFF_ROOM:120])


# ---- 数值段 [120,128) ----

def test_bits_numeric_decode():
    bits = 0x00123456
    vec = encode_candidate(_cand("CARD_REWARD"), bits=bits)
    seg = vec[_OFF_NUM:128]
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
    assert all(x == 0.0 for x in vec[_OFF_NUM:128])


# ---- label 哈希段 [128,368) ----

def test_label_hash_multihot():
    vec = encode_candidate(_cand(label="Take Strike upgraded"))
    seg = vec[_OFF_LABEL:368]
    assert sum(seg) >= 1.0
    assert all(x in (0.0, 1.0) for x in seg)


def test_different_labels_differ():
    v1 = encode_candidate(_cand(label="Take Strike"))
    v2 = encode_candidate(_cand(label="Take Bash"))
    assert v1[_OFF_LABEL:368] != v2[_OFF_LABEL:368]


def test_same_label_same_hash():
    v1 = encode_candidate(_cand("REWARD", "hello world"))
    v2 = encode_candidate(_cand("SHOP_ITEM", "hello world"))
    assert v1[_OFF_LABEL:368] == v2[_OFF_LABEL:368]


# ---- 地图前瞻段 [368,392) ----

def test_map_lookahead_onehot():
    rooms = ["ELITE", "REST", "SHOP", "MONSTER", "BOSS", "EVENT"]
    vec = encode_candidate(_cand("MAP_NODE"), lookahead_rooms=rooms)
    seg = vec[_OFF_LOOKAHEAD:_OFF_EFFECTS]
    # layer0 ELITE -> [1,0,0,0]
    assert seg[0:4] == [1.0, 0.0, 0.0, 0.0]
    # layer1 REST  -> [0,1,0,0]
    assert seg[4:8] == [0.0, 1.0, 0.0, 0.0]
    # layer2 SHOP  -> [0,0,1,0]
    assert seg[8:12] == [0.0, 0.0, 1.0, 0.0]
    # layer3-5 MONSTER/BOSS/EVENT -> OTHER [0,0,0,1]
    assert seg[12:16] == [0.0, 0.0, 0.0, 1.0]
    assert seg[16:20] == [0.0, 0.0, 0.0, 1.0]
    assert seg[20:24] == [0.0, 0.0, 0.0, 1.0]


def test_map_lookahead_partial_and_none():
    vec = encode_candidate(_cand("MAP_NODE"), lookahead_rooms=["ELITE"])
    assert vec[_OFF_LOOKAHEAD:_OFF_LOOKAHEAD + 4] == [1.0, 0.0, 0.0, 0.0]
    assert all(x == 0.0 for x in vec[_OFF_LOOKAHEAD + 4:_OFF_EFFECTS])

    vec0 = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec0[_OFF_LOOKAHEAD:_OFF_EFFECTS])


def test_compute_map_lookahead_greedy():
    """纯函数前瞻：每层取离当前 x 最近的子节点。"""
    # 三层梯形：0->{1,2}, 1->{0,1,2}, 2->{2}
    edges = {(0, 0): [1, 2], (1, 1): [0, 1, 2], (2, 2): [2]}
    rooms = {(1, 1): "ELITE", (2, 1): "REST", (0, 2): "MONSTER",
             (1, 2): "SHOP", (2, 2): "EVENT", (2, 3): "BOSS"}

    def edge_fn(x, y, x2):
        return x2 in edges.get((x, y), [])

    def room_fn(x, y):
        return rooms.get((x, y))

    # 从 (0,0) 出发：子节点 {1,2} 取离 0 最近的 1 -> (1,1) ELITE
    # 从 (1,1) 出发：子节点 {0,1,2} 取离 1 最近的 1 -> (1,2) SHOP
    # 从 (1,2) 出发：无子节点 -> 终止
    out = compute_map_lookahead(0, 0, room_fn, edge_fn)
    assert out == ["ELITE", "SHOP"]


# ---- 卡牌效果段 [392,408) ----

def test_card_effects_bash():
    card = CardInfo(name="BASH", card_type="ATTACK", rarity="BASIC",
                    cost=0.0, upgraded=False)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    seg = vec[_OFF_EFFECTS:408]
    assert seg[0] == pytest.approx(8 / 20)     # base_damage
    assert seg[1] == pytest.approx(0.0)        # base_block (None)
    assert seg[2] == pytest.approx(2 / 20)     # base_cost
    assert seg[11] == 1.0                      # applies_vulnerable
    assert seg[3] == 0.0 and seg[4] == 0.0     # 非 power/非 aoe


def test_card_effects_non_card_zero():
    vec = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec[_OFF_EFFECTS:408])


# ---- 环境集成（需要 pyd）----

def test_env_candidates_have_real_desc_vec():
    from env.lightspeed_env import LightspeedEnv
    env = LightspeedEnv(mcts_simulations=50)
    env.reset(seed=7)
    cands = env.candidates()
    assert cands
    for c in cands:
        assert len(c.desc_vec) == 408
        assert sum(c.desc_vec[:10]) == 1.0
        assert any(x != 0.0 for x in c.desc_vec)
