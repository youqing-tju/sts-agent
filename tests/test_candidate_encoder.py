"""候选动作 368 维描述向量编码的单元测试（不依赖 slaythespire pyd）。"""
import pytest

from env.base_env import Candidate
from models.candidate_encoder import (
    CardInfo,
    DESC_VEC_DIM,
    KIND_LIST,
    ROOM_LIST,
    decode_action_bits,
    encode_candidate,
)


def _cand(kind="ACTION", label="action_0x00000000", idx=0):
    return Candidate(idx=idx, kind=kind, desc_vec=[], label=label)


# ---- 维度 ----

def test_dim_is_368():
    vec = encode_candidate(_cand())
    assert len(vec) == 368
    assert len(vec) == DESC_VEC_DIM


def test_dim_matches_env_constant():
    from env.lightspeed_env import _DESC_VEC_DIM
    assert DESC_VEC_DIM == _DESC_VEC_DIM == 368


# ---- 确定性 ----

def test_deterministic():
    card = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC",
                    cost=1.0, upgraded=False)
    kwargs = dict(card=card, room="MONSTER", bits=0x12345678)
    v1 = encode_candidate(_cand("CARD_REWARD", "Take Strike"), **kwargs)
    v2 = encode_candidate(_cand("CARD_REWARD", "Take Strike"), **kwargs)
    assert v1 == v2


# ---- kind one-hot ----

def test_different_kinds_differ():
    vecs = {k: encode_candidate(_cand(k)) for k in KIND_LIST}
    # 两两不同（kind one-hot 必然不同）
    keys = list(vecs)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            assert vecs[keys[i]] != vecs[keys[j]]
    # kind 段恰有一个 1
    for k, v in vecs.items():
        assert sum(v[:10]) == 1.0
        assert v[KIND_LIST.index(k)] == 1.0


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        encode_candidate(_cand("NOPE"))


# ---- 全零输入不崩 ----

def test_zero_input_no_crash():
    vec = encode_candidate(_cand("ACTION", label=""), bits=None)
    assert len(vec) == 368
    assert sum(vec[:10]) == 1.0            # kind 仍有 one-hot
    assert all(x == 0.0 for x in vec[10:])  # 其余全零


# ---- 卡牌特征段 [10,110) ----

def test_card_features_set():
    card = CardInfo(name="BASH", card_type="ATTACK", rarity="BASIC",
                    cost=2.0, upgraded=True)
    vec = encode_candidate(_cand("CARD_REWARD"), card=card)
    seg = vec[10:110]
    assert sum(seg[0:5]) == 1.0     # CardType one-hot
    assert seg[0] == 1.0            # ATTACK 是词表第一个
    assert sum(seg[5:10]) == 1.0    # CardRarity one-hot
    assert seg[5] == 1.0            # BASIC 是词表第一个
    assert seg[10] == pytest.approx(2.0 / 3.0)  # 费用归一化
    assert seg[11] == 1.0           # upgraded
    assert sum(seg[12:100]) == 1.0  # id 哈希桶 one-hot


def test_no_card_block_all_zero():
    vec = encode_candidate(_cand("MAP_NODE"))
    assert all(x == 0.0 for x in vec[10:110])


def test_different_cards_differ():
    c1 = CardInfo(name="STRIKE_RED", card_type="ATTACK", rarity="BASIC")
    c2 = CardInfo(name="DEFEND_RED", card_type="SKILL", rarity="BASIC")
    v1 = encode_candidate(_cand(), card=c1)
    v2 = encode_candidate(_cand(), card=c2)
    assert v1[10:110] != v2[10:110]


# ---- 房间特征段 [110,120) ----

def test_room_one_hot():
    for room in ROOM_LIST:
        vec = encode_candidate(_cand("MAP_NODE"), room=room)
        seg = vec[110:120]
        assert sum(seg) == 1.0
        assert seg[ROOM_LIST.index(room)] == 1.0


def test_no_room_all_zero():
    vec = encode_candidate(_cand("EVENT_OPTION"))
    assert all(x == 0.0 for x in vec[110:120])


# ---- 数值段 [120,128) ----

def test_bits_numeric_decode():
    # reward_type=CARD(0), idx3=0x12, idx2=0x34, idx1=0x56
    bits = 0x00123456
    vec = encode_candidate(_cand("CARD_REWARD"), bits=bits)
    seg = vec[120:128]
    assert seg[0] == pytest.approx(0x56 / 255)
    assert seg[1] == pytest.approx(0x34 / 255)
    assert seg[2] == pytest.approx(0x12 / 255)
    assert seg[3] == pytest.approx(0 / 6)      # reward_type CARD=0
    assert seg[4] == 0.0                        # 非药水
    assert seg[5] == 0.0


def test_bits_potion_flags():
    bits = 0xC0000001  # 药水 + 丢弃，idx1=1
    d = decode_action_bits(bits)
    assert d["is_potion"] and d["is_potion_discard"] and d["idx1"] == 1
    vec = encode_candidate(_cand(), bits=bits)
    assert vec[124] == 1.0 and vec[125] == 1.0


def test_no_bits_numeric_zero():
    vec = encode_candidate(_cand())
    assert all(x == 0.0 for x in vec[120:128])


# ---- label 哈希段 [128,368) ----

def test_label_hash_multihot():
    vec = encode_candidate(_cand(label="Take Strike upgraded"))
    seg = vec[128:368]
    assert sum(seg) >= 3.0          # 三个词 → 至少三桶（可能撞桶，所以 >= 不成立时放宽）
    assert all(x in (0.0, 1.0) for x in seg)


def test_different_labels_differ():
    v1 = encode_candidate(_cand(label="Take Strike"))
    v2 = encode_candidate(_cand(label="Take Bash"))
    assert v1[128:] != v2[128:]


def test_same_label_same_hash():
    v1 = encode_candidate(_cand("REWARD", "hello world"))
    v2 = encode_candidate(_cand("SHOP_ITEM", "hello world"))
    assert v1[128:] == v2[128:]


# ---- 环境集成（需要 pyd，与现有测试同级要求）----

def test_env_candidates_have_real_desc_vec():
    from env.lightspeed_env import LightspeedEnv
    env = LightspeedEnv(mcts_simulations=50)
    env.reset(seed=7)
    cands = env.candidates()
    assert cands
    for c in cands:
        assert len(c.desc_vec) == 368
        assert sum(c.desc_vec[:10]) == 1.0       # kind one-hot 已填
        assert any(x != 0.0 for x in c.desc_vec)  # 不再是全零占位
