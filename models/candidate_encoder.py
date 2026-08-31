"""候选动作 368 维描述向量编码（M2）。

把 env.base_env.Candidate 编码成定长特征向量，给打分网络用。
纯函数：不 import slaythespire，不持有状态；同一输入必然得到同一输出
（哈希用 zlib.crc32，与 PYTHONHASHSEED 无关，跨进程确定）。

维度布局（总计 368，与 lightspeed_env._DESC_VEC_DIM 一致）：
  [0,   10)  kind one-hot（KIND_LIST）
  [10, 110)  卡牌特征：CardType one-hot(5) + CardRarity one-hot(5)
             + 费用归一化(1, cost/3 截断到 [0,1]，未知为 0) + 是否升级(1)
             + 卡牌名 crc32 % 88 哈希桶 one-hot(88)；非卡牌候选全零
  [110, 120) 地图节点 Room 类型 one-hot（ROOM_LIST）；非地图节点全零
  [120, 128) 数值特征（由 GameAction.bits 解码，见 decode_action_bits）：
             idx1/255, idx2/255, idx3/255, reward_type/6,
             is_potion, is_potion_discard, 2 维保留（绑定层拿不到价格，置 0）
  [128, 368) label 文本哈希：分词后每词 crc32 % 240 的 multi-hot（兜底语义）
"""
import re
import zlib
from dataclasses import dataclass

DESC_VEC_DIM = 368

KIND_LIST = [
    "MAP_NODE", "CARD_REWARD", "REWARD", "SKIP", "SHOP_ITEM",
    "REST_OPTION", "EVENT_OPTION", "BOSS_RELIC", "TREASURE", "ACTION",
]

CARD_TYPE_LIST = ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE"]
CARD_RARITY_LIST = ["BASIC", "COMMON", "UNCOMMON", "RARE", "SPECIAL"]
CARD_ID_BUCKETS = 88

ROOM_LIST = [
    "MONSTER", "ELITE", "REST", "SHOP", "TREASURE",
    "EVENT", "BOSS", "BOSS_TREASURE", "NONE", "INVALID",
]

# 各段偏移
_OFF_KIND = 0
_OFF_CARD = 10
_OFF_ROOM = 110
_OFF_NUM = 120
_OFF_LABEL = 128

_LABEL_BUCKETS = DESC_VEC_DIM - _OFF_LABEL  # 240

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 费用归一化分母：普通卡牌费用 0~3，X 费/特殊费截断到 1.0
_COST_NORM = 3.0


@dataclass
class CardInfo:
    """卡牌特征输入。由 env 从 sts.Card 提取（解耦，便于无 pyd 单测）。"""
    name: str            # CardId 枚举名，如 "STRIKE_RED"
    card_type: str       # CardType 名，未知值不置位
    rarity: str          # CardRarity 名，未知值不置位
    cost: float = 0.0    # 绑定层未暴露费用时保持 0
    upgraded: bool = False


def _hash_bucket(text: str, buckets: int) -> int:
    return zlib.crc32(text.encode("utf-8")) % buckets


def decode_action_bits(bits: int) -> dict:
    """解码 GameAction.bits（布局见 C++ search/GameAction.h）。

    bit31=药水动作，bit30=药水丢弃，bit27-29=RewardsActionType
    （CARD=0/GOLD=1/KEY=2/POTION=3/RELIC=4/CARD_REMOVE=5/SKIP=6），
    bit16-23=idx3，bit8-15=idx2，bit0-7=idx1。
    """
    return {
        "is_potion": bool(bits & 0x80000000),
        "is_potion_discard": bool(bits & 0x40000000),
        "reward_type": (bits >> 27) & 0x7,
        "idx3": (bits >> 16) & 0xFF,
        "idx2": (bits >> 8) & 0xFF,
        "idx1": bits & 0xFF,
    }


def _encode_kind(vec: list, kind: str):
    try:
        vec[_OFF_KIND + KIND_LIST.index(kind)] = 1.0
    except ValueError:
        raise ValueError(f"未知 candidate kind: {kind!r}（合法值见 KIND_LIST）")


def _encode_card(vec: list, card: CardInfo):
    base = _OFF_CARD
    if card.card_type in CARD_TYPE_LIST:
        vec[base + CARD_TYPE_LIST.index(card.card_type)] = 1.0
    if card.rarity in CARD_RARITY_LIST:
        vec[base + 5 + CARD_RARITY_LIST.index(card.rarity)] = 1.0
    vec[base + 10] = max(0.0, min(1.0, card.cost / _COST_NORM))
    vec[base + 11] = 1.0 if card.upgraded else 0.0
    vec[base + 12 + _hash_bucket(card.name, CARD_ID_BUCKETS)] = 1.0


def _encode_room(vec: list, room: str):
    if room in ROOM_LIST:
        vec[_OFF_ROOM + ROOM_LIST.index(room)] = 1.0


def _encode_numeric(vec: list, bits: int):
    d = decode_action_bits(bits)
    vec[_OFF_NUM + 0] = d["idx1"] / 255.0
    vec[_OFF_NUM + 1] = d["idx2"] / 255.0
    vec[_OFF_NUM + 2] = d["idx3"] / 255.0
    vec[_OFF_NUM + 3] = d["reward_type"] / 6.0
    vec[_OFF_NUM + 4] = 1.0 if d["is_potion"] else 0.0
    vec[_OFF_NUM + 5] = 1.0 if d["is_potion_discard"] else 0.0
    # +6/+7 保留：价格/数量绑定层未暴露，置 0


def _encode_label(vec: list, label: str):
    for tok in _TOKEN_RE.findall(label.lower()):
        vec[_OFF_LABEL + _hash_bucket(tok, _LABEL_BUCKETS)] = 1.0


def encode_candidate(candidate, *, card: CardInfo = None,
                     room: str = None, bits: int = None) -> list:
    """把 Candidate 编码为 368 维特征向量（纯函数）。

    candidate: env.base_env.Candidate（读 kind/label）。
    card: 候选对应的卡牌（卡牌奖励等），None 则卡牌段全零。
    room: 候选指向的地图节点 Room 类型名，None 则房间段全零。
    bits: GameAction.bits，None 则数值段全零。
    """
    vec = [0.0] * DESC_VEC_DIM
    _encode_kind(vec, candidate.kind)
    if card is not None:
        _encode_card(vec, card)
    if room is not None:
        _encode_room(vec, room)
    if bits is not None:
        _encode_numeric(vec, bits)
    if candidate.label:
        _encode_label(vec, candidate.label)
    return vec
