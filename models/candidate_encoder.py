"""候选动作 408 维描述向量编码。

把 env.base_env.Candidate 编码成定长特征向量，给打分网络用。
纯函数：不 import slaythespire，不持有状态；同一输入必然得到同一输出
（哈希用 zlib.crc32，与 PYTHONHASHSEED 无关，跨进程确定）。

维度布局（总计 408，与 lightspeed_env._DESC_VEC_DIM 一致）：
  [0,   10)  kind one-hot（KIND_LIST）
  [10, 110)  卡牌特征：CardType one-hot(5) + CardRarity one-hot(5)
             + 费用归一化(1, cost/3 截断到 [0,1]，cost=0 时回退查卡牌字典 base_cost)
             + 是否升级(1) + 卡牌名 crc32 % 88 哈希桶 one-hot(88)；非卡牌候选全零
  [110, 120) 地图节点 Room 类型 one-hot（ROOM_LIST）；非地图节点全零
  [120, 128) 数值特征（由 GameAction.bits 解码，见 decode_action_bits）：
             idx1/255, idx2/255, idx3/255, reward_type/6,
             is_potion, is_potion_discard, 2 维保留（绑定层拿不到价格，置 0）
  [128, 368) label 文本哈希：分词后每词 crc32 % 240 的 multi-hot（兜底语义）
  [368, 392) 地图路径前瞻（map-lookahead）：向下 6 层，每层 4 维 one-hot
             ELITE/REST/SHOP/OTHER（由调用方预计算 lookahead_rooms 序列传入）
  [392, 408) 卡牌效果（card-effects）16 维：按卡名查 models.card_data 字典，
             base_damage/block/cost ÷20，num_hits/draw ÷5，布尔 0/1
"""
import re
import zlib
from dataclasses import dataclass

from .card_data import get_base_cost, get_card_effects

DESC_VEC_DIM = 408

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
_OFF_LOOKAHEAD = 368
_OFF_EFFECTS = 392

_LABEL_BUCKETS = _OFF_LOOKAHEAD - _OFF_LABEL  # 240
_LOOKAHEAD_LAYERS = (_OFF_EFFECTS - _OFF_LOOKAHEAD) // 4  # 6

# 前瞻段房间类型 → 4 维 one-hot 下标；其余（MONSTER/TREASURE/EVENT/BOSS 等）归 OTHER
LOOKAHEAD_ROOM_MAP = {"ELITE": 0, "REST": 1, "SHOP": 2}

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# 费用归一化分母：普通卡牌费用 0~3，X 费/特殊费截断到 1.0
_COST_NORM = 3.0
# 效果段数值归一化分母
_DMG_NORM = 20.0
_HITS_NORM = 5.0


@dataclass
class CardInfo:
    """卡牌特征输入。由 env 从 sts.Card 提取（解耦，便于无 pyd 单测）。"""
    name: str            # CardId 枚举名，如 "STRIKE_RED"
    card_type: str       # CardType 名，未知值不置位
    rarity: str          # CardRarity 名，未知值不置位
    cost: float = 0.0    # 绑定层未暴露费用时保持 0；编码时回退查字典 base_cost
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


def compute_map_lookahead(start_x, start_y, room_fn, edge_fn, layers=6) -> list:
    """从 (start_x, start_y) 向下贪婪前瞻，返回每层房间类型名列表（纯函数）。

    room_fn(x, y) -> str|None：房间类型名，None 表示无效。
    edge_fn(x, y, x2) -> bool：(x,y) 节点是否有通向下一层 x2 的边。
    每层在所有子节点中取离当前 x 最近的；越界/无子节点提前终止。
    """
    out = []
    cx, cy = start_x, start_y
    for _ in range(layers):
        ny = cy + 1
        children = [x2 for x2 in range(7) if edge_fn(cx, cy, x2)]
        if not children:
            break
        nx = min(children, key=lambda v: abs(v - cx))
        room = room_fn(nx, ny)
        out.append(room)
        cx, cy = nx, ny
    return out


def _encode_kind(vec: list, kind: str):
    try:
        vec[_OFF_KIND + KIND_LIST.index(kind)] = 1.0
    except ValueError:
        raise ValueError(f"未知 candidate kind: {kind!r}（合法值见 KIND_LIST）")


def _card_cost(card: CardInfo) -> float:
    """卡牌段费用：CardInfo.cost 非 0 直接用；否则回退查字典 base_cost。"""
    if card.cost and card.cost > 0:
        return card.cost
    return get_base_cost(card.name)


def _encode_card(vec: list, card: CardInfo):
    base = _OFF_CARD
    if card.card_type in CARD_TYPE_LIST:
        vec[base + CARD_TYPE_LIST.index(card.card_type)] = 1.0
    if card.rarity in CARD_RARITY_LIST:
        vec[base + 5 + CARD_RARITY_LIST.index(card.rarity)] = 1.0
    vec[base + 10] = max(0.0, min(1.0, _card_cost(card) / _COST_NORM))
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


def _encode_lookahead(vec: list, rooms):
    """rooms: 预计算的每层房间类型名序列（len<=6）。"""
    if not rooms:
        return
    for i, room in enumerate(rooms[: _LOOKAHEAD_LAYERS]):
        if room is None:
            continue
        idx = LOOKAHEAD_ROOM_MAP.get(room, 3)  # 未列出的有效房间 → OTHER
        vec[_OFF_LOOKAHEAD + i * 4 + idx] = 1.0


def _encode_card_effects(vec: list, name: str):
    e = get_card_effects(name)
    if not e:
        return
    base = _OFF_EFFECTS
    vec[base + 0] = (e["base_damage"] or 0.0) / _DMG_NORM
    vec[base + 1] = (e["base_block"] or 0.0) / _DMG_NORM
    vec[base + 2] = (e["base_cost"] or 0.0) / _DMG_NORM
    vec[base + 3] = 1.0 if e["is_power"] else 0.0
    vec[base + 4] = 1.0 if e["is_aoe"] else 0.0
    vec[base + 5] = (e["num_hits"] or 0.0) / _HITS_NORM
    vec[base + 6] = (e["draw_count"] or 0.0) / _HITS_NORM
    for i, k in enumerate(("exhausts", "self_damage", "retain",
                           "scales_strength", "applies_vulnerable",
                           "applies_weak", "gains_block_scaling",
                           "is_status", "is_curse"), start=7):
        vec[base + i] = 1.0 if e[k] else 0.0


def encode_candidate(candidate, *, card: CardInfo = None,
                     room: str = None, bits: int = None,
                     lookahead_rooms=None) -> list:
    """把 Candidate 编码为 408 维特征向量（纯函数）。

    candidate: env.base_env.Candidate（读 kind/label）。
    card: 候选对应的卡牌（卡牌奖励等），None 则卡牌段/效果段全零。
    room: 候选指向的地图节点 Room 类型名，None 则房间段全零。
    bits: GameAction.bits，None 则数值段全零。
    lookahead_rooms: 地图前瞻的房间类型名序列（由调用方用 SpireMap 预计算），
        None/空则前瞻段全零。
    """
    vec = [0.0] * DESC_VEC_DIM
    _encode_kind(vec, candidate.kind)
    if card is not None:
        _encode_card(vec, card)
        _encode_card_effects(vec, card.name)
    if room is not None:
        _encode_room(vec, room)
    if bits is not None:
        _encode_numeric(vec, bits)
    if candidate.label:
        _encode_label(vec, candidate.label)
    if lookahead_rooms is not None:
        _encode_lookahead(vec, lookahead_rooms)
    return vec
