"""候选动作 347 维描述向量编码。

把 env.base_env.Candidate 编码成定长特征向量，给打分网络用。
纯函数：不 import slaythespire，不持有状态；同一输入必然得到同一输出。

维度布局（总计 369；0–333 段布局与 347 维版一致，[333,353) 为事件 KB 段，
[353,369) 为 V10 新增的 SKIP 专用牌组质量上下文段）：
  [0,   10)  kind one-hot（KIND_LIST）
  [10,  22)  卡牌属性：CardType one-hot(5) + CardRarity one-hot(5)
             + 费用归一化(1, cost/3 截断到 [0,1]，cost=0 时回退查卡牌字典 base_cost)
             + 是否升级(1)；非卡牌候选全零
  [22, 132)  卡牌精确 one-hot(110)：用 CARD_INDEX_MAP[card.name] 定位，
             与观测里 110 卡牌空间严格对齐；取代原 88 桶 crc32 哈希
  [132,176)  卡牌语义(44)：从 card_data.EFFECT_KEYS 取 48 字段，剔除
             铁甲卡池恒为 0 的 4 个死字段（DEAD_FIELDS），按 EFFECT_KEYS
             原序排列；数值字段归一化，布尔/rarity_score 直接写值；
             V10 在末尾追加 6 个防战/复利可分辨维度（per_turn_strength /
             block_per_turn_amt / block_leverage / couples_to_block /
             payoff_delay / exhaust_block_gain）
  [176,211)  流派交叉项(35)：a(5) ⊕ d(5) ⊕ outer(a,d)(25)，
             a=候选卡五流派归属，d=牌组流派强度（由外部传入 deck_arch），
             外积让"卡 × 牌组"协同直接成激活特征
  [211,221)  地图节点 Room 类型 one-hot（ROOM_LIST）；非地图节点全零
  [221,229)  数值特征（由 GameAction.bits 解码，见 decode_action_bits）：
             idx1/255, idx2/255, idx3/255, reward_type/6,
             is_potion, is_potion_discard, 2 维保留（绑定层拿不到价格，置 0）
  [229,277)  地图路径前瞻（map-lookahead）：向下 12 层，每层 4 维 one-hot
             ELITE/REST/SHOP/OTHER（由调用方预计算 lookahead_rooms 序列传入）
  ───── 以下为三段知识库段（与 347 维版一致，仅因语义段 +6 整体后移） ─────
  [277,301)  药水知识库特征（24 维，models.potion_kb）：
             候选对应药水奖励/商品时填充；potion_name→get_potion_features(name)，
             其余候选整段全零
  [301,333)  遗物知识库特征（32 维，models.relic_kb）：
             候选对应遗物时填充（含 BOSS_RELIC 屏全部候选），
             relic_name→get_relic_features(name)；其余全零
  [333,353)  事件知识库特征（20 维，models.event_kb）：
             kind=EVENT_OPTION 时，event_key(=event_name)+event_option_idx(=bits.idx1)
             →get_event_option_features；其余全零
  [353,369)  SKIP 专用牌组质量上下文(16)：仅 kind=SKIP 候选填充，
             内容为 encode_skip_deck_context 返回的 16 维
             （牌组流派强度5 + 牌组派生统计8 + 局势3：楼层/血量比/牌组规模），
             由训练侧(env)与真机侧(_rl_pick)共用同一函数构造，单一事实来源。
             解决"RL 不会跳牌"表征根因——旧 SKIP 向量仅 kind+reward_type 2 维非零。

注：流派交叉项需要"牌组流派强度" deck_arch，由 env/lightspeed_env.py 每步
预计算后传入；本模块保持纯函数，deck_arch=None 时交叉项 35 维全零。
KB 三段均为纯查询（models.potion_kb/relic_kb/event_kb），不持有状态；
传入的 name/key 为 None 时对应段全零，保持纯函数特性。
SKIP 牌组上下文段为纯函数：传入 deck_ctx（16 维 list）时填充，否则整段全零。
"""
from dataclasses import dataclass

from .card_data import (
    EFFECT_KEYS,
    get_base_cost,
    get_card_archetype,
    get_card_effects,
)
from .card_index import CARD_INDEX_MAP, CARD_INDEX_SIZE
from .potion_kb import get_potion_features, POTION_FEATURE_DIM
from .relic_kb import get_relic_features, RELIC_FEATURE_DIM
from .event_kb import get_event_option_features, EVENT_KB_DIM

DESC_VEC_DIM = 369

KIND_LIST = [
    "MAP_NODE", "CARD_REWARD", "REWARD", "SKIP", "SHOP_ITEM",
    "REST_OPTION", "EVENT_OPTION", "BOSS_RELIC", "TREASURE", "ACTION",
]

CARD_TYPE_LIST = ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE"]
CARD_RARITY_LIST = ["BASIC", "COMMON", "UNCOMMON", "RARE", "SPECIAL"]

ROOM_LIST = [
    "MONSTER", "ELITE", "REST", "SHOP", "TREASURE",
    "EVENT", "BOSS", "BOSS_TREASURE", "NONE", "INVALID",
]

# 流派顺序固定：力量 / 格挡 / 消耗 / 自伤 / 状态
ARCHETYPE_DIM = 5

# 铁甲卡池里没有任何一张卡用到的字段（恒为 0，留着只是噪音），编码时剔除。
# 顺序保持 EFFECT_KEYS 原序，仅删这 4 项 → 42 - 4 = 38 维语义段。
DEAD_FIELDS = ("dexterity_gain", "block_efficiency", "discard", "retain")
SEMANTIC_KEYS = [k for k in EFFECT_KEYS if k not in DEAD_FIELDS]
SEMANTIC_DIM = len(SEMANTIC_KEYS)  # 38

# 各段偏移
_OFF_KIND = 0
_OFF_CARD = 10            # 卡牌属性段：12 维
_OFF_CARD_ONEHOT = 22    # 卡牌精确 one-hot：110 维（= CARD_INDEX_SIZE）
_OFF_SEMANTIC = 132       # 卡牌语义：44 维（38 旧 + 6 V10 新增）
_OFF_CROSS = 176          # 流派交叉项：35 维
_OFF_ROOM = 211           # 房间类型 one-hot：10 维
_OFF_NUM = 221            # 数值特征：8 维
_OFF_LOOKAHEAD = 229      # 地图前瞻：48 维（12 层 × 4）
_OFF_POTION = 277         # 药水 KB：24 维（models.potion_kb）
_OFF_RELIC = 301          # 遗物 KB：32 维（models.relic_kb）
_OFF_EVENT = 333          # 事件 KB：20 维（models.event_kb）
_OFF_SKIP_CTX = 353       # SKIP 专用牌组质量上下文：16 维（V10 新增）
# 末尾 = 353 + 16 = 369 = DESC_VEC_DIM
_SKIP_CTX_DIM = 16

_CARD_ATTR_DIM = 12
_CARD_ONEHOT_DIM = CARD_INDEX_SIZE          # 110
_CROSS_DIM = ARCHETYPE_DIM + ARCHETYPE_DIM + ARCHETYPE_DIM * ARCHETYPE_DIM  # 35
_ROOM_DIM = len(ROOM_LIST)                  # 10
_NUM_DIM = 8
_LOOKAHEAD_LAYERS = 12
_LOOKAHEAD_DIM = _LOOKAHEAD_LAYERS * 4      # 48
_POTION_KB_DIM = POTION_FEATURE_DIM        # 24
_RELIC_KB_DIM = RELIC_FEATURE_DIM          # 32
_EVENT_KB_DIM = EVENT_KB_DIM                # 20

# 前瞻段房间类型 → 4 维 one-hot 下标；其余（MONSTER/TREASURE/EVENT/BOSS 等）归 OTHER
LOOKAHEAD_ROOM_MAP = {"ELITE": 0, "REST": 1, "SHOP": 2}

# 费用归一化分母：普通卡牌费用 0~3，X 费/特殊费截断到 1.0
_COST_NORM = 3.0
# 语义段数值归一化分母
_NUMERIC_NORM = {
    "base_damage": 20.0,
    "base_block": 20.0,
    "base_cost": 3.0,
    "num_hits": 5.0,
    "draw_count": 5.0,
    # V10 新增（防战/复利可分辨维度）
    "per_turn_strength": 10.0,
    "block_per_turn_amt": 10.0,
    "block_leverage": 3.0,
    "exhaust_block_gain": 10.0,
}


@dataclass
class CardInfo:
    """卡牌特征输入。由 env 从 sts.Card 提取（解耦，便于无 pyd 单测）。"""
    name: str            # CardId 枚举名，如 "STRIKE_RED"
    card_type: str       # CardType 名，未知值不置位
    rarity: str          # CardRarity 名，未知值不置位
    cost: float = 0.0    # 绑定层未暴露费用时保持 0；编码时回退查字典 base_cost
    upgraded: bool = False


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
    默认 layers=6；新版调用方传 12 取更深的路径前瞻。
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


def _encode_card_attrs(vec: list, card: CardInfo):
    base = _OFF_CARD
    if card.card_type in CARD_TYPE_LIST:
        vec[base + CARD_TYPE_LIST.index(card.card_type)] = 1.0
    if card.rarity in CARD_RARITY_LIST:
        vec[base + 5 + CARD_RARITY_LIST.index(card.rarity)] = 1.0
    vec[base + 10] = max(0.0, min(1.0, _card_cost(card) / _COST_NORM))
    vec[base + 11] = 1.0 if card.upgraded else 0.0


def _encode_card_onehot(vec: list, card: CardInfo):
    """卡牌精确 one-hot：与 CARD_INDEX 严格对齐，取代原 88 桶 crc32 哈希。"""
    idx = CARD_INDEX_MAP.get(card.name)
    if idx is not None:
        vec[_OFF_CARD_ONEHOT + idx] = 1.0
    # 不在 110 词表内的卡（如部分状态/诅咒/事件特殊卡）→ 整段保持全零


def _encode_semantic(vec: list, name: str):
    """卡牌语义 38 维：按 SEMANTIC_KEYS（EFFECT_KEYS 剔 4 死字段）。"""
    e = get_card_effects(name)
    if not e:
        return
    base = _OFF_SEMANTIC
    for i, k in enumerate(SEMANTIC_KEYS):
        v = e.get(k)
        if v is None:
            val = 0.0
        elif k in _NUMERIC_NORM:
            val = float(v) / _NUMERIC_NORM[k]
        else:
            # 布尔 0/1、rarity_score 0/0.5/1.0，以及 strength_gain=2 等计数值：直接写值
            val = float(v)
        vec[base + i] = val


def _encode_cross(vec: list, card: CardInfo, deck_arch):
    """流派交叉项 35 维：a(5) ⊕ d(5) ⊕ outer(a,d)(25)。

    a = 候选卡五流派归属（get_card_archetype）；
    d = 牌组流派强度（deck_arch，5 维，由外部传入）；
    outer[i][j] = a[i]*d[j]（含 i≠j 的有信息量交叉项）。
    非卡牌候选或 deck_arch=None 时整段保持全零。
    """
    if card is None or deck_arch is None:
        return
    a = get_card_archetype(card.name)
    d = deck_arch
    base = _OFF_CROSS
    for i in range(ARCHETYPE_DIM):
        vec[base + i] = float(a[i])
    for i in range(ARCHETYPE_DIM):
        vec[base + ARCHETYPE_DIM + i] = float(d[i])
    for i in range(ARCHETYPE_DIM):
        ai = float(a[i])
        for j in range(ARCHETYPE_DIM):
            vec[base + 2 * ARCHETYPE_DIM + i * ARCHETYPE_DIM + j] = ai * float(d[j])


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


def _encode_lookahead(vec: list, rooms):
    """rooms: 预计算的每层房间类型名序列（len<=12）。"""
    if not rooms:
        return
    for i, room in enumerate(rooms[:_LOOKAHEAD_LAYERS]):
        if room is None:
            continue
        idx = LOOKAHEAD_ROOM_MAP.get(room, 3)  # 未列出的有效房间 → OTHER
        vec[_OFF_LOOKAHEAD + i * 4 + idx] = 1.0


def _encode_potion_kb(vec: list, name):
    """药水知识库段 [271,295)：potion_name→get_potion_features（24 维）。

    name 非真值（None/空）时整段保持全零，保持纯函数。
    """
    if not name:
        return
    feats = get_potion_features(name)
    for i, v in enumerate(feats):
        vec[_OFF_POTION + i] = v


def _encode_relic_kb(vec: list, name):
    """遗物知识库段 [295,327)：relic_name→get_relic_features（32 维）。

    name 非真值（None/空）时整段保持全零，保持纯函数。
    """
    if not name:
        return
    feats = get_relic_features(name)
    for i, v in enumerate(feats):
        vec[_OFF_RELIC + i] = v


def _encode_event_kb(vec: list, event_key, option_idx):
    """事件知识库段 [333,353)：event_key+option_idx→get_event_option_features（20 维）。

    event_key 或 option_idx 为 None 时整段保持全零，保持纯函数。
    event_key 为 C++ 返回的 event_name（显示名风格），由 event_kb 内部
    normalize_event_key 对齐一级 key，未收录事件自然返回全零。
    """
    if event_key is None or option_idx is None:
        return
    feats = get_event_option_features(event_key, option_idx)
    for i, v in enumerate(feats):
        vec[_OFF_EVENT + i] = v


def _encode_skip_ctx(vec: list, deck_ctx):
    """SKIP 专用牌组质量上下文段 [353,369)：16 维。

    deck_ctx 为 encode_skip_deck_context 返回的 16 维（牌组流派5 + 牌组统计8
    + 局势3）。仅 kind=SKIP 候选传入；其余候选或 deck_ctx=None 时整段全零。
    训练侧(env)与真机侧(_rl_pick)共用同一 encode_skip_deck_context 构造，
    保证 SKIP 向量单一事实来源。
    """
    if not deck_ctx:
        return
    base = _OFF_SKIP_CTX
    for i, v in enumerate(deck_ctx[:_SKIP_CTX_DIM]):
        vec[base + i] = float(v)


def encode_candidate(candidate, *, card: CardInfo = None,
                     room: str = None, bits: int = None,
                     lookahead_rooms=None, deck_arch=None,
                     potion_name: str = None, relic_name: str = None,
                     event_key: str = None, event_option_idx: int = None,
                     deck_ctx: list = None) -> list:
    """把 Candidate 编码为 369 维特征向量（纯函数）。

    原 0–333 段布局不变；末尾 [333,353) 事件 KB 段按需填充；[353,369) 为 V10
    新增的 SKIP 专用牌组质量上下文段（仅 kind=SKIP 填充）：
      - 药水 KB [277,301)：potion_name 非 None 时填充
      - 遗物 KB [301,333)：relic_name 非 None 时填充（含 BOSS_RELIC 屏候选）
      - 事件 KB [333,353)：event_key+event_option_idx 均非 None 时填充
      - SKIP 上下文 [353,369)：deck_ctx 非 None 且 kind=SKIP 时填充
    所有新参数默认 None → 对应段全零，保持纯函数。

    candidate: env.base_env.Candidate（读 kind/label）。
    card: 候选对应的卡牌（卡牌奖励等），None 则卡牌段/语义段/交叉项全零。
    room: 候选指向的地图节点 Room 类型名，None 则房间段全零。
    bits: GameAction.bits，None 则数值段全零。
    lookahead_rooms: 地图前瞻的房间类型名序列（由调用方用 SpireMap 预计算），
        None/空则前瞻段全零。
    deck_arch: 牌组五流派强度向量 [力量,格挡,消耗,自伤,状态]（5 维），
        由 env 每步预计算传入；None 时交叉项段全零。
    potion_name / relic_name / event_key / event_option_idx:
        KB 三段的可选键；None 时对应段全零（纯函数）。
    deck_ctx: SKIP 专用 16 维牌组质量上下文（encode_skip_deck_context 产出）；
        仅 kind=SKIP 候选填充，其余候选整段全零（纯函数）。
    """
    vec = [0.0] * DESC_VEC_DIM
    _encode_kind(vec, candidate.kind)
    if card is not None:
        _encode_card_attrs(vec, card)
        _encode_card_onehot(vec, card)
        _encode_semantic(vec, card.name)
        _encode_cross(vec, card, deck_arch)
    if room is not None:
        _encode_room(vec, room)
    if bits is not None:
        _encode_numeric(vec, bits)
    if lookahead_rooms is not None:
        _encode_lookahead(vec, lookahead_rooms)
    # KB 三段（末尾追加，None 时全零）
    _encode_potion_kb(vec, potion_name)
    _encode_relic_kb(vec, relic_name)
    _encode_event_kb(vec, event_key, event_option_idx)
    # SKIP 专用牌组质量上下文段（仅 SKIP 填充）
    if candidate.kind == "SKIP":
        _encode_skip_ctx(vec, deck_ctx)
    return vec

