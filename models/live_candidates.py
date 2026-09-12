"""真机候选构造器：从 Communication Mod JSON 直接构造候选向量。

绕开"模拟器枚举不全"的瓶颈（2026-09-11 实测：initFromJson 灌入真机帧后
CARD_SELECT=0 / EVENT=0 / MAP=1，RL 在真机上从未真正上场——69 次调用 vs
351 次降级，宏观决策全被规则兜底接管）。但：
  · obs 核心段灌入后完全准确（HP/金币/楼层/牌组全对，实测）
  · 候选可以由真机数据手工构造（卡名/房间类型/事件选项都在帧里）
  · V9 对这样构造的候选打分有明确区分度（选牌实测 6.3 分差）
→ 用本模块绕开枚举瓶颈，让 V9 在真机上真正参与决策。

bits 反向构造（与训练时的 GameAction.bits 语义对齐，见
models/candidate_encoder.decode_action_bits）：
  bit31=药水 / bit30=药水丢弃 / bit27-29=reward_type / bit16-23=idx3 /
  bit8-15=idx2 / bit0-7=idx1
"""
from __future__ import annotations

from models.candidate_encoder import CardInfo

# 地图 symbol → ROOM_LIST 里的房间类型名（candidate_encoder.ROOM_LIST）
SYMBOL_TO_ROOM = {
    "M": "MONSTER", "E": "ELITE", "R": "REST", "$": "SHOP",
    "T": "TREASURE", "?": "EVENT", "B": "BOSS",
}


def _card_info_from_json(c: dict) -> CardInfo:
    try:
        cost = float(c.get("cost", 1))
    except Exception:
        cost = 1.0
    if cost is None or cost < 0:
        cost = 0.0
    return CardInfo(
        name=str(c.get("id") or ""),
        card_type=str(c.get("type") or "ATTACK"),
        rarity=str(c.get("rarity") or "COMMON"),
        cost=cost,
        upgraded=bool(c.get("upgrades") or 0),
    )


def build_card_reward_candidates(gs: dict) -> list:
    """选牌屏 → 候选规格列表。

    每个规格含：idx（候选序号）/select_idx（CHOOSE 用的下标）/kind/label/
    card/bits。bits 按训练语义构造（reward_type=0 卡牌、idx2=卡下标）。

    V10 修复：真正把 SKIP 候选加进列表（旧版只循环 3 张卡，真机结构上无法跳牌）。
    SKIP 候选的 kind="SKIP"、bits=0x30000000（reward_type=6，与训练侧
    env/lightspeed_env 生成的 SKIP 完全一致）、select_idx=-1（哨兵，由 main.py
    F4 分支识别为 "SKIP" 命令）。deck_ctx 由 _rl_pick 在打分时按帧灌入，
    与训练侧共用 encode_skip_deck_context，保证 SKIP 向量单一事实来源。
    """
    cards = ((gs.get("screen_state") or {}).get("cards") or [])
    out = []
    for i, c in enumerate(cards):
        if not isinstance(c, dict):
            continue
        out.append({
            "idx": i,
            "select_idx": i,
            "kind": "CARD_REWARD",
            "label": str(c.get("id") or ""),
            "card": _card_info_from_json(c),
            "bits": ((0 & 0x7) << 27) | ((i & 0xFF) << 8),
        })
    # V10：补 SKIP 候选（让真机 RL 真正能跳牌）
    out.append({
        "idx": len(out),
        "select_idx": -1,   # 哨兵：F4 分支识别为 "SKIP" 命令
        "kind": "SKIP",
        "label": "SKIP",
        "bits": (6 & 0x7) << 27,   # 0x30000000 → reward_type=6（与训练侧一致）
    })
    return out


def build_map_candidates(gs: dict) -> list:
    """地图屏 → 候选规格（room 由 next_nodes 的 symbol 映射；bits=节点 x）。"""
    nn = ((gs.get("screen_state") or {}).get("next_nodes") or [])
    out = []
    for i, n in enumerate(nn):
        if not isinstance(n, dict):
            continue
        sym = str(n.get("symbol") or "").upper()
        out.append({
            "idx": i,
            "select_idx": i,
            "kind": "MAP_NODE",
            "label": f"map_{sym}_{n.get('x')}",
            "room": SYMBOL_TO_ROOM.get(sym),
            "bits": int(n.get("x") or 0) & 0xFF,
        })
    return out


def build_event_candidates(gs: dict) -> list:
    """事件屏 → 候选规格（event_key + 选项序号；bits=选项序号）。"""
    ss = gs.get("screen_state") or {}
    eid = ss.get("event_id")
    opts = ss.get("options") or []
    out = []
    for i, o in enumerate(opts):
        if not isinstance(o, dict):
            continue
        out.append({
            "idx": i,
            "select_idx": i,
            "kind": "EVENT_OPTION",
            "label": str(o.get("text") or f"event_opt_{i}")[:40],
            "event_key": str(eid) if eid else None,
            "event_option_idx": i,
            "bits": i & 0xFF,
        })
    return out
