"""候选动作的「屏幕分发」共享逻辑：kind 判定 + KB 三段键解析。

**为什么单独抽出来**：这段逻辑原先只存在于 `env/lightspeed_env.py` 的
`candidates()` 里。真实游戏桥接（`main.py`）需要**完全相同**的分发规则，
否则真机喂给模型的候选向量与训练时不一致 → 模型行为整体错位
（这正是"部署后表现异常但查不出原因"的典型来源）。

所以这里做成**单一事实来源**：env 与 main.py 都调用本模块。
任何一处改动都会同时作用于训练与真机两条路径。

与 C++ 绑定的对应关系（详见 models/candidate_encoder.py 段表注释）：
  REWARDS            : reward_type 区分类别，idx1 = 该类别子列表下标
                       （POTION=3→potions[idx1]，RELIC=4→relics[idx1]）
  SHOP_ROOM          : get_shop_contents 扁平元组列表，idx1 = 该类别子列表下标
  BOSS_RELIC_REWARDS : idx1 = 0..N-1 → contents[idx1]；==len 为跳过项
  EVENT_SCREEN       : event_name 作 event_key，bits.idx1 = option_indices 序号
  REST_ROOM          : 不在 KB 范围（get_rest_options 不产出 KB 特征）
"""

# screen_state 枚举名 → Candidate.kind（粗映射；REWARDS/SHOP 屏另有细化）
KIND_BY_SCREEN = {
    "MAP_SCREEN": "MAP_NODE",
    "REWARDS": "REWARD",
    "SHOP_ROOM": "SHOP_ITEM",
    "REST_ROOM": "REST_OPTION",
    "EVENT_SCREEN": "EVENT_OPTION",
    "BOSS_RELIC_REWARDS": "BOSS_RELIC",
}

# RewardsActionType 编码（见 candidate_encoder.decode_action_bits）
REWARD_CARD = 0
REWARD_POTION = 3
REWARD_RELIC = 4
REWARD_SKIP = 6

# 唱歌碗特殊动作（不拿牌）：idx2 == 5
SINGING_BOWL_IDX2 = 5


def candidate_kind(screen: str, d: dict) -> str:
    """按屏幕 + bits 的 reward_type 细化 kind；兜底 ACTION。

    d 为 decode_action_bits(bits) 的结果。
    """
    if screen == "REWARDS":
        rt = d["reward_type"]
        if rt == REWARD_CARD:
            return "CARD_REWARD"
        if rt == REWARD_SKIP:
            return "SKIP"
        return "REWARD"
    if screen == "SHOP_ROOM":
        return "SKIP" if d["reward_type"] == REWARD_SKIP else "SHOP_ITEM"
    if screen == "TREASURE_ROOM":
        return "TREASURE"
    return KIND_BY_SCREEN.get(screen, "ACTION")


def resolve_candidate_kb(screen: str, d: dict, gc) -> dict:
    """解析候选对应的 KB 三段键（药水/遗物/事件）。拿不到则为 None。

    每个候选调用一次；`gc` 需已灌入当前真实/模拟状态（init_from_json 或
    模拟器原生状态）。非对应屏 / 下标越界 / 段不存在时安静返回 None，
    对应 KB 段在编码时保持全零（与训练一致）。

    返回 dict：potion_name / relic_name / event_key / event_option_idx
    """
    out = {"potion_name": None, "relic_name": None,
           "event_key": None, "event_option_idx": None}

    if screen == "REWARDS":
        contents = gc.get_reward_contents()
        if contents is None:
            return out
        if d["reward_type"] == REWARD_POTION:
            pl = contents.get("potions") or []
            if d["idx1"] < len(pl):
                out["potion_name"] = pl[d["idx1"]]
        elif d["reward_type"] == REWARD_RELIC:
            rl = contents.get("relics") or []
            if d["idx1"] < len(rl):
                out["relic_name"] = rl[d["idx1"]]
        # 卡牌（idx2 处理）/金币/钥匙：无 KB

    elif screen == "SHOP_ROOM":
        contents = gc.get_shop_contents()
        if contents is None:
            return out
        if d["reward_type"] == REWARD_POTION:
            pl = [t[1] for t in contents if t[0] == "potion"]
            if d["idx1"] < len(pl):
                out["potion_name"] = pl[d["idx1"]]
        elif d["reward_type"] == REWARD_RELIC:
            rl = [t[1] for t in contents if t[0] == "relic"]
            if d["idx1"] < len(rl):
                out["relic_name"] = rl[d["idx1"]]
        # 卡牌（绑定未暴露价格/名称→段全零）/删卡/跳过：无 KB

    elif screen == "BOSS_RELIC_REWARDS":
        contents = gc.get_boss_relic_contents()
        if contents is None:
            return out
        # 候选 reward_type 固定为 0（绑定怪异），idx1==len(contents) 为跳过项
        if d["idx1"] < len(contents):
            out["relic_name"] = contents[d["idx1"]]

    elif screen == "EVENT_SCREEN":
        info = gc.get_event_info()
        if info is None:
            return out
        out["event_key"] = info.get("event_name")
        out["event_option_idx"] = d["idx1"]

    return out
