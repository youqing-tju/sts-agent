# -*- coding: utf-8 -*-
"""ShopBrain 回归测试 —— 全部围绕真机 2026-09-11 血案的真实帧。

fixtures（取自真机原始日志 logs/realtime/0911_001847，已复制到 tests/fixtures/）：
  shop_screen.json  金币 146 的商店界面：choice_list = [purge(75)] + 6 卡 + 3 药水
  shop_room.json    同一访问的商店房间帧：choice_list = ["shop"]（重置陷阱！）
  grid.json         买完删卡服务后的选牌帧（for_purge=True，12 张牌组）

锁死的不变量：
  A. 绝不在 SHOP_ROOM 发 CHOOSE（那是"点击商人"= 重开商店）；
  B. 一次访问购买尝试 ≤ max_purchases、每件物品只尝试一次；
  C. 离开后 ≤2 条命令（LEAVE -> PROCEED）走出商店；
  D. 命令数硬上限，任何序列下不无限循环。
"""
import json
import os

from shop_brain import ShopBrain, card_buy_score, is_shop_state

_FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _load(name):
    with open(os.path.join(_FIX, name), encoding="utf-8") as f:
        return json.load(f)


def _mk_screen(gold=146, avail=None, items=None):
    """构造一个极简商店界面帧（用于序列测试）。"""
    return {
        "available_commands": avail or ["choose", "potion", "leave", "key",
                                        "click", "wait", "state"],
        "ready_for_command": True, "in_game": True,
        "game_state": {"screen_type": "SHOP_SCREEN", "screen_name": "SHOP",
                       "floor": 3, "act": 1, "current_hp": 80, "max_hp": 80,
                       "gold": gold, "room_phase": "COMPLETE",
                       "choice_list": items or [],
                       "screen_state": {"cards": [], "potions": [], "relics": [],
                                        "purge_available": False,
                                        "purge_cost": 75}},
    }


class Rec:
    """假日志器：收集 L3 事件。"""
    def __init__(self):
        self.events = []

    def event(self, kind, **kw):
        self.events.append((kind, kw))

    def kinds(self):
        return [k for k, _ in self.events]

    def find(self, kind, **match):
        for k, kw in self.events:
            if k == kind and all(kw.get(m) == v for m, v in match.items()):
                return kw
        return None


# ---------------- 基础：识别与打分 ----------------

def test_is_shop_state():
    assert is_shop_state(_load("shop_screen.json"))
    assert is_shop_state(_load("shop_room.json"))
    combat = {"game_state": {"screen_type": "NONE",
                             "combat_state": {"hand": []}}}
    assert not is_shop_state(combat)


def test_card_score_calibration():
    """锁定 v1 买卡策略（改权重时必须同步改这里，防止悄悄漂移）。"""
    assert card_buy_score("Shrug It Off") >= 0.35      # 8 格挡+1 抽牌，好卡
    assert card_buy_score("Wild Strike") < 0.35        # 塞伤口，不买
    assert card_buy_score("Havoc") < 0.35              # 花活，不裸买
    assert card_buy_score("Dark Embrace") < 0.35       # 消耗流组件，需体系
    assert card_buy_score("Purity", "UNCOMMON") < 0.35     # KB 外不盲买罕见
    assert card_buy_score("Secret Technique", "RARE") >= 0.35  # KB 外敢买稀有


# ---------------- 房间：绝不放 CHOOSE（核心不变量 A） ----------------

def test_entry_room_waits_then_bounded_open():
    """刚进店：先等待（界面自动打开），有限次后尝试点击商人，最终放弃走人。

    绝不无限循环、绝不在房间盲发 CHOOSE。
    """
    brain = ShopBrain()
    room = _load("shop_room.json")
    replies = [brain.decide(room) for _ in range(12)]
    assert replies[0] == "STATE" and replies[1] == "STATE", replies[:2]
    assert not replies[0].startswith("CHOOSE"), "前两帧不得盲发 CHOOSE"
    assert sum(1 for r in replies if r.startswith("CHOOSE")) <= 3, replies
    assert "PROCEED" in replies[7:], f"尝试打开失败后应放弃并走人: {replies}"


def test_historical_loop_cannot_reproduce():
    """真机死循环逐帧重建：SHOP_SCREEN(无可买) -> SHOP_ROOM 交替。

    旧版在 SHOP_ROOM 发 CHOOSE 0（点击商人）→ 商店重开 → 29.5 分钟死循环。
    新版必须：屏幕帧 LEAVE、房间帧 PROCEED，全程零 CHOOSE。
    """
    brain = ShopBrain()
    screen = _mk_screen(gold=25,
                        avail=["potion", "leave", "key", "click", "wait", "state"],
                        items=[])
    room = _load("shop_room.json")          # choice_list=["shop"]，重置陷阱
    seq = []
    for _ in range(4):
        seq.append(brain.decide(screen))
        seq.append(brain.decide(room))
    assert not any(r.startswith("CHOOSE") for r in seq), f"出现 CHOOSE: {seq}"
    room_replies = seq[1::2]
    assert all(r == "PROCEED" for r in room_replies), room_replies
    assert all(r == "LEAVE" for r in seq[0::2]), seq[0::2]


# ---------------- 购买：预算 / 上限 / 去重（不变量 B） ----------------

def test_buy_purge_then_card_then_leave():
    """完整购买链：先删卡（purge）→ 再买一张过阈值的卡 → 没得买就 LEAVE。

    全部用真机 fixture 与真实价格（purge 75 / Shrug It Off 54）。
    """
    rec = Rec()
    brain = ShopBrain(log=rec)
    fr = _load("shop_screen.json")
    gs = fr["game_state"]
    ss = gs["screen_state"]

    # 第 1 帧：删卡服务优先级最高（75 <= 146）
    assert brain.decide(fr) == "CHOOSE 0"
    assert rec.find("shop_buy", type="purge", price=75) is not None
    assert brain.purge_pending is True

    # 模拟游戏推进：删卡 75、purge 从列表消失、金币 146 -> 71
    gs["gold"] = 71
    affordable = [c["name"] for c in ss["cards"]
                  if c["price"] <= 71 and c["name"] in gs["choice_list"]]
    affordable += [p["name"] for p in ss["potions"]
                   if p["price"] <= 71 and p["name"] in gs["choice_list"]]
    gs["choice_list"] = affordable           # [Wild Strike, Headbutt, Shrug It Off, Havoc, Attack Potion]

    # 第 2 帧：Shrug It Off（0.38 过阈值、54 买得起）应被选中——只买对的，不买列表第一个
    assert brain.decide(fr) == "CHOOSE 2", gs["choice_list"]
    buy = rec.find("shop_buy", id="Shrug It Off")
    assert buy is not None and buy.get("price") == 54

    # 第 3 帧：没得买 -> 关闭界面（两步出口第一步）
    gs["gold"] = 17
    gs["choice_list"] = []
    assert brain.decide(fr) == "LEAVE"
    assert "shop_leave" in rec.kinds()

    # 第 4 帧：房间 -> 走出（第二步，核心修复）
    assert brain.decide(_load("shop_room.json")) == "PROCEED"
    assert "shop_exit" in rec.kinds()


def test_leave_mode_never_buys():
    """--shop-mode leave：有钱也不买，直接离开（且仍能正确走出商店）。"""
    brain = ShopBrain(allow_buys=False)
    assert brain.decide(_load("shop_screen.json")) == "LEAVE"
    assert brain.decide(_load("shop_room.json")) == "PROCEED"


def test_purchase_cap_and_dedup():
    """购买硬上限 3 次；已买过的物品不再重复；之后必须 LEAVE。"""
    brain = ShopBrain(max_purchases=3, log=Rec())
    fr = _load("shop_screen.json")
    gs = fr["game_state"]
    ss = gs["screen_state"]
    gs["gold"] = 9999
    gs["choice_list"] = (["purge"] + [c["name"] for c in ss["cards"]]
                         + [p["name"] for p in ss["potions"]])
    picked, cmds = [], []
    for _ in range(8):
        cmd = brain.decide(fr)
        cmds.append(cmd)
        if cmd.startswith("CHOOSE "):
            i = int(cmd.split()[1])
            picked.append(gs["choice_list"][i])
            gs["choice_list"] = [x for j, x in enumerate(gs["choice_list"])
                                 if j != i]
    assert len(picked) <= 3, picked
    assert len(set(picked)) == len(picked), picked        # 不重复购买
    assert sum(1 for c in cmds if c.startswith("CHOOSE")) == 3
    assert cmds[-1] == "LEAVE", cmds


def test_failed_purchase_not_retried():
    """购买被无声拒绝（状态原样重发）时，绝不能重复发同一条 CHOOSE。

    这是防死循环的关键：真机上"无效购买 + 确定性重复决策"就是 29.5 分钟的成因。
    """
    brain = ShopBrain(log=Rec())
    fr = _load("shop_screen.json")
    gs = fr["game_state"]
    ss = gs["screen_state"]
    sio_name = ss["cards"][2]["name"]           # Shrug It Off 的显示名
    gs["choice_list"] = [sio_name]              # 只剩一个可选项
    assert brain.decide(fr) == "CHOOSE 0"
    # 状态原样重发（购买未生效）：同物品不重试 -> 直接离开
    assert brain.decide(fr) == "LEAVE"


# ---------------- 删卡 GRID 流程 ----------------

def test_grid_purge_flow():
    rec = Rec()
    brain = ShopBrain(log=rec)
    assert brain.decide(_load("shop_screen.json")) == "CHOOSE 0"   # 买 purge
    assert brain.purge_pending is True

    grid = _load("grid.json")
    assert brain.decide(grid) == "CHOOSE 0"      # 优先删打击（grid[0] 即 Strike_R）
    assert rec.find("shop_purge_pick", idx=0) is not None

    grid2 = json.loads(json.dumps(grid))
    grid2["available_commands"] = ["potion", "confirm", "cancel", "key",
                                   "click", "wait", "state"]
    grid2["game_state"]["screen_state"]["selected_cards"] = \
        [grid2["game_state"]["screen_state"]["cards"][0]]
    assert brain.decide(grid2) == "CONFIRM"
    assert brain.purge_pending is False
    assert "shop_purge_done" in rec.kinds()

    # 未被 purge_pending 关联的 GRID 不接管（避免劫持其它选牌屏）
    assert ShopBrain().decide(grid) is None


def test_room_never_unbounded_under_weird_avail():
    """房间帧在缺失 proceed、缺失 choose 等异常组合下也不得空转或乱发。"""
    brain = ShopBrain()
    room = _load("shop_room.json")
    room = json.loads(json.dumps(room))
    room["available_commands"] = ["wait", "state"]      # 既无 choose 也无 proceed
    replies = [brain.decide(room) for _ in range(12)]
    assert all(r in ("STATE", "PROCEED") for r in replies), replies
    assert not any(r.startswith("CHOOSE") for r in replies), replies
