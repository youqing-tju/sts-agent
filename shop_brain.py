"""商店决策大脑（ShopBrain）：买得聪明、走得干净、绝不卡死。

—— 为什么需要它（真机 2026-09-11 血案，已在原始帧日志上逐帧验证）——

1) 商店的出口是**两步**（Communication Mod 官方语义 + 原始帧实证）：
       SHOP_SCREEN --leave--> SHOP_ROOM --proceed--> 地图
   · `leave` 属于 RETURN 家族：点击左侧返回按钮 → 关闭商店界面；
   · `proceed` 点击右侧前进按钮 → 走出商店房间（原始帧 f1168→f1170 实证：
     SHOP_ROOM 收到非-choice 命令后直接进入 MAP）。

2) 死循环的引擎：`SHOP_ROOM` 的 `choice_list` 是 `["shop"]`（房间里的商人
   实体，点击即可重开商店界面）。旧版 decide() 的「P0 安全网 #1」（任何非空
   choice_list → `CHOOSE 0`）把它当普通选项屏 → `CHOOSE 0` = 点击商人 →
   **把刚用 leave 关掉的商店重新打开** →
       SHOP_SCREEN --leave--> SHOP_ROOM --choose 0--> SHOP_SCREEN --leave--> ...
   逐帧交替，实测两段卡死合计 29.5 分钟（448 次 + 2349 次 LEAVE），且该路径
   不写 L2 日志——1868 个 SHOP_ROOM 帧在决策日志里完全消失，排查瞎了两轮。

3) 购买机制（实测）：
   · `choice_list` 只含当前**买得起**的东西：
       ["purge"（删卡服务）] + 买得起的卡牌名 + 买得起的药水名
     （金币变化后列表会变短、下标会移动，因此索引必须按**当前帧**计算，
     不能缓存）；
   · `screen_state.cards[].{id,price,rarity}` 与 `purge_cost` 全部可读；
     卡牌 id 是稳定英文名（"Wild Strike"），与 deck / GRID 的 id 同源；
   · relics 不在 choice_list 里（本期不买）；
   · 买删卡服务后进入 `GRID`：第一帧选牌（CHOOSE）、第二帧确认（CONFIRM，
     实测 `confirm` 会出现在 available_commands），随后回到商店。

—— 本模块的不变量（tests/test_shop_brain.py 逐条锁死）——
   A. 绝不在 SHOP_ROOM 发 CHOOSE（那是"点击商人"= 重开商店）；
   B. 一次商店访问：购买尝试 ≤ max_purchases，每件物品只尝试一次；
   C. 决定离开后 ≤2 条命令（LEAVE → PROCEED）走出商店；
   D. 命令总数有硬上限（CMD_CAP），任何情况下不允许无限循环。

后续升级路径：模拟器已支持从真机 JSON 灌入 SHOP_ROOM（实测 12 个动作），
可用 RL 模型替换 `_pick_purchase` 的打分（需先做商店候选编码等价性验证）。
"""
from __future__ import annotations

import logging

log = logging.getLogger("sts-agent.shop")

# ---------------------------------------------------------------------------
# v1 购卡打分（权重集中在这里，方便据真机数据调参；所有分数都会写进 L3 日志）
# 设计取舍：宁可少买（漏掉一张好牌只损失机会），不要乱买（白花金币 + 污染牌组）。
# ---------------------------------------------------------------------------
DEFAULT_BUY_THRESHOLD = 0.35

_W_RARITY = 0.30          # × rarity_score（C=0 / U=0.5 / R=1）
_W_DRAW = 0.20            # × min(1, draw_count)
_W_ENERGY = 0.18          # 有 energy_gain
_W_BLOCK = 0.18           # base_block >= 8
_W_DAMAGE = 0.10          # × min(1, base_damage/14)
_W_STRENGTH = 0.15        # × min(1, strength_gain)
_W_AOE = 0.10
_W_POWER = 0.06
_W_TRIGGER = 0.12         # on_exhaust_trigger / block_per_turn 等长线机制
_DUP_PENALTY = -0.15      # 牌组已有 ≥2 张同名
# 不在 KB 里的牌（如部分无色牌）按稀有度兜底：只敢盲买"稀有"，不盲买"罕见"。
_FALLBACK_RARITY = {"RARE": 0.55, "UNCOMMON": 0.30, "COMMON": 0.15}

# 删卡优先级：先删诅咒，再删打击/防御
CURSE_HINTS = ("Curse", "AscendersBane", "Necronomicurse", "Parasite",
               "Injury", "Regret", "Shame", "Doubt", "Pain", "Writhe",
               "Normality", "Clumsy", "Decay")

_EFFECT_GETTER = None


def _get_effects(card_id):
    """懒加载 card_data；失败返回 None（打分退回稀有度兜底）。"""
    global _EFFECT_GETTER
    if _EFFECT_GETTER is None:
        try:
            from models.card_data import get_card_effects as _ge
            _EFFECT_GETTER = _ge
        except Exception:            # pragma: no cover - 环境缺 models 时兜底
            _EFFECT_GETTER = False
    if _EFFECT_GETTER is False:
        return None
    try:
        return _EFFECT_GETTER(card_id) or None
    except Exception:
        return None


def card_buy_score(card_id, rarity=None, deck_counts=None) -> float:
    """买卡打分（v1 规则版；≥ buy_threshold 才买）。"""
    eff = _get_effects(card_id)
    if isinstance(eff, dict) and eff:
        s = _W_RARITY * float(eff.get("rarity_score") or 0.0)
        s += _W_DRAW * min(1.0, float(eff.get("draw_count") or 0))
        if eff.get("energy_gain"):
            s += _W_ENERGY
        if (eff.get("base_block") or 0) >= 8:
            s += _W_BLOCK
        s += _W_DAMAGE * min(1.0, float(eff.get("base_damage") or 0) / 14.0)
        s += _W_STRENGTH * min(1.0, float(eff.get("strength_gain") or 0))
        if eff.get("is_aoe"):
            s += _W_AOE
        if eff.get("is_power"):
            s += _W_POWER
        if eff.get("on_exhaust_trigger") or eff.get("block_per_turn"):
            s += _W_TRIGGER
    else:
        s = _FALLBACK_RARITY.get(str(rarity or "").upper(), 0.0)
    if deck_counts and deck_counts.get(card_id, 0) >= 2:
        s += _DUP_PENALTY
    return s


def is_shop_state(state: dict) -> bool:
    """该帧是否属于商店（SHOP_SCREEN / SHOP_ROOM，含 screen_name=SHOP 变体）。"""
    gs = state.get("game_state") or {}
    if not isinstance(gs, dict):
        return False
    scr = str(gs.get("screen_type") or "").upper()
    nm = str(gs.get("screen_name") or "").upper()
    return scr in ("SHOP_SCREEN", "SHOP_ROOM") or nm == "SHOP"


def _deck_counts(gs):
    from collections import Counter
    return Counter(str(c.get("id") or "")
                   for c in (gs.get("deck") or []) if isinstance(c, dict))


def _deck_has_junk(gs) -> bool:
    cnt = _deck_counts(gs)
    if any(any(h in cid for h in CURSE_HINTS) for cid in cnt):
        return True
    return (cnt.get("Strike_R", 0) + cnt.get("Defend_R", 0)) >= 1


def _purge_target(gs) -> int:
    """删卡 GRID 的选牌下标：诅咒 > 打击 > 防御 > 0 号位。"""
    ss = gs.get("screen_state") or {}
    cl = [str(x) for x in (gs.get("choice_list") or [])]
    by_name = {}
    for c in ss.get("cards") or []:
        if isinstance(c, dict):
            by_name.setdefault(str(c.get("name")), c)

    def card_at(i):
        return by_name.get(cl[i]) if i < len(cl) else None

    for i in range(len(cl)):
        c = card_at(i)
        if c and any(h in str(c.get("id") or "") for h in CURSE_HINTS):
            return i
    for want in ("Strike_R", "Defend_R"):
        for i in range(len(cl)):
            c = card_at(i)
            if c and str(c.get("id") or "") == want:
                return i
    return 0


class ShopBrain:
    """商店 / 商店房间 / 删卡 GRID 三态决策器（有记忆、有上限、绝不空转）。"""

    SCREEN, ROOM, GRID = "SHOP_SCREEN", "SHOP_ROOM", "GRID"
    CMD_CAP = 30               # 一次商店访问的命令硬上限（超出后只做退出）

    def __init__(self, allow_buys=True, buy_threshold=DEFAULT_BUY_THRESHOLD,
                 max_purchases=3, log=None):
        self.allow_buys = bool(allow_buys)
        self.buy_threshold = float(buy_threshold)
        self.max_purchases = int(max_purchases)
        self.log = log
        self._visit = ()
        self._reset_visit(())

    # ---------------- 对外接口 ----------------

    def decide(self, state: dict):
        """返回商店相关命令；None = 本帧与商店无关（交回主流程）。"""
        gs = state.get("game_state") or {}
        if not isinstance(gs, dict):
            return None
        scr = str(gs.get("screen_type") or "").upper()

        key = (gs.get("act"), gs.get("floor"))
        if scr in (self.SCREEN, self.ROOM) and key != self._visit:
            self._reset_visit(key)          # 新一次商店访问（按 act+floor 识别）

        cmds = [str(c).lower() for c in (state.get("available_commands") or [])]
        if scr == self.SCREEN:
            return self._decide_screen(gs, cmds)
        if scr == self.ROOM:
            return self._decide_room(gs, cmds)
        if scr == self.GRID:
            return self._decide_grid(gs, cmds)
        return None

    # ---------------- 内部 ----------------

    def _reset_visit(self, key):
        self._visit = key
        self.seen_screen = False
        self.leaving = False
        self.room_frames = 0
        self.purchases = 0
        self.attempted = set()
        self.purge_pending = False
        self.grid_selected = False
        self.grid_steps = 0
        self.cmds = 0
        self._exit_reported = False

    def _emit(self, kind, **kw):
        if self.log is not None:
            try:
                self.log.event(kind, **kw)
            except Exception:
                pass

    # ---- 商店界面 ----

    def _decide_screen(self, gs, cmds):
        self.seen_screen = True
        self.cmds += 1
        gold = gs.get("gold") or 0

        if self.cmds > self.CMD_CAP:        # 硬上限：只做退出，绝不无限循环
            return "LEAVE" if "leave" in cmds else "STATE"
        if self.leaving:                    # 已决定离开：重复 LEAVE 无害
            return "LEAVE" if "leave" in cmds else "STATE"

        pick = self._pick_purchase(gs) if self.allow_buys else None
        if pick is not None and "choose" not in cmds:
            pick = None                     # 命令不可用则不买（防错误帧）
        if pick is not None:
            idx, meta = pick
            key = meta.pop("key", None)
            if key is not None:
                self.attempted.add(key)     # 每件物品只尝试一次（防死循环核心）
            self.purchases += 1
            if meta.get("type") == "purge":
                self.purge_pending = True
                self.grid_selected = False
                self.grid_steps = 0
            self._emit("shop_buy", idx=idx, gold=gold, **meta)
            return f"CHOOSE {idx}"

        # 没有值得买/买得起的 → 关闭商店界面（两步出口的第一步）
        self.leaving = True
        self._emit("shop_leave", gold=gold, purchases=self.purchases)
        if "leave" in cmds:
            return "LEAVE"
        if "proceed" in cmds:
            return "PROCEED"
        return "STATE"

    def _pick_purchase(self, gs):
        """返回 (choice_idx, meta) 或 None。按当前帧的 choice_list 计算下标。"""
        if self.purchases >= self.max_purchases:
            return None
        ss = gs.get("screen_state") or {}
        cl = [str(x) for x in (gs.get("choice_list") or [])]
        gold = gs.get("gold") or 0
        if not cl:
            return None

        # 1) 删卡服务：商店里性价比最高的动作（尤其前期），优先级最高
        pc = ss.get("purge_cost") or 0
        if (ss.get("purge_available") and "purge" in cl and pc and pc <= gold
                and "purge" not in self.attempted and _deck_has_junk(gs)):
            return cl.index("purge"), {"type": "purge", "price": pc,
                                       "key": "purge"}

        # 2) 卡牌：取分数最高且过阈值的（买得起 & 没试过 & 在可选列表里）
        deck_counts = _deck_counts(gs)
        best = None
        for c in ss.get("cards") or []:
            if not isinstance(c, dict):
                continue
            nm = str(c.get("name"))
            if nm not in cl:                # 不在列表 = 买不起/不可选
                continue
            cid = str(c.get("id") or "")
            price = c.get("price") or 0
            if price > gold or ("card", cid) in self.attempted:
                continue
            sc = card_buy_score(cid, c.get("rarity"), deck_counts)
            if best is None or sc > best[0]:
                best = (sc, c)
        if best is not None and best[0] >= self.buy_threshold:
            sc, c = best
            return (cl.index(str(c.get("name"))),
                    {"type": "card", "id": c.get("id"),
                     "price": c.get("price"), "score": round(float(sc), 3),
                     "key": ("card", str(c.get("id") or ""))})
        # 3) 药水：v1 不买（性价比一般且优先让位给卡牌/删卡）
        return None

    # ---- 商店房间（两步出口的第二步）----

    def _decide_room(self, gs, cmds):
        self.cmds += 1
        can_proceed = "proceed" in cmds
        if self.leaving:
            # **核心修复**：这里必须 PROCEED（走出房间），绝不能 CHOOSE ——
            # CHOOSE 0 = ["shop"] = 点击商人 = 把刚关掉的商店重新打开。
            if can_proceed and not self._exit_reported:
                self._exit_reported = True
                self._emit("shop_exit")
            return "PROCEED" if can_proceed else "STATE"

        # 还没逛完却回到了房间：界面要么正在自动打开（实测会），要么需要点击
        # 商人。有限次尝试后放弃并走人——绝不在这里无限循环。
        self.room_frames += 1
        if self.room_frames <= 2:
            return "STATE"
        if self.room_frames <= 7:
            if self.room_frames % 2 == 1 and "choose" in cmds:
                return "CHOOSE 0"           # 点击商人 → 打开商店界面
            return "STATE"
        self.leaving = True
        self._emit("shop_giveup", room_frames=self.room_frames)
        return "PROCEED" if can_proceed else "STATE"

    # ---- 删卡 GRID（买完 purge 后的选牌/确认）----

    def _decide_grid(self, gs, cmds):
        if not self.purge_pending:          # 只接管由我们发起的删卡流程
            return None
        self.cmds += 1
        self.grid_steps += 1
        if "confirm" in cmds:               # 已选好 → 确认
            self.purge_pending = False
            self._emit("shop_purge_done")
            return "CONFIRM"
        if "choose" in cmds and not self.grid_selected:
            idx = _purge_target(gs)
            self.grid_selected = True
            self._emit("shop_purge_pick", idx=idx)
            return f"CHOOSE {idx}"
        if self.grid_steps >= 6:            # 兜底：别把流程吊死
            self.purge_pending = False
            if "cancel" in cmds:
                self._emit("shop_purge_cancel")
                return "CANCEL"
            if "confirm" in cmds:
                return "CONFIRM"
        return "STATE"
