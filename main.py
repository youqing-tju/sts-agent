"""sts-agent 真实游戏入口（路线 B：模拟器当大脑外挂）。

协议（Communication Mod）：
- 启动后向 stdout 打印 "ready"
- 之后每个稳定游戏状态会通过 stdin 收到一行 JSON
- 向 stdout 写动作命令（PLAY / END / CHOOSE / START 等）

决策架构：
- 宏观决策（选路/卡牌奖励/篝火/事件/商店/遗物/药水）：
  真实游戏 JSON → GameContext.init_from_json 灌入模拟器 → get_all_actions
  枚举动作 → **477 维观测 + 347 维候选** → Scorer 打分 argmax →
  把选中的 GameAction.bits 翻译成 Communication Mod 命令发回真实游戏。
- 战斗出牌：CombatBrain（ISMCTS：K 个平行未来 + 按卡牌身份投票），
  失败回退 main_rule 的启发式。
- 降级：init_from_json / 编码 / 打分 / 翻译任一步失败 → 回退规则决策。

【与训练严格对齐】观测经 `enhance_observation`（412→477），候选经
`encode_candidate` 并带上 `deck_arch` 与 KB 三段键（药水/遗物/事件）——
屏幕分发逻辑走 `models/candidate_kb.py` 这一份**单一事实来源**，与
`env/lightspeed_env.py` 完全同源。等价性由 `tests/test_bridge_parity.py` 回归保证。

【日志】三层，落在 `logs/realtime/`：
  L1 `*_states.jsonl`    每条真实状态原样落盘（无损重放，事后排障唯一依据）
  L2 `*_decisions.jsonl` 每步决策：屏幕/候选/模型打分 top3/所选/命令/是否降级
  L3 `*_events.jsonl`    结局/异常/心跳（心跳线程每 60s 一条）
"""

import argparse
import json
import logging
import os
import sys
import threading
import time

# 真机由 Communication Mod 直接拉起（不经过 shell），拿不到外部环境变量：
# torch + slaythespire.pyd 各自带一个 OpenMP 运行时，不设这个会报 OMP Error #15。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_ROOT = os.path.dirname(os.path.abspath(__file__))
# Mod 拉起时 cwd 是游戏目录；切到项目根，保证相对路径（日志/last_state.json）落在这里
os.chdir(_ROOT)

logging.basicConfig(
    filename="sts_agent_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("sts-agent")

import slaythespire as sts
import torch

from models.candidate_encoder import (
    DESC_VEC_DIM,
    CardInfo,
    compute_map_lookahead,
    decode_action_bits,
    encode_candidate,
)
from models.candidate_kb import (
    REWARD_CARD,
    REWARD_SKIP,
    SINGING_BOWL_IDX2,
    candidate_kind,
    resolve_candidate_kb,
)
from models.card_data import CARD_ARCHETYPE, CARD_EFFECTS
from models.card_index import CARD_INDEX
from models.obs_encoder import OBS_DIM, encode_deck_stats, enhance_observation
from models.scorer import Scorer
from env.base_env import Candidate
from combat_brain import CombatBrain

# 降级路径：规则 bot（战斗启发式 + 宏观规则）
from main_rule import combat_action, macro_action

_DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints", "ppo_v9_10000.pt")
_DEFAULT_LOG_DIR = os.path.join(_ROOT, "logs", "realtime")


def _sanitize(obj):
    """递归清洗 Communication Mod JSON 里的残缺 UTF-16 代理对。

    CJK 版游戏用 GBK 编码中文，转换时产生 lone surrogate，
    json.dumps 会因此抛异常导致 RL 路径每次都回退规则。全部替换为 U+FFFD。
    """
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj


def _has_surrogate(s: str) -> bool:
    """字符串是否含孤立代理字符（U+DC80..U+DCFF 等 lone surrogate）。"""
    return any(0xD800 <= ord(c) <= 0xDFFF for c in s)


def _purge_surrogates(obj, path: str = "", found: list = None):
    """递归把孤立代理字符替换成 U+FFFD，保证 json.dumps 后是合法 UTF-8 JSON。

    用 surrogatepass 把代理当有效字节编码、再 decode('utf-8','replace') 把非法
    UTF-8 字节（即原代理）换成 U+FFFD。found 非空时收集出问题的字段路径（只用于
    首次告警，避免刷屏）。
    """
    if isinstance(obj, str):
        if _has_surrogate(obj):
            if found is not None:
                found.append(path or "<root>")
            return obj.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
        return obj
    if isinstance(obj, list):
        return [_purge_surrogates(x, f"{path}[{i}]", found)
                for i, x in enumerate(obj)]
    if isinstance(obj, dict):
        return {k: _purge_surrogates(v, f"{path}.{k}" if path else str(k), found)
                for k, v in obj.items()}
    return obj


def _is_victory(state: dict) -> bool:
    """通关画面：screen_type=GAME_OVER 且 screen_state.victory=true。"""
    gs = state.get("game_state", {})
    if (gs.get("screen_type") or "").upper() != "GAME_OVER":
        return False
    return bool((gs.get("screen_state") or {}).get("victory"))


def _is_game_over(state: dict) -> bool:
    return (state.get("game_state", {}).get("screen_type") or "").upper() == "GAME_OVER"


# ---- 防卡死守卫 ----
# 真机历史上死过一次：agent 发 `PLAY 1` 被拒 → 回 `STATE` → 游戏状态不变 →
# 再发 `PLAY 1` …… 死循环 5853 次、整局一步没动（`communication_mod_errors.log`）。
# 根因是「命令无效 + 状态不变」没人检测。下面这套守卫专门堵这个洞。
_STALL_LIMIT = 5        # 状态指纹连续相同 / 同一真实状态反复出现几次 → 判卡死，强制换动作
_STALL_HARD_LIMIT = 20  # 换动作后仍卡死 → 退出保命（宁可停下让人接管，不要空转）


def _fingerprint(state: dict) -> tuple:
    """游戏状态指纹：能反映「有没有真的推进」。

    刻意不含 HP 之外的高频抖动字段；战斗中打出一张牌会改变 hand/energy，
    因此指纹变化 = 有进展；指纹不变 = 卡住。
    """
    gs = state.get("game_state", {}) or {}
    cs = gs.get("combat_state") or {}
    hand = cs.get("hand") or []
    return (
        (gs.get("screen_type") or "").upper(),
        gs.get("floor"), gs.get("act"), gs.get("current_hp"),
        gs.get("gold"), gs.get("room_type"),
        cs.get("turn"), len(hand), cs.get("player", {}).get("energy")
        if isinstance(cs.get("player"), dict) else None,
        bool(state.get("ready_for_command")),
    )


def _alternate_cmd(state: dict, avoid: str = None):
    """卡死时强制换一个『几乎总能推进』的动作。

    avoid：尽量避开这条命令（否则"换了等于没换"，守卫失效）。
    """
    cmds = [c.lower() for c in state.get("available_commands", [])]
    gs = state.get("game_state", {}) or {}
    screen = (gs.get("screen_type") or "").upper()
    avoid_u = (avoid or "").upper()

    candidates = []
    if "play" in cmds:                       # 战斗中：结束回合永远合法
        candidates.append("END")
    if "proceed" in cmds:
        candidates.append("PROCEED")
    if "skip" in cmds:
        candidates.append("SKIP")
    # 商店例外：SHOP_SCREEN 的 choice_list 是商品列表，CHOOSE 0 = 买东西 /
    # 开删除卡牌服务（真机实测金币 110→14、状态在商店/房间/GRID 间死循环）。
    # 守卫的强制换动作也绝不能把"离开商店"改成"购物"。
    _gs2 = state.get("game_state") or {}
    _is_shop2 = ((_gs2.get("screen_type") or "").upper() == "SHOP_SCREEN"
                 or (_gs2.get("screen_name") or "").upper() == "SHOP")
    if "choose" in cmds and not _is_shop2:
        candidates.append("CHOOSE 0")        # 事件/篝火/地图：退到第一个选项
    if "leave" in cmds:
        candidates.append("LEAVE")
    if "start" in cmds:
        candidates.append("START IRONCLAD")
    for c in cmds:                           # 兜底：任何非 state 命令
        # 兜底候选也要过滤：key/click/play/potion 需要参数，裸发会被拒
        # （真机实测 81 次 `Argument missing in command "key"`）。
        if c not in ("state", "key", "click", "play", "potion", "choose"):
            candidates.append(c.upper())

    for cand in candidates:
        if cand != avoid_u:
            return cand
    # 找不到真正不同的替代命令：战斗内强制 END（至少推进回合/放弃出牌），
    # 非战斗回 STATE（避免假装行动却原地空转）。绝不可返回原命令（avoid）。
    if "play" in cmds:
        return "END"
    return "STATE"


class LoopDetector:
    """卡死检测：三种信号，与「发了什么命令」协同判断游戏有没有真的推进。

    信号 1（连续相同指纹）：状态指纹逐帧完全相同（连 error 帧都算）→ 游戏完全不动。
    信号 2（同一真实状态反复出现）：允许中间夹着 error 帧——本次真机死循环正是
        「COMBAT 状态 → error 帧 → 同一 COMBAT 状态 → error 帧」交替，命令恒为
        `PLAY 1 0`、状态指纹从不连续重复，旧 StallGuard 永不触发。这里只统计真实
        游戏状态（error 帧 has_game_state=False，不计数），同一状态出现 ≥limit 次即判卡死。
    信号 3（同一命令连续输出）：与状态指纹无关——真机上「命令恒同、状态指纹却在变」
        是真实存在的模式（如 SHOP_SCREEN 反复发 LEAVE 无效）。同一非 STATE 命令
        连续输出 ≥limit 次（允许中间夹 STATE）即判卡死，强制改发不同命令。

    返回当前 severity（三信号取大），main 循环据此强制换动作 / 主动退出。
    """

    def __init__(self, limit=_STALL_LIMIT, hard_limit=_STALL_HARD_LIMIT):
        self.limit = limit
        self.hard_limit = hard_limit
        self._fp = None
        self._consec = 0          # 连续相同指纹（含 error 帧）
        self._game_fp = None
        self._game_repeats = 0    # 同一真实游戏状态出现次数（error 帧不计数）
        self._cmd_last = None
        self._cmd_streak = 0      # 同一非 STATE 命令连续输出次数（STATE 夹在中间不计）
        self.total_stalls = 0

    def observe(self, fp, has_game_state=False, cmd=None) -> int:
        # 信号 1：连续相同指纹
        if fp == self._fp:
            self._consec += 1
        else:
            self._consec = 0
        self._fp = fp
        # 信号 2：同一真实状态反复出现（忽略 error 帧）
        if has_game_state:
            if fp == self._game_fp:
                self._game_repeats += 1
            else:
                self._game_fp = fp
                self._game_repeats = 1
        # 信号 3：同一命令连续输出（与指纹无关；STATE 夹在中间不算打断也不计数）
        if cmd is not None and cmd != "STATE":
            if cmd == self._cmd_last:
                self._cmd_streak += 1
            else:
                self._cmd_last = cmd
                self._cmd_streak = 1
        # 注意：cmd == "STATE" 时既不增也不清（允许「有效命令 STATE 无效命令」交替）
        return max(self._consec, self._game_repeats, self._cmd_streak)


class RunLog:
    """三层日志（L1 状态流 / L2 决策 / L3 事件+心跳）。"""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%m%d_%H%M%S")
        # errors="backslashreplace"：即便净化漏网，写不进的字符也转义成 \\udcaa
        # 形式落盘，绝不抛异常丢记录（双保险）。
        _open = lambda name: open(os.path.join(log_dir, f"{stamp}_{name}.jsonl"),
                                  "a", encoding="utf-8", errors="backslashreplace")
        self.f_states = _open("states")
        self.f_dec = _open("decisions")
        self.f_evt = _open("events")
        # L0：原始 stdin 行（解析前原样落盘，含解析失败的行）——"Mod 到底发了什么"
        # 的不可辩驳证据。22:32 死循环的根因之一就是 L1 只落解析成功的对象。
        self.f_raw = _open("raw")
        self.stamp = stamp
        self.n_states = 0
        self._surrogate_warned = False  # 首次净化只告警一次，不刷屏
        self._stop = threading.Event()
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()

    def _write(self, f, rec: dict):
        try:
            paths = []
            clean = _purge_surrogates(rec, "", paths)
            if paths and not self._surrogate_warned:
                self._surrogate_warned = True
                log.warning("日志含孤立代理字符（GBK/CJK 解码产物），已替换为 "
                            "U+FFFD 并落盘，不丢记录；字段: %s", paths[:5])
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
            f.flush()
        except Exception as exc:
            # 绝不静默吞掉：记 ERROR 且带上异常类型，方便事后定位
            log.error("写日志失败: %s: %s", type(exc).__name__, exc, exc_info=True)

    def raw(self, line: str):
        """原样追加一行原始 stdin（解析前），但先净化代理字符保证可解析。"""
        try:
            clean = line.encode("utf-8", "surrogatepass").decode("utf-8",
                                                                  "replace")
            if clean != line and not self._surrogate_warned:
                self._surrogate_warned = True
                log.warning("原始日志含孤立代理字符，已替换为 U+FFFD 落盘")
            self.f_raw.write(clean + "\n")
            self.f_raw.flush()
        except Exception as exc:
            log.error("写原始日志失败: %s: %s", type(exc).__name__, exc,
                      exc_info=True)

    def state(self, state: dict):
        self.n_states += 1
        self._write(self.f_states, state)

    def decision(self, **kw):
        kw["ts"] = time.time()
        self._write(self.f_dec, kw)

    def event(self, kind: str, **kw):
        kw["ts"] = time.time()
        kw["kind"] = kind
        self._write(self.f_evt, kw)

    def _heartbeat(self):
        while not self._stop.wait(60):
            self.event("heartbeat", states=self.n_states)

    def close(self):
        self._stop.set()
        for f in (getattr(self, "f_states", None), getattr(self, "f_dec", None),
                  getattr(self, "f_evt", None), getattr(self, "f_raw", None)):
            try:
                if f:
                    f.close()
            except Exception:
                pass


class SimBrain:
    """模拟器大脑：把真实游戏状态灌进 lightspeed 模拟器，用 RL Scorer 决策。"""

    def __init__(self, checkpoint_path=_DEFAULT_CKPT, obs_dim=OBS_DIM,
                 cand_dim=DESC_VEC_DIM, hidden=(512, 256),
                 combat_sims=2000, determinizations=10, log_dir=None,
                 min_cmd_interval_ms=300, stuck_grace_ms=600):
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
        self._nni = sts.getNNInterface()  # 缓存单例（C++ 所有权 bug 已修复）
        self._reject_streak = 0  # 连续被拒计数（仅 error 帧自增，真实状态清零）
        self._last_cmd = ""    # 上一次发的命令（target 被拒不带目标重发用）
        # 命令重复轮换（Bug 3）：同一个有效命令连续发出且屏幕未变 ≥3 次 → 轮换
        self._rule_last_cmd = None
        self._rule_repeat = 0
        self._rule_last_screen = None
        # 命令响应节奏（0910 第三轮真机）：相邻命令最小间隔 + 同指纹不重复发同命令
        self._min_cmd_interval_ms = min_cmd_interval_ms
        self._stuck_grace_ms = stuck_grace_ms
        self._last_cmd_time = 0.0     # 上一条命令写出的时间（节流用）
        self._cad_fp = None           # 当前已回应的状态指纹
        self._cad_fp_cmds = set()     # 该指纹下已发过的命令（每条最多一次）
        self._cad_fp_first = 0.0      # 首次看到该指纹的时间（grace 窗口用）
        self._cad_fp_exhausted = False  # 该指纹下所有可用命令都试过
        self._last_interval_ms = 0.0  # 本次与上一条命令的实际间隔（L2 诊断）
        # 模拟器新鲜度：只有本帧带完整真实 game_state 且 init_from_json 成功才为真。
        # 一条真实状态只允许产生一条模拟器命令（发出后置 False）。不新鲜时所有
        # 模拟器分支（ISMCTS / _sim_decide）被禁止，必须回 STATE 请求重发——
        # 绝不拿模拟器自己的假对局状态当事实来源（22:32 死循环主因）。
        self._sim_fresh = False
        self._gs_keys_logged = False  # 一次性诊断：真实 game_state 有哪些键
        self.obs_dim = obs_dim
        self.cand_dim = cand_dim
        self.log = RunLog(log_dir) if log_dir else None
        self.decisions = 0
        self.fallbacks = 0

        self.scorer = None
        if os.path.exists(checkpoint_path):
            try:
                self.scorer = Scorer(obs_dim=obs_dim, cand_dim=cand_dim,
                                     hidden=tuple(hidden))
                state_dict = torch.load(checkpoint_path, weights_only=True)
                self.scorer.load_state_dict(state_dict)
                self.scorer.eval()
                log.info("scorer loaded: %s (obs=%d cand=%d hidden=%s)",
                         checkpoint_path, obs_dim, cand_dim, hidden)
                print(f"[brain] 权重已加载: {checkpoint_path} "
                      f"obs={obs_dim} cand={cand_dim} hidden={hidden}",
                      file=sys.stderr, flush=True)
            except Exception as exc:
                log.exception("scorer 加载失败，宏观决策将全程走规则降级")
                print(f"[brain] !!! 权重加载失败，将全程走规则降级: {exc}",
                      file=sys.stderr, flush=True)
                self.scorer = None
        else:
            log.warning("checkpoint 不存在: %s，宏观决策将全程走规则降级",
                        checkpoint_path)
            print(f"[brain] !!! 权重不存在: {checkpoint_path}，将全程走规则降级",
                  file=sys.stderr, flush=True)

        self._ascension = 0
        self._map_cache = None       # (seed, act) 对应的 SpireMap
        self._map_cache_key = None
        self.combat_brain = CombatBrain(simulation_count=combat_sims,
                                        determinizations=determinizations)

    # ---- 决策入口 ----

    def _handle_error_frame(self, state: dict, err) -> str:
        """处理游戏的 error 帧。

        关键错误：「Selected card cannot be played with the selected target」——
        Communication Mod 语义：不需要目标的牌（Defend/技能/能力/群体攻击）一旦
        带上目标下标，整条命令被拒且游戏不执行任何操作。这种情况我们不回 STATE、
        也不计入熔断，而是把上一条 PLAY 命令去掉末尾目标下标后立刻重发。

        其它错误：计入连续被拒计数，≥3 次强制 END 回合（熔断保命）。
        """
        log.warning("game error: %s", err)
        low = (err or "").lower()
        if "cannot be played with the selected target" in low:
            prev = self._last_cmd
            if isinstance(prev, str) and prev.startswith("PLAY"):
                parts = prev.split()
                if len(parts) >= 3:
                    new_cmd = " ".join(parts[:2])  # PLAY 1 0 -> PLAY 1
                    if self.log:
                        self.log.event("target_rejected", original=prev,
                                       retried=new_cmd, error=err)
                    log.info("目标被拒，换形式重发: %r -> %r", prev, new_cmd)
                    self._last_cmd = new_cmd
                    return new_cmd
            # 上一条不是带目标的 PLAY（无真实状态可重试）→ 回 STATE 请求重发，
            # 绝不拿模拟器假状态猜命令，也不计入熔断。
            return "STATE"
        # 通用错误（非目标错误）：计入连续被拒，超限强 END 回合（熔断保命）。
        self._reject_streak += 1
        if self._reject_streak >= 3:
            n = self._reject_streak
            self._reject_streak = 0
            if self.log:
                self.log.event("force_end", reason="reject_streak", count=n,
                               error=err)
            log.warning("命令连续被拒 %d 次，强制 END 回合", n)
            self._last_cmd = "END"
            return "END"
        return "STATE"

    def decide(self, state: dict, apply_repeat_guard: bool = True) -> str:
        # --- error 帧处理（先于一切分支，避免被后面「无 game_state → STATE」吞掉） ---
        err = state.get("error")
        if err:
            return self._handle_error_frame(state, err)

        # 收到真实游戏状态（带 game_state 且就绪）→ 清零连续被拒计数。
        # 不再依赖 _last_cmd，避免「被拒后插 STATE 把计数清零」导致熔断失效。
        if state.get("game_state") and state.get("ready_for_command"):
            self._reject_streak = 0

        if _is_game_over(state):
            gs = state.get("game_state", {})
            win = _is_victory(state)
            if self.log:
                self.log.event("game_over", victory=win,
                               floor=gs.get("floor"), act=gs.get("act"),
                               hp=gs.get("current_hp"), max_hp=gs.get("max_hp"))
            log.info("GAME_OVER victory=%s floor=%s", win, gs.get("floor"))

        if not state.get("ready_for_command"):
            return "STATE"

        # Communication Mod 的 available_commands 全是小写
        cmds = [c.lower() for c in state.get("available_commands", [])]

        # 模拟器新鲜度（P1：缺 seed 时用占位 seed=0 灌入；见 _refresh_sim）
        self._refresh_sim(state)

        # 主菜单：规则命令，不依赖模拟器（即使不新鲜也照发）
        if "start" in cmds:
            cmd = "START IRONCLAD"
            if self.log:
                self.log.event("run_start", command=cmd)
            return cmd

        # ---- 商店例外（必须在 choose 分支之前）：优先离开，绝不"购物" ----
        # 真机 2026-09-11 血案：SHOP_SCREEN 的 choice_list 是商品列表
        # （'purge' + 卡牌/药水名），且 avail 含 choose → 下面的 choose 分支
        # 永远命中 → 一直 CHOOSE 0 买/开删除卡牌服务，金币 110 → 14，
        # 状态在 SHOP_SCREEN/SHOP_ROOM/GRID 之间死循环，永远出不了商店。
        # 模拟器侧不支持购买决策，买了纯浪费金币，所以商店一律先离开。
        _gs = state.get("game_state") or {}
        _scr = (_gs.get("screen_type") or "").upper()
        _nm = (_gs.get("screen_name") or "").upper()
        if (_scr == "SHOP_SCREEN" or _nm == "SHOP") and "leave" in cmds:
            # 仍走 _repeat_guard：商店真能关掉时屏幕会变、计数自动清零；
            # 若 LEAVE 连续 3 次无效则轮换（否则会一直重发 LEAVE 打转过场动画）。
            cmd = self._repeat_guard(state, "LEAVE")
            self._log_decision(state, screen=_scr or None, kind="RULE",
                               cmd=cmd, n_cands=0, degraded=True)
            self._last_cmd = cmd
            return cmd

        # ---- P0 安全网 #1：选项类屏幕（有 choice_list）永远选第一个 ----
        # 优先级最高：确保「缺 seed 等导致模拟器不新鲜」时也给出可执行命令，绝不
        # 回 STATE 空转（本任务要修的新 failure mode）。CHOOSE 0 对事件/篝火/地图/
        # 选卡永远合法且一定推进游戏。
        if "choose" in cmds and self._has_choice_list(state):
            return "CHOOSE 0"

        # 战斗：ISMCTS（灌状态进模拟器搜索最优出牌），需模拟器新鲜。不新鲜
        # （缺 seed 且占位 seed 也灌入失败 / 无真实 game_state）→ P0 真实状态规则
        # 兜底（出牌或 END），绝不用假状态决策、绝不空转。
        if "play" in cmds:
            if not self._sim_fresh:
                log.warning("战斗命令但模拟器不新鲜（无真实 game_state/缺 seed 等），"
                            "P0 真实状态规则兜底（绝不用假状态决策，也绝不空转）")
                cmd = self._real_state_rule(state, apply_repeat_guard)
                self._log_decision(state, screen="COMBAT", kind="RULE",
                                   cmd=cmd, n_cands=0, degraded=True)
                self._last_cmd = cmd
                return cmd
            try:
                cmd = self.combat_brain.decide_with_fallback(
                    state, self.gc, combat_action)
                self._log_decision(state, screen="COMBAT", kind="COMBAT",
                                   cmd=cmd, n_cands=0, degraded=False)
            except Exception:
                log.exception("MCTS 战斗异常，回退规则")
                cmd = combat_action(state)
                self._log_decision(state, screen="COMBAT", kind="COMBAT",
                                   cmd=cmd, n_cands=0, degraded=True)
            cmd = self._repeat_guard(state, cmd) if apply_repeat_guard else cmd  # Bug 3
            self._sim_fresh = False  # 一条真实状态只允许一条命令
            self._last_cmd = cmd
            return cmd

        # 战斗奖励：模拟器没有真实奖励内容，RL 会错误跳过——规则逐个领取（不依赖模拟器）
        gs = state.get("game_state", {})
        if (gs.get("screen_type") or "").upper() == "COMBAT_REWARD":
            cmd = macro_action(state)
            self._log_decision(state, screen="COMBAT_REWARD", kind="REWARD",
                               cmd=cmd, n_cands=0, degraded=True)
            return cmd

        # 宏观决策：模拟器 + RL；需模拟器新鲜。不新鲜 → P0 真实状态规则兜底
        # （绝不回 STATE 空转：会卡死）。
        if not self._sim_fresh:
            log.warning("宏观命令但模拟器不新鲜（缺 seed 且占位 seed 也失败等），"
                        "P0 真实状态规则兜底（绝不用假状态决策，也绝不空转）")
            cmd = self._real_state_rule(state, apply_repeat_guard)
            self._log_decision(state, screen=gs.get("screen_type"), kind="RULE",
                               cmd=cmd, n_cands=0, degraded=True)
            self._last_cmd = cmd
            return cmd
        try:
            cmd = self._sim_decide(state, cmds)
        except Exception as exc:
            log.exception("sim 决策异常，回退规则决策")
            if self.log:
                self.log.event("sim_error", error=f"{type(exc).__name__}: {exc}")
            cmd = None
        if cmd is None:
            cmd = macro_action(state)
            self.fallbacks += 1
            log.info("rule fallback -> %s", cmd)
            self._log_decision(state, screen=gs.get("screen_type"), kind="-",
                               cmd=cmd, n_cands=0, degraded=True)
        self._sim_fresh = False  # 一条真实状态只允许一条命令
        cmd = self._repeat_guard(state, cmd)  # Bug 3：宏观同命令同屏 ≥3 次→轮换
        self._last_cmd = cmd
        return cmd

    # ---- 模拟器新鲜度 ----

    # 判定「真实 game_state 是否完整」所需的必备字段（缺任一即视为不完整，
    # init_from_json 会崩或产出的模拟器状态不可信）。
    _REQUIRED_GS_KEYS = ("seed", "floor", "act", "current_hp", "max_hp", "gold")

    def _has_complete_game_state(self, state) -> bool:
        gs = state.get("game_state")
        if not isinstance(gs, dict):
            return False
        return all(gs.get(k) is not None for k in self._REQUIRED_GS_KEYS)

    def _missing_only_seed(self, state) -> bool:
        """除 seed 外其它必备字段都在（seed 缺失或为 null）。

        用于 P1：这种状态其它字段齐全，只是 Mod 没发 seed；用占位 seed=0
        灌模拟器即可（实测可行，见 _refresh_sim）。若连其它字段也缺，则不是
        「只缺 seed」，不能占位，回 STATE 等重发。
        """
        gs = state.get("game_state")
        if not isinstance(gs, dict):
            return False
        others = [k for k in self._REQUIRED_GS_KEYS if k != "seed"]
        return (all(gs.get(k) is not None for k in others)
                and gs.get("seed") is None)

    def _has_choice_list(self, state) -> bool:
        """真实状态是否带「可选选项列表」（事件/篝火/地图/选卡屏）。"""
        gs = state.get("game_state") or {}
        cl = gs.get("choice_list")
        if isinstance(cl, list) and cl:
            return True
        ro = (gs.get("screen_state") or {}).get("rest_options")
        return isinstance(ro, list) and bool(ro)

    def _refresh_sim(self, state):
        """尝试把本帧真实状态灌入模拟器。仅当状态完整且灌入成功才置 _sim_fresh。

        P1 占位 seed：若状态只缺 seed（其它必备字段齐全），用占位 seed=0 灌入——
        实测 init_from_json 在 seed 缺失/为 null 时会抛 RuntimeError，但给占位
        seed 后能成功且 get_all_actions 返回合理动作（MAP_SCREEN=5 个、BATTLE=0
        个战斗动作，战斗动作本就来自 ISMCTS 而非 get_all_actions）。这样既不拿假
        状态决策，也不会因缺 seed 永远不新鲜而空转。

        错误帧（无 game_state）/ 连其它字段也缺的状态不会灌入，模拟器保持不新鲜。
        """
        self._sim_fresh = False
        payload = state
        if not self._has_complete_game_state(state):
            if self._missing_only_seed(state):
                # 只缺 seed：占位 seed=0 灌模拟器（P1）
                gs = dict(state.get("game_state") or {})
                gs["seed"] = 0
                payload = dict(state)
                payload["game_state"] = gs
                log.info("[P1] 真实状态缺 seed，用占位 seed=0 灌模拟器")
            else:
                return
        try:
            self.gc.init_from_json(json.dumps(_sanitize(payload)))
        except Exception:
            log.exception("init_from_json 失败（缺 seed 且占位 seed 也失败/格式异常），"
                          "模拟器不新鲜 → 走 P0 真实状态规则兜底")
            self._sim_fresh = False
            return
        self._sim_fresh = True
        self._maybe_log_game_state_keys(state)

    def _maybe_log_game_state_keys(self, state):
        """一次性诊断：首次拿到带 game_state 的真实状态，打印其键名 + 首张手牌字段。

        这次真机日志里根本没抓到过真实状态，无法确认 game_state 到底有哪些键、
        hand 里有没有 has_target；打印一次便于下次真机立刻确认（只打一次，不刷屏）。
        """
        if self._gs_keys_logged:
            return
        self._gs_keys_logged = True
        gs = state.get("game_state", {})
        if not isinstance(gs, dict):
            return
        log.info("[diag] 真实 game_state 键名: %s",
                 list(gs.keys()))
        combat = gs.get("combat_state")
        if isinstance(combat, dict):
            log.info("[diag] combat_state 键名: %s", list(combat.keys()))
            hand = combat.get("hand") or []
            if hand and isinstance(hand[0], dict):
                log.info("[diag] 首张手牌字段: %s ; 完整内容: %s",
                         list(hand[0].keys()), hand[0])

    # ---- P0 真实状态规则（安全网，不碰模拟器） ----

    def _real_state_rule(self, state: dict, apply_repeat_guard: bool = True) -> str:
        """P0 安全网：真实状态规则，只读 game_state / available_commands。

        只要有真实状态且 ready_for_command，务必给出一个**可执行**命令，绝不
        允许一直回 STATE 导致静默死锁（本任务要修的新 failure mode：缺 seed 等
        导致 _sim_fresh 永远 False → 宏观/战斗分支都回 STATE → 空转）。

        优先级：
          choice_list 非空        → CHOOSE 0   （事件/篝火/地图/选卡，永远合法推进）
          proceed 可用            → PROCEED
          end 可用                → END
          play 可用               → 战斗规则出牌，规则哑火则 END（推进回合）
          skip / leave 可用       → 对应命令
        以上都不满足（确实没有任何可用命令）→ 才回 STATE（绝不假装行动原地空转）。
        """
        natural = self._natural_rule_cmd(state)
        return self._repeat_guard(state, natural) if apply_repeat_guard else natural

    def _natural_rule_cmd(self, state: dict) -> str:
        """真实状态规则的「自然」命令（未做重复轮换）。"""
        cmds = [c.lower() for c in state.get("available_commands", [])]

        # 商店屏必须**优先离开**：SHOP_SCREEN 的 choice_list 是商品列表
        # （'purge' + 卡牌/药水名），而 choose 分支排在 leave 前面 → 会一直
        # CHOOSE 0 去购物/开删除卡牌服务，永远轮不到 LEAVE。真机 2026-09-11
        # 实测：金币 110 → 14，状态在 SHOP_SCREEN/SHOP_ROOM/GRID 之间循环。
        # 模拟器侧不支持购买决策，买了纯浪费金币，所以规则一律先离开。
        gs0 = state.get("game_state") or {}
        screen0 = (gs0.get("screen_type") or "").upper()
        name0 = (gs0.get("screen_name") or "").upper()
        if screen0 == "SHOP_SCREEN" or name0 == "SHOP":
            if "leave" in cmds:
                return "LEAVE"

        if "choose" in cmds and self._has_choice_list(state):
            return "CHOOSE 0"
        if "proceed" in cmds:
            return "PROCEED"
        if "end" in cmds:
            return "END"
        if "play" in cmds:
            cmd = combat_action(state)
            if cmd and cmd.upper() != "STATE":
                return cmd
            return "END"
        if "skip" in cmds:
            return "SKIP"
        if "leave" in cmds:
            return "LEAVE"
        # 确实没有任何可执行命令
        return "STATE"

    # 可以**不带参数**直接发的命令白名单。其余（key/click/play/potion/choose…）
    # 需要参数，原样发出去只会换来 "Argument missing in command" 拒绝 ——
    # 真机实测 81 次 `Argument missing in command "key"` 就是这么来的。
    _SAFE_NOARG = ("proceed", "leave", "skip", "end", "start", "wait",
                   "confirm", "return", "state")

    def _rotate_available(self, state: dict, avoid: str):
        """挑一个 ≠ avoid 的**可安全发出**的命令做轮换；没有返回 None。

        只从 _SAFE_NOARG 白名单里挑（key/click/play/potion 需要参数，不能裸发）；
        `choose` 若可用则回 CHOOSE 0（它一定有选项可退）。
        """
        cmds = [c.lower() for c in state.get("available_commands", [])]
        if not cmds:
            return None
        others = [c.upper() for c in cmds
                  if c in self._SAFE_NOARG and c != "STATE"
                  and c.upper() != (avoid or "").upper()]
        if others:
            return others[0]
        # 商店里绝不轮换到 CHOOSE：SHOP_SCREEN 的 choice_list 是商品列表，
        # CHOOSE 0 = 买东西 / 打开删除卡牌服务（会白花金币、开新屏）。
        gs = state.get("game_state") or {}
        is_shop = ((gs.get("screen_type") or "").upper() == "SHOP_SCREEN"
                   or (gs.get("screen_name") or "").upper() == "SHOP")
        if "choose" in cmds and not is_shop and (avoid or "").upper() != "CHOOSE 0":
            return "CHOOSE 0"
        if avoid != "STATE":
            return "STATE"
        return None

    def _repeat_guard(self, state: dict, cmd: str) -> str:
        """Bug 3：同一有效命令连续 ≥3 次且屏幕未变 → 轮换为其它可用命令。

        轮换只在该帧已有「同屏同命令重复」时才发生；STATE 是 P0 兜底（无命令
        可用），不算有效命令重复，不轮换、清零计数。返回可能已轮换的命令。
        """
        if cmd == "STATE" or cmd is None:
            self._rule_last_cmd = None
            self._rule_repeat = 0
            return cmd
        screen = (state.get("game_state", {}) or {}).get("screen_type") or ""
        screen = screen.upper()
        if cmd == self._rule_last_cmd and screen == self._rule_last_screen:
            self._rule_repeat += 1
        else:
            self._rule_last_cmd = cmd
            self._rule_last_screen = screen
            self._rule_repeat = 1
        if self._rule_repeat >= 3:
            alt = self._rotate_available(state, avoid=cmd)
            if alt is not None and alt != cmd:
                if self.log:
                    self.log.event("cmd_rotate", from_cmd=cmd, to_cmd=alt,
                                   screen=screen, count=self._rule_repeat)
                # 轮换后把当前命令记为 alt，避免下一帧又因「同命令」立刻再轮换
                self._rule_last_cmd = alt
                self._rule_repeat = 1
                return alt
        return cmd

    # ---- 模拟器决策路径 ----

    def _sim_decide(self, state: dict, cmds: list):
        """灌入模拟器 → RL 选动作 → 翻译成 comm 命令；不可翻译返回 None。

        前置：调用方已通过 _refresh_sim 灌入并确认 _sim_fresh，故此处不再灌。
        """
        if self.scorer is None:
            return None
        game = state.get("game_state", {})

        # 兜底：若不知为何在不新鲜时被调用，直接放弃（不靠假状态打分）
        if not self._sim_fresh:
            log.warning("sim 决策被调用但模拟器不新鲜，放弃")
            return None

        gc = self.gc
        self._ascension = int(game.get("ascension_level", 0) or 0)
        screen = gc.screen_state.name

        actions = sts.get_all_actions(gc)
        if not actions:
            log.warning("sim 无可用动作（screen=%s），回退规则", screen)
            return None

        # ---- 观测：412 原始 + 65 增强 = 477（与训练严格一致） ----
        raw = list(self._nni.getObservation(gc))
        obs = enhance_observation(raw, gc, card_index=CARD_INDEX,
                                  arch_table=CARD_ARCHETYPE,
                                  card_effects=CARD_EFFECTS)
        # 牌组流派强度（5 维）：喂给候选的流派交叉项
        deck_arch = encode_deck_stats(raw, card_index=CARD_INDEX,
                                      arch_table=CARD_ARCHETYPE,
                                      card_effects=CARD_EFFECTS)[:5]

        encoded = [self._encode(a, screen, i, deck_arch, gc)
                   for i, a in enumerate(actions)]
        with torch.no_grad():
            logits = self.scorer(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(encoded, dtype=torch.float32),
            )
        scores = [float(x) for x in logits]
        best = int(torch.argmax(logits))
        chosen = actions[best]
        log.debug("screen=%s n_act=%d best=%d bits=0x%08x logits=%s",
                  screen, len(actions), best, chosen.bits, scores)

        cmd = self._to_comm(chosen, actions, screen, cmds, state)
        top3 = sorted(range(len(scores)), key=lambda i: -scores[i])[:3]
        self._log_decision(
            state, screen=screen, kind=candidate_kind(screen,
                                                      decode_action_bits(chosen.bits)),
            cmd=cmd, n_cands=len(actions),
            scores=[round(scores[i], 4) for i in top3],
            bits=[f"0x{actions[i].bits:08x}" for i in top3],
            chosen_idx=best, degraded=(cmd is None),
        )
        return cmd

    # ---- bits → Communication Mod 命令翻译 ----

    def _to_comm(self, action, actions: list, screen: str, cmds: list,
                 state: dict = None):
        """把选中的 GameAction 翻译成 comm 命令；无法翻译返回 None（走规则）。"""
        d = decode_action_bits(action.bits)

        # v1 不翻译药水动作（bit31/30）
        if d["is_potion"] or d["is_potion_discard"]:
            return None

        # 真实游戏的选项数量上限（防止模拟器选项多于真实游戏导致越界）
        real_choice_count = None
        if state:
            gs = state.get("game_state", {})
            cl = gs.get("choice_list") or gs.get("screen_state", {}).get("rest_options")
            if cl is not None:
                real_choice_count = len(cl)

        def _safe_choose(idx):
            """CHOOSE 命令加上界检查：模拟器的选项可能比真实游戏多（比如
            模拟器篝火有 dig 但真实游戏没有铲子遗物），越界会导致游戏报错卡死。"""
            if "choose" not in cmds:
                return None
            if real_choice_count is not None and idx >= real_choice_count:
                log.warning("choose idx %d 越界（真实游戏只有 %d 个选项），回退规则",
                            idx, real_choice_count)
                return None
            return f"CHOOSE {idx}"

        if screen == "MAP_SCREEN":
            return self._map_to_comm(actions, d, cmds)

        if screen in ("REWARDS", "CARD_SELECT"):
            rt = d["reward_type"]
            if rt == REWARD_CARD:
                idx = d["idx2"]
                if idx == SINGING_BOWL_IDX2:
                    return None  # 唱歌碗：v1 不会翻译，回退规则
                cards = self.gc.get_card_reward()
                if idx >= len(cards):
                    return None
                return f"CHOOSE {idx}" if "choose" in cmds else None
            if rt == REWARD_SKIP:
                if "skip" in cmds:
                    return "SKIP"
                if "proceed" in cmds:
                    return "PROCEED"
                return None
            # GOLD/KEY/POTION/RELIC/CARD_REMOVE：comm mod 通常自动收取，v1 不翻译
            return None

        if screen == "REST_ROOM":
            return _safe_choose(d["idx1"])

        if screen == "EVENT_SCREEN":
            return _safe_choose(d["idx1"])

        if screen == "SHOP_ROOM":
            # v1 直接离开商店（不买）
            if "proceed" in cmds:
                return "PROCEED"
            if "leave" in cmds:
                return "LEAVE"
            return None

        return None

    @staticmethod
    def _map_to_comm(actions: list, d: dict, cmds: list):
        """MAP 屏：idx1 = 目标节点 x 坐标。

        comm mod 的 choice_list 是可选节点列表，顺序与下一层节点从左到右
        （x 升序）一致；模拟器 MAP 动作与可选节点一一对应，因此把所有地图
        动作的 x 排序后，选中 x 的位次即 CHOOSE 下标。
        """
        if "choose" not in cmds:
            return None
        xs = sorted(a.bits & 0xFF for a in actions)
        x = d["idx1"]
        if x not in xs:
            return None
        return f"CHOOSE {xs.index(x)}"

    # ---- 候选编码（与 env/lightspeed_env 同源：kind/KB 走共享模块） ----

    def _encode(self, action, screen: str, idx: int, deck_arch, gc) -> list:
        d = decode_action_bits(action.bits)
        label = action.describe(gc) or f"action_0x{action.bits:08x}"
        kb = resolve_candidate_kb(screen, d, gc)
        proto = Candidate(idx=idx, kind=candidate_kind(screen, d),
                          desc_vec=[], label=label)
        return encode_candidate(
            proto,
            card=self._candidate_card(screen, d),
            room=self._candidate_room(screen, d),
            bits=action.bits,
            lookahead_rooms=self._candidate_lookahead(screen, d),
            deck_arch=deck_arch,
            **kb,
        )

    def _candidate_card(self, screen: str, d: dict):
        if screen != "REWARDS" or d["reward_type"] != 0 or d["idx2"] == 5:
            return None
        cards = self.gc.get_card_reward()
        if d["idx2"] >= len(cards):
            return None
        c = cards[d["idx2"]]
        return CardInfo(
            name=c.id.name, card_type=c.type.name, rarity=c.rarity.name,
            cost=0.0,  # get_card_reward 返回 Card 无 cost；编码时回退查 card_data.base_cost
            upgraded=c.upgraded,
        )

    def _candidate_room(self, screen: str, d: dict):
        if screen != "MAP_SCREEN":
            return None
        y = self.gc.cur_map_node_y
        if y == 14:
            return "BOSS"
        if self.gc.act >= 4:
            return None
        m = self._spire_map()
        target_y = 0 if y == -1 else y + 1
        name = m.get_room_type(d["idx1"], target_y).name
        return None if name in ("INVALID", "NONE") else name

    def _candidate_lookahead(self, screen: str, d: dict):
        if screen != "MAP_SCREEN":
            return None
        y = self.gc.cur_map_node_y
        if y == 14 or self.gc.act >= 4:
            return []
        m = self._spire_map()
        target_y = 0 if y == -1 else y + 1

        def room_fn(x, yy):
            name = m.get_room_type(x, yy).name
            return None if name in ("INVALID", "NONE") else name

        return compute_map_lookahead(d["idx1"], target_y, room_fn, m.has_edge)

    def _spire_map(self):
        key = (self.gc.seed, self.gc.act)
        if self._map_cache_key != key:
            self._map_cache = sts.SpireMap(
                self.gc.seed, self._ascension, self.gc.act, False)
            self._map_cache_key = key
        return self._map_cache

    # ---- 日志辅助 ----

    def _log_decision(self, state, *, screen, kind, cmd, n_cands,
                      scores=None, bits=None, chosen_idx=None, degraded=False):
        self.decisions += 1
        if not self.log:
            return
        gs = state.get("game_state", {}) or {}
        self.log.decision(
            n=self.decisions, floor=gs.get("floor"), act=gs.get("act"),
            hp=gs.get("current_hp"), max_hp=gs.get("max_hp"),
            gold=gs.get("gold"), screen=screen, kind=kind,
            n_cands=n_cands, cmd=cmd, scores=scores, bits=bits,
            chosen_idx=chosen_idx, degraded=degraded,
            total_fallbacks=self.fallbacks,
            # Bug 3：把该帧的 available_commands 与 screen_type 写进 L2 决策记录，
            # 便于事后复盘「同一命令被轮换」到底发生在哪个屏、当时有哪些可选命令。
            avail=state.get("available_commands"),
            screen_raw=gs.get("screen_type"),
        )


def _parse_hidden(s: str):
    return tuple(int(x) for x in s.split(",") if x.strip())


class CommandPacer:
    """命令节奏与升级控制（真机 2026-09-10 晚实测教训，必须保留）。

    现象：agent 每 **167ms** 回一条命令（每秒 6 条），而游戏"关闭商店"的过场
    动画需要几百毫秒——命令在过场进行中又到达，动画被打断并重启，商店永远
    关不掉，界面在开/关之间反复**闪烁**（用户原话）。
    结论：**命令语义是对的，节奏是错的。**

    本类做两件事：
      1) **最小命令间隔**：写 stdout 前 sleep 补齐。不能靠"不回"节流——
         Mod 在等回应，不回会把游戏挂住。
      2) **同一状态指纹下每条命令最多发一次**：指纹相同说明上一条命令没能
         推动状态；先等 `stuck_grace_ms`（给过场留时间），仍不变则**升级**到
         该指纹下还没试过的下一条可用命令；全部试过 → 只回 STATE 并停止对该
         指纹发命令（事件里体现）。
    """

    def __init__(self, min_interval_ms=300, stuck_grace_ms=600):
        self.min_interval = max(0.0, float(min_interval_ms) / 1000.0)
        self.grace = max(0.0, float(stuck_grace_ms) / 1000.0)
        self.last_cmd_time = 0.0
        self.last_interval_ms = None
        self.waits = 0
        self.logged_wait = False
        self._fp = None
        self._fp_t = 0.0
        self._tried = set()
        self._exhausted = False

    def pace(self, fp, cmd, avail=None):
        """只做**最小命令间隔节流**，返回应写出的命令（内容不变）。

        分工：命令语义与"同屏同命令轮换"由 SimBrain._repeat_guard 负责；
        本类只管**节奏**——真机实测每条 167ms 会把游戏过场动画反复打断，
        表现为商店界面闪烁，所以这里强制两条命令之间至少间隔 min_interval。
        必须 sleep 而不是"不回"：Mod 在等回应，不回会把游戏挂住。
        """
        now = time.monotonic()
        wait = 0.0
        if self.last_cmd_time and self.min_interval:
            wait = self.min_interval - (now - self.last_cmd_time)
        if wait > 0:
            time.sleep(wait)
            self.waits += 1
        self._fp = fp
        return cmd

    def mark_sent(self):
        """在**真正写出命令之后**调用：记录时间戳与实际间隔。

        必须在 write 之后调用——主循环在写之前还会为 STATE 额外 sleep 0.1s，
        若在 pace() 里记时间戳，实际间隔会比 min_interval 少最多 100ms
        （实测最小值 200ms 而非 300ms，就是这么来的）。
        """
        now = time.monotonic()
        if self.last_cmd_time:
            self.last_interval_ms = int((now - self.last_cmd_time) * 1000)
        self.last_cmd_time = now


def main(argv=None):
    p = argparse.ArgumentParser(prog="main", description="sts-agent 真实游戏桥接")
    p.add_argument("--checkpoint", default=_DEFAULT_CKPT,
                   help=f"Scorer 权重路径（默认 {_DEFAULT_CKPT}）")
    p.add_argument("--obs-dim", type=int, default=OBS_DIM)
    p.add_argument("--cand-dim", type=int, default=DESC_VEC_DIM)
    p.add_argument("--hidden", default="512,256",
                   help="与训练时一致：PPO=512,256")
    p.add_argument("--combat-sims", type=int, default=2000,
                   help="战斗 ISMCTS 每个平行未来的模拟次数")
    p.add_argument("--determinizations", type=int, default=10,
                   help="战斗 ISMCTS 平行未来数 K")
    p.add_argument("--log-dir", default=_DEFAULT_LOG_DIR,
                   help="三层日志目录；传空串关闭")
    p.add_argument("--min-cmd-interval-ms", type=int, default=300,
                   help="两条命令之间的最小间隔（毫秒）。真机实测：默认节奏约 167ms/条，"
                        "会打断游戏过场动画导致界面闪烁，故默认节流到 300ms")
    p.add_argument("--stuck-grace-ms", type=int, default=600,
                   help="同一状态指纹下同一命令重复时的等待宽限（毫秒）；"
                        "超过后升级到该指纹下还没试过的下一条可用命令")
    p.add_argument("--dry-run", action="store_true",
                   help="只做编码自检（不读写 stdin/stdout），用于启动前验证")
    args = p.parse_args(argv)

    brain = SimBrain(
        checkpoint_path=args.checkpoint, obs_dim=args.obs_dim,
        cand_dim=args.cand_dim, hidden=_parse_hidden(args.hidden),
        combat_sims=args.combat_sims,
        determinizations=args.determinizations,
        log_dir=(args.log_dir or None),
    )

    if args.dry_run:
        # 自检：用模拟器跑一小段，确认编码链路维度正确
        from env.lightspeed_env import LightspeedEnv
        env = LightspeedEnv(mcts_simulations=200)
        env.reset(20260831)
        cands = env.candidates()
        print(f"[dry-run] 模拟器决策点 OK: screen={env.gc.screen_state.name} "
              f"候选数={len(cands)}")
        print(f"[dry-run] env 观测维度={len(env._nn_vec())} "
              f"候选维度={len(cands[0].desc_vec)}")
        print(f"[dry-run] brain 期望 obs_dim={brain.obs_dim} "
              f"cand_dim={brain.cand_dim} scorer={'已加载' if brain.scorer else '未加载'}")
        if brain.log:
            brain.log.event("dry_run", ok=True)
            brain.log.close()
        return 0

    print(f"[brain] 就绪 | 权重={args.checkpoint} | "
          f"obs={args.obs_dim} cand={args.cand_dim} | "
          f"战斗 ISMCTS {args.determinizations}×{args.combat_sims} | "
          f"日志={args.log_dir}", file=sys.stderr, flush=True)
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    log.info("sts-agent bridge ready (sim brain)")
    guard = LoopDetector()
    pacer = CommandPacer(args.min_cmd_interval_ms, args.stuck_grace_ms)

    for line in sys.stdin:
        raw_line = line.strip()
        if not raw_line:
            continue
        # L0：原始 stdin 行原样落盘（解析前），含解析失败的行——"Mod 到底发了什么"
        if brain.log:
            brain.log.raw(raw_line)
        try:
            state = json.loads(raw_line)
        except json.JSONDecodeError:
            log.error("bad json: %s", raw_line[:200])
            continue

        # L1：真实状态原样落盘（无损重放）
        if brain.log:
            brain.log.state(state)
        # 兼容旧习惯：把最近一次状态单独写盘（同样抗代理字符，绝不静默丢）
        try:
            with open("last_state.json", "w", encoding="utf-8",
                      errors="backslashreplace") as f:
                json.dump(_purge_surrogates(state), f, ensure_ascii=False,
                          indent=1)
        except Exception as exc:
            log.error("写 last_state.json 失败: %s: %s", type(exc).__name__, exc,
                      exc_info=True)

        try:
            cmd = brain.decide(state)
        except Exception as exc:
            log.exception("decide 顶层异常，发 STATE 保命")
            if brain.log:
                brain.log.event("fatal", error=f"{type(exc).__name__}: {exc}")
            cmd = "STATE"

        # 防卡死：连续相同状态指纹 / 同一真实状态反复出现 / 同一命令连续输出
        # （含夹 error 帧、STATE 中间）→ 强制换动作；仍未恢复 → 退出保命
        strikes = guard.observe(_fingerprint(state),
                                has_game_state=bool(state.get("game_state")),
                                cmd=cmd)
        if strikes >= guard.limit:
            alt = _alternate_cmd(state, avoid=cmd)
            guard.total_stalls += 1
            screen = (state.get("game_state", {}) or {}).get("screen_type")
            if brain.log:
                brain.log.event("stall", strikes=strikes, prev_cmd=cmd,
                                forced=alt, screen=screen)
            print(f"[guard] 卡死检测：状态连续 {strikes} 次未变，"
                  f"强制换动作 {cmd!r} -> {alt!r} (screen={screen})",
                  file=sys.stderr, flush=True)
            cmd = alt
            if strikes >= guard.hard_limit:
                if brain.log:
                    brain.log.event("stall_abort", strikes=strikes, screen=screen)
                print(f"[guard] !!! 卡死 {strikes} 次仍无法恢复，主动退出。"
                      f"原始状态已存 logs/realtime/*_states.jsonl，可复盘。",
                      file=sys.stderr, flush=True)
                break

        # 节奏与去重：真机实测 167ms/条会打断游戏过场 → 界面闪烁；见 CommandPacer
        avail = state.get("available_commands", []) or []
        paced = pacer.pace(_fingerprint(state), cmd, avail)
        if paced != cmd and brain.log:
            brain.log.event("cmd_paced", wanted=cmd, sent=paced,
                            interval_ms=pacer.last_interval_ms,
                            tried=sorted(pacer._tried))
        if pacer.waits and not pacer.logged_wait and brain.log:
            brain.log.event("cadence_wait",
                            min_interval_ms=int(pacer.min_interval * 1000))
            pacer.logged_wait = True
        cmd = paced

        log.debug("screen=%s -> %s",
                  state.get("game_state", {}).get("screen_type"), cmd)
        if cmd == "STATE":
            time.sleep(0.1)  # 避免在主菜单等无操作状态下疯狂自旋
        sys.stdout.write(cmd + "\n")
        sys.stdout.flush()
        pacer.mark_sent()

    if brain.log:
        brain.log.event("bridge_exit", decisions=brain.decisions,
                        fallbacks=brain.fallbacks,
                        stalls=guard.total_stalls,
                        cadence_waits=pacer.waits)
        brain.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
