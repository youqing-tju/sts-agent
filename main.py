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
_STALL_LIMIT = 4        # 状态指纹连续相同几次 → 判定卡死，强制换动作
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
    if "choose" in cmds:
        candidates.append("CHOOSE 0")        # 事件/篝火/地图：退到第一个选项
    if "leave" in cmds:
        candidates.append("LEAVE")
    if "start" in cmds:
        candidates.append("START IRONCLAD")
    for c in cmds:                           # 兜底：任何非 state 命令
        if c != "state":
            candidates.append(c.upper())

    for cand in candidates:
        if cand != avoid_u:
            return cand
    return candidates[0] if candidates else "STATE"


class StallGuard:
    """连续相同状态指纹检测（与发了什么命令无关，只看游戏有没有动）。"""

    def __init__(self, limit=_STALL_LIMIT, hard_limit=_STALL_HARD_LIMIT):
        self.limit = limit
        self.hard_limit = hard_limit
        self._fp = None
        self.strikes = 0
        self.total_stalls = 0

    def observe(self, fp) -> int:
        if fp == self._fp:
            self.strikes += 1
        else:
            self.strikes = 0
        self._fp = fp
        return self.strikes


class RunLog:
    """三层日志（L1 状态流 / L2 决策 / L3 事件+心跳）。"""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%m%d_%H%M%S")
        self.f_states = open(os.path.join(log_dir, f"{stamp}_states.jsonl"),
                             "a", encoding="utf-8")
        self.f_dec = open(os.path.join(log_dir, f"{stamp}_decisions.jsonl"),
                          "a", encoding="utf-8")
        self.f_evt = open(os.path.join(log_dir, f"{stamp}_events.jsonl"),
                          "a", encoding="utf-8")
        self.stamp = stamp
        self.n_states = 0
        self._stop = threading.Event()
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()

    def _write(self, f, rec: dict):
        try:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        except Exception:
            log.exception("写日志失败")

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


class SimBrain:
    """模拟器大脑：把真实游戏状态灌进 lightspeed 模拟器，用 RL Scorer 决策。"""

    def __init__(self, checkpoint_path=_DEFAULT_CKPT, obs_dim=OBS_DIM,
                 cand_dim=DESC_VEC_DIM, hidden=(512, 256),
                 combat_sims=2000, determinizations=10, log_dir=None):
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
        self._nni = sts.getNNInterface()  # 缓存单例（C++ 所有权 bug 已修复）
        self._error_count = 0  # 连续游戏错误计数（熔断用）
        self._last_cmd = ""    # 上一次发的命令（检测重复拒绝）
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

    def decide(self, state: dict) -> str:
        if state.get("error"):
            log.warning("game error: %s", state["error"])
            # 熔断：PLAY 被拒后游戏报错帧可能没有 play 命令，必须在分支前拦截
            if self._last_cmd.startswith("PLAY"):
                self._error_count += 1
                if self._error_count >= 3:
                    log.warning("PLAY 连续被拒 %d 次，强制 END 回合", self._error_count)
                    self._error_count = 0
                    self._last_cmd = "END"
                    return "END"
        else:
            self._error_count = 0

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

        # 战斗：ISMCTS 优先（灌状态进模拟器搜索最优出牌），失败回退规则
        if "play" in cmds:
            try:
                self.gc.init_from_json(json.dumps(_sanitize(state)))
                cmd = self.combat_brain.decide_with_fallback(
                    state, self.gc, combat_action)
                self._log_decision(state, screen="COMBAT", kind="COMBAT",
                                   cmd=cmd, n_cands=0, degraded=False)
            except Exception:
                log.exception("MCTS 战斗异常，回退规则")
                cmd = combat_action(state)
                self._log_decision(state, screen="COMBAT", kind="COMBAT",
                                   cmd=cmd, n_cands=0, degraded=True)
            self._last_cmd = cmd
            return cmd

        # 战斗奖励：模拟器没有真实奖励内容，RL 会错误跳过——直接走规则（逐个领取）
        gs = state.get("game_state", {})
        if (gs.get("screen_type") or "").upper() == "COMBAT_REWARD":
            cmd = macro_action(state)
            self._log_decision(state, screen="COMBAT_REWARD", kind="REWARD",
                               cmd=cmd, n_cands=0, degraded=True)
            return cmd

        # 游戏未开始（主菜单）
        if "start" in cmds:
            cmd = "START IRONCLAD"
            if self.log:
                self.log.event("run_start", command=cmd)
            return cmd

        # 宏观决策：模拟器 + RL；任何失败回退规则
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
        return cmd

    # ---- 模拟器决策路径 ----

    def _sim_decide(self, state: dict, cmds: list):
        """灌入模拟器 → RL 选动作 → 翻译成 comm 命令；不可翻译返回 None。"""
        if self.scorer is None:
            return None
        game = state.get("game_state", {})

        # 前置检查：关键字段为 null 时直接跳过（init_from_json 会崩）
        _REQUIRED_NUMERIC = ["seed", "floor", "act", "current_hp", "max_hp", "gold"]
        for field in _REQUIRED_NUMERIC:
            if game.get(field) is None:
                log.warning("game_state.%s 为 null，跳过模拟器决策", field)
                return None

        self.gc.init_from_json(json.dumps(_sanitize(state)))
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
        )


def _parse_hidden(s: str):
    return tuple(int(x) for x in s.split(",") if x.strip())


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
    guard = StallGuard()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            log.error("bad json: %s", line[:200])
            continue

        # L1：真实状态原样落盘（无损重放）
        if brain.log:
            brain.log.state(state)
        # 兼容旧习惯：把最近一次状态单独写盘
        try:
            with open("last_state.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

        try:
            cmd = brain.decide(state)
        except Exception as exc:
            log.exception("decide 顶层异常，发 STATE 保命")
            if brain.log:
                brain.log.event("fatal", error=f"{type(exc).__name__}: {exc}")
            cmd = "STATE"

        # 防卡死：状态指纹连续不变 → 强制换动作；仍未恢复 → 退出保命
        strikes = guard.observe(_fingerprint(state))
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

        log.debug("screen=%s -> %s",
                  state.get("game_state", {}).get("screen_type"), cmd)
        if cmd == "STATE":
            time.sleep(0.1)  # 避免在主菜单等无操作状态下疯狂自旋
        sys.stdout.write(cmd + "\n")
        sys.stdout.flush()

    if brain.log:
        brain.log.event("bridge_exit", decisions=brain.decisions,
                        fallbacks=brain.fallbacks,
                        stalls=guard.total_stalls)
        brain.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
