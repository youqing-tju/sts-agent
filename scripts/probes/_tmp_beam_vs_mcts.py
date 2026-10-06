# -*- coding: utf-8 -*-
"""Beam Search vs MCTS —— 同帧同起点，各自连续决策到战斗结束。

对比设计（关键：**控制起点**）：
    从 `logs/mcts_budget_probe.json` 的每个真机战斗帧出发，各自连续决策
    （Beam 取方案第一步 / MCTS 取 best_action），直到战斗结束。
    同一帧 → 两条路线 → 比「胜负 / 剩余 HP / 决策耗时」。

为什么要"打到结束"而不是"只比第一个动作"：
    第一个动作的好坏无法直接判定（同一动作在不同后续下结果不同）；
    只有走完整场才能拿到真实结果（胜/负/剩血）。

⚠️ 公平性说明：
    · MCTS 的动作空间包含 POTION / 选牌类动作，而 Beam P1 只枚举 CARD+END_TURN。
      若帧里出现这些特殊情况，双方不等价 → 记录在 `special` 字段里，不掩盖。
    · MCTS 用 8000 sims（≈1.2s/决策），Beam 用 W=8（≈0.5s/决策），时间量级接近。

环境变量：N_FRAMES / MCTS_SIMS / BEAM_W / BEAM_DEPTH / BEAM_MAXNODES / MAX_STEPS
"""
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _find_project_root(start):
    """从 start 向上逐级查找含 .git 的目录作为项目根。

    本脚本已移入 scripts/probes/，不能再假设 __file__ 所在即项目根；
    根目录下才有 logs/（含 realtime/、mcts_budget_probe.json）与 combat_beam 包。
    """
    d = os.path.dirname(os.path.abspath(start))
    while not os.path.isdir(os.path.join(d, ".git")):
        parent = os.path.dirname(d)
        if parent == d:            # 到盘根仍未找到，退回脚本所在目录
            return os.path.dirname(os.path.abspath(start))
        d = parent
    return d


_HERE = _find_project_root(__file__)   # 项目根（含 logs/）
sys.path.insert(0, _HERE)

import slaythespire as sts

from combat_beam import BeamSolver

N_FRAMES = int(os.environ.get("N_FRAMES", "5"))
MCTS_SIMS = int(os.environ.get("MCTS_SIMS", "8000"))
BEAM_W = int(os.environ.get("BEAM_W", "8"))
BEAM_DEPTH = int(os.environ.get("BEAM_DEPTH", "16"))
BEAM_MAXNODES = int(os.environ.get("BEAM_MAXNODES", "20000"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "200"))
OUT = os.path.join(_HERE, "logs", "beam_vs_mcts.json")

_AT = {
    "CARD": sts.ActionType.CARD, "POTION": sts.ActionType.POTION,
    "END_TURN": sts.ActionType.END_TURN,
    "SINGLE_CARD_SELECT": sts.ActionType.SINGLE_CARD_SELECT,
    "MULTI_CARD_SELECT": sts.ActionType.MULTI_CARD_SELECT,
}


def load_bc(frame: dict):
    """从 (file, frame_idx) 载入 BattleContext。"""
    lines = open(os.path.join(_HERE, "logs", "realtime", frame["file"]),
                 encoding="utf-8").read().splitlines()
    raw = json.loads(lines[frame["frame_idx"]])
    gs = raw["game_state"]
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(json.dumps(gs))
    bc = sts.BattleContext()
    bc.init_from_json(gc, json.dumps(gs))
    return bc


def _special(bc) -> str:
    """标记 Beam 无法正确处理的状态（选牌屏等）。InputState: 1 = PLAYER_NORMAL。"""
    try:
        return "" if int(bc.input_state) == 1 else "input_state=%d" % bc.input_state
    except Exception:
        return "input_state=?"


def play_out(bc0, decider, name: str):
    """从 bc0 出发用 decider 连续决策，返回结果统计。"""
    bc = sts.BattleContext(bc0)
    n_steps = 0
    specials: list[str] = []
    aborted = ""
    t0 = time.perf_counter()
    for _ in range(MAX_STEPS):
        if bc.is_terminal():
            break
        sp = _special(bc)
        if sp:
            specials.append(sp)
        try:
            act = decider(bc)
        except Exception as e:
            aborted = "%s: %s" % (type(e).__name__, e)
            break
        if act is None:
            aborted = "decider 无动作"
            break
        try:
            act.execute(bc)
        except Exception as e:
            aborted = "execute: %s" % type(e).__name__
            break
        n_steps += 1
    dt = time.perf_counter() - t0
    return {
        "won": bool(bc.is_terminal() and bc.outcome == sts.BattleOutcome.PLAYER_VICTORY),
        "terminal": bool(bc.is_terminal()),
        "hp": int(bc.player.hp),
        "max_hp": int(bc.player.max_hp),
        "turn": int(bc.turn),
        "steps": n_steps,
        "secs": round(dt, 2),
        "secs_per_decision": round(dt / max(1, n_steps) * 1000, 1),
        "special_states": len(specials),
        "aborted": aborted,
    }


def beam_decider(bc):
    s = BeamSolver(beam_width=BEAM_W, max_depth=BEAM_DEPTH, max_nodes=BEAM_MAXNODES)
    plan = s.solve(bc)
    if not plan:
        return None
    t, i, j = plan[0]
    return sts.BattleAction(_AT[t], i, j)


def mcts_decider(bc):
    s = sts.BattleSearcher(bc, MCTS_SIMS)
    return s.best_action


def main():
    frames = json.load(open(os.path.join(_HERE, "logs", "mcts_budget_probe.json"),
                             encoding="utf-8"))[:N_FRAMES]
    print("对比：%d 帧 | Beam(W=%d,depth=%d,max=%d) vs MCTS(sims=%d)"
          % (len(frames), BEAM_W, BEAM_DEPTH, BEAM_MAXNODES, MCTS_SIMS), flush=True)

    rows = []
    for k, fr in enumerate(frames):
        tag = "%s#%d" % (fr["file"][:13], fr["frame_idx"])
        try:
            bc0 = load_bc(fr)
        except Exception as e:
            print("[%d] %s 载入失败: %s" % (k, tag, e), flush=True)
            continue
        r_beam = play_out(bc0, beam_decider, "beam")
        r_mcts = play_out(bc0, mcts_decider, "mcts")
        rows.append({"frame": tag, "turn0": fr.get("turn"), "hp0": fr.get("hp"),
                     "beam": r_beam, "mcts": r_mcts})
        print("[%2d] %-20s | Beam %s hp=%-3d %5.1fs | MCTS %s hp=%-3d %5.1fs"
              % (k, tag,
                 "WIN " if r_beam["won"] else ("--  " if not r_beam["terminal"] else "LOSS"),
                 r_beam["hp"], r_beam["secs"],
                 "WIN " if r_mcts["won"] else ("--  " if not r_mcts["terminal"] else "LOSS"),
                 r_mcts["hp"], r_mcts["secs"]), flush=True)

    if not rows:
        print("无有效样本"); return
    n = len(rows)
    bw = sum(r["beam"]["won"] for r in rows)
    mw = sum(r["mcts"]["won"] for r in rows)
    bh = sum(r["beam"]["hp"] for r in rows) / n
    mh = sum(r["mcts"]["hp"] for r in rows) / n
    bs = sum(r["beam"]["secs"] for r in rows) / n
    ms = sum(r["mcts"]["secs"] for r in rows) / n
    print("\n=== 汇总（n=%d）===" % n)
    print("  Beam : 通关 %d/%d (%.0f%%) | 平均剩余 HP %.1f | 平均耗时 %.1fs" % (bw, n, bw/n*100, bh, bs))
    print("  MCTS : 通关 %d/%d (%.0f%%) | 平均剩余 HP %.1f | 平均耗时 %.1fs" % (mw, n, mw/n*100, mh, ms))
    print("  差   : 通关 %+d | HP %+.1f | 耗时 %+.1fs" % (bw-mw, bh-mh, bs-ms))

    json.dump({"config": {"n_frames": N_FRAMES, "mcts_sims": MCTS_SIMS, "beam_w": BEAM_W,
                          "beam_depth": BEAM_DEPTH, "beam_max_nodes": BEAM_MAXNODES},
               "summary": {"n": n, "beam_wins": bw, "mcts_wins": mw,
                           "beam_avg_hp": bh, "mcts_avg_hp": mh,
                           "beam_avg_secs": bs, "mcts_avg_secs": ms},
               "rows": rows},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\n已保存 %s" % OUT)


if __name__ == "__main__":
    main()
