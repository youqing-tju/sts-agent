# -*- coding: utf-8 -*-
"""验证 Beam 求解器能否直接工作在**真实游戏状态**上（无游戏，用 L1 日志重放）。

回答的问题：**现在把 Beam 接进 `main.py`，它在真机状态下能跑起来吗？**

关键差异（真机 vs 训练）：
- 真机：`bc.init_from_json(gc, json.dumps(sanitize(state)))` + `alias_power_ids`（见 `combat_brain.py:222-227`）
- 训练：`bc.init_from_gc(gc)`（见 `env/lightspeed_env.py`）
→ 本探针按**真机路径**构造 bc，再用 `BeamSolver` 求解，统计成功率与耗时。

用法：.venv/Scripts/python.exe scripts/probes/_tmp_beam_realtime_probe.py
"""
import glob
import json
import os
import statistics as st
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts  # noqa: E402

from combat_beam import BeamSolver, PRESETS, Weights, legal_actions  # noqa: E402
from combat_brain import _sanitize  # noqa: E402（真机同源预处理）
from models.power_alias import alias_power_ids  # noqa: E402


def collect_combat_states(limit=150):
    """从 L1 日志里挑出「真实战斗帧」（game_state.combat_state.monsters 非空）。"""
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "logs", "realtime", "*_states.jsonl"))):
        try:
            fh = open(f, encoding="utf-8")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    s = json.loads(line)
                except Exception:                     # noqa: BLE001
                    continue
                cs = ((s.get("game_state") or {}).get("combat_state")) or {}
                if isinstance(cs, dict) and cs.get("monsters"):
                    out.append(s)
                    if len(out) >= limit:
                        return out
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=150, help="重放多少真实战斗帧")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--max-nodes", type=int, default=300)
    ap.add_argument("--value-mode", choices=["static", "rollout"], default="static")
    a = ap.parse_args()

    states = collect_combat_states(a.frames)
    print("提取到真实战斗帧: %d" % len(states))
    if not states:
        print("（没找到战斗帧，无法验证）")
        return

    w = Weights.from_overrides(PRESETS["defensive"])
    solver = BeamSolver(beam_width=a.width, max_depth=a.depth, max_nodes=a.max_nodes,
                        weights=w, value_mode=a.value_mode)
    print("配置: W=%d d=%d n=%d value_mode=%s" % (a.width, a.depth, a.max_nodes, a.value_mode))

    ok = 0
    fail_gc = fail_bc = fail_solve = 0
    times, firsts, n_legal = [], Counter(), []
    samples = []

    for i, s in enumerate(states):
        try:
            gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
            gc.init_from_json(json.dumps(s))
        except Exception as e:                        # noqa: BLE001
            fail_gc += 1
            if fail_gc <= 2:
                print("  [gc 失败] %s: %s" % (type(e).__name__, e))
            continue
        try:
            clean = _sanitize(s)
            alias_power_ids(clean)
            bc = sts.BattleContext()
            bc.init_from_json(gc, json.dumps(clean))
        except Exception as e:                        # noqa: BLE001
            fail_bc += 1
            if fail_bc <= 2:
                print("  [bc 失败] %s: %s" % (type(e).__name__, e))
            continue
        if bc.is_terminal():
            continue
        try:
            n_legal.append(len(legal_actions(bc)))
            t0 = time.perf_counter()
            plan = solver.solve(bc)
            times.append((time.perf_counter() - t0) * 1000.0)
        except Exception as e:                        # noqa: BLE001
            fail_solve += 1
            if fail_solve <= 2:
                print("  [solve 失败] %s: %s" % (type(e).__name__, e))
            continue
        ok += 1
        firsts[plan[0][0] if plan else "EMPTY"] += 1
        if len(samples) < 3:
            samples.append((plan[0] if plan else None, solver.last_stats))

    print("\n== 结果 ==")
    print("成功 %d / 尝试 %d   失败(gc=%d bc=%d solve=%d)"
          % (ok, len(states), fail_gc, fail_bc, fail_solve))
    if times:
        times_sorted = sorted(times)
        print("单步耗时: 均值 %.2f ms  中位 %.2f ms  P95 %.2f ms  最大 %.2f ms"
              % (st.mean(times), st.median(times),
                 times_sorted[int(len(times_sorted) * 0.95) - 1], max(times)))
        print("  （真机限速 --min-cmd-interval-ms 默认 300 ms → 余量充足）")
    if n_legal:
        print("合法动作数/帧: 均值 %.1f  最大 %d" % (st.mean(n_legal), max(n_legal)))
    print("首动作类型分布:", dict(firsts))
    print("\n== 样例 ==")
    for act, stats in samples:
        print("  首动作=%s  nodes=%s  plan_len=%s  won=%s"
              % (act, stats.get("nodes"), stats.get("plan_len"), stats.get("won")))


if __name__ == "__main__":
    main()
