# -*- coding: utf-8 -*-
"""给 Beam 加「终局信息」的两条路：成本 + 信息量探针（无游戏，用 L1 真实战斗帧）。

背景（`logs/sts1-solver-comparison-report.md` 纠错清单 1）：
    Beam 的 `evaluate()` 打分的是**静态中间状态**，没有终局信息 —— 这是「更宽/更深/
    更大预算全部变差」的根本解释。MCTS 那条路有 `evaluateEndState`（终局），但 Beam 没沾到光。

本探针量化两条可选实现路径（都不需要重编译 pyd）：

  A. **Python rollout**：从叶节点用一个贪心策略打到终局，再用终局评估函数打分。
      成本 = 每步 `legal_actions` + `execute`；信息量 = 终局胜负 + 剩余血量。
  B. **C++ MCTS 当价值函数**：`sts.BattleSearcher(bc, sims)` 在低 sims 下跑，
      它内部会把 rollout 打到终局并用 `evaluateEndState` 打分 —— 直接复用已验证的终局评估。
      成本 = sims 的线性函数；信息量 = `root_action_stats` 的 avg_eval。

用法：
    .venv/Scripts/python.exe scripts/probes/_tmp_terminal_value_cost.py [帧数]
"""
import glob
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts  # noqa: E402

from combat_beam import PRESETS, Weights, evaluate, legal_actions, _make  # noqa: E402
from combat_brain import _sanitize                    # noqa: E402
from models.power_alias import alias_power_ids        # noqa: E402


def collect_combat_states(limit=40):
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


def build_bc(s):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(json.dumps(s))
    clean = _sanitize(s)
    alias_power_ids(clean)
    bc = sts.BattleContext()
    bc.init_from_json(gc, json.dumps(clean))
    return bc


def greedy_rollout(bc, w, max_steps=400):
    """从 bc 出发用「贪心 by evaluate」打到终局，返回 (outcome, steps, ms)。"""
    t0 = time.perf_counter()
    cur = sts.BattleContext(bc)
    steps = 0
    while not cur.is_terminal() and steps < max_steps:
        acts = legal_actions(cur)
        if not acts:
            break
        # 贪心：选 evaluate 最高的动作（**只做一步前瞻**，不含 rollout —— 这就是"策略"）
        best, best_s = None, None
        for a in acts:
            nb = sts.BattleContext(cur)
            try:
                _make(a).execute(nb)
            except Exception:                          # noqa: BLE001
                continue
            sc = evaluate(nb, w)
            if best_s is None or sc > best_s:
                best, best_s = a, sc
        if best is None:
            break
        _make(best).execute(cur)
        steps += 1
    return str(cur.outcome), steps, (time.perf_counter() - t0) * 1000.0


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    states = collect_combat_states(n)
    print("真实战斗帧: %d" % len(states))
    if not states:
        return

    w = Weights.from_overrides(PRESETS["defensive"])
    bcs = []
    for s in states:
        try:
            bc = build_bc(s)
        except Exception:                              # noqa: BLE001
            continue
        if not bc.is_terminal():
            bcs.append(bc)
    print("可用（非终局）: %d\n" % len(bcs))

    # ---------- 原子操作成本 ----------
    bc0 = bcs[0]
    t0 = time.perf_counter()
    for _ in range(2000):
        sts.BattleContext(bc0)
    fork_us = (time.perf_counter() - t0) / 2000 * 1e6
    t0 = time.perf_counter()
    for _ in range(500):
        legal_actions(bc0)
    la_us = (time.perf_counter() - t0) / 500 * 1e6
    acts0 = legal_actions(bc0)
    t0 = time.perf_counter()
    for _ in range(2000):
        nb = sts.BattleContext(bc0)
        _make(acts0[0]).execute(nb)
    exec_us = (time.perf_counter() - t0) / 2000 * 1e6
    print("== 原子操作（单次）==")
    print("  BattleContext 拷贝   %8.1f us" % fork_us)
    print("  legal_actions        %8.1f us" % la_us)
    print("  拷贝+execute 一步     %8.1f us" % exec_us)

    # ---------- A: Python 贪心 rollout ----------
    outs, steps_l, times_l = [], [], []
    for bc in bcs:
        o, s, ms = greedy_rollout(bc, w)
        outs.append(o)
        steps_l.append(s)
        times_l.append(ms)
    print("\n== A. Python 贪心 rollout（到终局）==")
    print("  耗时/次: 均值 %.2f ms  中位 %.2f ms  P95 %.2f ms  最大 %.2f ms"
          % (st.mean(times_l), st.median(times_l),
             sorted(times_l)[int(len(times_l) * 0.95) - 1], max(times_l)))
    print("  步数/次: 均值 %.1f  最大 %d" % (st.mean(steps_l), max(steps_l)))
    from collections import Counter
    print("  终局分布:", dict(Counter(outs)))

    # ---------- B: C++ MCTS 当价值函数 ----------
    print("\n== B. C++ BattleSearcher（低 sims）==")
    print("  %6s %12s %12s %14s" % ("sims", "均值ms", "P95ms", "有best?"))
    for sims in (1, 4, 10, 30, 100):
        ts, okc = [], 0
        for bc in bcs[:20]:
            t0 = time.perf_counter()
            try:
                srch = sts.BattleSearcher(bc, sims)
                _ = srch.best_action
                okc += 1
            except Exception:                          # noqa: BLE001
                pass
            ts.append((time.perf_counter() - t0) * 1000.0)
        p95 = sorted(ts)[int(len(ts) * 0.95) - 1]
        print("  %6d %12.2f %12.2f %8d/%d" % (sims, st.mean(ts), p95, okc, len(bcs[:20])))


if __name__ == "__main__":
    main()
