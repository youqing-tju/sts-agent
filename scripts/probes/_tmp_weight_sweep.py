# -*- coding: utf-8 -*-
"""启发式权重扫描（P0，`logs/beam-advanced-mechanisms.md` §5 的下一步）。

**为什么扫描是可行的**：本项目的评估是**确定性**的 —— 同种子 + 同配置两次运行，
`avg_floor / std_floor / win_rate` **逐项完全相同**（2026-10-15 实测：nodes=300 vs 1500）。
因此可以对每个种子做**配对比较**（配对 t 检验 + McNemar），
比"两组独立比例检验"灵敏得多，100 局就足够。

用法（务必在项目根运行）：
  .venv/Scripts/python.exe scripts/probes/_tmp_weight_sweep.py --seeds 100 --workers 6
  .venv/Scripts/python.exe scripts/probes/_tmp_weight_sweep.py --only intent
"""
import argparse
import json
import os
import sys
import time
from statistics import mean

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.evaluate import GreedyScorerPolicy, evaluate  # noqa: E402

#: (名称, 权重覆盖)。每组只动**一个**维度 → 可归因。
CONFIGS = [
    ("baseline",        {}),
    ("intent_x0.5",     {"intent": 3_000}),
    ("intent_x2",       {"intent": 12_000}),
    ("intent_0",        {"intent": 0}),
    ("enemy_hp_x0.5",   {"enemy_hp": 6_000}),
    ("enemy_hp_x2",     {"enemy_hp": 24_000}),
    ("block_x0.5",      {"block": 9_000}),
    ("block_x2",        {"block": 36_000}),
    ("hp_x0.5",         {"hp": 15_000}),
    ("energy_0",        {"energy": 0}),
    ("energy_x3",       {"energy": 27_000}),
    ("hand_x4",         {"hand": 10_000}),
    ("strength_x2",     {"strength": 30_000}),
    ("dexterity_x2",    {"dexterity": 24_000}),
    # ---- 组合（基于单维结果的"各自最优方向"；单维扫描的盲区）----
    ("combo_A",         {"enemy_hp": 6_000, "intent": 12_000}),
    ("combo_B",         {"enemy_hp": 4_000, "intent": 16_000, "block": 24_000}),
    ("combo_C",         {"enemy_hp": 3_000, "intent": 20_000}),
]


def paired_stats(cur, base):
    """返回 (Δ均值, 配对 t, McNemar b/c)。cur/base 为 per-seed 列表。"""
    if isinstance(cur[0], bool) or isinstance(base[0], bool):
        raise TypeError("数值列表请用 floors，布尔列表请用 wins")
    d = [x - y for x, y in zip(cur, base)]
    n = len(d)
    m = mean(d)
    if n < 2:
        return m, 0.0
    sd = (sum((x - m) ** 2 for x in d) / (n - 1)) ** 0.5
    t = (m / (sd / n ** 0.5)) if sd > 1e-12 else (float("inf") if abs(m) > 1e-12 else 0.0)
    return m, t


def mcnemar(cur_w, base_w):
    """(b, c)：b = 变体赢且基线输；c = 基线赢且变体输。"""
    b = sum(1 for x, y in zip(cur_w, base_w) if x and not y)
    c = sum(1 for x, y in zip(cur_w, base_w) if (not x) and y)
    return b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/v12_beam_500.pt")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mcts-sim", type=int, default=20000)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--beam-depth", type=int, default=6)
    ap.add_argument("--beam-max-nodes", type=int, default=300)
    ap.add_argument("--only", default=None, help="只跑名称含该子串的配置")
    ap.add_argument("--out", default=None, help="结果 JSON 输出路径")
    a = ap.parse_args()

    policy = GreedyScorerPolicy(a.checkpoint)
    base_env = {"combat_solver": "beam", "beam_width": a.beam_width,
                "beam_depth": a.beam_depth, "beam_max_nodes": a.beam_max_nodes}

    # ⚠️ baseline **始终跑** —— 配对比较必须有基准（否则 --only 出来的表全是空的）
    cfgs = [(n, o) for n, o in CONFIGS
            if n == "baseline" or not a.only or a.only in n]
    print("扫描 %d 组 | seeds=%d workers=%d | W=%d depth=%d nodes=%d\n"
          % (len(cfgs), a.seeds, a.workers, a.beam_width, a.beam_depth, a.beam_max_nodes))

    rows = {}
    for name, ov in cfgs:
        t0 = time.time()
        kw = dict(base_env)
        kw["weights"] = ov or None
        res = evaluate(policy, n_seeds=a.seeds, workers=a.workers,
                       mcts_sim=a.mcts_sim, env_kwargs=kw)
        floors = [r["final_floor"] for r in res]
        wins = [bool(r["win"]) for r in res]
        rows[name] = {"floors": floors, "wins": wins,
                      "avg_floor": mean(floors),
                      "win_rate": sum(wins) / len(wins),
                      "sec": round(time.time() - t0, 1)}
        print("%-16s floor=%5.2f  win=%5.1f%%  (%4.0fs)"
              % (name, rows[name]["avg_floor"], 100 * rows[name]["win_rate"],
                 rows[name]["sec"]), flush=True)

    if "baseline" in rows and len(rows) > 1:
        b = rows["baseline"]
        print("\n== 配对比较（vs baseline，n=%d 同种子）==" % a.seeds)
        print("%-16s %9s %9s %9s %9s" % ("config", "Δfloor", "Δwin", "配对t", "McNem b/c"))
        for name, r in rows.items():
            if name == "baseline":
                continue
            d, t = paired_stats(r["floors"], b["floors"])
            bb, cc = mcnemar(r["wins"], b["wins"])
            print("%-16s %+9.2f %+8.1f%% %9.2f %6d/%-3d"
                  % (name, d, 100 * (r["win_rate"] - b["win_rate"]), t, bb, cc))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("\n已写入 %s" % a.out)


if __name__ == "__main__":
    main()
