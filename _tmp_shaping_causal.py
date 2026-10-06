# -*- coding: utf-8 -*-
"""V11 奖励塑形因果诊断：E1（无塑形）vs V11（塑形）的牌组指标分层对比。

## 为什么要做这个诊断

V11 的奖励塑形（`G += 1.0×升级数 + 1.0×遗物数`）在 100 局配对评估中被判**无效**
（楼层差 -0.42，SE 1.41，t=-0.30；通关率 -2.0pp）。而它的动机来自一份后视观察
（`logs/deck-strength-report.md`）：通关组 vs 死 Act3 组**进入 Act3 时**的
升级卡数 6.07 vs 3.63（p=0.030）、遗物数 10.82 vs 9.58（p=0.038）。

本诊断区分两种解释：
  **A 信号没被学到**：V11 的升级数/遗物数 ≈ E1
     → bonus 太小（+2.4 分 vs 楼层 0-57 + 通关 50），或被 PPO 的 advantage
       标准化吸收（baseline V 学到该偏置后 advantage 归零）。
  **B 伪相关**：V11 的升级数/遗物数 > E1，但通关率没变
     → "多升级"是"活得久"的**结果**而非原因 → 必须换方向，别再调奖励。

## 关键方法：比较「进入 Act3 时」而非终局

终局指标会被生存时间污染（死得早 → 牌组自然更弱）。因此在每局**首次进入 Act3**
的那一刻采样牌组指标。这样"通关组 vs Act3 死亡组"的比较都建立在
「已经活到 Act3」的同等前提上 —— 与原始诊断口径一致。

## 口径

- 两模型各 N 局，**同种子同顺序**（`make_eval_seeds(N)`）→ 可逐局配对
- 8 workers 并行，一进程一局（`max_tasks_per_child=1`，同 evaluation/evaluate.py）
- greedy 策略，MCTS 20000（与 100 局评估口径一致）
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np

from evaluation.evaluate import GreedyScorerPolicy, make_eval_seeds

MODELS = [
    ("checkpoints/v10_scratch.pt", "E1"),    # 无塑形基线
    ("checkpoints/v11_shaped.pt", "V11"),    # 有塑形
]
N = int(os.environ.get("N_SEEDS", "60"))
MCTS = int(os.environ.get("MCTS", "20000"))
WORKERS = int(os.environ.get("WORKERS", "8"))
OUT = os.path.join(_HERE, "logs", "shaping_causal.json")

_ACT3_MIN_FLOOR = 34   # Act3 死亡局的楼层下界（沿用 logs/deck-strength-report.md 口径）
_ACT3_MAX_FLOOR = 50   # Act3 通关是 floor 51+；<=50 且未通关 = Act3 内死亡


def _snap(gc):
    """采样当前牌组指标（字段名已实测：upgrade_count / upgraded；relics 为列表）。"""
    deck = list(gc.deck)
    up = sum(1 for c in deck
             if (getattr(c, "upgrade_count", 0) or 0) > 0 or getattr(c, "upgraded", False))
    return {"n_cards": len(deck), "n_upgraded": up, "n_relics": len(list(gc.relics))}


def run_one_ext(args):
    """顶层 worker（可 pickle）：跑一局，记录「进 Act3 时」与「终局」两组指标。"""
    seed, policy, mcts_sim = args
    from env.lightspeed_env import LightspeedEnv

    env = LightspeedEnv(mcts_simulations=mcts_sim)
    obs = env.reset(seed)
    act3_snap, prev_act = None, 1
    done, steps, res = False, 0, None
    while not done and steps < 500:
        gc = env.gc
        try:
            act = int(gc.act)
        except Exception:
            act = prev_act
        if act >= 3 and prev_act < 3 and act3_snap is None:
            act3_snap = _snap(gc)
        prev_act = act
        cands = env.candidates()
        idx = policy.choose(obs, cands) if cands else -1
        res = env.step(idx)
        obs, done = res.obs, res.done
        steps += 1

    outcome = res.info.get("outcome", "") if res else ""
    return {"seed": seed, "floor": int(obs.floor), "win": outcome == "PLAYER_VICTORY",
            "act3": act3_snap, "final": _snap(env.gc)}


def _subset(rows, kind):
    """分层：通关 / Act3 死亡 / 前期死亡。"""
    if kind == "win":
        return [r for r in rows if r["win"]]
    if kind == "act3dead":
        return [r for r in rows if not r["win"]
                and _ACT3_MIN_FLOOR <= r["floor"] <= _ACT3_MAX_FLOOR]
    if kind == "earlydead":
        return [r for r in rows if not r["win"] and r["floor"] < _ACT3_MIN_FLOOR]
    return rows


def _mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def main():
    seeds = make_eval_seeds(N)
    print(f"因果诊断：{len(seeds)} 种子 × {len(MODELS)} 模型 | MCTS {MCTS} | "
          f"{WORKERS} workers | greedy", flush=True)

    data = {}
    for path, name in MODELS:
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在", flush=True)
            continue
        t0 = time.time()
        pol = GreedyScorerPolicy(path)
        args = [(s, pol, MCTS) for s in seeds]
        with ProcessPoolExecutor(max_workers=WORKERS, max_tasks_per_child=1) as pool:
            rows = list(pool.map(run_one_ext, args))
        dt = time.time() - t0
        data[name] = rows
        fl = [r["floor"] for r in rows]
        wn = [1 if r["win"] else 0 for r in rows]
        print(f"[{name}] 楼层 {np.mean(fl):.2f} | 通关率 {np.mean(wn)*100:.1f}% | "
              f"{dt:.0f}s", flush=True)

    names = list(data.keys())
    if len(names) < 2:
        print("模型不足，退出"); return

    a, b = names[0], names[1]
    lines = []
    lines.append("| 分层 | 指标 | %s | %s | 差(B-A) |" % (a, b))
    lines.append("|---|---|---|---|---|")
    for kind, kname in [("win", "通关局"), ("act3dead", "Act3死亡局"), ("earlydead", "前期死亡局")]:
        ra, rb = _subset(data[a], kind), _subset(data[b], kind)
        if min(len(ra), len(rb)) < 8:
            lines.append(f"| {kname} | (n={len(ra)}/{len(rb)}) | 样本不足 | 样本不足 | — |")
            continue
        for field, fname in [("n_upgraded", "进Act3时升级数"), ("n_relics", "进Act3时遗物数"),
                             ("n_cards", "进Act3时牌组张数")]:
            va = [r["act3"][field] for r in ra if r["act3"]]
            vb = [r["act3"][field] for r in rb if r["act3"]]
            if not va or not vb:
                continue
            lines.append(f"| {kname} | {fname} | {_mean(va):.2f} (n={len(va)}) | "
                         f"{_mean(vb):.2f} (n={len(vb)}) | {_mean(vb)-_mean(va):+.2f} |")
        lines.append(f"| {kname} | 楼层 | {_mean([r['floor'] for r in ra]):.1f} | "
                     f"{_mean([r['floor'] for r in rb]):.1f} | "
                     f"{_mean([r['floor'] for r in rb])-_mean([r['floor'] for r in ra]):+.1f} |")

    # 逐局配对（同种子）
    d_fl = np.array([r["floor"] for r in data[b]], float) - np.array([r["floor"] for r in data[a]], float)
    se = d_fl.std(ddof=1) / np.sqrt(len(d_fl))
    pair = (f"楼层配对差 {d_fl.mean():+.2f} (SE {se:.2f}, t={d_fl.mean()/se:+.2f}) | "
            f"胜 {int((d_fl>0).sum())} / 负 {int((d_fl<0).sum())}")

    print("\n=== 分层对比（进 Act3 时）===")
    for ln in lines:
        print(ln)
    print("\n" + pair)

    os.makedirs(os.path.join(_HERE, "logs"), exist_ok=True)
    json.dump({"n_seeds": N, "mcts_sim": MCTS, "pair": pair, "table": lines, "data": data},
              open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n已保存 {OUT}")


if __name__ == "__main__":
    main()
