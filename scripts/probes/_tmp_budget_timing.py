# -*- coding: utf-8 -*-
"""Part 2 — 测速 + 冒烟：每档跑 1 局（seed=3000..3003），记录每局耗时与结果。
确认 V9 GreedyScorerPolicy 在 LightspeedEnv 下能跑通完整对局，并估算总时长。
单进程串行。落盘日志到 stdout + logs/_tmp_timing.log。
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from evaluation.evaluate import run_one
from evaluation.evaluate import GreedyScorerPolicy

CKPT = os.path.join(ROOT, "checkpoints", "ppo_v9_10000.pt")
BUDGETS = [2000, 8000, 20000, 50000]
SEEDS = [3000, 3001, 3002, 3003]


def main():
    policy = GreedyScorerPolicy(CKPT)  # 默认 obs/cand/hidden 与 V9 匹配
    print(f"[init] policy loaded from {CKPT}", flush=True)
    results = {}
    # 每档一个 seed
    for i, b in enumerate(BUDGETS):
        s = SEEDS[i]
        t0 = time.time()
        try:
            r = run_one(s, policy, mcts_sim=b)
        except Exception as e:
            print(f"[FAIL] budget={b} seed={s}: {e!r}", flush=True)
            results[b] = {"seed": s, "error": str(e), "seconds": round(time.time()-t0,1)}
            continue
        dt = time.time() - t0
        results[b] = {"seed": s, "final_floor": r["final_floor"],
                      "win": r["win"], "seconds": round(dt, 1)}
        print(f"[timing] budget={b} seed={s} floor={r['final_floor']} "
              f"win={r['win']} time={dt:.1f}s", flush=True)

    # 估算
    est = {}
    for b in BUDGETS:
        rec = results.get(b)
        if rec and "seconds" in rec and "error" not in rec:
            est[b] = rec["seconds"]
    print("=== per-game timing (s) ===")
    for b in BUDGETS:
        print(f"  {b}: {est.get(b, 'NA')}s")
    if est:
        fastest, slowest = min(est.values()), max(est.values())
        print(f"[estimate] 20 seeds × 4 budgets: min≈{20*4*fastest/60:.1f}min "
              f"max≈{20*4*slowest/60:.1f}min")
    with open(os.path.join(ROOT, "logs", "_tmp_timing.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
