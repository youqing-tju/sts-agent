# -*- coding: utf-8 -*-
"""Part 2：用 V9 GreedyScorerPolicy + LightspeedEnv 跑完整对局，比较战斗预算。
档位 [2000, 8000, 20000, 50000] × 20 种子 (3000..3019)。串行、增量落盘。
复用 evaluation.evaluate.run_one / GreedyScorerPolicy（不重造）。
落盘 logs/mcts_budget_tuning.json 的 part2 段。
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from evaluation.evaluate import run_one, GreedyScorerPolicy

CKPT = os.path.join(ROOT, "checkpoints", "ppo_v9_10000.pt")
OUT_JSON = os.path.join(ROOT, "logs", "mcts_budget_tuning.json")
BUDGETS = [2000, 8000, 20000, 50000]
SEEDS = list(range(3000, 3020))  # 3000..3019


def load_blob():
    if os.path.exists(OUT_JSON):
        try:
            return json.load(open(OUT_JSON, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_blob(blob):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    if os.path.exists(OUT_JSON):
        os.remove(OUT_JSON)
    os.rename(tmp, OUT_JSON)


def main():
    import logging
    logging.disable(logging.CRITICAL)
    policy = GreedyScorerPolicy(CKPT)
    print(f"[init] V9 policy loaded: {CKPT}", flush=True)

    blob = load_blob()
    part2 = blob.get("part2", {})
    # part2 keyed by budget -> {seed: {final_floor, win, seconds}}
    for b in BUDGETS:
        part2.setdefault(str(b), {})

    total_planned = len(BUDGETS) * len(SEEDS)
    done = sum(len(v) for v in part2.values())
    print(f"[info] 已完成 {done}/{total_planned}", flush=True)

    t_start = time.time()
    for b in BUDGETS:
        for s in SEEDS:
            if str(s) in part2[str(b)]:
                continue
            t0 = time.time()
            try:
                r = run_one(s, policy, mcts_sim=b)
                dt = time.time() - t0
                rec = {"final_floor": r["final_floor"], "win": bool(r["win"]),
                       "seconds": round(dt, 2)}
            except Exception as e:
                dt = time.time() - t0
                rec = {"error": str(e)[:300], "seconds": round(dt, 2)}
            part2[str(b)][str(s)] = rec
            done += 1
            save_blob({**blob, "part2": part2})
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total_planned - done) / rate if rate > 0 else 0
            print(f"[done] budget={b} seed={s} {rec} "
                  f"progress={done}/{total_planned} eta={eta/60:.1f}min", flush=True)

    # 汇总
    summary = {}
    for b in BUDGETS:
        recs = list(part2[str(b)].values())
        floors = [r["final_floor"] for r in recs if "final_floor" in r]
        wins = [r["win"] for r in recs if "win" in r]
        secs = [r["seconds"] for r in recs if "seconds" in r]
        import statistics
        n = len(floors)
        summary[str(b)] = {
            "n": n,
            "avg_floor": round(statistics.fmean(floors), 2) if floors else None,
            "std_floor": round(statistics.pstdev(floors), 2) if len(floors) > 1 else 0.0,
            "win_rate": round(statistics.fmean(wins), 3) if wins else None,
            "total_seconds": round(sum(secs), 1) if secs else None,
            "games_per_sec": round(n / sum(secs), 3) if secs else None,
            "avg_seconds_per_game": round(statistics.fmean(secs), 2) if secs else None,
        }
    blob = load_blob()
    blob["part2"] = part2
    blob["part2_summary"] = summary
    save_blob(blob)
    print("=== Part2 summary ===")
    for b in BUDGETS:
        print(f"  {b}: {summary[str(b)]}", flush=True)
    print("[complete] Part2 写入 " + OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
