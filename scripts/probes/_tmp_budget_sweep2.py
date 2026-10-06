# -*- coding: utf-8 -*-
"""Part 2（append-only 版）：避免反复删除重写触发安全钩子。
每局追加一行 JSON 到 logs/mcts_budget_part2.jsonl（从不删除/改名），可断点续跑。
档位 [2000, 8000, 20000, 50000] × 20 种子 (3000..3019)。
V9 GreedyScorerPolicy + LightspeedEnv（复用 evaluation.evaluate.run_one）。
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from evaluation.evaluate import run_one, GreedyScorerPolicy

CKPT = os.path.join(ROOT, "checkpoints", "ppo_v9_10000.pt")
JSONL = os.path.join(ROOT, "logs", "mcts_budget_part2.jsonl")
TUNING_JSON = os.path.join(ROOT, "logs", "mcts_budget_tuning.json")
BUDGETS = [2000, 8000, 20000, 50000]
SEEDS = list(range(3000, 3020))


def load_done():
    done = {}  # (budget, seed) -> rec
    if os.path.exists(JSONL):
        with open(JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                done[(r["budget"], r["seed"])] = r
    return done


def migrate_existing(done):
    """把当前 tuning json 里已有的 part2（budget 2000=20 局）补进 jsonl。"""
    if not os.path.exists(TUNING_JSON):
        return 0
    try:
        blob = json.load(open(TUNING_JSON, encoding="utf-8"))
    except Exception:
        return 0
    p2 = blob.get("part2", {})
    cnt = 0
    for bstr, seedmap in p2.items():
        b = int(bstr)
        for sstr, rec in seedmap.items():
            s = int(sstr)
            if (b, s) in done:
                continue
            line = {"budget": b, "seed": s,
                    "final_floor": rec.get("final_floor"),
                    "win": rec.get("win"),
                    "seconds": rec.get("seconds"),
                    "migrated": True}
            if "error" in rec:
                line["error"] = rec["error"]
            with open(JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            done[(b, s)] = line
            cnt += 1
    return cnt


def main():
    import logging
    logging.disable(logging.CRITICAL)
    policy = GreedyScorerPolicy(CKPT)
    print(f"[init] V9 policy loaded: {CKPT}", flush=True)

    done = load_done()
    migrated = migrate_existing(done)
    if migrated:
        print(f"[migrate] 从 tuning json 补回 {migrated} 局", flush=True)

    total = len(BUDGETS) * len(SEEDS)
    print(f"[info] 已完成 {len(done)}/{total}", flush=True)

    t_start = time.time()
    for b in BUDGETS:
        for s in SEEDS:
            if (b, s) in done:
                continue
            t0 = time.time()
            try:
                r = run_one(s, policy, mcts_sim=b)
                dt = time.time() - t0
                rec = {"budget": b, "seed": s,
                       "final_floor": r["final_floor"], "win": bool(r["win"]),
                       "seconds": round(dt, 2)}
            except Exception as e:
                dt = time.time() - t0
                rec = {"budget": b, "seed": s,
                       "error": str(e)[:300], "seconds": round(dt, 2)}
            # 追加写，绝不删除/改名
            with open(JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done[(b, s)] = rec
            elapsed = time.time() - t_start
            rate = len(done) / elapsed if elapsed > 0 else 0
            eta = (total - len(done)) / rate if rate > 0 else 0
            print(f"[done] budget={b} seed={s} {rec} "
                  f"progress={len(done)}/{total} eta={eta/60:.1f}min", flush=True)

    print("[complete] Part2 jsonl 写入 " + JSONL, flush=True)


if __name__ == "__main__":
    main()
