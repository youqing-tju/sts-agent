# -*- coding: utf-8 -*-
"""宏观指标 JSON 的**配对比较**（同一批留出种子）。

背景：本项目评估是**确定性**的（同种子同配置两次运行结果逐项相同）→ 可以对每个种子
做配对比较（楼层用**配对 t**、通关率用 **McNemar**），灵敏度远高于两组独立比例检验。
100~200 局即可，不必被迫跑到 500 局（`logs/weight-tuning-report.md` §2）。

输出：每个变体相对 baseline 的
    Δfloor / 配对 t / 逐种子 delta 分布 / McNemar b,c / 粗略 p
    以及"通关局牌组张数"等宏观指标对比（若有）。

用法：
    .venv/Scripts/python.exe scripts/probes/_tmp_paired_compare.py \
        logs/macro-v14_defbeam_2k.json  logs/macro-v14-def-rollout-d6.json [更多...]
第一个文件为 baseline；可用 --n 限制用前 N 个种子（种子嵌套，故不同 n 也可配对）。
"""
import json
import math
import sys


def _load(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    return {r["seed"]: r for r in rows}


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mcnemar_exact_p(b, c):
    """McNemar 精确检验（双侧，二项分布）。b/c = 两个方向的不一致对数。"""
    n = b + c
    if n == 0:
        return 1.0
    # 双侧：2 * P(X <= min(b,c))，截断到 1
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def paired(base, cur, n=None):
    seeds = sorted(set(base) & set(cur))
    if n:
        seeds = seeds[:n]
    d = [cur[s]["final_floor"] - base[s]["final_floor"] for s in seeds]
    m = sum(d) / len(d)
    var = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    sd = math.sqrt(var)
    t = m / (sd / math.sqrt(len(d))) if sd else float("nan")
    p = 2 * (1 - _norm_cdf(abs(t))) if sd else 1.0
    won = [s for s in seeds if cur[s]["win"] and not base[s]["win"]]
    lost = [s for s in seeds if base[s]["win"] and not cur[s]["win"]]
    return {
        "n": len(seeds), "dfloor": m, "t": t, "p_t": p,
        "sd": sd, "max_up": max(d), "max_dn": min(d),
        "same": sum(1 for x in d if x == 0),
        "up": sum(1 for x in d if x > 0), "dn": sum(1 for x in d if x < 0),
        "b": len(won), "c": len(lost), "p_mcn": mcnemar_exact_p(len(won), len(lost)),
    }


def agg(rows, n=None):
    seeds = sorted(rows)[:n] if n else sorted(rows)
    fl = [rows[s]["final_floor"] for s in seeds]
    w = sum(1 for s in seeds if rows[s]["win"])
    return sum(fl) / len(fl), w / len(seeds), w, len(seeds)


def _deck(row, key="deck"):
    v = row.get(key)
    return len(v) if isinstance(v, list) else (v if isinstance(v, (int, float)) else None)


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else None


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    n = None
    for a in sys.argv[1:]:
        if a.startswith("--n="):
            n = int(a.split("=")[1])
    if len(args) < 2:
        print(__doc__)
        return

    base = _load(args[0])
    bf, bw, bc, bt = agg(base, n)
    print("baseline: %s" % args[0])
    print("  楼层 %.2f | 通关率 %.1f%% (%d/%d)%s"
          % (bf, 100 * bw, bc, bt, "" if not n else "   [前 %d 种子]" % n))

    print("\n%-34s %8s %7s %7s %9s %7s %14s %9s %s"
          % ("变体", "楼层", "Δfloor", "配对t", "p(t)", "Δwin", "McNemar b/c", "p(McN)", "delta 分布"))
    for path in args[1:]:
        cur = _load(path)
        cf, cw, cc, ct = agg(cur, n)
        r = paired(base, cur, n)
        print("%-34s %8.2f %+7.2f %+7.2f %9.4f %+6.1f%% %14s %9.4f  +%d/=%d/-%d (max %+d/%+d)"
              % (path.split("macro-")[-1][:33], cf, r["dfloor"], r["t"], r["p_t"],
                 100 * (cw - bw), "%d/%d" % (r["b"], r["c"]), r["p_mcn"],
                 r["up"], r["same"], r["dn"], r["max_up"], r["max_dn"]))

    # 宏观指标（通关局的牌组/遗物/Act3 血量 —— v14 那轮证明这些比楼层更能说明"赢法"）
    print("\n== 通关局宏观指标（v14 那轮的关键证据）==")
    print("  %-32s %7s %8s %8s %10s %9s" % ("配置", "通关局", "牌组", "遗物", "Act3血量", "金币"))
    for path in args:
        label = path.split("macro-")[-1][:32]
        rows = _load(path)
        # ⚠️ 先按「前 n 个种子」定口径，**再**筛通关局 —— 否则会把种子序列更靠后的
        #    通关局也算进来，与上面主表的口径不一致（首版就踩了这个坑）。
        allow = set(sorted(rows)[:n]) if n else set(rows)
        seeds = [s for s in sorted(rows) if s in allow and rows[s]["win"]]
        if not seeds:
            print("  %-32s %7d %8s %8s %10s %9s" % (label, 0, "-", "-", "-", "-"))
            continue
        dk = _mean([d for d in (_deck(rows[s]) for s in seeds) if d is not None])
        rl = _mean([rows[s].get("relics") for s in seeds if rows[s].get("relics") is not None])
        gl = _mean([rows[s].get("gold") for s in seeds if rows[s].get("gold") is not None])
        a3 = _mean([rows[s]["act_snap"]["3"]["hp"] for s in seeds
                    if isinstance(rows[s].get("act_snap"), dict) and "3" in rows[s]["act_snap"]])
        print("  %-32s %7d %8s %8s %10s %9s"
              % (label, len(seeds),
                 "-" if dk is None else "%.1f" % dk,
                 "-" if rl is None else "%.1f" % rl,
                 "-" if a3 is None else "%.1f" % a3,
                 "-" if gl is None else "%.1f" % gl))


if __name__ == "__main__":
    main()
