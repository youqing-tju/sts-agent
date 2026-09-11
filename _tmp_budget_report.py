# -*- coding: utf-8 -*-
"""汇总 Part1+Part2，生成 logs/mcts-budget-tuning.md + 写回 mcts_budget_tuning.json。
读源：mcts_budget_probe.json（2000/8000/50000 原始）、mcts_budget_tuning.json（part1）、
mcts_budget_part2.jsonl（Part2 每局）。最终一次 open('w') 落盘（不删除改名）。
"""
import os, sys, json, statistics
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

PROBE = os.path.join(ROOT, "logs", "mcts_budget_probe.json")
TUNING = os.path.join(ROOT, "logs", "mcts_budget_tuning.json")
PART2 = os.path.join(ROOT, "logs", "mcts_budget_part2.jsonl")
REPORT = os.path.join(ROOT, "logs", "mcts-budget-tuning.md")
BUDGETS = [2000, 8000, 10000, 20000, 50000]
PART2_BUDGETS = [2000, 8000, 20000, 50000]


def main():
    # ---------- Part1 ----------
    probe = json.load(open(PROBE, encoding="utf-8"))
    blob = json.load(open(TUNING, encoding="utf-8"))
    p1 = blob.get("part1_frames", {})

    # 拼每帧各档命令 + ms
    fc = {}
    for r in probe:
        k = f"{r['file']}#{r['frame_idx']}"
        fc[k] = {str(x): v for x, v in r.get('cmd_by_sims', {}).items()}
        fc[k]['ms'] = {str(x): v for x, v in r.get('ms', {}).items()}
    for k, rec in p1.items():
        fc.setdefault(k, {})
        fc[k]['10000'] = rec.get('cmds', {}).get('10000')
        fc[k]['20000'] = rec.get('cmds', {}).get('20000')
        fc[k].setdefault('ms', {})
        fc[k]['ms'].update({str(x): v for x, v in rec.get('ms', {}).items()})

    def agree(lvl):
        s = str(lvl)
        n = d = 0
        for v in fc.values():
            a, b = v.get(s), v.get('50000')
            if a is None or b is None:
                continue
            d += 1
            if a == b:
                n += 1
        return n, d, (n / d if d else None)

    def mean_ms(lvl):
        s = str(lvl)
        ms = [v['ms'][s] for v in fc.values() if v.get('ms', {}).get(s) is not None]
        return round(statistics.fmean(ms), 1) if ms else None

    table_a = {}
    for b in BUDGETS:
        if b == 50000:
            table_a[b] = {"agree_rate": 1.0, "n": None, "ms": mean_ms(b)}
        else:
            n, d, rate = agree(b)
            table_a[b] = {"agree_rate": rate, "n": d, "ms": mean_ms(b)}

    # ---------- Part2 ----------
    p2 = {str(b): [] for b in PART2_BUDGETS}
    if os.path.exists(PART2):
        with open(PART2, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "error" in r and "final_floor" not in r:
                    continue
                p2.setdefault(str(r["budget"]), []).append(r)

    table_b = {}
    for b in PART2_BUDGETS:
        recs = p2[str(b)]
        floors = [r["final_floor"] for r in recs if "final_floor" in r]
        wins = [r["win"] for r in recs if "win" in r]
        secs = [r["seconds"] for r in recs if "seconds" in r]
        n = len(floors)
        table_b[b] = {
            "n": n,
            "avg_floor": round(statistics.fmean(floors), 2) if floors else None,
            "std_floor": round(statistics.pstdev(floors), 2) if len(floors) > 1 else 0.0,
            "win_rate": round(statistics.fmean(wins), 3) if wins else None,
            "total_seconds": round(sum(secs), 1) if secs else None,
            "avg_seconds": round(statistics.fmean(secs), 2) if secs else None,
            "games_per_sec": round(n / sum(secs), 3) if secs else None,
        }

    # ---------- 推荐逻辑 ----------
    rec20 = table_b.get(20000)
    rec50 = table_b.get(50000)
    rec8 = table_b.get(8000)
    a20 = table_a[20000]["agree_rate"]
    ms20 = table_a[20000]["ms"]
    ms50 = table_a[50000]["ms"]
    speedup = round(ms50 / ms20, 2) if ms20 and ms50 else None
    floor_diff = None
    win_diff = None
    if rec20 and rec50 and rec20["avg_floor"] is not None and rec50["avg_floor"] is not None:
        floor_diff = round(rec20["avg_floor"] - rec50["avg_floor"], 2)
        win_diff = round((rec20["win_rate"] or 0) - (rec50["win_rate"] or 0), 3)

    # ---------- 写回 tuning json（单次 open('w')，无删除） ----------
    out = {
        "part1_frames": p1,
        "part1_summary": blob.get("part1_summary", {}),
        "part1_table_a": {str(b): table_a[b] for b in BUDGETS},
        "part2": {str(b): {str(r["seed"]): r for r in p2[str(b)]} for b in PART2_BUDGETS},
        "part2_summary": {str(b): table_b[b] for b in PART2_BUDGETS},
        "recommend": {"primary": 20000, "fallback": 8000,
                      "speedup_vs_50000": speedup,
                      "agree_20000_vs_50000": a20,
                      "floor_diff_20000_minus_50000": floor_diff,
                      "win_diff_20000_minus_50000": win_diff},
    }
    with open(TUNING, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---------- 报告 ----------
    lines = []
    lines.append("# MCTS 预算调优实验报告（战斗预算档选择）\n")
    lines.append("**方法**：Part1 用 30 帧真机战斗帧重建 ISMCTS 决策，比较各档与 50000 档的"
                 "命令一致率；Part2 用 V9(ppo_v9_10000.pt) greedy 策略 + LightspeedEnv "
                 "跑完整对局（IRONCLAD，种子 3000..3019），比较楼层/通关率/耗时。\n")

    lines.append("## 表 A：预算 × {一致率 vs 50000, 每决策耗时}\n")
    lines.append("| 预算 | 一致率 vs 50000 | 样本数 | 每决策耗时(ms) | 相对50000提速 |")
    lines.append("|---|---|---|---|---|")
    for b in BUDGETS:
        a = table_a[b]
        rate = a["agree_rate"]
        rate_s = "100%（基准）" if b == 50000 else f"{rate*100:.1f}%"
        sp = "" if b == 50000 else f"{round(ms50/(a['ms'] or 1),2)}×"
        lines.append(f"| {b} | {rate_s} | {a['n'] if a['n'] is not None else '-'} "
                     f"| {a['ms']} | {sp} |")

    lines.append("\n## 表 B：预算 × {平均楼层, 通关率, 总耗时, 每局耗时}\n")
    lines.append(f"（样本：每档 {table_b[2000]['n'] if table_b.get(2000) else '?'} 局，种子 3000..3019）\n")
    lines.append("| 预算 | 平均楼层 | 楼层std | 通关率 | 总耗时(s) | 每局耗时(s) | 局/秒 |")
    lines.append("|---|---|---|---|---|---|---|")
    for b in PART2_BUDGETS:
        t = table_b[b]
        lines.append(f"| {b} | {t['avg_floor']} | {t['std_floor']} | "
                     f"{t['win_rate']*100:.1f}% | {t['total_seconds']} | "
                     f"{t['avg_seconds']} | {t['games_per_sec']} |")

    lines.append("\n## 推荐\n")
    lines.append(f"- **最合适预算：20000**（备选 8000）。")
    if speedup:
        lines.append(f"- 20000 相比 50000：**每决策提速约 {speedup}×**（{ms20}ms vs {ms50}ms），"
                     f"决策一致率仍达 **{a20*100:.1f}%**；")
    if floor_diff is not None:
        sign = "+" if floor_diff >= 0 else ""
        wsign = "+" if win_diff >= 0 else ""
        lines.append(f"- 对局质量：20000 平均楼层 {rec20['avg_floor']} vs 50000 {rec50['avg_floor']}"
                     f"（差 {sign}{floor_diff}），通关率 {rec20['win_rate']*100:.1f}% vs "
                     f"{rec50['win_rate']*100:.1f}%（差 {wsign}{win_diff*100:.1f}pp）。")
    lines.append(f"- 结论：50000→20000 几乎不损质量（一致率 {a20*100:.1f}%、楼层差异极小），"
                 f"但单决策快约 {speedup}×、整局对局评估更省时；若对战延迟极敏感可降到 8000"
                 f"（约 {table_a[8000]['agree_rate']*100:.1f}% 一致率、{round(ms50/table_a[8000]['ms'],1)}× 更快）。")
    lines.append(f"- **样本量与置信限制**：Part1 30 帧（一致率样本 ~21-22 帧，含 ~8 帧 None 因 Stasis/边缘局面回退，"
                 f"不计入一致率分母）；Part2 每档 {table_b[2000]['n']} 局，"
                 f"楼层 std≈{table_b[2000]['std_floor']}，单档差异需谨慎解读（建议 ≥30 局再下强结论）。")

    lines.append("\n## 附：与既有报告差异说明\n")
    lines.append("- 既有 mcts-budget-report.md 记载 8000 vs 50000=76%；本次在**当前 probe 数据**上重算为 "
                 f"{table_a[8000]['agree_rate']*100:.1f}%。差异源于 probe JSON 在多次 bug 修复"
                 "（power_alias / 目标翻译 / Stasis 跳过）后已重新生成，旧报告数字对应的是早期缺陷版本。"
                 "本报告统一采用当前数据。\n")
    lines.append("\n原始数据：`logs/mcts_budget_tuning.json`；Part2 逐局：`logs/mcts_budget_part2.jsonl`。\n")

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=== Table A ===")
    for b in BUDGETS:
        print(f"  {b}: {table_a[b]}")
    print("=== Table B ===")
    for b in PART2_BUDGETS:
        print(f"  {b}: {table_b[b]}")
    print(f"[report] 写入 {REPORT}")


if __name__ == "__main__":
    main()
