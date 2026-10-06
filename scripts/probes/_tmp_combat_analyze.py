# -*- coding: utf-8 -*-
"""分析 logs/combat_raw.json：产出四张统计表 + Act 死亡率复核 + 失败 Boss 战摘要。
纯分析，不依赖生产代码。输出到 stdout 并写入 logs/combat_stats.json。
"""
import os, sys, json
from collections import defaultdict, Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "logs", "combat_raw.json")
OUT = os.path.join(ROOT, "logs", "combat_stats.json")

# 已知基线（30 固定种子 / MCTS 20000 / greedy）：48.43 层 / 70% 通关
KNOWN = {"floor": 48.43, "win": 0.70, "act_death": {1: 0.0, 2: 0.20, 3: 0.23}, "clear": 0.57}


def main():
    data = json.load(open(RAW, encoding="utf-8"))
    games = data["games"]
    N = data["n"]
    combats = [c for g in games for c in g["combats"]]
    print(f"# 载入 {len(games)} 局，共 {len(combats)} 场战斗\n")

    # ---- 总体口径校验 ----
    wins = sum(1 for g in games if g["win"])
    avg_floor = sum(g["final_floor"] for g in games) / N
    print(f"# 总体：通关 {wins}/{N} = {wins/N:.1%} | 平均楼层 {avg_floor:.2f}")
    print(f"# 对照已知基线：通关 {KNOWN['win']:.0%} / {KNOWN['floor']:.1f} 层 "
          f"（本次 {N} 种子，MCTS={data['mcts']}）\n")

    # ---- 1. 死亡遭遇 TOP10（按致死次数）----
    lethal_by_enc = defaultdict(lambda: {"n": 0, "floors": [], "acts": [], "loss": []})
    for c in combats:
        if c["lethal"]:
            d = lethal_by_enc[c["encounter"]]
            d["n"] += 1
            d["floors"].append(c["floor"])
            d["acts"].append(c["act"])
            d["loss"].append(c["hp_loss"])
    death_sorted = sorted(lethal_by_enc.items(), key=lambda kv: (-kv[1]["n"], -max(kv[1]["loss"])))
    print("## 表1：死亡遭遇 TOP10（致死次数 / 典型 floor·act / 该战平均失血）")
    print(f"{'遭遇':22s} {'致死':>4s} {'floor':>7s} {'act':>4s} {'avg_loss':>8s}")
    for enc, d in death_sorted[:10]:
        fl = Counter(d["floors"]).most_common(1)[0][0]
        ac = Counter(d["acts"]).most_common(1)[0][0]
        print(f"{enc:22s} {d['n']:>4d} {fl:>7d} {ac:>4d} {sum(d['loss'])/len(d['loss']):>8.1f}")
    if not death_sorted:
        print("(无致死战斗——全部通关)")

    # ---- 2. 平均 HP 损失 TOP10（含未致死）----
    loss_by_enc = defaultdict(lambda: {"n": 0, "loss": []})
    for c in combats:
        loss_by_enc[c["encounter"]]["n"] += 1
        loss_by_enc[c["encounter"]]["loss"].append(c["hp_loss"])
    loss_sorted = sorted(loss_by_enc.items(),
                         key=lambda kv: -(sum(kv[1]["loss"]) / len(kv[1]["loss"])))[:10]
    print("\n## 表2：平均 HP 损失 TOP10（含未致死——隐性吃血点；n=出现次数）")
    print(f"{'遭遇':22s} {'avg_loss':>8s} {'max_loss':>8s} {'n':>4s} {'lethal':>6s}")
    lethal_set = set(lethal_by_enc.keys())
    for enc, d in loss_sorted:
        avg = sum(d["loss"]) / len(d["loss"])
        print(f"{enc:22s} {avg:>8.1f} {max(d['loss']):>8d} {d['n']:>4d} {('Y' if enc in lethal_set else ''):>6s}")

    # ---- 3. Boss 战统计（room_type==BOSS）----
    boss = defaultdict(lambda: {"n": 0, "loss": [], "deaths": 0})
    for c in combats:
        if c["room_type"] == "BOSS":
            b = boss[c["encounter"]]
            b["n"] += 1
            b["loss"].append(c["hp_loss"])
            if c["lethal"]:
                b["deaths"] += 1
    print("\n## 表3：Boss 战统计（按 act / 遭遇；win = 到达且未致死）")
    print(f"{'act':>3s} {'遭遇':22s} {'次数':>4s} {'胜率':>6s} {'avg_loss':>8s}")
    boss_rows = sorted(boss.items(), key=lambda kv: (kv[1] and True,))
    # 按 act 分组
    by_act = defaultdict(list)
    for enc, d in boss.items():
        # 找该 boss 的 act（取首次出现）
        act = next((c["act"] for c in combats if c["room_type"] == "BOSS" and c["encounter"] == enc), 0)
        by_act[act].append((enc, d))
    for act in sorted(by_act):
        for enc, d in sorted(by_act[act], key=lambda kv: -kv[1]["n"]):
            wr = (d["n"] - d["deaths"]) / d["n"] if d["n"] else 0
            avg = sum(d["loss"]) / len(d["loss"])
            print(f"{act:>3d} {enc:22s} {d['n']:>4d} {wr:>5.0%} {avg:>8.1f}")

    # ---- 4. Act 死亡率复核 ----
    act_death = Counter()
    cleared = 0
    for g in games:
        if g["win"]:
            cleared += 1
        else:
            # 找致死战斗的 act
            lac = next((c["act"] for c in g["combats"] if c["lethal"]), None)
            if lac:
                act_death[lac] += 1
            else:
                act_death["unknown"] += 1
    print("\n## 表4：Act 死亡率复核（本次 vs 已知基线）")
    print(f"{'类别':10s} {'本次':>8s} {'已知':>8s}")
    print(f"{'Act1 死':10s} {act_death.get(1,0):>6d}/{N}  {KNOWN['act_death'][1]:>6.0%}")
    print(f"{'Act2 死':10s} {act_death.get(2,0):>6d}/{N}  {KNOWN['act_death'][2]:>6.0%}")
    print(f"{'Act3 死':10s} {act_death.get(3,0):>6d}/{N}  {KNOWN['act_death'][3]:>6.0%}")
    print(f"{'通关':10s} {cleared:>6d}/{N}  {KNOWN['clear']:>6.0%}")

    # ---- 失败 Boss 实例（供 pass2 复现逐回合）----
    lethal_bosses = [(g["seed"], c) for g in games for c in g["combats"]
                     if c["room_type"] == "BOSS" and c["lethal"]]
    print("\n## 失败 Boss 实例（seed, 遭遇, floor, act, hp_before->after）：")
    for seed, c in lethal_bosses:
        print(f"  seed={seed} {c['encounter']} f{c['floor']} A{c['act']} "
              f"hp {c['hp_before']}->{c['hp_after']}")

    json.dump({
        "n": N, "wins": wins, "avg_floor": avg_floor,
        "act_death": {str(k): v for k, v in act_death.items()},
        "death_encounters": {k: v for k, v in death_sorted},
        "boss": {k: v for k, v in boss.items()},
        "lethal_bosses": [{"seed": s, **c} for s, c in lethal_bosses],
    }, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n# 统计已写入 {OUT}")


if __name__ == "__main__":
    main()
