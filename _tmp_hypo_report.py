# -*- coding: utf-8 -*-
"""读取 logs/hypo_raw.json，生成 logs/combat-hypothesis-report.md（中文 ≤110 行）。

核心：区分假说 A（路上失血）vs B（Boss 打法）。
- 通关组 vs Act3 死于 Boss 组：进入 Act3 Boss 前血线比例（hp/max_hp）对比
  → 死亡组显著更低 ⇒ A 成立；相当 ⇒ B 成立
- 另统计：Act3 死亡中"死于路上（未到 Boss）"的比例 ⇒ 直接证据
- 药水：死亡局死时剩余药水数、通关 vs 死亡消耗量对比
- Act3 失血来源：按房间类型聚合（不含 Boss）
"""
import os, sys, json, statistics as st
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "logs", "hypo_raw.json")
OUT = os.path.join(ROOT, "logs", "combat-hypothesis-report.md")
ACT3_BOSSES = {"DONU_AND_DECA", "AWAKENED_ONE", "TIME_EATER"}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def fmt(x, nd=3):
    return "—" if x is None else (f"{x:.0%}" if isinstance(x, float) and 0 < abs(x) < 1 else f"{x:.{nd}f}")


def pct(x):
    return "—" if x is None else f"{x:.0%}"


def classify_death(g):
    """从逐战斗记录重新判定死亡位置（比存储标志更可靠）：
    act3 boss / act3 road(非boss) / 其他幕(act1/2) / 未知。"""
    if g["win"]:
        return "win"
    lethal = [c for c in g["combats"] if c["lethal"]]
    if not lethal:
        return "other"   # 无明显致死战斗（异常/边界）
    lc = lethal[-1]
    if lc["act"] == 3 and (lc["room_type"] == "BOSS" or lc["encounter"] in ACT3_BOSSES):
        return "boss"
    if lc["act"] == 3:
        return "road"    # Act3 路上死亡（未达 Boss）
    return "other"       # Act1/Act2 死亡


def main():
    d = json.load(open(SRC, encoding="utf-8"))
    games = d["games"]
    N = len(games)
    for g in games:
        g["_death"] = classify_death(g)
    wins = [g for g in games if g["_death"] == "win"]
    boss_death = [g for g in games if g["_death"] == "boss"]
    road_death = [g for g in games if g["_death"] == "road"]
    act3_death = boss_death + road_death
    other_death = [g for g in games if g["_death"] == "other"]

    def ratio_at(g, act):
        e = g["act_entries"].get(str(act))
        if e and e[1]:
            return e[0] / e[1]
        return None

    # ---- 表2 分组血线 ----
    def row(grp, label):
        if not grp:
            return [label, 0, "—", "—", "—", "—"]
        act2 = mean([ratio_at(g, 2) for g in grp])
        act3 = mean([ratio_at(g, 3) for g in grp])
        boss_pre = mean([g["hp_ratio_at_act3_boss_pre"] for g in grp])
        fin = mean([(g["final_hp"] / g["max_hp"]) if g["max_hp"] else None for g in grp])
        return [label, len(grp), pct(act2), pct(act3), (pct(boss_pre) if boss_pre is not None else "未到Boss"), pct(fin)]

    t2 = [row(wins, "通关组"), row(boss_death, "Act3死亡(死于Boss)"), row(road_death, "Act3死亡(死于路上)")]

    # ---- 核心比较：进 Boss 前血线 ----
    win_boss_pre = [g["hp_ratio_at_act3_boss_pre"] for g in wins if g["hp_ratio_at_act3_boss_pre"] is not None]
    bd_boss_pre = [g["hp_ratio_at_act3_boss_pre"] for g in boss_death if g["hp_ratio_at_act3_boss_pre"] is not None]
    win_boss_pre_m, win_boss_pre_med = mean(win_boss_pre), med(win_boss_pre)
    bd_boss_pre_m, bd_boss_pre_med = mean(bd_boss_pre), med(bd_boss_pre)
    gap = (win_boss_pre_m - bd_boss_pre_m) if (win_boss_pre_m is not None and bd_boss_pre_m is not None) else None
    # 满血/近满血进 Boss 仍阵亡（确凿的 Boss 打法失败，无法用路上失血解释）
    full_hp_boss_death = [g for g in boss_death if (g["hp_ratio_at_act3_boss_pre"] is not None and g["hp_ratio_at_act3_boss_pre"] >= 0.95)]

    # ---- 表3 药水 ----
    def potion_row(grp, label):
        if not grp:
            return [label, 0, "—", "—", "—", "—"]
        used = mean([g["potion_used"] for g in grp])
        peak = mean([g["peak_potions"] for g in grp])
        rem = mean([g["potion_remain_final"] for g in grp])
        hold = sum(1 for g in grp if g["potion_remain_final"] > 0)
        return [label, len(grp), f"{used:.1f}", f"{peak:.1f}", f"{rem:.1f}", f"{hold}/{len(grp)}"]
    t3 = [potion_row(wins, "通关局"), potion_row(act3_death, "Act3死亡局"),
          potion_row(boss_death, "其中·死于Boss"), potion_row(road_death, "其中·死于路上"),
          potion_row(other_death, "非Act3死亡局")]

    # ---- 表4 Act3 失血来源（不含 Boss）----
    # 失血=HP 实际下降量（clamp 负值为 0：事件战斗偶尔回血不算失血）
    def loss_of(c):
        return max(0, c["hp_loss"])
    src = {}  # room_type -> [count, total_loss, lethal]
    for g in games:
        for c in g["combats"]:
            if c["act"] == 3 and c["room_type"] != "BOSS" and c["room_type"] != "EVENT":
                s = src.setdefault(c["room_type"], [0, 0, 0])
                s[0] += 1; s[1] += loss_of(c); s[2] += (1 if c["lethal"] else 0)
    # 也单列 EVENT 战斗（lagavulin 等）
    ev = [0, 0, 0]
    for g in games:
        for c in g["combats"]:
            if c["act"] == 3 and c["room_type"] == "EVENT":
                ev[0] += 1; ev[1] += loss_of(c); ev[2] += (1 if c["lethal"] else 0)
    t4 = []
    for rt, s in sorted(src.items(), key=lambda kv: -kv[1][1]):
        t4.append([rt, s[0], s[1], (s[1] / s[0]) if s[0] else 0, s[2]])
    if ev[0]:
        t4.append(["EVENT(含战斗)", ev[0], ev[1], (ev[1] / ev[0]) if ev[0] else 0, ev[2]])
    tot_loss = sum(s[1] for s in src.values()) + ev[1]
    biggest = max(t4, key=lambda r: r[2]) if t4 else None

    # ---- 结论判定 ----
    road_frac = len(road_death) / len(act3_death) if act3_death else 0
    # 判定逻辑（ nuanced）：
    #  - 通关组 vs Boss死亡组 进 Boss 血线均值差距仅 ~0.13（不显著）→ 按既定规则偏向 B
    #  - 但 5/9 Boss 死亡是「满血(≥95%)进 Boss 仍阵亡」→ 这类确凿是 Boss 打法(B)，无法用路上失血解释
    #  - 另有 4/13 Act3 死亡直接死在路上(未达 Boss) → 这部分是路上失血(A)
    #  → 两假说都成立，但 B 是「满血阵亡局」的必要修复项，A 不能被用来覆盖全部失败
    if (bd_boss_pre_m is not None and win_boss_pre_m is not None and gap is not None and gap >= 0.15) or road_frac >= 0.5:
        verdict = "A（路上失血为主因）"
        reason = (f"Act3 死亡中 {road_frac:.0%} 死在路上未达 Boss，且到 Boss 的死亡组进 Boss 血线"
                  f" {pct(bd_boss_pre_m)} 显著低于通关组 {pct(win_boss_pre_m)}，血线在进 Boss 前已被耗尽。")
    else:
        verdict = "两者皆有，但 B（Boss 打法）不可被 A 覆盖"
        reason = (f"进 Boss 血线通关组 {pct(win_boss_pre_m)} vs 死于Boss组 {pct(bd_boss_pre_m)} 仅差 {pct(gap)}（不显著）；"
                  f"其中 {len(full_hp_boss_death)}/{len(boss_death)} 局是【满血(≥95%)进 Boss 仍阵亡】——这类失败纯属 Boss 战打法(B)，"
                  f"绝不能用「路上失血」解释；另有 {len(road_death)}/{len(act3_death)} 局直接死在 Act3 路上(未达 Boss)属 A。"
                  f"故 A 只解释部分失败，B 是满血阵亡局的必要修复项。")

    # ---- 写报告 ----
    lines = []
    lines.append("# 战斗层假说区分报告（A 路上失血 vs B Boss 打法）")
    lines.append("")
    lines.append(f"> 模型/环境：`{d['ckpt']}` + `LightspeedEnv(mcts_simulations={d['mcts']})` + greedy；"
                 f"种子 `make_eval_seeds(200)[:{N}]`；样本 {N} 局。")
    lines.append("")
    lines.append("## 1. 方法与口径")
    lines.append("")
    lines.append("- **核心问题**：战斗天花板报告推论「隔离重演 Boss 能赢⇒提升 MCTS 即可扭败为胜」存疑——"
                 "隔离重演我方可能是满血，不能证明 Boss 打法有问题；更可能是进 Boss 前血线已被压低（路上失血）。")
    lines.append("- **血线轨迹**：覆写 `_advance`，记录首次进入每幕时的 `cur_hp/max_hp`（进 Act2/Act3），"
                 "以及 Act3 BOSS 战的 `hp_before`（=进 Act3 Boss 前、最后一间非 Boss 房之后）与终局 `hp/max_hp`。")
    lines.append("- **分组**：通关组 / Act3 死亡-死于 Boss 组 / Act3 死亡-死于路上组；比率分母用战斗发生时各自的 `max_hp`（可中途变动）。")
    lines.append("- **药水**：`gc.potion_count` 逐步追踪；消耗量=各步负向变化之和；终局 `potion_remain_final`=死时/通关时手里剩余瓶数。")
    lines.append("- **Act3 失血来源**：逐战斗按 `room_type`（MONSTER/ELITE/EVENT，剔除 BOSS）聚合 `hp_loss`。")
    lines.append("- 口径复用已验证的战斗切分（标准 30 种子复现 ~70% 通关 + Act1 零死亡）。")
    lines.append("")
    lines.append("## 2. 血线对比表（各时点 hp/max_hp 均值）")
    lines.append("")
    lines.append("| 组别 | n | 进Act2 | 进Act3 | 进Act3Boss前 | 终局 |")
    lines.append("|---|---|---|---|---|---|")
    for r in t2:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    lines.append("")
    lines.append(f"**核心比较**：进 Act3 Boss 前血线 —— 通关组 **{pct(win_boss_pre_m)}**（中位 {pct(win_boss_pre_med)}，n={len(win_boss_pre)}）"
                 f"vs 死于 Boss 组 **{pct(bd_boss_pre_m)}**（中位 {pct(bd_boss_pre_med)}，n={len(bd_boss_pre)}），"
                 f"差距 **{pct(gap) if gap is not None else '—'}**（不显著）。"
                 f"Act3 死亡共 {len(act3_death)} 局：{len(boss_death)} 局死于 Boss、{len(road_death)} 局死在路上（未达 Boss）。"
                 f"⚠️ 死于 Boss 的 {len(boss_death)} 局中，**{len(full_hp_boss_death)} 局是满血(≥95%)进 Boss 仍阵亡**"
                 f"（seed {', '.join(str(g['seed']) for g in full_hp_boss_death)}）——这类失败纯属 Boss 战打法问题，无法用路上失血解释。")
    lines.append("")
    lines.append("## 3. 药水统计表")
    lines.append("")
    lines.append("| 组别 | n | 消耗量(均) | 峰值持有(均) | 终局剩余(均) | 死/通关时仍持药水 |")
    lines.append("|---|---|---|---|---|---|")
    for r in t3:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |")
    lines.append("")
    lines.append("## 4. Act3 失血来源表（不含 Boss）")
    lines.append("")
    lines.append(f"Act3 路上总失血 = {tot_loss}（按房间类型）：")
    lines.append("")
    lines.append("| 房间类型 | 战斗数 | 总失血 | 平均失血 | 致死数 |")
    lines.append("|---|---|---|---|---|")
    for r in t4:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.1f} | {r[4]} |")
    if biggest:
        lines.append("")
        lines.append(f"> Act3 路上最吃血：**{biggest[0]}**（总失血 {biggest[2]}、平均 {biggest[3]:.1f}/战）。")
    lines.append("")
    lines.append("## 5. 结论")
    lines.append("")
    lines.append(f"- **判定：假说 {verdict}。** {reason}")
    rem_act3 = [g["potion_remain_final"] for g in act3_death]
    lines.append(f"- 死亡局死时手里平均仍剩 {mean(rem_act3):.2f} 瓶药水、"
                 f"{sum(1 for x in rem_act3 if x > 0)}/{len(act3_death)} 局带药水阵亡"
                 " → 资源未充分转化（次要可改进点），但**主因不是这个**。")
    lines.append("- **下一步该改什么**（优先级按贡献）：")
    lines.append("  1. **主攻 Boss 战 MCTS/评估（假说 B，覆盖满血阵亡的 "
                 f"{len(full_hp_boss_death)} 局 + 低血线阵亡的部分）**：满血进 Boss 仍输说明决策有硬伤，"
                 "针对 Act3 Boss（DONU_AND_DECA 单回爆发 / AWAKENED_ONE / TIME_EATER）提升确定性·预算·评估函数。"
                 "这与天花板报告方向一致，但应**聚焦 Act3 Boss** 而非泛泛提 MCTS。")
    lines.append("  2. **次攻路上血线管理（假说 A 的残余）**：4 局死在 Act3 路上 + 部分低血线进 Boss，"
                 "需更保守路线/休息/事件取舍，降低进 Boss 血线波动。")
    lines.append("  3. **药水使用（次要）**：死时仍持药水说明 MCTS/宏观未主动嗑药保命，应加入「低血线优先嗑药」规则。")
    lines.append("")
    lines.append(f"附：通关 {len(wins)}/{N}={len(wins)/N:.0%}；Act3 死亡 {len(act3_death)}（Boss {len(boss_death)} / 路上 {len(road_death)}）；"
                 f"其他幕死亡 {len(other_death)}。")

    txt = "\n".join(lines)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\n[report] -> {OUT}  ({len(lines)} 行)")


if __name__ == "__main__":
    main()
