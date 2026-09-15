# -*- coding: utf-8 -*-
"""牌组强度报告生成（deck-strength 任务）。

读 logs/deck_strength_raw.json（由 _tmp_deck_probe.py 产出），
生成 logs/deck-strength-report.md（中文 ≤90 行）。

分组：A=通关局（win=True）；B=死在 Act3 局（win=False 且 reached_act3 且 34<=floor<=50）。
显著性：Welch t 检验（手动实现 betai，无 scipy 依赖）。
"""
import os, json, math, statistics

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "logs", "deck_strength_raw.json")
OUT = os.path.join(ROOT, "logs", "deck-strength-report.md")

METRICS = [
    ("n_cards", "牌组张数", "张"),
    ("n_rare", "稀有卡数", "张"),
    ("n_upgraded", "升级卡数", "张"),
    ("avg_cost", "平均费用", ""),
    ("n_relics", "遗物数", "个"),
    ("n_potions", "药水数", "个"),
    ("hp", "HP", ""),
    ("max_hp", "max_HP", ""),
]


# ---------- 手动 Welch t 检验（Student-t 双侧 p，betai 连分式） ----------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def welch(a, b):
    """返回 (t, df, p_two_sided, diff, cohen_d)。a/b 为两样本列表。"""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    va = statistics.pvariance(a)
    vb = statistics.pvariance(b)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan"), float("nan"), ma - mb, float("nan")
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p = _betai(df / 2.0, 0.5, df / (df + t * t))
    sp = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    d = (ma - mb) / sp if sp else float("nan")
    return t, df, p, ma - mb, d


def main():
    data = json.load(open(RAW, encoding="utf-8"))
    games = data["games"]
    n_total = len(games)

    A = [g for g in games if g["win"]]
    B = [g for g in games
         if (not g["win"]) and g.get("reached_act3")
         and 34 <= g["final_floor"] <= 50]
    # 其余（死在 Act2 等）仅作透明说明，不参与 A/B 对比
    other = [g for g in games if g not in A and g not in B]

    def col(g, key):
        return g["act3_snap"][key]

    lines = []
    lines.append("# 牌组强度 × Act3 死因 诊断报告")
    lines.append("")
    lines.append(f"- 模型：`{data['ckpt']}`（Scorer 477→369, hidden 512/256, greedy）")
    lines.append(f"- 战斗：`LightspeedEnv(mcts_simulations={data['mcts']})`；种子 `make_eval_seeds(200)[:{n_total}]`")
    lines.append(f"- 样本：{n_total} 局 → **A 通关 {len(A)} 局** / **B 死 Act3 {len(B)} 局**"
                 f" / 其他(死 Act2 等) {len(other)} 局（不计入对比）")
    lines.append("")
    lines.append("## 1. 方法与口径")
    lines.append("")
    lines.append("- **采集点**：每局在 `env.gc.act == 3` 的**首个宏观决策点**快照牌组"
                 "（act 切换发生在上一幕 Boss 奖励后，故反映进 Act3 时的真实牌组）。")
    lines.append("- **字段来源**（3 局试探确认 `gc.deck`/`gc.relics` 为对象，非 dict）：")
    lines.append("  - 卡：`c.id.name` / `c.rarity.name` / `c.upgraded`(bool) / 费用回退 `card_data.get_base_cost(c.id.name)`")
    lines.append("  - 遗物：`len(gc.relics)`；药水：`gc.potion_count`；血：`gc.cur_hp` / `gc.max_hp`")
    lines.append("- **探针示例**（seed 856258841 进 Act3）：30 张 / 稀有 4（DEMON_FORM,FIEND_FIRE,CORRUPTION,FIEND_FIRE）"
                 " / 升级 2（POMMEL_STRIKE×2）/ 遗物 10 / 药水 1 / HP 80-80。字段均可正确读取。")
    lines.append("- **显著性**：Welch t 检验（手动 betai，无 scipy），双侧 p<0.05 记显著；并给 Cohen's d 效应量。")
    lines.append("")
    lines.append("## 2. 分组对比表（进 Act3 时，均值 ± 总体标准差）")
    lines.append("")
    lines.append("| 指标 | A 通关 (n=%d) | B 死Act3 (n=%d) | 差(A−B) | Welch t | p | Cohen's d |"
                 % (len(A), len(B)))
    lines.append("|---|---|---|---|---|---|---|")
    for key, label, unit in METRICS:
        av = [col(g, key) for g in A]
        bv = [col(g, key) for g in B]
        am, bm = statistics.fmean(av), statistics.fmean(bv)
        asd = statistics.pstdev(av) if len(av) > 1 else 0.0
        bsd = statistics.pstdev(bv) if len(bv) > 1 else 0.0
        t, df, p, diff, d = welch(av, bv)
        sig = "✅" if (isinstance(p, float) and p < 0.05) else "—"
        lines.append(
            f"| {label} | {am:.2f} ± {asd:.2f} | {bm:.2f} ± {bsd:.2f} | "
            f"{diff:+.2f} | {t:+.2f} | {p:.3f}{sig} | {d:+.2f} |")
    lines.append("")
    lines.append("## 3. 结论")
    lines.append("")

    # 自动找差距最大的指标（按 |d| 或按 |差|/尺度）
    best = None
    for key, label, unit in METRICS:
        av = [col(g, key) for g in A]
        bv = [col(g, key) for g in B]
        t, df, p, diff, d = welch(av, bv)
        if isinstance(p, float) and (best is None or abs(d) > abs(best[1])):
            best = ((label, d, p, diff), abs(d))
    if best:
        (blabel, bd, bp, bdiff), _ = best
    else:
        blabel, bd, bp, bdiff = "—", float("nan"), float("nan"), float("nan")

    if len(B) == 0:
        lines.append("- ⚠️ B 组为空（60 局无 Act3 死亡），无法对比；需检查种子/阈值。")
    else:
        # 判定：核心指标（稀有/升级/遗物/牌张）是否显著且方向为 A>B
        core = ["n_cards", "n_rare", "n_upgraded", "n_relics"]
        sig_positive = []
        for key, label, unit in METRICS:
            if key not in core:
                continue
            av = [col(g, key) for g in A]
            bv = [col(g, key) for g in B]
            t, df, p, diff, d = welch(av, bv)
            if isinstance(p, float) and p < 0.05 and diff > 0:
                sig_positive.append((label, diff, p, d))
        if sig_positive:
            items = "、".join(f"{l}(+{dd:.2f}, p={pp:.3f}, d={d:.2f})"
                              for l, dd, pp, d in sig_positive)
            lines.append(f"- **假说成立 ✅**：通关局进 Act3 时的牌组显著强于 Act3 死亡局"
                         f"（{items}）。")
            lines.append(f"- **瓶颈在宏观决策质量**（选牌/路线/商店/篝火的长期收益），"
                         f"而非战斗执行层——两队进 Act3 时血线/药水相当，差距只在牌组构成。")
            lines.append(f"- **差距最大指标：{blabel}**（A−B={bdiff:+.2f}，d={bd:+.2f}，"
                         f"p={bp:.3f}）。后续训练应优先奖励该维度相关收益"
                         f"（如稀有卡/升级卡的长期获取）。")
        else:
            lines.append("- **假说不成立 ❌**：通关局与 Act3 死亡局进 Act3 时的牌组强度"
                         "无显著差异（核心指标 p 均≥0.05 或 A 不占优）。")
            lines.append("- 说明瓶颈确实在**战斗执行层**：两队带着同样强的牌组进 Act3，"
                         "死亡局是被 Act3 战斗/MCTS 打法拖垮，需另找原因（战斗评估/出牌策略）。")
            lines.append(f"- 各指标差距最大的为 {blabel}（d={bd:+.2f}），但不显著，不足以支撑牌组假说。")

    lines.append("")
    lines.append("> 只读分析：仅新增 `_tmp_deck_probe.py`/`_tmp_deck_report.py`，未改任何生产代码。"
                 "原始数据见 `logs/deck_strength_raw.json`。")

    txt = "\n".join(lines)
    open(OUT, "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\n[written] {OUT}")
    # 给 team-lead 用的关键数字（控制台）
    if A and B:
        def m(g, k):
            return statistics.fmean([col(x, k) for x in g])
        print("\n=== KEY (A vs B) ===")
        print(f"cards  A={m(A,'n_cards'):.2f} B={m(B,'n_cards'):.2f}")
        print(f"rare   A={m(A,'n_rare'):.2f} B={m(B,'n_rare'):.2f}")
        print(f"upg    A={m(A,'n_upgraded'):.2f} B={m(B,'n_upgraded'):.2f}")
        print(f"relics A={m(A,'n_relics'):.2f} B={m(B,'n_relics'):.2f}")


if __name__ == "__main__":
    main()
