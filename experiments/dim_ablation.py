"""推理期维度消融：删掉候选向量 [271,367) 那 96 维 label 哈希段会不会变差？

只读推理实验（不训练、不改任何已有文件、不写 checkpoints/）。权重固定用
checkpoints/ppo_v6_5000.pt（已知最好，Scorer(obs_dim=468, cand_dim=443, hidden=(512,256))）。

三个条件（同一份权重、同一批种子/决策点）：
  A 基线   ：不遮蔽
  B 实验   ：遮蔽候选向量 [271,367)（96 维 label 哈希）
  C 正对照 ：遮蔽候选向量 [22,118)（卡牌 one-hot 前 96 维，真实有信息量，校准"置零 96 维"本身的破坏力）

Part 1（先跑，便宜）：RulePolicy 跑 N 局（MCTS@200）收集决策点（obs.vec 468 + 各候选 desc_vec 443），
         之后只做前向，统计 A→B / A→C 的 argmax 翻转率、打分离散度、按 kind 分组翻转率。
Part 2（仅当 A→B 翻转率 > 5% 才跑）：3 条件 × 200 种子 × MCTS@2000 × 4 worker 配对对比，
         报 win_rate/avg_floor 与 A↔B、A↔C 两比例 z 检验 p 值。

运行：
  KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python.exe -m experiments.dim_ablation
  （可选：--part1-games 20 --mcts 200 --skip-part2）

产出：
  logs/dim_ablation.json         原始聚合数据（含决策点）
  logs/dim-ablation-report.md     中文报告
"""
import argparse
import json
import math
import os
import random
import sys
import datetime

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

# 限制 torch 线程，避免与后台 V8 训练（GPU）抢占过多 CPU
torch.set_num_threads(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from env.lightspeed_env import LightspeedEnv
from agents.rule_policy import RulePolicy
from models.scorer import Scorer
from models.candidate_encoder import DESC_VEC_DIM

# ⚠️ 本脚本只适用于 **443 维候选向量**（含 [271,367) 的 96 维 label 哈希段）——
# 即 2026-09-10 删除该段之前的表征版本。改成 347 维后，MASK_B 的区间含义已变
# （271 起现在是药水 KB 段），重跑会遮蔽错误的段并给出错误结论。
# 要复现本实验，先把编码器回滚到删段前：git checkout a3472a5 -- models/candidate_encoder.py
assert DESC_VEC_DIM == 443, (
    f"dim_ablation 只适用于 443 维候选表征（当前 {DESC_VEC_DIM} 维）。"
    "该段已被删除，请勿直接重跑；见文件头说明。"
)

WEIGHTS = os.path.join(ROOT, "checkpoints", "ppo_v6_5000.pt")
OBS_DIM = 468
CAND_DIM = 443
HIDDEN = (512, 256)

# 三段遮蔽范围（[lo, hi)）
MASK_B = (271, 367)   # label 哈希段
MASK_C = (22, 118)    # 卡牌 one-hot 前 96 维（正对照）

PART2_FLIP_THRESHOLD = 0.05   # A→B 翻转率阈值，超过才跑 Part 2


def _load_scorer():
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=CAND_DIM, hidden=HIDDEN)
    sd = torch.load(WEIGHTS, weights_only=True, map_location="cpu")
    scorer.load_state_dict(sd)
    scorer.eval()
    return scorer


def _apply_mask(vec, mask):
    """返回 vec 的副本，[lo,hi) 置零（mask=None 时原样返回新列表）。"""
    if mask is None:
        return list(vec)
    lo, hi = mask
    v = list(vec)
    for i in range(lo, hi):
        v[i] = 0.0
    return v


# --------------------------------------------------------------------------
# Part 1：收集决策点（跑 env） + 前向统计
# --------------------------------------------------------------------------

def collect_decision_points(n_games, mcts, max_decisions, seeds):
    """用 RulePolicy 跑 n_games 局（MCTS@mcts），逐步记录决策点。

    每个决策点：{"obs": [468], "cands": [[443]...], "kinds": [kind...]}
    返回决策点列表（最多 max_decisions 个，先到先得）。
    """
    env = LightspeedEnv(mcts_simulations=mcts)
    policy = RulePolicy()
    points = []
    for gi in range(n_games):
        seed = seeds[gi] if gi < len(seeds) else (100000 + gi * 13)
        obs = env.reset(seed)
        done = False
        steps = 0
        try:
            while not done and steps < 500:
                cands = env.candidates()
                if cands:
                    points.append({
                        "obs": list(obs.vec),
                        "cands": [list(c.desc_vec) for c in cands],
                        "kinds": [c.kind for c in cands],
                    })
                    if len(points) >= max_decisions:
                        break
                    idx = policy.choose(obs, cands)
                else:
                    idx = -1
                res = env.step(idx)
                obs, done = res.obs, res.done
                steps += 1
        except Exception as e:  # 单局异常不拖垮整批
            print(f"[collect] game {gi} seed={seed} 异常: {type(e).__name__}: {e}",
                  flush=True)
        if len(points) >= max_decisions:
            break
    return points


def forward_stats(points, scorer):
    """对决策点做 A/B/C 三条件前向，返回统计。

    - flips_ab / flips_ac：argmax 翻转的决策点数
    - div_b / div_c：|max_score_cond - max_score_A| 的列表
    - per_kind_ab：按 A 所选候选 kind 分组的翻转率
    """
    flips_ab = 0
    flips_ac = 0
    div_b = []
    div_c = []
    total = len(points)

    per_kind_count = {}
    per_kind_flip = {}

    with torch.no_grad():
        for pt in points:
            obs_t = torch.tensor(pt["obs"], dtype=torch.float32)
            base = torch.tensor(pt["cands"], dtype=torch.float32)
            sA = scorer(obs_t, base)

            masked_b = torch.tensor([_apply_mask(v, MASK_B) for v in pt["cands"]],
                                    dtype=torch.float32)
            sB = scorer(obs_t, masked_b)

            masked_c = torch.tensor([_apply_mask(v, MASK_C) for v in pt["cands"]],
                                    dtype=torch.float32)
            sC = scorer(obs_t, masked_c)

            aA = int(sA.argmax().item())
            aB = int(sB.argmax().item())
            aC = int(sC.argmax().item())

            if aA != aB:
                flips_ab += 1
            if aA != aC:
                flips_ac += 1

            div_b.append(abs(float(sB[aA].item()) - float(sA[aA].item())))
            div_c.append(abs(float(sC[aA].item()) - float(sA[aA].item())))

            # 按 A 所选候选的 kind 分组（看翻转是否集中某类）
            kind = pt["kinds"][aA]
            per_kind_count[kind] = per_kind_count.get(kind, 0) + 1
            if aA != aB:
                per_kind_flip[kind] = per_kind_flip.get(kind, 0) + 1

    def _rate(n, d):
        return (100.0 * n / d) if d else 0.0

    def _median(xs):
        return statistics.median(xs) if xs else 0.0

    import statistics
    per_kind = {
        k: {"count": per_kind_count[k],
            "flip_ab": per_kind_flip.get(k, 0),
            "flip_ab_pct": round(_rate(per_kind_flip.get(k, 0), per_kind_count[k]), 2)}
        for k in sorted(per_kind_count)
    }

    return {
        "total_decision_points": total,
        "flips_ab": flips_ab,
        "flips_ac": flips_ac,
        "flip_ab_pct": round(_rate(flips_ab, total), 2),
        "flip_ac_pct": round(_rate(flips_ac, total), 2),
        "score_div_b_mean": round(statistics.fmean(div_b), 6) if div_b else 0.0,
        "score_div_b_median": round(_median(div_b), 6),
        "score_div_c_mean": round(statistics.fmean(div_c), 6) if div_c else 0.0,
        "score_div_c_median": round(_median(div_c), 6),
        "per_kind_ab": per_kind,
    }


def run_part1(n_games, mcts, max_decisions, seeds):
    print(f"[Part1] 收集 {n_games} 局决策点 (MCTS@{mcts})...", flush=True)
    points = collect_decision_points(n_games, mcts, max_decisions, seeds)
    print(f"[Part1] 收集到 {len(points)} 个决策点；加载权重并前向...", flush=True)
    scorer = _load_scorer()
    stats = forward_stats(points, scorer)
    print(f"[Part1] A→B 翻转率={stats['flip_ab_pct']:.2f}%，"
          f"A→C 翻转率={stats['flip_ac_pct']:.2f}%", flush=True)
    return points, stats


# --------------------------------------------------------------------------
# Part 2：博弈级配对对比（仅当 A→B 翻转率 > 5% 时调用）
# --------------------------------------------------------------------------

class MaskedScorerPolicy:
    """顶层类（可 pickle）：按 mask 遮蔽候选向量后喂 Scorer 取 argmax。

    scorer 在 worker 内惰性加载（存路径，首次 choose 时建），避免跨进程传 torch 模块。
    """

    def __init__(self, path, mask=None, obs_dim=OBS_DIM, cand_dim=CAND_DIM, hidden=HIDDEN):
        self.path = path
        self.mask = mask
        self.obs_dim = obs_dim
        self.cand_dim = cand_dim
        self.hidden = tuple(hidden)
        self._scorer = None

    def _ensure(self):
        if self._scorer is None:
            self._scorer = _load_scorer_path(self.path, self.obs_dim,
                                             self.cand_dim, self.hidden)

    def choose(self, obs, candidates):
        self._ensure()
        obs_t = torch.tensor(obs.vec, dtype=torch.float32)
        vecs = [_apply_mask(c.desc_vec, self.mask) for c in candidates]
        cands_t = torch.tensor(vecs, dtype=torch.float32)
        with torch.no_grad():
            scores = self._scorer(obs_t, cands_t)
        return candidates[int(scores.argmax().item())].idx


def _load_scorer_path(path, obs_dim, cand_dim, hidden):
    scorer = Scorer(obs_dim=obs_dim, cand_dim=cand_dim, hidden=hidden)
    sd = torch.load(path, weights_only=True, map_location="cpu")
    scorer.load_state_dict(sd)
    scorer.eval()
    return scorer


def _two_proportion_z(x1, n1, x2, n2):
    """两比例 z 检验，返回 (z, p_value)。"""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = x1 / n1, x2 / n2
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # 正态分布双尾 p 值
    from math import erf, sqrt
    pval = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    return z, pval


_POLICY_CACHE = {}


def _get_part2_policy(mask):
    """worker 进程内按 mask 缓存策略，避免每个任务都重新 torch.load。"""
    key = tuple(mask) if mask else None
    pol = _POLICY_CACHE.get(key)
    if pol is None:
        pol = MaskedScorerPolicy(WEIGHTS, mask=mask)
        _POLICY_CACHE[key] = pol
    return pol


def _part2_worker(args):
    """进程池 worker —— 必须是**模块级**函数，嵌套函数无法 pickle（会报
    AttributeError: Can't get local object）。"""
    seed, cond, mask, mcts = args
    from evaluation.evaluate import run_one
    r = run_one(seed, _get_part2_policy(mask), mcts_sim=mcts)
    return (seed, cond, r["win"], r["final_floor"])


def run_part2(n_seeds, mcts, workers):
    """3 条件 × n_seeds × MCTS@mcts × workers 配对对比。复用 evaluate 的并行范式。"""
    import statistics as _st
    from concurrent.futures import ProcessPoolExecutor
    from evaluation.evaluate import make_eval_seeds

    seeds = make_eval_seeds(n_seeds)
    conditions = {"A": None, "B": MASK_B, "C": MASK_C}
    # 提交所有 (seed, cond, mask, mcts) 任务
    tasks = [(seed, cond, mask, mcts)
             for seed in seeds for cond, mask in conditions.items()]

    results = {"A": [], "B": [], "C": []}
    floors = {"A": [], "B": [], "C": []}
    # 注意：不要用 max_tasks_per_child=1 —— 那会让每个任务重启进程、
    # 每次重新 import torch + 载入模型，并行收益被进程启动开销吃掉。
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for seed, cond, win, floor in pool.map(_part2_worker, tasks):
            results[cond].append(1 if win else 0)
            floors[cond].append(floor)

    out = {}
    n = len(seeds)
    for cond in ("A", "B", "C"):
        wins = results[cond]
        out[cond] = {
            "win_rate": round(100.0 * _st.fmean(wins), 2),
            "wins": sum(wins),
            "avg_floor": round(_st.fmean(floors[cond]), 2),
        }
    z_ab, p_ab = _two_proportion_z(sum(results["A"]), n, sum(results["B"]), n)
    z_ac, p_ac = _two_proportion_z(sum(results["A"]), n, sum(results["C"]), n)
    out["ztest_A_vs_B"] = {"z": round(z_ab, 3), "p": round(p_ab, 4)}
    out["ztest_A_vs_C"] = {"z": round(z_ac, 3), "p": round(p_ac, 4)}
    return out


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------

def build_report(meta, p1, p2, part2_ran):
    L = []
    L.append("# 推理期维度消融报告：删掉候选向量 [271,367) 96 维会变差吗？\n")
    L.append(f"- 生成时间：{meta['generated_at']}")
    L.append(f"- 权重：{os.path.relpath(WEIGHTS, ROOT)}（Scorer obs_dim={OBS_DIM}, cand_dim={CAND_DIM}, hidden={HIDDEN}）")
    L.append(f"- 资源约束：并行度≤{meta['workers']}，总时长≈{meta['elapsed_min']:.1f} 分钟；未训练、未改已有文件、未写 checkpoints。\n")

    L.append("## 一句话结论\n")
    if not part2_ran:
        rel = "高" if p1["flip_ac_pct"] > p1["flip_ab_pct"] else "低"
        L.append(f"**Part 1 已完成，但 Part 2 未运行，因此不给出'变差/不变差'的结论。** "
                 f"A→B 翻转率 **{p1['flip_ab_pct']:.2f}%**（触发阈值 "
                 f"{meta['flip_threshold'] * 100:.0f}%），打分变化中位数 "
                 f"**{p1['score_div_b_median']}**。正对照 C（删卡牌 one-hot 前 96 维）翻转率 "
                 f"**{p1['flip_ac_pct']:.2f}%，比 B {rel}**。"
                 f"注意两点：(1) C 只在「候选本身是卡」的决策上非零，天然是弱对照，不能当量级基准；"
                 f"(2) 翻转率本身只衡量『排序是否被扰动』，不衡量『扰动是否伤害成绩』。\n")
    else:
        L.append(f"**A→B 翻转率 {p1['flip_ab_pct']:.2f}% > 5%，已跑 Part 2 博弈级对比**（见 §3）。\n")

    L.append("## Part 1：决策点前向消融（无模拟，纯推理）\n")
    L.append(f"- 决策点采集：{meta['part1_games']} 局 RulePolicy（MCTS@{meta['mcts']}），"
             f"共 {p1['total_decision_points']} 个决策点。")
    L.append(f"- **A→B 翻转率（删 label 哈希段）**：**{p1['flip_ab_pct']:.2f}%**"
             f"（{p1['flips_ab']}/{p1['total_decision_points']}）")
    L.append(f"- **A→C 翻转率（删卡牌 one-hot 前 96 维，正对照）**：**{p1['flip_ac_pct']:.2f}%**"
             f"（{p1['flips_ac']}/{p1['total_decision_points']}）")
    L.append(f"- 打分离散度 |max_score_条件 − max_score_A|：")
    L.append(f"  - B（label 哈希置零）：均值 {p1['score_div_b_mean']}，中位数 {p1['score_div_b_median']}")
    L.append(f"  - C（卡牌前 96 维置零）：均值 {p1['score_div_c_mean']}，中位数 {p1['score_div_c_median']}")
    L.append("")
    L.append("### 按候选 kind 分组的 A→B 翻转率\n")
    L.append("| kind（A 所选候选） | 决策点数 | 翻转数 | 翻转率 |")
    L.append("|---|---:|---:|---:|")
    for k, v in p1["per_kind_ab"].items():
        L.append(f"| {k} | {v['count']} | {v['flip_ab']} | {v['flip_ab_pct']:.2f}% |")
    L.append("")

    if part2_ran:
        L.append(f"## Part 2：博弈级配对对比"
                 f"（3 条件 × {meta.get('part2_seeds', 0)} 种子 × "
                 f"MCTS@{meta.get('part2_mcts', 0)} × {meta.get('part2_workers', 0)} worker）\n")
        L.append("| 条件 | win_rate | wins | avg_floor |")
        L.append("|---|---:|---:|---:|")
        for cond in ("A", "B", "C"):
            c = p2[cond]
            label = {"A": "A 基线", "B": "B 删label哈希", "C": "C 删卡牌前96维"}[cond]
            L.append(f"| {label} | {c['win_rate']:.2f}% | {c['wins']} | {c['avg_floor']} |")
        L.append("")
        L.append(f"- A vs B 两比例 z 检验：z={p2['ztest_A_vs_B']['z']}，p={p2['ztest_A_vs_B']['p']}")
        L.append(f"- A vs C 两比例 z 检验：z={p2['ztest_A_vs_C']['z']}，p={p2['ztest_A_vs_C']['p']}")
        L.append("")
        L.append("### 解读\n")
        L.append(f"- B 与 A 的 win_rate 差异 {p2['A']['win_rate'] - p2['B']['win_rate']:+.2f} 个百分点，"
                 f"p={p2['ztest_A_vs_B']['p']}（p>0.05 即无统计显著差异）。")
        L.append(f"- C 与 A 的 win_rate 差异 {p2['A']['win_rate'] - p2['C']['win_rate']:+.2f} 个百分点，"
                 f"p={p2['ztest_A_vs_C']['p']}。")
    else:
        L.append("## Part 2：未运行\n")
        L.append(f"A→B 翻转率 {p1['flip_ab_pct']:.2f}%（触发阈值 "
                 f"{meta['flip_threshold'] * 100:.0f}%）。"
                 f"**注意：翻转率高于阈值时，不得以『无影响』作结**——此时应补跑 Part 2，"
                 f"或明确标注『未做博弈级验证』。只有翻转率低于阈值，才可据『模型基本不依赖该段』"
                 f"下结论；且即便如此，该结论的强度也弱于 Part 2 的配对通关率对比。\n")

    L.append("## 说明 / 局限\n")
    L.append("- Part 1 用 RulePolicy 生成决策点（覆盖真实对局状态分布），但打分来自已训练好的 ppo_v6_5000；"
             "翻转率衡量「同一状态下模型排序是否因置零而改变」。")
    L.append("- 该权重训练时候选即 443 维（含 label 哈希段），故测量的是「已训练模型对这段的推理期依赖」，"
             "非「训练期信息增益」。")
    L.append("- 遮蔽只在喂 Scorer 前对候选向量副本置零，不改 Candidate.desc_vec 本体，也不改任何模型文件。")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(prog="experiments.dim_ablation")
    ap.add_argument("--part1-games", type=int, default=20)
    ap.add_argument("--mcts", type=int, default=200)
    ap.add_argument("--max-decisions", type=int, default=1500)
    ap.add_argument("--part2-seeds", type=int, default=200)
    ap.add_argument("--part2-mcts", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-part2", action="store_true")
    ap.add_argument("--force-part2", action="store_true")
    ap.add_argument("--seed-base", type=int, default=20001)
    args = ap.parse_args()

    t0 = datetime.datetime.now()
    seeds = [args.seed_base + i * 17 for i in range(max(args.part1_games, args.part2_seeds))]

    points, p1 = run_part1(args.part1_games, args.mcts, args.max_decisions, seeds)

    flip_ab = p1["flip_ab_pct"] / 100.0
    part2_ran = False
    p2 = None
    if (not args.skip_part2) and (args.force_part2 or flip_ab > PART2_FLIP_THRESHOLD):
        print(f"[Part2] A→B 翻转率 {flip_ab*100:.2f}% > 阈值，启动博弈级对比...",
              flush=True)
        p2 = run_part2(args.part2_seeds, args.part2_mcts, args.workers)
        part2_ran = True
    else:
        print(f"[Part2] 跳过（A→B 翻转率 {flip_ab*100:.2f}% ≤ {PART2_FLIP_THRESHOLD*100:.0f}%"
              f" 或 --skip-part2）。", flush=True)

    elapsed = (datetime.datetime.now() - t0).total_seconds() / 60.0

    meta = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "weights": os.path.relpath(WEIGHTS, ROOT),
        "part1_games": args.part1_games,
        "mcts": args.mcts,
        "workers": args.workers,
        "part2_seeds": args.part2_seeds,
        "part2_mcts": args.part2_mcts,
        "part2_workers": args.workers,
        "elapsed_min": round(elapsed, 2),
        "flip_threshold": PART2_FLIP_THRESHOLD,
    }

    # 原始数据
    out = {
        "meta": meta,
        "part1": p1,
        "part2_ran": part2_ran,
        "part2": p2,
        "decision_points": points,   # 存盘以便只做前向复查
    }
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    json_path = os.path.join(ROOT, "logs", "dim_ablation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[save] 原始数据：{json_path}", flush=True)

    md = build_report(meta, p1, p2, part2_ran)
    md_path = os.path.join(ROOT, "logs", "dim-ablation-report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[save] 报告：{md_path}", flush=True)
    print(f"[done] 总时长 {elapsed:.1f} 分钟；A→B 翻转率 {p1['flip_ab_pct']:.2f}%，"
          f"Part2={'已跑' if part2_ran else '跳过'}。", flush=True)


if __name__ == "__main__":
    main()
