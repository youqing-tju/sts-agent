"""V11（奖励塑形）vs E1（基线）配对评估 —— 100 固定种子 / 多进程并行 / greedy。

为什么这样比：
- E1 = checkpoints/v10_scratch.pt（从零 7984 局，**无奖励塑形**，历史最佳 RL 模型）
- V11 = checkpoints/v11_shaped.pt（同配方 8000 局，**唯一变量 = 回报多了
  upgrade_bonus×升级数 + relic_bonus×遗物数**）
=> 二者是干净的配对对照，唯一差异就是"有没有奖励塑形"。

复用 evaluation.evaluate 的多进程能力（ProcessPoolExecutor，一进程一局），
比旧的 _tmp_eval_v10.py 串行版快约 8 倍。
"""
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _find_project_root(start):
    """从 start 向上逐级查找含 .git 的目录作为项目根。

    本脚本已移入 scripts/probes/，不能再依赖 CWD；
    根目录下才有可导入的包与 checkpoints/、logs/。
    """
    d = os.path.dirname(os.path.abspath(start))
    while not os.path.isdir(os.path.join(d, ".git")):
        parent = os.path.dirname(d)
        if parent == d:            # 到盘根仍未找到，退回脚本所在目录
            return os.path.dirname(os.path.abspath(start))
        d = parent
    return d


_ROOT = _find_project_root(__file__)   # 项目根（含 checkpoints/、logs/）
sys.path.insert(0, _ROOT)

import numpy as np

from evaluation.evaluate import (GreedyScorerPolicy, evaluate, make_eval_seeds)

N_SEEDS = int(os.environ.get("N_SEEDS", "100"))
MCTS = int(os.environ.get("MCTS", "20000"))
WORKERS = int(os.environ.get("WORKERS", "8"))

CKPTS = [
    (os.path.join(_ROOT, "checkpoints", "v10_scratch.pt"), "E1基线(从零7984局,无塑形)"),
    (os.path.join(_ROOT, "checkpoints", "v11_shaped.pt"), "V11塑形(8000局,up/relic=1.0)"),
    (os.path.join(_ROOT, "checkpoints", "v11_shaped_g6000.pt"), "V11_g6000快照"),
]


def main():
    seeds = make_eval_seeds(N_SEEDS)
    print(f"配对评估：{len(seeds)} 固定种子 | MCTS {MCTS} | greedy | {WORKERS} workers",
          flush=True)
    out = {}
    for path, name in CKPTS:
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在", flush=True)
            continue
        t0 = time.time()
        pol = GreedyScorerPolicy(path)          # 维度不匹配会在此立即报错
        res = evaluate(pol, n_seeds=N_SEEDS, workers=WORKERS, mcts_sim=MCTS)
        dt = time.time() - t0
        floors = [int(r["final_floor"]) for r in res]
        wins = [1 if r["win"] else 0 for r in res]
        out[name] = {"path": path, "floors": floors, "wins": wins, "secs": dt}
        print(f"[{name}] 楼层 {np.mean(floors):.2f} ± {np.std(floors):.2f} | "
              f"通关率 {np.mean(wins)*100:.1f}% | {dt:.0f}s", flush=True)

    names = list(out.keys())
    print("\n=== 汇总 ===")
    for nm in names:
        f = np.array(out[nm]["floors"], float)
        w = np.array(out[nm]["wins"], float)
        print(f"  {nm:<32} 楼层 {f.mean():6.2f} ± {f.std():5.2f} | "
              f"通关率 {w.mean()*100:5.1f}%")

    if len(names) >= 2:
        base = np.array(out[names[0]]["floors"], float)
        print(f"\n=== 配对差（vs {names[0]}）===")
        for nm in names[1:]:
            f = np.array(out[nm]["floors"], float)
            d = f - base
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            t = d.mean() / se if se else float("nan")
            wn = np.mean(out[nm]["wins"]) - np.mean(out[names[0]]["wins"])
            print(f"  {nm:<32} 楼层差 {d.mean():+6.2f} (SE {se:.2f}, t={t:+.2f}) | "
                  f"通关率 {wn*100:+.1f}pp | 胜 {int((d > 0).sum())} / 负 {int((d < 0).sum())}")

    os.makedirs(os.path.join(_ROOT, "logs"), exist_ok=True)
    json.dump({"n_seeds": N_SEEDS, "mcts_sim": MCTS, "results": out},
              open(os.path.join(_ROOT, "logs", "v11_vs_e1_eval.json"), "w", encoding="utf-8"),
              indent=1)
    print("已保存 " + os.path.join(_ROOT, "logs", "v11_vs_e1_eval.json"))


if __name__ == "__main__":
    main()
