"""V10 vs BC-only 配对评估（同种子 / 同预算 / greedy）。

为什么这样比：V9 是旧维度（347），与 V10（369）不兼容、无法同台评估。
所以用 **BC-only（纯模仿人类，未 RL）** 作对照——它正是 V10 的起点。
二者只差"有没有经过 8000 局 PPO 微调"，是干净的配对对照。
"""
import json
import os
import sys
import time

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.getcwd())

import torch

from evaluation.evaluate import make_eval_seeds, run_one
from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer
from training.run_training import _GreedyPolicy

N_SEEDS = int(os.environ.get("N_SEEDS", "30"))
MCTS = int(os.environ.get("MCTS", "20000"))
HIDDEN = (512, 256)

CKPTS = [("checkpoints/v10_seg3.pt", "段3 + 新战斗评估")]


def main():
    seeds = make_eval_seeds(100)[:N_SEEDS]
    print(f"配对评估：{len(seeds)} 个固定种子 | MCTS 预算 {MCTS} | greedy 策略",
          flush=True)

    results = {}
    for path, name in CKPTS:
        if not os.path.exists(path):
            print(f"[skip] {path} 不存在"); continue
        sd = torch.load(path, map_location="cpu", weights_only=True)
        scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=HIDDEN)
        scorer.load_state_dict(sd)
        scorer.eval()
        pol = _GreedyPolicy(scorer)
        floors, wins, t0 = [], [], time.time()
        for i, s in enumerate(seeds):
            r = run_one(s, pol, mcts_sim=MCTS)
            floors.append(int(r["final_floor"]))
            wins.append(1 if r.get("win") else 0)
            if (i + 1) % 10 == 0:
                print(f"  [{name}] {i+1}/{len(seeds)} "
                      f"当前均值 {np.mean(floors):.1f} 层", flush=True)
        dt = time.time() - t0
        results[name] = {"floors": floors, "wins": wins, "secs": dt}
        print(f"[{name}] 平均楼层 {np.mean(floors):.2f} ± {np.std(floors):.2f}"
              f"｜通关率 {np.mean(wins) * 100:.1f}%｜用时 {dt:.0f}s", flush=True)

    names = list(results.keys())
    print("\n=== 汇总 ===")
    base = np.array(results[names[0]]["floors"], dtype=float)
    for nm in names:
        fl = np.array(results[nm]["floors"], dtype=float)
        wn = np.mean(results[nm]["wins"])
        print(f"  {nm:<26} 楼层 {fl.mean():6.2f} ± {fl.std():5.2f} | 通关率 {wn*100:5.1f}%")
    print("\n=== 配对差（各模型 − BC-only 基线）===")
    for nm in names[1:]:
        fl = np.array(results[nm]["floors"], dtype=float)
        d = fl - base
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        t = d.mean() / se if se else float("nan")
        w = int((d > 0).sum()); l = int((d < 0).sum())
        print(f"  {nm:<26} 楼层差 {d.mean():+6.2f} (SE {se:.2f}, t={t:+.2f}) | "
              f"通关率 {np.mean(results[nm]['wins'])*100 - np.mean(results[names[0]]['wins'])*100:+.1f}pp | "
              f"胜 {w} / 负 {l} / 平 {len(d)-w-l}")
    print("\n=== 第2个模型 vs 第3个模型 ===")
    if len(names) >= 3:
        a = np.array(results[names[1]]["floors"], dtype=float)
        b = np.array(results[names[2]]["floors"], dtype=float)
        d = a - b
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
        print(f"  V10 − E1 楼层差 {d.mean():+.2f} (SE {se:.2f}, t={d.mean()/se if se else float('nan'):+.2f})"
              f" | 胜 {int((d>0).sum())} / 负 {int((d<0).sum())}")
    os.makedirs("logs", exist_ok=True)
    json.dump({"n_seeds": N_SEEDS, "mcts_sim": MCTS, "results": results},
              open("logs/v10_vs_bc_eval.json", "w", encoding="utf-8"), indent=1)
    print("\n已保存 logs/v10_vs_bc_eval.json")


if __name__ == "__main__":
    main()
