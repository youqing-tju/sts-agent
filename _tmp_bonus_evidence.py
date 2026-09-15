# -*- coding: utf-8 -*-
"""奖励塑形生效证据（临时验证脚本）。

构造 PPOTrainer（随机权重即可，collect_episode 不依赖好权重），
用 upgrade_bonus/relic_bonus=1.0 跑若干局，直接调用 collect_episode 拿
(traj, floor, win, upgrade_count, relic_count)，按 update() 中相同的公式
算 G（开/关 bonus 两版），证明 bonus 确实加进回报且数值正确。

不修改任何生产代码。
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
from models.obs_encoder import OBS_DIM
from models.candidate_encoder import DESC_VEC_DIM
from models.scorer import Scorer, ValueNet
from training.ppo import PPOTrainer

WIN_BONUS = 50.0


def main():
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=(512, 256))
    value_net = ValueNet(obs_dim=OBS_DIM)
    # 开 bonus 的 trainer
    trainer = PPOTrainer(scorer, value_net, win_bonus=WIN_BONUS,
                         upgrade_bonus=1.0, relic_bonus=1.0, mcts_sim=2000)
    from env.lightspeed_env import LightspeedEnv
    env = LightspeedEnv(mcts_simulations=2000)

    seeds = [11, 22, 33, 44, 55]
    print(f"{'seed':>5} {'floor':>5} {'win':>5} {'up':>3} {'rel':>3} "
          f"{'G(noB)':>8} {'G(b1.0)':>8} {'Δ':>6}", flush=True)
    all_ok = True
    for s in seeds:
        traj, floor, win, up, rel = trainer.collect_episode(
            env, seed=s)  # collect_episode 内部会用 seed 重置 env
        g_nob = float(floor) + WIN_BONUS * float(win)
        g_b = (float(floor) + WIN_BONUS * float(win)
               + 1.0 * float(up) + 1.0 * float(rel))
        delta = g_b - g_nob
        expect = float(up) + float(rel)
        ok = abs(delta - expect) < 1e-6 and up >= 0 and rel >= 0
        all_ok = all_ok and ok
        print(f"{s:>5} {floor:>5} {int(win):>5} {up:>3} {rel:>3} "
              f"{g_nob:>8.1f} {g_b:>8.1f} {delta:>6.1f} {'OK' if ok else 'FAIL'}",
              flush=True)
    print("\n结论: bonus 生效 =", all_ok, flush=True)
    print("G(b1.0) = floor + 50*win + 1*upgrade_count + 1*relic_count", flush=True)


if __name__ == "__main__":
    main()
