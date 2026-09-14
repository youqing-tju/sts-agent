"""V10 行为体检：它到底学会"跳牌"了吗？（用户最关心的能力之一）

方法：用 V10 + LightspeedEnv 跑若干局，在 CARD_REWARD 屏统计
  · 有 SKIP 候选可选的次数
  · 实际选了 SKIP 的次数（= 真·跳牌）
  · 人类数据里"选牌"的偏好是否保留（对照组：BC-only）
"""
import os
import sys
import collections

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.getcwd())

import torch

from env.lightspeed_env import LightspeedEnv
from evaluation.evaluate import make_eval_seeds
from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer
from training.run_training import _GreedyPolicy

N_GAMES = int(os.environ.get("N_GAMES", "20"))
MCTS = int(os.environ.get("MCTS", "8000"))
MAX_STEPS = 400
CKPTS = [("checkpoints/v10_scratch.pt", "E1（从零+RL，当前最佳）")]


def probe(path, name):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sc = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=(512, 256))
    sc.load_state_dict(sd)
    sc.eval()
    pol = _GreedyPolicy(sc)

    env = LightspeedEnv(mcts_simulations=MCTS)
    n_reward_screens = n_skip_offered = n_skip_taken = n_card_taken = 0
    picked = collections.Counter()
    floors = []
    for s in make_eval_seeds(100)[:N_GAMES]:
        obs = env.reset(s)
        done, steps = False, 0
        while not done and steps < MAX_STEPS:
            cands = env.candidates()
            if not cands:
                res = env.step(-1)
            else:
                kinds = [c.kind for c in cands]
                pos = pol.choose(obs, cands)
                chosen = cands[pos]
                if "CARD_REWARD" in kinds:
                    n_reward_screens += 1
                    if "SKIP" in kinds:
                        n_skip_offered += 1
                    if chosen.kind == "SKIP":
                        n_skip_taken += 1
                    else:
                        n_card_taken += 1
                        picked[chosen.label] += 1
                res = env.step(chosen.idx)
            obs, done = res.obs, res.done
            steps += 1
        floors.append(obs.floor)
    print(f"\n[{name}] {N_GAMES} 局 | 平均楼层 {sum(floors)/len(floors):.1f}")
    print(f"  选牌屏出现 {n_reward_screens} 次，其中 {n_skip_offered} 次提供 SKIP 选项")
    print(f"  → **实际跳牌 {n_skip_taken} 次**（跳牌率 "
          f"{n_skip_taken / max(1, n_skip_offered) * 100:.1f}%）｜拿牌 {n_card_taken} 次")
    if picked:
        print(f"  最常拿的卡 TOP10：{picked.most_common(10)}")


if __name__ == "__main__":
    for p, n in CKPTS:
        if os.path.exists(p):
            probe(p, n)
