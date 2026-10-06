# -*- coding: utf-8 -*-
"""药水使用探针：验证 MCTS（BattleScumSearcher2）是否在战斗中消耗药水。

仅只读分析 + 本探针脚本；不改生产代码、不重编译 pyd。
单进程；用 v10_seg3.pt + greedy 宏观策略 + mcts_simulations=8000 驱动战斗。

测量法（决定性）：
- 覆写 _advance：每次 self._agent.playout(gc)（= 一整场战斗）前后读 gc.potion_count。
  战斗内消耗（drinkPotion 会 --potionCount，且 exitBattle 同步回 gc）使 delta<0。
  => 累计每场战斗的负数 delta = 战斗中消耗的药水数。
- 覆写 step：每次宏观决策前后读 gc.potion_count。
  药水只在非战斗屏（奖励/事件/商店/篝火）获得 => 正数 delta = 获得数。
"""
import os
import sys
import time
import json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import slaythespire as sts
from env.lightspeed_env import LightspeedEnv, _MAX_ADVANCE_ROUNDS
from models.obs_encoder import OBS_DIM
from models.candidate_encoder import DESC_VEC_DIM
from models.scorer import Scorer


class PotionProbeEnv(LightspeedEnv):
    def __init__(self, mcts_simulations=8000):
        super().__init__(mcts_simulations=mcts_simulations)
        self.combat_deltas = []   # 每场 playout(战斗) 的 potion_count 变化
        self.macro_deltas = []    # 每次 step(宏观决策) 的 potion_count 变化
        self.max_held = 0
        self.final_count = 0

    def _advance(self):
        gc = self.gc
        for _ in range(_MAX_ADVANCE_ROUNDS):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            before = gc.potion_count
            self._agent.playout(gc)
            after = gc.potion_count
            self.combat_deltas.append(after - before)
            self.max_held = max(self.max_held, after)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            if sts.get_all_actions(gc):
                return

    def step(self, idx):
        before = self.gc.potion_count
        res = super().step(idx)
        after = self.gc.potion_count
        self.macro_deltas.append(after - before)
        self.max_held = max(self.max_held, after)
        self.final_count = after
        return res


def make_greedy(path="checkpoints/v10_seg3.pt", hidden=(512, 256)):
    import torch
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=hidden)
    scorer.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    scorer.eval()

    class Greedy:
        def choose(self, obs, candidates):
            with torch.no_grad():
                obs_t = torch.tensor(obs.vec, dtype=torch.float32)
                cands_t = torch.tensor([c.desc_vec for c in candidates],
                                       dtype=torch.float32)
                scores = scorer(obs_t, cands_t)
            return candidates[int(scores.argmax().item())].idx
    return Greedy()


def run_one_game(seed, policy, mcts_sim=8000):
    env = PotionProbeEnv(mcts_simulations=mcts_sim)
    obs = env.reset(seed)
    done, steps = False, 0
    while not done and steps < 500:
        cands = env.candidates()
        idx = policy.choose(obs, cands) if cands else -1
        res = env.step(idx)
        obs, done = res.obs, res.done
        steps += 1
    acquired = sum(d for d in env.macro_deltas if d > 0)
    consumed = sum(-d for d in env.combat_deltas if d < 0)
    # 战斗中是否出现"用掉药水"的净减（决定性信号）
    combat_used = any(d < 0 for d in env.combat_deltas)
    return {
        "seed": seed,
        "outcome": obs.screen,  # 终局 screen（已无动作）
        "result": res.info.get("outcome"),
        "final_floor": obs.floor,
        "acquired": acquired,
        "consumed": consumed,
        "final_held": env.final_count,
        "max_held": env.max_held,
        "combat_used": combat_used,
        "n_playouts": len(env.combat_deltas),
        "n_macro_steps": len(env.macro_deltas),
    }


def main():
    n = int(os.environ.get("N_GAMES", "18"))
    mcts = int(os.environ.get("MCTS", "8000"))
    seeds = [20260831 + i for i in range(n)]  # 确定性种子，便于复现
    policy = make_greedy()
    t0 = time.time()
    rows = []
    for i, s in enumerate(seeds):
        r = run_one_game(s, policy, mcts_sim=mcts)
        rows.append(r)
        print(f"[{i+1}/{n}] seed={s} {r['result']:<16} floor={r['final_floor']} "
              f"获得={r['acquired']} 消耗={r['consumed']} 终局剩={r['final_held']} "
              f"峰值={r['max_held']} 战斗内用过={r['combat_used']}", flush=True)
    dt = time.time() - t0
    tot_acq = sum(r["acquired"] for r in rows)
    tot_con = sum(r["consumed"] for r in rows)
    games_with_consumption = sum(1 for r in rows if r["consumed"] > 0)
    deaths_holding = sum(1 for r in rows
                         if r["result"] != "PLAYER_VICTORY" and r["final_held"] > 0)
    print("\n=== 汇总 ===")
    print(f"局数={n} MCTS={mcts} 用时={dt:.0f}s")
    print(f"总获得={tot_acq} 总消耗={tot_con}")
    print(f"有消耗药水的局数={games_with_consumption}/{n}")
    print(f"死亡时仍持药水局数={deaths_holding}")
    print(f"结论: MCTS {'会' if tot_con>0 else '不会'}在战斗中使用药水")
    os.makedirs("logs", exist_ok=True)
    json.dump({"n": n, "mcts": mcts, "total_acquired": tot_acq,
               "total_consumed": tot_con,
               "games_with_consumption": games_with_consumption,
               "deaths_holding": deaths_holding, "rows": rows},
              open("logs/_potion_probe_raw.json", "w", encoding="utf-8"),
              indent=1)
    print("原始数据已存 logs/_potion_probe_raw.json")


if __name__ == "__main__":
    main()
