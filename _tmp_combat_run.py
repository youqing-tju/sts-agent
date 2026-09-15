# -*- coding: utf-8 -*-
"""战斗层天花板探针（Pass 1）：用 v10_seg3.pt + LightspeedEnv(MCTS 20000) greedy 跑 N 局，
逐战斗记录 {遭遇名, floor, act, room_type, hp_before, hp_after, hp_loss, 是否致死}。

口径（如何界定"一场战斗"）：
- env.step() 内部 agent.playout(gc) 会一次性把整场战斗打完（实测每个战斗 = 1 次 playout）。
- 覆写 _advance：每次 playout 前若 screen==BATTLE（=刚进入战斗），记一场战斗，
  hp_before = 进战前 cur_hp，playout 后 hp_after = 出战后 cur_hp；lethal = 之后 outcome==PLAYER_LOSS。
- 遭遇名用 gc.encounter.name（MonsterEncounter 枚举字符串）；Boss/Elite 用 gc.cur_room.name 区分。

单进程、只读分析、不改生产代码。落盘 logs/combat_raw.json。
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import slaythespire as sts
from env.lightspeed_env import LightspeedEnv, _MAX_ADVANCE_ROUNDS
from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer
from training.run_training import _GreedyPolicy
from evaluation.evaluate import make_eval_seeds

CKPT = "checkpoints/v10_seg3.pt"
MCTS = 20000
HIDDEN = (512, 256)
N = int(os.environ.get("N_GAMES", "50"))
OUT = os.path.join(ROOT, "logs", "combat_raw.json")


def enc_name(gc):
    e = getattr(gc, "encounter", None)
    return e.name if e is not None else "None"


class CombatEnv(LightspeedEnv):
    """覆写 _advance：在每次 playout 边界采集逐战斗数据。"""
    def __init__(self, mcts_simulations=20000):
        super().__init__(mcts_simulations=mcts_simulations)
        self.combats = []          # 本局战斗列表
        self._run_deaths = []      # 调试

    def _advance(self):
        gc = self.gc
        for _ in range(_MAX_ADVANCE_ROUNDS):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            pre_screen = gc.screen_state.name
            pre_enc = enc_name(gc)
            pre_hp = gc.cur_hp
            pre_room = getattr(gc.cur_room, "name", str(gc.cur_room))
            pre_act = gc.act
            pre_floor = gc.floor_num
            self._agent.playout(gc)
            post_hp = gc.cur_hp
            post_outcome = gc.outcome.name
            if pre_screen == "BATTLE":
                self.combats.append({
                    "encounter": pre_enc,
                    "floor": pre_floor,
                    "act": pre_act,
                    "room_type": pre_room,
                    "hp_before": pre_hp,
                    "hp_after": post_hp,
                    "hp_loss": pre_hp - post_hp,
                    "lethal": (gc.outcome == sts.GameOutcome.PLAYER_LOSS),
                    "outcome_after": post_outcome,
                })
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            if sts.get_all_actions(gc):
                return


def main():
    t0 = time.time()
    print(f"[init] 载入 {CKPT} (obs={OBS_DIM}, cand={DESC_VEC_DIM}, hidden={HIDDEN})", flush=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=HIDDEN)
    scorer.load_state_dict(sd)
    scorer.eval()
    policy = _GreedyPolicy(scorer)

    env = CombatEnv(mcts_simulations=MCTS)
    seeds = make_eval_seeds(100)[:N]
    print(f"[init] {N} 个留出种子（前3: {seeds[:3]}），MCTS={MCTS} greedy", flush=True)

    games = []          # 每局：{seed, win, final_floor, final_act, combats:[...]}
    for gi, seed in enumerate(seeds):
        env.combats = []
        obs = env.reset(seed)
        done = False
        steps = 0
        while not done and steps < 500:
            cands = env.candidates()
            idx = policy.choose(obs, cands) if cands else -1
            res = env.step(idx)
            obs, done = res.obs, res.done
            steps += 1
        win = (obs.hp > 0 and env.gc.outcome == sts.GameOutcome.PLAYER_VICTORY)
        rec = {
            "seed": seed,
            "win": win,
            "final_floor": obs.floor,
            "final_act": env.gc.act,
            "final_hp": obs.hp,
            "outcome": env.gc.outcome.name,
            "combats": [dict(c) for c in env.combats],
        }
        games.append(rec)
        n_combat = len(rec["combats"])
        print(f"  [{gi+1}/{N}] seed={seed} win={win} floor={obs.floor} "
              f"act={env.gc.act} hp={obs.hp} 战斗数={n_combat} "
              f"({time.time()-t0:.0f}s)", flush=True)
        # 前 3 局打印原始战斗记录（口径确认）
        if gi < 3:
            print(f"    >>> 第{gi+1}局原始战斗记录：")
            for c in rec["combats"]:
                tag = " [致死]" if c["lethal"] else ""
                print(f"      {c['encounter']:18s} f{c['floor']:2d} A{c['act']} "
                      f"{c['room_type']:8s} hp {c['hp_before']:3d}->{c['hp_after']:3d} "
                      f"loss={c['hp_loss']:3d}{tag}")

    # 落盘
    os.makedirs("logs", exist_ok=True)
    payload = {"ckpt": CKPT, "mcts": MCTS, "n": N, "seeds": seeds, "games": games}
    json.dump(payload, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    wins = sum(1 for g in games if g["win"])
    print(f"\n[done] {N} 局完成，通关 {wins}/{N} = {wins/N:.1%}，"
          f"总耗时 {time.time()-t0:.0f}s，原始数据 -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
