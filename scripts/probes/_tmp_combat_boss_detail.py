# -*- coding: utf-8 -*-
"""Pass 2：对 2-3 个失败的 Boss 战做逐回合 HP 复现。

方法：对每个目标实例（seed + 遭遇 + 楼层），用 BossDetailEnv 重跑该 seed。
BossDetailEnv 覆写 _advance：当进入目标 Boss 战（pre_screen==BATTLE 且遭遇/楼层匹配）的瞬间，
用 BattleContext.init_from_gc(gc) 拷贝隔离战斗，再用 BattleSearcher(bc, MCTS).best_action
逐回合驱动，记录我方/敌方每回合 HP（隔离副本，不影响真实对局），之后照常 playout 推进真实战斗。

输出：logs/combat_boss_turnbytrun.json（每个实例的完整逐回合轨迹）。
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import slaythespire as sts
from collections import defaultdict
from env.lightspeed_env import LightspeedEnv, _MAX_ADVANCE_ROUNDS
from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer
from training.run_training import _GreedyPolicy

CKPT = "checkpoints/v10_seg3.pt"
MCTS = 20000
HIDDEN = (512, 256)
K = int(os.environ.get("K_BOSS", "3"))
RAW = os.path.join(ROOT, "logs", "combat_raw.json")
OUT = os.path.join(ROOT, "logs", "combat_boss_turnbytrun.json")


def enc_name(gc):
    e = getattr(gc, "encounter", None)
    return e.name if e is not None else "None"


def resim_boss(gc, mcts):
    """隔离复现 Boss 战逐回合 HP。返回 [{php, pmax, mon_hp:[...]}, ...]。"""
    try:
        bc = sts.BattleContext()
        bc.init_from_gc(gc)
    except Exception as e:
        print("   init_from_gc 失败:", e); return None
    traj = []
    t = 0
    searcher = sts.BattleSearcher
    while not bc.is_terminal() and t < 400:
        action = searcher(bc, mcts).best_action
        if action is None:
            break
        try:
            if callable(action):
                action(bc)
            else:
                action.execute(bc)
        except Exception as e:
            print("   apply 失败:", e); break
        mon_hp = [getattr(m, "hp", None) for m in bc.monsters]
        php = getattr(bc.player, "hp", None)
        pmax = getattr(bc.player, "max_hp", None)
        traj.append({"php": php, "pmax": pmax, "mon_hp": mon_hp, "turn": bc.turn})
        t += 1
    return traj


class BossDetailEnv(LightspeedEnv):
    def __init__(self, mcts_simulations, target):
        super().__init__(mcts_simulations=mcts_simulations)
        self._target = target            # (encounter, floor)
        self.captured = None

    def _advance(self):
        gc = self.gc
        for _ in range(_MAX_ADVANCE_ROUNDS):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            pre_screen = gc.screen_state.name
            pre_enc = enc_name(gc)
            pre_floor = gc.floor_num
            if (pre_screen == "BATTLE" and pre_enc == self._target[0]
                    and pre_floor == self._target[1] and self.captured is None):
                self.captured = resim_boss(gc, self._agent.simulation_count_base)
            self._agent.playout(gc)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            if sts.get_all_actions(gc):
                return


def main():
    data = json.load(open(RAW, encoding="utf-8"))
    by_enc = defaultdict(list)
    for g in data["games"]:
        for c in g["combats"]:
            if c["room_type"] == "BOSS" and c["lethal"]:
                by_enc[c["encounter"]].append({"seed": g["seed"], "encounter": c["encounter"],
                                               "floor": c["floor"], "act": c["act"],
                                               "hp_before": c["hp_before"], "hp_after": c["hp_after"]})
    ranked = sorted(by_enc.items(),
                    key=lambda kv: (-len(kv[1]), -max(x["hp_before"] - x["hp_after"] for x in kv[1])))
    targets_meta = []
    for enc, lst in ranked[:K]:
        best = max(lst, key=lambda x: x["hp_before"] - x["hp_after"])
        targets_meta.append(best)
    print(f"挑选 {len(targets_meta)} 个失败 Boss 实例做逐回合复现：")
    for t in targets_meta:
        print(f"  seed={t['seed']} {t['encounter']} f{t['floor']} A{t['act']} "
              f"hp {t['hp_before']}->{t['hp_after']}")

    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=HIDDEN)
    scorer.load_state_dict(sd); scorer.eval()
    policy = _GreedyPolicy(scorer)

    results = []
    for t in targets_meta:
        env = BossDetailEnv(mcts_simulations=MCTS, target=(t["encounter"], t["floor"]))
        obs = env.reset(t["seed"])
        done = False; steps = 0
        while not done and steps < 500:
            cands = env.candidates()
            idx = policy.choose(obs, cands) if cands else -1
            res = env.step(idx)
            obs, done = res.obs, res.done
            steps += 1
            if env.captured is not None and done:
                break
            if env.captured is not None:
                break   # 已捕获目标 Boss 轨迹，无需继续
        turns = env.captured
        results.append({"target": t, "turns": turns})
        if turns:
            print(f"\n>>> {t['encounter']} f{t['floor']} 复现 {len(turns)} 回合：")
            for i, row in enumerate(turns):
                print(f"   t{i:2d} 我方HP={row['php']:3d}/{row['pmax']} 敌方HP={row['mon_hp']}")
        else:
            print(f"\n[miss] seed={t['seed']} 未捕获 {t['encounter']}")

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n# 逐回合轨迹已写入 {OUT}")


if __name__ == "__main__":
    main()
