# -*- coding: utf-8 -*-
"""假说区分探针（boss-hypothesis 任务）。

目标：区分两个假说
  A：问题在「路上的失血」——进 Act3 Boss 前血线就被压低，故 Boss 战打不过是结果不是原因
  B：问题在「Boss 战打法」——MCTS 在满血/正常血线时也会输 Boss 战

方法：用 v10_seg3.pt + LightspeedEnv(MCTS 20000) + greedy 跑 50 局（种子 make_eval_seeds(200)[:50]）。
每局采集：
  1) 血线轨迹：进入 Act2 / 进入 Act3 / 进入 Act3 Boss 前（=Act3 BOSS 战的 hp_before）/ 终局 的 hp,max_hp
  2) 药水轨迹：每步 potion_count（据此算消耗量=负向 delta 之和，及终局剩余）
  3) 逐战斗记录（act/encounter/room_type/hp_before/hp_after/hp_loss/lethal）——用于 Act3 失血来源

只读分析：仅新增本脚本，不改任何生产代码。单进程。
落盘：logs/hypo_raw.json（原始），报告由 _tmp_hypo_report.py 读取生成。
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
OUT = os.path.join(ROOT, "logs", "hypo_raw.json")

# Act3 Boss 遭遇（普通胜利路径；THE_HEART 为隐藏第4幕，单独标记）
ACT3_BOSSES = {"DONU_AND_DECA", "AWAKENED_ONE", "TIME_EATER"}


def enc_name(gc):
    e = getattr(gc, "encounter", None)
    return e.name if e is not None else "None"


class HypoEnv(LightspeedEnv):
    """覆写 _advance：在每次 playout 边界采集逐战斗数据 + 记录进幕血线。"""
    def __init__(self, mcts_simulations=20000):
        super().__init__(mcts_simulations=mcts_simulations)
        self.combats = []
        self.act_entries = {}      # act -> (cur_hp, max_hp) 首次见到的幕内血线
        self._prev_act = None

    def _advance(self):
        gc = self.gc
        for _ in range(_MAX_ADVANCE_ROUNDS):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            # 记录进幕血线（首次见到该 act）
            if gc.act not in self.act_entries:
                self.act_entries[gc.act] = (gc.cur_hp, gc.max_hp)
            pre_screen = gc.screen_state.name
            pre_enc = enc_name(gc)
            pre_hp = gc.cur_hp
            pre_room = getattr(gc.cur_room, "name", str(gc.cur_room))
            pre_act = gc.act
            pre_floor = gc.floor_num
            pre_max = gc.max_hp
            self._agent.playout(gc)
            post_hp = gc.cur_hp
            if pre_screen == "BATTLE":
                self.combats.append({
                    "encounter": pre_enc,
                    "floor": pre_floor,
                    "act": pre_act,
                    "room_type": pre_room,
                    "hp_before": pre_hp,
                    "hp_after": post_hp,
                    "hp_loss": pre_hp - post_hp,
                    "max_hp": pre_max,
                    "lethal": (gc.outcome == sts.GameOutcome.PLAYER_LOSS),
                })
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            if sts.get_all_actions(gc):
                return


def main():
    t0 = time.time()
    print(f"[init] 载入 {CKPT}", flush=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=HIDDEN)
    scorer.load_state_dict(sd)
    scorer.eval()
    policy = _GreedyPolicy(scorer)

    env = HypoEnv(mcts_simulations=MCTS)
    seeds = make_eval_seeds(200)[:N]
    print(f"[init] {N} 种子（前3: {seeds[:3]}），MCTS={MCTS} greedy", flush=True)

    games = []
    for gi, seed in enumerate(seeds):
        env.combats = []
        env.act_entries = {}
        env._prev_act = None
        obs = env.reset(seed)
        potion_series = [env.gc.potion_count]   # 开局药水数
        step_deltas = []                          # 每步 hp_delta
        potion_deltas = []                        # 每步 potion_count 变化
        done = False
        steps = 0
        while not done and steps < 500:
            cands = env.candidates()
            idx = policy.choose(obs, cands) if cands else -1
            pre_p = env.gc.potion_count
            pre_hp = env.gc.cur_hp
            res = env.step(idx)
            post_p = env.gc.potion_count
            potion_series.append(post_p)
            potion_deltas.append(post_p - pre_p)
            step_deltas.append(env.gc.cur_hp - pre_hp)
            obs, done = res.obs, res.done
            steps += 1

        gc = env.gc
        win = (obs.hp > 0 and gc.outcome == sts.GameOutcome.PLAYER_VICTORY)

        # 进入 Act3 Boss 前血线 = Act3 BOSS 战 hp_before（若有）
        act3_boss = None
        for c in env.combats:
            if c["act"] == 3 and (c["room_type"] == "BOSS" or c["encounter"] in ACT3_BOSSES):
                act3_boss = c
                break
        # Act1/2 Boss 战（用于上下文，不纳入主比较）
        act1_boss = next((c for c in env.combats
                          if c["act"] == 1 and (c["room_type"] == "BOSS" or c["encounter"] in {"SLIME_BOSS","THE_GUARDIAN","HEXAGHOST"})), None)
        act2_boss = next((c for c in env.combats
                          if c["act"] == 2 and (c["room_type"] == "BOSS" or c["encounter"] in {"COLLECTOR","AUTOMATON","CHAMP"})), None)

        # 死亡定位：终局 + 最后一战（lethal）决定死在 boss 还是 road
        death_at_boss = False
        death_at_road = False
        road_death_act = None
        if not win:
            lethal_combats = [c for c in env.combats if c["lethal"]]
            if lethal_combats:
                lc = lethal_combats[-1]
                if lc["act"] == 3 and (lc["room_type"] == "BOSS" or lc["encounter"] in ACT3_BOSSES):
                    death_at_boss = True
                else:
                    death_at_road = True
                    road_death_act = lc["act"]

        # 药水消耗：负向 delta 之和 = 整局用掉的药水数；终局剩余 = potion_series[-1]
        potion_used = sum(d for d in potion_deltas if d < 0)
        potion_gained = sum(d for d in potion_deltas if d > 0)
        potion_remain_final = potion_series[-1]
        peak_potions = max(potion_series)

        rec = {
            "seed": seed,
            "win": win,
            "outcome": gc.outcome.name,
            "final_act": gc.act,
            "final_floor": obs.floor,
            "final_hp": obs.hp,
            "max_hp": obs.max_hp,
            "act_entries": {str(k): list(v) for k, v in env.act_entries.items()},
            "act1_boss_pre": list((act1_boss["hp_before"],)) if act1_boss else None,
            "act2_boss_pre": list((act2_boss["hp_before"],)) if act2_boss else None,
            "act3_boss_pre": [act3_boss["hp_before"], act3_boss["hp_after"]] if act3_boss else None,
            "death_at_boss": death_at_boss,
            "death_at_road": death_at_road,
            "road_death_act": road_death_act,
            "potion_used": potion_used,
            "potion_gained": potion_gained,
            "potion_remain_final": potion_remain_final,
            "peak_potions": peak_potions,
            "combats": [dict(c) for c in env.combats],
            # 诊断：进 act3 时的血线、进 act3 boss 前血线比例（max_hp 取自终局，全程不变）
            "hp_ratio_at_act3_entry": None,
            "hp_ratio_at_act3_boss_pre": None,
        }
        if 3 in env.act_entries and env.act_entries[3][1]:
            rec["hp_ratio_at_act3_entry"] = env.act_entries[3][0] / env.act_entries[3][1]
        else:
            rec["hp_ratio_at_act3_entry"] = None
        if act3_boss and act3_boss["max_hp"]:
            rec["hp_ratio_at_act3_boss_pre"] = act3_boss["hp_before"] / act3_boss["max_hp"]
        else:
            rec["hp_ratio_at_act3_boss_pre"] = None

        games.append(rec)
        n_c = len(rec["combats"])
        abp = rec["act3_boss_pre"]
        print(f"  [{gi+1}/{N}] seed={seed} win={win} A{rec['final_act']} f{obs.floor} "
              f"hp={obs.hp}/{obs.max_hp} 战斗={n_c} 进A3={rec['hp_ratio_at_act3_entry']} "
              f"进Boss={rec['hp_ratio_at_act3_boss_pre']} 死Boss={death_at_boss} 死路={death_at_road} "
              f"药水用={potion_used} 剩={potion_remain_final} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs("logs", exist_ok=True)
    payload = {"ckpt": CKPT, "mcts": MCTS, "n": N, "seeds": seeds, "games": games}
    json.dump(payload, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    wins = sum(1 for g in games if g["win"])
    print(f"\n[done] {N} 局完成，通关 {wins}/{N}={wins/N:.1%}，耗时 {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
