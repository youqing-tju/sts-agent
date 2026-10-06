# -*- coding: utf-8 -*-
"""测试 pause_on_combat 是否能让 playout 逐回合推进（每回合返回一次）。"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import slaythespire as sts

MCTS = 2000
SEED = 20260831 + 5

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, SEED, 0)
agent = sts.Agent()
agent.simulation_count_base = MCTS
agent.pause_on_noncombat = True
agent.pause_on_combat = True   # 期望：战斗中每回合暂停

def enc_name():
    e = getattr(gc, "encounter", None)
    return e.name if e is not None else "None"

# 推进到第一个战斗
turn = 0
combat_log = []
while gc.outcome == sts.GameOutcome.UNDECIDED and turn < 300:
    acts = sts.get_all_actions(gc)
    if acts:
        a = acts[0]
        if not a.is_valid(gc):
            a = next((x for x in acts if x.is_valid(gc)), acts[0])
        a.execute(gc)
        if gc.screen_state.name == "BATTLE":
            # 逐回合 playout
            t = 0
            traj = [gc.cur_hp]
            while gc.screen_state.name == "BATTLE" and gc.outcome == sts.GameOutcome.UNDECIDED and t < 200:
                agent.playout(gc)
                traj.append(gc.cur_hp)
                t += 1
            combat_log.append((enc_name(), traj))
            print(f"[combat {len(combat_log)}] enc={enc_name()} turns={t} hp_traj={traj} "
                  f"post_scr={gc.screen_state.name} outcome={gc.outcome.name}")
            if len(combat_log) >= 4:
                break
    else:
        agent.playout(gc)
    turn += 1

print("\n首个战斗的逐回合 hp 轨迹如上（若 turns>1 且每回合返回一次，则 pause_on_combat 生效）。")
