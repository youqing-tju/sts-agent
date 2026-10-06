# -*- coding: utf-8 -*-
"""验证：用 BattleContext.init_from_gc + BattleSearcher 对 Boss 战做逐回合复现，
记录我方/敌方每回合 HP。复现是隔离的副本，不影响真实 gc。"""
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

def enc_name():
    e = getattr(gc, "encounter", None)
    return e.name if e is not None else "None"

def mon_hps(bc):
    out = []
    for m in bc.monsters:
        out.append(getattr(m, "hp", None))
    return out

turn = 0
boss_seen = 0
while gc.outcome == sts.GameOutcome.UNDECIDED and turn < 400:
    acts = sts.get_all_actions(gc)
    if acts:
        a = acts[0]
        if not a.is_valid(gc):
            a = next((x for x in acts if x.is_valid(gc)), acts[0])
        a.execute(gc)
        if gc.screen_state.name == "BATTLE" and getattr(gc, "boss", False):
            # 进入 Boss 战：隔离复现逐回合
            bc = sts.BattleContext()
            bc.init_from_gc(gc)
            print(f"\n>>> Boss 战 enc={enc_name()} act={gc.act} floor={gc.floor_num} "
                  f"玩家HP={gc.cur_hp}/{gc.max_hp} 怪物={mon_hps(bc)}")
            searcher = sts.BattleSearcher
            traj = []
            t = 0
            while (not bc.is_terminal) and t < 300:
                act = searcher(bc, MCTS).best_action
                if act is None:
                    print("   best_action=None, 终止"); break
                try:
                    act(bc) if callable(act) else bc.apply_action(act)
                except Exception as e:
                    print("   apply err:", e); break
                traj.append((bc.player.hp if hasattr(bc.player,'hp') else '?',
                             mon_hps(bc), bc.turn))
                t += 1
            print(f"   复现回合={t} 末态 玩家HP={traj[-1][0] if traj else '?'} 怪物={traj[-1][1] if traj else '?'}")
            for i, (php, mhp, tn) in enumerate(traj[:25]):
                print(f"     t{i}: 我方HP={php} 敌方HP={mhp}")
            boss_seen += 1
            if boss_seen >= 2:
                break
    else:
        agent.playout(gc)
    turn += 1

print("\nBoss 复现测试完成，boss_seen=", boss_seen)
