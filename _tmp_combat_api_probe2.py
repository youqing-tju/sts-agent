# -*- coding: utf-8 -*-
"""精确探测战斗步进行为：
- 单局、低预算，手动驱动：循环 get_all_actions + execute/playout，打印每步
  screen / encounter / cur_hp / 动作数 / outcome，直到看清一场战斗由多少次 playout 组成。
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import slaythespire as sts

MCTS = 500
SEED = 20260831 + 5

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, SEED, 0)
agent = sts.Agent()
agent.simulation_count_base = MCTS
agent.pause_on_noncombat = True

def enc_name():
    e = getattr(gc, "encounter", None)
    return e.name if e is not None else "None"

def act_name():
    try:
        return gc.act
    except Exception:
        return "?"

# 先把游戏推进到第一个战斗
print("seed=%d mcts=%d" % (SEED, MCTS))
combat_steps = []
in_combat = False
total = 0
combat_idx = 0
while gc.outcome == sts.GameOutcome.UNDECIDED and total < 400:
    acts = sts.get_all_actions(gc)
    if acts:
        # 宏观决策点：选第一个合法动作
        a = acts[0]
        if not a.is_valid(gc):
            a = next((x for x in acts if x.is_valid(gc)), acts[0])
        pre_hp = gc.cur_hp
        pre_scr = gc.screen_state.name
        pre_enc = enc_name()
        a.execute(gc)
        # execute 后若进入战斗，playout 会自动打完
        if gc.screen_state.name == "BATTLE":
            # 进入战斗：连续 playout 直到离开 BATTLE
            csteps = []
            csteps.append(("enter", pre_enc, gc.cur_hp, gc.max_hp))
            guard = 0
            while gc.screen_state.name == "BATTLE" and gc.outcome == sts.GameOutcome.UNDECIDED and guard < 200:
                ph = gc.cur_hp
                agent.playout(gc)
                csteps.append(("playout", gc.screen_state.name, gc.cur_hp, gc.max_hp,
                               len(sts.get_all_actions(gc))))
                guard += 1
            combat_steps.append((combat_idx, pre_enc, pre_scr, pre_hp, csteps))
            combat_idx += 1
            print(f"[combat {combat_idx}] enc={pre_enc} pre_hp={pre_hp} -> "
                  f"playouts={len(csteps)-1} post_hp={gc.cur_hp} post_scr={gc.screen_state.name}")
            for s in csteps:
                print("    ", s)
        total += 1
    else:
        # 无动作：playout 推进（理论上不该在宏观层出现）
        agent.playout(gc)
        total += 1
        if total % 20 == 0:
            print("  (no-act playout) scr=%s enc=%s hp=%d" % (gc.screen_state.name, enc_name(), gc.cur_hp))

print("\n总 step 数:", total, "战斗数:", len(combat_steps))
