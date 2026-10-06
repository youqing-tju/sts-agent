# -*- coding: utf-8 -*-
"""快速探测 gc 的药水/房间相关属性名，供 _tmp_hypo_run.py 使用。只读。"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import slaythespire as sts
from env.lightspeed_env import LightspeedEnv

env = LightspeedEnv(mcts_simulations=300)
obs = env.reset(20260831 + 1)
gc = env.gc
print("=== dir(gc) 含 potion/room/floor/hp 的键 ===")
for k in dir(gc):
    if not k.startswith("__"):
        if any(t in k.lower() for t in ("potion", "room", "hp", "floor", "act", "screen", "max")):
            print("  ", k)
print("\n=== 尝试读取可能的水药属性 ===")
for attr in ("potion_count", "potion_slots", "potion_capacity", "potions", "get_potion", "potion_list"):
    try:
        v = getattr(gc, attr, "<MISSING>")
        print(f"  gc.{attr} = {type(v).__name__}: {repr(v)[:120]}")
    except Exception as e:
        print(f"  gc.{attr} -> ERR {type(e).__name__}: {e}")

print("\n=== 驱动几步，打印每步 screen/act/floor/room/cur_hp + 药水数 ===")
done, steps = False, 0
def potion_num(g):
    for attr in ("potion_count", "potion_slots"):
        try:
            v = getattr(g, attr, None)
            if v is not None:
                if isinstance(v, (list, tuple)):
                    return len(v)
                return v
        except Exception:
            pass
    return None
while not done and steps < 40:
    cands = env.candidates()
    idx = cands[0].idx if cands else -1
    before_hp = gc.cur_hp
    res = env.step(idx)
    print(f"  step={steps:2d} screen={obs.screen:14s}-> act={gc.act} floor={gc.floor_num} "
          f"room={getattr(gc.cur_room,'name','?')} hp={gc.cur_hp}/{gc.max_hp} potion={potion_num(gc)}")
    obs, done = res.obs, res.done
    steps += 1
print("done")
