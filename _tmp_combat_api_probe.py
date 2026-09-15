# -*- coding: utf-8 -*-
"""API 探测：确认 gc.encounter 形态 + 战斗中如何读敌方 HP。

用 LightspeedEnv，但覆写 _advance 在每次 playout 前后打印 gc 状态，
并探测可能的敌方 HP 访问方式。单局、低 MCTS 预算，仅看前 ~60 次 playout。
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import slaythespire as sts
from env.lightspeed_env import LightspeedEnv

# ---- 覆写 _advance 做探测 ----
class ProbeEnv(LightspeedEnv):
    def __init__(self, mcts_simulations=300, max_playout=70):
        super().__init__(mcts_simulations=mcts_simulations)
        self._pp_max = max_playout
        self._log = []

    def _advance(self):
        gc = self.gc
        for i in range(self._pp_max):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                self._log.append(("OUTCOME", gc.outcome.name))
                return
            screen = gc.screen_state.name
            enc = getattr(gc, "encounter", "<?>")
            enc_repr = repr(enc)
            enc_name = enc if isinstance(enc, str) else (getattr(enc, "name", None) if enc is not None else None)
            # 尝试读敌方 HP
            enemy_info = self._probe_enemy(gc)
            rec = {
                "i": i, "screen": screen, "enc": enc_name, "enc_repr": enc_repr,
                "floor": gc.floor_num, "act": gc.act,
                "php": gc.cur_hp, "pmax": gc.max_hp,
                "enemy": enemy_info,
            }
            self._log.append(rec)
            self._agent.playout(gc)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                self._log.append(("OUTCOME2", gc.outcome.name))
                return
            if sts.get_all_actions(gc):
                self._log.append(("MACRO", gc.screen_state.name, repr(getattr(gc,"encounter","?"))))
                return

    @staticmethod
    def _probe_enemy(gc):
        out = {}
        for attr in ("battle", "combat", "monsters", "get_monsters", "battle_context"):
            try:
                v = getattr(gc, attr, None)
                if callable(v):
                    v = v()
                out[attr] = type(v).__name__ if not isinstance(v, (list, dict, int, float, str, type(None))) else (f"len={len(v)}" if isinstance(v,(list,dict)) else v)
            except Exception as e:
                out[attr] = f"ERR:{type(e).__name__}"
        return out


def main():
    env = ProbeEnv(mcts_simulations=300, max_playout=80)
    obs = env.reset(20260831 + 1)
    print(">>> reset 后 screen =", obs.screen, "| floor =", obs.floor,
          "| hp =", obs.hp, "/", obs.max_hp)
    print(">>> dir(gc) 片段（含 hp/encounter/room 相关的）:")
    gc = env.gc
    keys = [k for k in dir(gc) if not k.startswith("__")]
    print("   ", keys)
    # 驱动：每次宏观决策点用第一个候选（任何合法动作都行），旨在尽快进入战斗
    done, steps = False, 0
    while not done and steps < 60:
        cands = env.candidates()
        if cands:
            idx = 0  # 选第一个候选，尽快推进
        else:
            idx = -1
        res = env.step(idx)
        obs, done = res.obs, res.done
        steps += 1
    # 打印日志
    print("\n=== playout 日志（前 80）===")
    for rec in env._log:
        if isinstance(rec, tuple):
            print("  ", rec)
            continue
        print(f"  i={rec['i']:2d} scr={rec['screen']:14s} enc={str(rec['enc'])[:28]:28s} "
              f"f{rec['floor']}A{rec['act']} php={rec['php']:3d}/{rec['pmax']} "
              f"enemy={rec['enemy']}")
    print(f"\n总 playout 记录 {len(env._log)} 条，宏观步数 {steps}")

if __name__ == "__main__":
    main()
