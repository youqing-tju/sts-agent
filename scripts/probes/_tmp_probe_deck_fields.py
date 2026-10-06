# -*- coding: utf-8 -*-
"""奖励塑形探针：确认 env.gc.deck / env.gc.relics 的实际元素结构。

只读分析脚本，跑 1 局（随机策略即可，只为拿到终局状态）。
目标：确定升级卡数字段名（upgrades / upgrade / upgraded）与遗物元素结构。
"""
import os, sys, random
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import slaythespire as sts
from env.lightspeed_env import LightspeedEnv

torch = None  # 不需要 scorer，随机策略即可


def main():
    env = LightspeedEnv(mcts_simulations=2000)
    obs = env.reset(12345)
    done = False
    steps = 0
    while not done and steps < 500:
        cands = env.candidates()
        idx = random.randrange(len(cands)) if cands else -1
        res = env.step(idx)
        obs, done = res.obs, res.done
        steps += 1

    gc = env.gc
    print("=== 终局 ===", flush=True)
    print("outcome:", gc.outcome.name, "| final_floor:", obs.floor, flush=True)

    deck = list(gc.deck)
    print(f"\n=== deck: len={len(deck)} ===", flush=True)
    if deck:
        c = deck[0]
        print("card.dir:", [a for a in dir(c) if not a.startswith("_")], flush=True)
        print("card.id:", repr(c.id), "-> .name:", c.id.name, flush=True)
        # 升级相关字段逐个试
        for attr in ("upgrades", "upgrade", "upgraded", "timesUpgraded", "times_upgraded"):
            print(f"card.{attr}:", repr(getattr(c, attr, "<MISSING>")), flush=True)
        # dict 兼容写法（万一绑定返回 dict）
        if isinstance(c, dict):
            print("card is dict; keys:", list(c.keys()), flush=True)
        # 打印几张卡的升级字段真实值
        print("\n前 8 张卡: name / upgraded / upgrades:", flush=True)
        for cc in deck[:8]:
            up = getattr(cc, "upgraded", None)
            ups = getattr(cc, "upgrades", None)
            print(f"  {cc.id.name:24s} upgraded={up!r} upgrades={ups!r}", flush=True)
        # 统计升级张数（三种口径）
        n_up_bool = sum(1 for cc in deck if getattr(cc, "upgraded", False))
        n_up_cnt = sum(1 for cc in deck if (getattr(cc, "upgrade_count", 0) or 0) > 0)
        print(f"\n升级卡数(布尔 upgraded): {n_up_bool}", flush=True)
        print(f"升级卡数(计数 upgrade_count>0): {n_up_cnt}", flush=True)
        print("逐卡: name / upgraded / upgrade_count:", flush=True)
        for cc in deck:
            print(f"  {cc.id.name:24s} upgraded={cc.upgraded!r} upgrade_count={getattr(cc, 'upgrade_count', '?')!r}", flush=True)

    relics = list(gc.relics)
    print(f"\n=== relics: len={len(relics)} ===", flush=True)
    if relics:
        r = relics[0]
        print("relic.dir:", [a for a in dir(r) if not a.startswith("_")], flush=True)
        print("relic.id:", repr(r.id), "-> .name:", r.id.name, flush=True)
        print("relic names:", [rr.id.name for rr in relics], flush=True)


if __name__ == "__main__":
    main()
