"""探测各决策屏在模拟器里的候选结构（kind / label / 数量）。

用途：决定"扩展 BC 到全部决策屏"时，每屏的候选该用模拟器枚举、还是手工构造。
"""
import os
import sys
import collections

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.getcwd())

from env.lightspeed_env import LightspeedEnv

N_GAMES = int(os.environ.get("N_GAMES", "8"))
MAX_STEPS = 400

seen = collections.defaultdict(lambda: {"n_seen": 0, "counts": collections.Counter(),
                                        "samples": []})
env = LightspeedEnv(mcts_simulations=2000)

for s in range(N_GAMES):
    obs = env.reset(1000 + s)
    done, steps = False, 0
    while not done and steps < MAX_STEPS:
        cands = env.candidates()
        if cands:
            key = (obs.screen, tuple(sorted({c.kind for c in cands})))
            rec = seen[key]
            rec["n_seen"] += 1
            rec["counts"][len(cands)] += 1
            if len(rec["samples"]) < 5:
                rec["samples"].append([c.label for c in cands])
        res = env.step(cands[0].idx if cands else -1)
        obs, done = res.obs, res.done
        steps += 1

print(f"跑完 {N_GAMES} 局，遇到的屏/候选类型组合：\n")
for (screen, kinds), rec in sorted(seen.items(), key=lambda x: -x[1]["n_seen"]):
    print(f"=== screen={screen} | kinds={kinds} | 出现 {rec['n_seen']} 次 "
          f"| 候选数分布 {dict(rec['counts'])}")
    for smp in rec["samples"][:3]:
        print(f"      {smp}")
