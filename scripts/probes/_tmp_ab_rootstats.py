"""A/B 验证：ISMCTS 旧逻辑（best_action 投票）vs 新逻辑（合并根边统计）。

用真机帧跑 15 帧，比较两者选择的命令，并统计 END 次数变化。
预算用 1000×16（低预算下差异更容易观察到）。
"""
import json
import logging
import os
import random
import sys
from collections import OrderedDict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

import slaythespire as sts
from main import _sanitize, SimBrain
from models.power_alias import alias_power_ids
from combat_brain import CombatBrain

SIMS = 1000
K = 16

brain = SimBrain()

frames = []
for L in ("0911_013551", "0911_022209", "0911_025542"):
    p = f"logs/realtime/{L}_states.jsonl"
    if not os.path.exists(p):
        continue
    for l in open(p, encoding="utf-8"):
        try:
            o = json.loads(l)
        except Exception:
            continue
        gs = o.get("game_state") or {}
        cs = gs.get("combat_state") or {}
        if (gs.get("screen_type") == "NONE" and (cs.get("hand") or [])
                and (cs.get("player") or {}).get("energy", 0) > 0):
            frames.append((L, o))
    if len(frames) >= 60:
        break

random.seed(7)
sample = random.sample(frames, min(15, len(frames)))
print(f"候选帧 {len(frames)}，抽样 {len(sample)}", flush=True)


def old_logic(frame):
    """复刻旧逻辑：K 个未来各自 best_action 投票，取最多票。"""
    clean = _sanitize(frame)
    alias_power_ids(clean)
    try:
        gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
        gc.init_from_json(json.dumps(clean))
        bc0 = sts.BattleContext()
        bc0.init_from_json(gc, json.dumps(clean))
    except Exception as e:
        return f"<灌入失败 {e}>"
    cb = CombatBrain(simulation_count=SIMS, determinizations=K)
    base_seed = cb._base_seed(frame, gc)
    votes = OrderedDict()
    winners = {}
    for k in range(K):
        bc_k = sts.BattleContext(bc0)
        bc_k.randomize_rng(base_seed + k)
        a = sts.BattleSearcher(bc_k, SIMS).best_action
        if callable(a):
            a = a()
        if a is None:
            continue
        key = cb._vote_key(a, bc0)
        votes[key] = votes.get(key, 0) + 1
        winners.setdefault(key, a)
    if not votes:
        return None
    wk = max(votes, key=lambda kk: votes[kk])
    return cb._to_comm(winners[wk], frame)


agree = 0
both = 0
end_old = 0
end_new = 0
for i, (L, fr) in enumerate(sample):
    brain._refresh_sim(fr)
    cb = CombatBrain(simulation_count=SIMS, determinizations=K)
    cmd_new = cb.decide(fr, brain.gc)
    cmd_old = old_logic(fr)
    if cmd_old and cmd_new:
        both += 1
        if cmd_old == cmd_new:
            agree += 1
    if str(cmd_old) == "END":
        end_old += 1
    if str(cmd_new) == "END":
        end_new += 1
    print(f"  [{i+1:2d}] {L}: old={cmd_old!r:16s} new={cmd_new!r}", flush=True)

print(f"\n一致 {agree}/{both}；END 次数 old={end_old} new={end_new}", flush=True)
if cb.last_root_stats:
    top = sorted(cb.last_root_stats.items(), key=lambda kv: -kv[1][0])[:5]
    print("最后一帧根边统计 Top5:", [(str(k), v[0], round(v[1], 1)) for k, v in top], flush=True)
