import os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import slaythespire as sts
from main import _sanitize
from models.power_alias import alias_power_ids

path="logs/realtime/0911_025542_states.jsonl"
frames=[]
with open(path,encoding="utf-8") as f:
    for line in f:
        line=line.strip()
        if not line: continue
        try: o=json.loads(line)
        except: continue
        gs=o.get("game_state",o)
        cs=gs.get("combat_state")
        if not cs: continue
        mons=cs.get("monsters",[])
        if not mons: continue
        ids=[m.get("id") for m in mons]
        # skip the bronze automaton boss fight
        if "BronzeAutomaton" in ids: continue
        # require at least one non-gone monster
        if all(m.get("is_gone") for m in mons): continue
        frames.append((ids,o))

print(f"found {len(frames)} non-boss combat frames", flush=True)
import random
random.seed(1)
picks=random.sample(frames, min(3,len(frames)))
for i,(ids,o) in enumerate(picks):
    clean=_sanitize(o); alias_power_ids(clean)
    gc=sts.GameContext(sts.CharacterClass.IRONCLAD,0,0); gc.init_from_json(json.dumps(clean))
    bc=sts.BattleContext(); bc.init_from_json(gc, json.dumps(clean))
    t=time.time(); s=sts.BattleSearcher(bc,500); a=s.best_action
    if callable(a): a=a()
    print(f"frame{i} monsters={ids} 500sims {time.time()-t:.1f}s best={getattr(a,'action_type',a)}", flush=True)
print("REGRESS DONE", flush=True)
