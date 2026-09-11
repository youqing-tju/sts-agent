import os, json, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE")
import slaythespire as sts
from main import _sanitize          # main.py 顶部会重定向 stderr，没关系
from models.power_alias import alias_power_ids
frame=json.load(open("logs/realtime/0911_113720_lastboss.json",encoding="utf-8"))
clean=_sanitize(frame); alias_power_ids(clean)
gc=sts.GameContext(sts.CharacterClass.IRONCLAD,0,0); gc.init_from_json(json.dumps(clean))
bc0=sts.BattleContext(); bc0.init_from_json(gc, json.dumps(clean))
print("init ok", flush=True)
for sims in (500, 5000, 50000):
    t=time.time(); s=sts.BattleSearcher(bc0, sims); a=s.best_action
    if callable(a): a=a()
    print(f"sims={sims} {time.time()-t:.1f}s best={getattr(a,'action_type',a)}", flush=True)
# 再复刻"多随机种子"（这是崩溃易发路径）：
for k in range(8):
    bc_k=sts.BattleContext(bc0); bc_k.randomize_rng(12345+k)
    t=time.time(); sk=sts.BattleSearcher(bc_k, 1000); ak=sk.best_action
    if callable(ak): ak=ak()
    print(f"seed12345+{k} {time.time()-t:.2f}s", flush=True)
print("ALL DONE", flush=True)
