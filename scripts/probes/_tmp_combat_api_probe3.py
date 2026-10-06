# -*- coding: utf-8 -*-
"""探查是否有"逐回合"驱动战斗的 API：BattleContext / Agent 方法 / gc 序列化。"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import slaythespire as sts

print("=== sts 顶层符号 ===")
print([x for x in dir(sts) if not x.startswith("_")])

print("\n=== Agent 实例方法/属性 ===")
ag = sts.Agent()
print([x for x in dir(ag) if not x.startswith("_")])

print("\n=== BattleContext 构造与方法 ===")
try:
    bc = sts.BattleContext
    print("BattleContext:", [x for x in dir(bc) if not x.startswith("_")])
except Exception as e:
    print("no BattleContext:", e)

print("\n=== BattleSearcher 构造与方法 ===")
try:
    bs = sts.BattleSearcher
    print("BattleSearcher:", [x for x in dir(bs) if not x.startswith("_")])
except Exception as e:
    print("no BattleSearcher:", e)

print("\n=== GameContext 中与 battle / json / state / serialize 相关 ===")
gcsyms = [x for x in dir(sts.GameContext) if not x.startswith("_")]
print(gcsyms)
for kw in ("json","state","battle","serialize","to_dict","snapshot","clone","copy"):
    hits = [x for x in gcsyms if kw in x.lower()]
    if hits:
        print("  hit[%s]:" % kw, hits)

print("\n=== GameContext 实例方法签名（help）===")
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 123, 0)
for m in gcsyms:
    if any(k in m.lower() for k in ("json","state","battle","serialize","clone","copy","get_","to_")):
        try:
            print("  ", m, "->", getattr(gc, m).__doc__)
        except Exception as e:
            print("  ", m, "ERR", e)
