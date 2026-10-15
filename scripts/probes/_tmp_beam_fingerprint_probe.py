# -*- coding: utf-8 -*-
"""探测 pybind11 绑定暴露的字段 + 分解 fingerprint() 的性能热点。

目的：为 P0-①「fingerprint 占单节点 98% 耗时」找可优化的点。
用法：.venv/Scripts/python.exe scripts/probes/_tmp_beam_fingerprint_probe.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts  # noqa: E402

import combat_beam  # noqa: E402
from tests.test_combat_brain import _combat_state  # noqa: E402


def make_bc():
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(json.dumps(_combat_state()))
    bc = sts.BattleContext()
    # 两种构造方式都试
    try:
        bc.init_from_gc(gc)
        return bc, "init_from_gc"
    except Exception as e:
        bc = sts.BattleContext()
        bc.init_from_json(gc, json.dumps(_combat_state()))
        return bc, "init_from_json(%s)" % e


def public_attrs(obj, cap=200):
    return [a for a in dir(obj) if not a.startswith("_")][:cap]


def main():
    bc, how = make_bc()
    print("== bc 构造方式:", how)
    print("== bc attrs:", public_attrs(bc))
    print()
    print("== player attrs:", public_attrs(bc.player))
    print()
    if bc.monsters:
        m = bc.monsters[0]
        print("== monster attrs:", public_attrs(m))
        print("   monster.name =", repr(m.name))
    print()
    if bc.hand:
        c = bc.hand[0]
        print("== card attrs:", public_attrs(c))
        print("   card.id attrs:", public_attrs(c.id))
        print("   card.id.name =", repr(c.id.name))
        for probe in ("value", "int", "index", "ordinal"):
            if hasattr(c.id, probe):
                try:
                    print("   card.id.%s = %r" % (probe, getattr(c.id, probe)))
                except Exception as e:
                    print("   card.id.%s -> %s" % (probe, e))
        print("   len(bc.hand) =", len(bc.hand), " len(bc.draw_pile) =", len(bc.draw_pile),
              " len(bc.discard_pile) =", len(bc.discard_pile),
              " len(bc.exhaust_pile) =", len(bc.exhaust_pile))
    print()

    # ---------------- 基准分解 ----------------
    N = 2000
    print("== 基准（N=%d 次） ==" % N)

    def bench(label, fn, n=N):
        # 预热
        for _ in range(50):
            fn()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        dt = (time.perf_counter() - t0) / n * 1e6  # us
        print("  %-42s %8.2f us/次" % (label, dt))
        return dt

    total = bench("fingerprint() 整体", lambda: combat_beam.fingerprint(bc))
    bench("hash(fingerprint(bc))", lambda: hash(combat_beam.fingerprint(bc)))
    bench("BattleContext(bc) 拷贝构造", lambda: sts.BattleContext(bc))

    p = bc.player
    bench("p.hp..p.artifact 9 个属性", lambda: (p.hp, p.block, p.energy, p.strength,
                                               p.dexterity, p.focus, p.artifact))
    bench("tuple(c.id.name for c in draw_pile)", lambda: tuple(c.id.name for c in bc.draw_pile))
    bench("hand: (name,cost,upgraded) 未排序", lambda: [(c.id.name, c.cost_for_turn, bool(c.upgraded))
                                                        for c in bc.hand])
    bench("hand: sorted(...)", lambda: sorted((c.id.name, c.cost_for_turn, bool(c.upgraded))
                                              for c in bc.hand))
    bench("discard: sorted(name)", lambda: sorted(c.id.name for c in bc.discard_pile))
    bench("monsters: 5 元组含 intent", lambda: [(m.name, m.hp, m.block,
                                                 combat_beam._intent_damage(m, bc),
                                                 bool(m.is_alive)) for m in bc.monsters])
    bench("monsters: 不含 intent", lambda: [(m.name, m.hp, m.block, bool(m.is_alive))
                                            for m in bc.monsters])
    bench("  └ _intent_damage 单次", lambda: [combat_beam._intent_damage(m, bc) for m in bc.monsters])
    bench("evaluate(bc)", lambda: combat_beam.evaluate(bc))
    bench("legal_actions(bc)", lambda: combat_beam.legal_actions(bc))

    # ---------------- 逐元素细测：定位 c.id.name 到底贵在哪 ----------------
    print()
    print("== 逐元素细测（hand 5 张牌） ==")
    bench("c.id 对象本身（[c.id for c in hand]）", lambda: [c.id for c in bc.hand])
    bench("c.id.value 整数（★候选替代）", lambda: [c.id.value for c in bc.hand])
    bench("c.id.name 字符串（当前实现）", lambda: [c.id.name for c in bc.hand])
    bench("str(c.id)（对照）", lambda: [str(c.id) for c in bc.hand])
    bench("c.cost_for_turn", lambda: [c.cost_for_turn for c in bc.hand])
    bench("bool(c.upgraded)", lambda: [bool(c.upgraded) for c in bc.hand])
    bench("c.type", lambda: [c.type for c in bc.hand])
    bench("hand 整体换 value", lambda: sorted((c.id.value, c.cost_for_turn, bool(c.upgraded))
                                              for c in bc.hand))
    bench("draw 整体换 value", lambda: tuple(c.id.value for c in bc.draw_pile))
    print()
    print("== 单张卡各字段对比（1 张牌，循环 20000 次） ==")
    c0 = bc.hand[0]
    bench("[c0.id]", lambda: [c0.id], n=20000)
    bench("[c0.id.value]", lambda: [c0.id.value], n=20000)
    bench("[c0.id.name]", lambda: [c0.id.name], n=20000)
    bench("[type(c0.id)]", lambda: [type(c0.id)], n=20000)
    print("   type(card.id) =", type(c0.id))
    print("   card.id == CardId? dir 里有没有 __hash__ __eq__:", 
          hasattr(c0.id, "__hash__"), hasattr(c0.id, "__eq__"))


if __name__ == "__main__":
    main()
