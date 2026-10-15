# -*- coding: utf-8 -*-
"""Beam 战斗求解器测试（`combat_beam.py`）。

覆盖四层：
1. **CardId.value ↔ name 双射** —— 指纹改用整数 id 的**前提**（破了就去重语义变了）
2. **fingerprint** —— 稳定性 / 拷贝不变性 / 区分性 / **与字符串实现的等价类划分一致**
3. **行为契约** —— legal_actions / evaluate / BeamSolver.solve（含 energy=0 死锁回归、
   solve 不得改动传入 bc）
4. **性能回归** —— 防止 fingerprint 退回字符串实现（约 2500 µs → 会低于阈值）

背景见 `docs/next-chat-handoff.md` §6 与 `combat_beam.fingerprint` docstring。
"""
import copy
import json
import os
import sys
import time
from collections import defaultdict

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts  # noqa: E402

import combat_beam  # noqa: E402
from combat_beam import BeamSolver, evaluate, fingerprint, legal_actions  # noqa: E402
from tests.test_combat_brain import _combat_state  # noqa: E402


# --------------------------------------------------------------------------
# 构造
# --------------------------------------------------------------------------
def _state(hp=None):
    """`_combat_state()` 的深拷贝，可选覆盖玩家生命。

    ⚠️ 生命必须改**顶层** `game_state.current_hp` —— `init_from_gc` **忽略**
    `combat_state.player.current_hp`（实测：改了它 `bc.player.hp` 仍是原值）。
    """
    st = copy.deepcopy(_combat_state())
    if hp is not None:
        st["game_state"]["current_hp"] = hp
        st["game_state"]["max_hp"] = max(hp, int(st["game_state"]["max_hp"]))
    return st


def _bc(**kw):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(json.dumps(_state(**kw)))
    bc = sts.BattleContext()
    bc.init_from_gc(gc)
    return bc


def _bc_drained():
    """反复出牌直到按不出（能量不足），得到「**只剩 END_TURN**」的局面。

    ⚠️ 不能靠 JSON 直接设 energy —— `init_from_gc` 同样忽略
    `combat_state.player.energy` 与 `player.energy_per_turn`（实测恒为 3）。
    出牌耗尽才是真实触发场景（这也正是「energy=0 死锁」bug 的复现条件）。
    """
    bc = _bc()
    for _ in range(20):
        cards = [a for a in legal_actions(bc) if a[0] == "CARD"]
        if not cards:
            break
        cb = sts.BattleContext(bc)
        combat_beam._make(cards[0]).execute(cb)
        bc = cb
    return bc


def _fp_by_name(bc):
    """**参考实现**：用 `CardId.name`（字符串）的旧指纹。

    与 `fingerprint` 逐位不等（int vs str），但**等价类划分必须一致**。
    """
    p = bc.player
    return (
        bc.turn,
        bc.input_state,
        p.hp, p.block, p.energy, p.strength, p.dexterity, p.focus, p.artifact,
        tuple(sorted((c.id.name, c.cost_for_turn, bool(c.upgraded)) for c in bc.hand)),
        tuple(c.id.name for c in bc.draw_pile),
        tuple(sorted(c.id.name for c in bc.discard_pile)),
        tuple(sorted(c.id.name for c in bc.exhaust_pile)),
        tuple((m.name, m.hp, m.block, combat_beam._intent_damage(m, bc), bool(m.is_alive))
              for m in bc.monsters),
    )


def _collect_states(limit=250):
    """从初始战斗状态**确定性展开**一批可达状态（含终局），用于等价性 / 区分性测试。"""
    out = [_bc()]
    frontier = [out[0]]
    while frontier and len(out) < limit:
        nxt = []
        for b in frontier:
            for act in legal_actions(b):
                cb = sts.BattleContext(b)
                try:
                    combat_beam._make(act).execute(cb)
                except Exception:                       # noqa: BLE001
                    continue
                out.append(cb)
                if not cb.is_terminal():
                    nxt.append(cb)
                if len(out) >= limit:
                    break
            if len(out) >= limit:
                break
        frontier = nxt
    return out


# --------------------------------------------------------------------------
# 1. CardId.value ↔ name 双射（指纹用整数 id 的前提）
# --------------------------------------------------------------------------
def test_cardid_value_name_bijection():
    names = [a for a in dir(sts.CardId) if a.isupper() and not a.startswith("_")]
    pairs = []
    for n in names:
        v = getattr(sts.CardId, n)
        if isinstance(v, sts.CardId):
            pairs.append((int(v.value), v.name))

    assert len(pairs) >= 300, "CardId 常量数异常：%d" % len(pairs)
    assert len({v for v, _ in pairs}) == len(pairs), "CardId.value 出现重复 → 不可作指纹"
    assert len({n for _, n in pairs}) == len(pairs), "CardId.name 出现重复"


# --------------------------------------------------------------------------
# 2. fingerprint 语义
# --------------------------------------------------------------------------
def test_fingerprint_stable_and_copy_invariant():
    """同一状态两次调用一致；**拷贝副本**指纹相同（Fork 不改变状态）。"""
    bc = _bc()
    assert fingerprint(bc) == fingerprint(bc)
    assert fingerprint(bc) == fingerprint(sts.BattleContext(bc))


def test_fingerprint_is_hashable():
    hash(fingerprint(_bc()))          # 不抛异常即可


def test_fingerprint_equivalent_partition_matches_name_impl():
    """🔴 核心：整数实现与字符串实现的**等价类划分逐一致**。

    即 `fp_new(a)==fp_new(b)` ⟺ `fp_old(a)==fp_old(b)`（双向单射）。
    满足则「用 value 替代 name」对去重语义**零影响**。
    """
    states = _collect_states(180)
    assert len(states) > 50, "展开出的状态太少，测试无意义"

    new_to_old = defaultdict(set)
    old_to_new = defaultdict(set)
    for s in states:
        fn, fo = fingerprint(s), _fp_by_name(s)
        new_to_old[fn].add(fo)
        old_to_new[fo].add(fn)

    bad_new = {k: v for k, v in new_to_old.items() if len(v) > 1}
    bad_old = {k: v for k, v in old_to_new.items() if len(v) > 1}
    assert not bad_new, "有 %d 个新指纹把不同旧状态合并了" % len(bad_new)
    assert not bad_old, "有 %d 个旧指纹被拆成了多个新指纹" % len(bad_old)


def test_fingerprint_distinguishes_played_card():
    """打出一张牌后指纹必须变（否则去重会把不同局面当同一态）。"""
    bc = _bc()
    cards = [a for a in legal_actions(bc) if a[0] == "CARD"]
    assert cards, "初始状态应有可出的牌"
    cb = sts.BattleContext(bc)
    combat_beam._make(cards[0]).execute(cb)
    assert fingerprint(cb) != fingerprint(bc)


def test_fingerprint_distinguishes_hp():
    a, b = _bc(hp=72), _bc(hp=60)
    assert fingerprint(a) != fingerprint(b)


# --------------------------------------------------------------------------
# 3. legal_actions / evaluate / solve 行为契约
# --------------------------------------------------------------------------
def test_legal_actions_has_end_turn_and_only_known_types():
    acts = legal_actions(_bc())
    assert ("END_TURN", 0, 0) in acts
    assert all(a[0] in combat_beam.ACTION_TYPE_BY_NAME for a in acts)


def test_legal_actions_drained_has_only_end_turn():
    """把能量打光后只剩 END_TURN（= 死锁 bug 的触发局面）。"""
    bc = _bc_drained()
    assert not bc.is_terminal()
    assert legal_actions(bc) == [("END_TURN", 0, 0)]


def test_evaluate_returns_finite_float_for_open_state():
    s = evaluate(_bc())
    assert isinstance(s, float) and s == s and abs(s) < 1e8


def test_solve_returns_legal_first_action():
    s = BeamSolver(beam_width=2, max_depth=4, max_nodes=200)
    plan = s.solve(_bc())
    assert plan, "solve 不得返回空方案（否则调用方死锁）"
    assert plan[0] in legal_actions(_bc())


def test_solve_no_deadlock_when_only_end_turn():
    """🔴 回归：`best_open` 初值曾误设为 root → 该局面下 solve 返回**空方案** → 调用方死锁。"""
    bc = _bc_drained()
    plan = BeamSolver(beam_width=2, max_depth=4, max_nodes=200).solve(bc)
    assert plan, "solve 不得返回空方案"
    assert plan[0] == ("END_TURN", 0, 0)


def test_solve_does_not_mutate_input():
    """solve 全程在分叉副本上操作，**不得改动传入的 bc**。"""
    bc = _bc()
    before = fingerprint(bc)
    BeamSolver(beam_width=2, max_depth=5, max_nodes=300).solve(bc)
    assert fingerprint(bc) == before


def test_solve_respects_max_nodes_budget():
    s = BeamSolver(beam_width=4, max_depth=30, max_nodes=50)
    s.solve(_bc())
    assert s.last_stats["nodes"] <= 50


def test_solve_stats_shape():
    s = BeamSolver(beam_width=2, max_depth=4, max_nodes=200)
    s.solve(_bc())
    for k in ("nodes", "ms", "depth", "score", "won", "outcome", "plan_len"):
        assert k in s.last_stats


def test_decide_returns_single_action():
    act = BeamSolver(beam_width=2, max_depth=4, max_nodes=200).decide(_bc())
    assert act is not None and act in legal_actions(_bc())


# --------------------------------------------------------------------------
# 4. 性能回归（防退回字符串实现）
# --------------------------------------------------------------------------
def test_fingerprint_perf_regression():
    """当前约 20 µs/次；字符串实现约 2500 µs/次。阈值 300 µs 足以抓住退化。"""
    bc = _bc()
    for _ in range(20):
        fingerprint(bc)
    t0 = time.perf_counter()
    n = 200
    for _ in range(n):
        fingerprint(bc)
    us = (time.perf_counter() - t0) / n * 1e6
    assert us < 300, ("fingerprint 退化到 %.0f µs/次（疑退回 c.id.name，约 2500 µs）" % us)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
