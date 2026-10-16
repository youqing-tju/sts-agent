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


# --------------------------------------------------------------------------
# 5. P0-②：泳道必保席 / 同分打破 / 束宽 Portfolio
# --------------------------------------------------------------------------
def _expand_once(bc):
    """展开一层，返回非终局子节点（`_Node`），供 `_select_beam` 直接测试。"""
    out = []
    for act in legal_actions(bc):
        cb = sts.BattleContext(bc)
        combat_beam._make(act).execute(cb)
        if cb.is_terminal():
            continue
        out.append(combat_beam._Node(bc=cb, path=(act,), score=evaluate(cb), depth=1,
                                     traits=combat_beam.lane_values(cb)))
    return out


def test_lane_values_shape_and_semantics():
    bc = _bc()
    v = combat_beam.lane_values(bc)
    assert len(v) == len(combat_beam.TRAITS)
    i_block = combat_beam.TRAITS.index("block")
    i_off = combat_beam.TRAITS.index("offense")
    assert v[i_block] == bc.player.block
    assert v[i_off] == -sum(m.hp for m in bc.monsters if m.is_alive)


def test_lane_values_offense_improves_after_attack():
    """打出一张攻击牌后 offense 泳道值应变高（= 敌人总血降低）。"""
    bc = _bc()
    attacks = [a for a in legal_actions(bc) if a[0] == "CARD"]
    i_off = combat_beam.TRAITS.index("offense")
    cb = sts.BattleContext(bc)
    combat_beam._make(attacks[0]).execute(cb)
    if sum(m.hp for m in cb.monsters if m.is_alive) == sum(m.hp for m in bc.monsters):
        pytest.skip("该牌未造成伤害（手牌随机），跳过")
    assert combat_beam.lane_values(cb)[i_off] > combat_beam.lane_values(bc)[i_off]


def _expand_multi(bc, rounds=3, cap=40):
    """多层展开，返回足够多的非终局节点（用于「节点数 > 束宽」的场景）。"""
    nodes = _expand_once(bc)
    for _ in range(rounds - 1):
        nxt = []
        for n in nodes:
            for act in legal_actions(n.bc):
                cb = sts.BattleContext(n.bc)
                try:
                    combat_beam._make(act).execute(cb)
                except Exception:                       # noqa: BLE001
                    continue
                if cb.is_terminal():
                    continue
                nxt.append(combat_beam._Node(
                    bc=cb, path=n.path + (act,), score=evaluate(cb), depth=n.depth + 1,
                    traits=combat_beam.lane_values(cb)))
        nodes = nxt
        if len(nodes) > cap:
            break
    return nodes


def test_lane_champions_are_seated_in_beam():
    """启用泳道时，**每条泳道的冠军必须进入 beam**（顶替语义的核心保证）。

    前提：展开节点数必须 **超过** 束宽，否则 `_select_beam` 直接全收、无需顶替。
    """
    width = 6
    expanded = _expand_multi(_bc(), rounds=3)
    if len(expanded) <= width:
        pytest.skip("展开节点不足以超过束宽（%d <= %d）" % (len(expanded), width))
    s = BeamSolver(beam_width=width, route_lanes=True, lane_budget=len(combat_beam.TRAITS))
    beam, seats = s._select_beam(list(expanded), width)
    assert seats >= 1
    for idx in range(len(combat_beam.TRAITS)):
        champ = combat_beam._find_best_lane(expanded, idx)
        assert champ is not None
        assert any(b is champ for b in beam), "泳道 %s 的冠军未进入 beam" % combat_beam.TRAITS[idx]


def test_route_lanes_off_is_pure_top_w():
    bc = _bc()
    expanded = _expand_once(bc)
    if len(expanded) < 3:
        pytest.skip("展开节点太少")
    s = BeamSolver(beam_width=3, route_lanes=False)
    beam, seats = s._select_beam(list(expanded), 3)
    assert seats == 0
    expect = sorted(expanded, key=combat_beam._beam_sort_key)[:3]
    assert [id(n) for n in beam] == [id(n) for n in expect]


def test_route_lanes_on_can_differ_from_top_w():
    """泳道开启后可能改变 beam 组成（否则机制是空转）。"""
    bc = _bc()
    expanded = _expand_once(bc)
    if len(expanded) < 4:
        pytest.skip("展开节点太少")
    on, seats = BeamSolver(beam_width=3, route_lanes=True,
                           lane_budget=2)._select_beam(list(expanded), 3)
    assert seats >= 1
    assert len(on) == 3


def test_beam_sort_key_score_desc_then_depth_asc():
    bc = _bc()
    tr = combat_beam.lane_values(bc)
    n1 = combat_beam._Node(bc=bc, score=10.0, depth=2, traits=tr)
    n2 = combat_beam._Node(bc=bc, score=10.0, depth=1, traits=tr)
    n3 = combat_beam._Node(bc=bc, score=11.0, depth=9, traits=tr)
    order = sorted([n1, n2, n3], key=combat_beam._beam_sort_key)
    assert order[0] is n3, "分数高者应排前"
    assert order[1] is n2, "同分时步数少者应排前"


def test_lane_tiebreak_prefers_higher_hp():
    a = combat_beam._Node(bc=_bc(hp=70), score=1.0, depth=1,
                          traits=combat_beam.lane_values(_bc(hp=70)))
    b = combat_beam._Node(bc=_bc(hp=50), score=1.0, depth=1,
                          traits=combat_beam.lane_values(_bc(hp=50)))
    assert combat_beam._dominant_by_tiebreak(a, b)


def test_portfolio_widths_dedup_and_positive():
    s = BeamSolver(beam_width=8, width_portfolio=True)
    ws = s.portfolio_widths()
    assert ws[0] == 8
    assert len(ws) == len(set(ws))
    assert all(x >= 1 for x in ws)
    assert len(ws) >= 3


def test_portfolio_solve_reports_chosen_width():
    s = BeamSolver(beam_width=4, max_depth=4, max_nodes=150, width_portfolio=True)
    plan = s.solve(_bc())
    assert plan
    st = s.last_stats
    assert st["portfolio_chosen_width"] in st["portfolio_widths"]
    assert "width" in st and "lane_seats" in st


def test_lane_seats_zero_when_width_one():
    """W=1 时泳道不占席（顶替会让 beam 全是冠军，反而失去常规席位）。"""
    bc = _bc()
    expanded = _expand_once(bc)
    if len(expanded) < 2:
        pytest.skip("展开节点太少")
    s = BeamSolver(beam_width=2, route_lanes=True, lane_budget=0)
    _, seats = s._select_beam(list(expanded), 2)
    assert seats == 0


def test_portfolio_shares_node_budget():
    """Portfolio 的各成员**共享一份节点预算**（总展开数 ≤ max_nodes）。"""
    s = BeamSolver(beam_width=8, max_depth=6, max_nodes=200, width_portfolio=True)
    calls = []
    orig = s._solve_once

    def spy(bc, width, max_nodes=None):
        plan, st = orig(bc, width, max_nodes=max_nodes)
        calls.append(st["nodes"])
        return plan, st

    s._solve_once = spy
    s.solve(_bc())
    assert calls, "至少应跑一个 portfolio 成员"
    assert sum(calls) <= 200, "各成员节点之和 %d 超预算" % sum(calls)
    assert s.last_stats["portfolio_nodes_left"] >= 0


def test_solve_stats_has_hit_budget():
    s = BeamSolver(beam_width=2, max_depth=4, max_nodes=200)
    s.solve(_bc())
    assert "hit_budget" in s.last_stats


# --------------------------------------------------------------------------
# 6. P0：可配置权重（Weights）
# --------------------------------------------------------------------------
def test_weights_default_matches_legacy_values():
    """🔴 默认权重必须与改造前的模块级常量**逐位一致**（保证改造零行为变化）。"""
    w = combat_beam.DEFAULT_WEIGHTS
    assert (w.hp, w.block, w.energy, w.enemy_hp, w.intent,
            w.strength, w.dexterity, w.hand, w.draw) == (
        30_000, 18_000, 9_000, 12_000, 6_000, 15_000, 12_000, 2_500, 500)
    assert w.enemy_block == 4_000
    assert w.win_base == 1e9 and w.loss_base == -1e9 and w.turn_penalty == 500.0


def test_weights_from_overrides_partial():
    w = combat_beam.Weights.from_overrides({"intent": 3_000})
    assert w.intent == 3_000
    assert w.hp == 30_000, "未覆盖的字段必须保持默认"
    assert combat_beam.Weights.from_overrides(None) == combat_beam.DEFAULT_WEIGHTS
    assert combat_beam.Weights.from_overrides({}) == combat_beam.DEFAULT_WEIGHTS


def test_weights_unknown_field_raises():
    """拼错字段名必须报错 —— 否则调参会静默失效。"""
    with pytest.raises(ValueError):
        combat_beam.Weights.from_overrides({"intnet": 1})


def test_evaluate_none_equals_default_weights():
    bc = _bc()
    assert evaluate(bc) == evaluate(bc, combat_beam.DEFAULT_WEIGHTS)


def test_evaluate_respects_weights_override():
    bc = _bc()
    # 玩家 hp > 0，把 hp 权重置 0 → 分数必然变化（不依赖意图伤害是否非零）
    assert evaluate(bc, combat_beam.Weights.from_overrides({"hp": 0})) != evaluate(bc)


def test_beam_solver_accepts_and_stores_weights():
    w = combat_beam.Weights.from_overrides({"intent": 0})
    s = BeamSolver(beam_width=2, max_depth=4, max_nodes=200, weights=w)
    assert s.weights is w
    assert s.solve(_bc())
    assert BeamSolver(beam_width=2).weights is None


def test_presets_all_constructible():
    """每个预设都必须能构造出 `Weights`（键名拼错会在此暴露）。"""
    assert combat_beam.PRESETS, "预设表不应为空"
    for name, ov in combat_beam.PRESETS.items():
        w = combat_beam.Weights.from_overrides(ov)
        assert isinstance(w, combat_beam.Weights), name


def test_preset_defensive_matches_measured_config():
    """`defensive` 必须与 200 局实测的那组配置一致（见 logs/weight-tuning-report.md）。"""
    assert combat_beam.PRESETS["default"] == {}
    assert combat_beam.PRESETS["defensive"] == {"enemy_hp": 6_000, "intent": 12_000}
    assert combat_beam.PRESETS["defensive_max"] == {"enemy_hp": 3_000, "intent": 20_000}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
