"""验证 BattleContext.randomize_rng（ISMCTS determinization）：
- 相同 seed 两次 randomize 产生相同抽牌堆顺序
- 不同 seed 产生不同抽牌堆顺序
- 洗牌只影响抽牌堆，不动手牌/怪物
"""
import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts

from tests.test_battle_searcher import _card, _combat_state, _new_bc


def _distinct_draw_pile_state():
    """抽牌堆用互不相同的卡，洗牌结果可直接按 id 序列比较。"""
    state = _combat_state()
    state["game_state"]["combat_state"]["draw_pile"] = [
        _card("Strike_R"),
        _card("Defend_R", ctype="SKILL"),
        _card("Bash", cost=2),
        _card("Cleave"),
        _card("Thunderclap"),
        _card("Twin Strike"),
        _card("Pommel Strike"),
        _card("Shrug It Off", ctype="SKILL"),
    ]
    return state


def _new_bc_from(state_dict):
    text = json.dumps(state_dict)
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(text)
    bc = sts.BattleContext()
    bc.init_from_json(gc, text)
    return bc


def _draw_pile_ids(bc):
    return [str(c.id) for c in bc.draw_pile]


def test_same_seed_same_draw_pile():
    bc0 = _new_bc_from(_distinct_draw_pile_state())

    bc1 = sts.BattleContext(bc0)
    bc1.randomize_rng(7)
    bc2 = sts.BattleContext(bc0)
    bc2.randomize_rng(7)

    assert _draw_pile_ids(bc1) == _draw_pile_ids(bc2)


def test_different_seed_different_draw_pile():
    bc0 = _new_bc_from(_distinct_draw_pile_state())
    original = _draw_pile_ids(bc0)

    bc1 = sts.BattleContext(bc0)
    bc1.randomize_rng(1)
    bc2 = sts.BattleContext(bc0)
    bc2.randomize_rng(2)

    ids1 = _draw_pile_ids(bc1)
    ids2 = _draw_pile_ids(bc2)

    # 同一集合的重排
    assert sorted(ids1) == sorted(original)
    assert sorted(ids2) == sorted(original)
    # 不同 seed 产生不同顺序
    assert ids1 != ids2
    # 原 bc 不被修改
    assert _draw_pile_ids(bc0) == original


def test_randomize_keeps_hand_and_monsters():
    bc0 = _new_bc()
    hand_before = [str(c.id) for c in bc0.hand]
    monster_hp_before = [m.hp for m in bc0.monsters]

    bc1 = sts.BattleContext(bc0)
    bc1.randomize_rng(42)

    assert [str(c.id) for c in bc1.hand] == hand_before
    assert [m.hp for m in bc1.monsters] == monster_hp_before
    assert len(bc1.draw_pile) == len(bc0.draw_pile)


def test_randomize_empty_draw_pile_no_crash():
    state = _combat_state()
    state["game_state"]["combat_state"]["draw_pile"] = []
    bc = _new_bc_from(state)
    bc.randomize_rng(1)  # 不应崩溃
    assert len(bc.draw_pile) == 0
