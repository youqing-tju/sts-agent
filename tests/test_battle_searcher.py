"""验证 BattleSearcher / BattleAction 绑定：对 init_from_json 加载的战斗状态跑 MCTS。"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts

VALID_ACTION_TYPES = {
    "CARD",
    "POTION",
    "SINGLE_CARD_SELECT",
    "MULTI_CARD_SELECT",
    "END_TURN",
}


def _card(cid, upgrades=0, cost=1, ctype="ATTACK"):
    return {
        "id": cid,
        "name": cid,
        "cost": cost,
        "upgrades": upgrades,
        "type": ctype,
        "rarity": "BASIC",
    }


def _starter_deck():
    return (
        [_card("Strike_R") for _ in range(5)]
        + [_card("Defend_R", ctype="SKILL") for _ in range(4)]
        + [_card("Bash", cost=2)]
    )


def _combat_state():
    return {
        "game_state": {
            "seed": 12345,
            "class": "IRONCLAD",
            "ascension_level": 0,
            "act": 1,
            "floor": 1,
            "current_hp": 72,
            "max_hp": 80,
            "gold": 99,
            "room_type": "MonsterRoom",
            "screen_type": "NONE",
            "deck": _starter_deck(),
            "relics": [{"id": "Burning Blood"}],
            "potions": [],
            "act_boss": None,
            "combat_state": {
                "turn": 1,
                "player": {
                    "current_hp": 72,
                    "max_hp": 80,
                    "block": 0,
                    "energy": 3,
                    "powers": [],
                    "orbs": [],
                },
                "monsters": [
                    {
                        "id": "JawWorm",
                        "name": "Jaw Worm",
                        "current_hp": 42,
                        "max_hp": 44,
                        "block": 0,
                        "intent": "ATTACK",
                        "half_dead": False,
                        "is_gone": False,
                        "powers": [],
                    }
                ],
                "hand": [
                    _card("Strike_R"),
                    _card("Strike_R"),
                    _card("Defend_R", ctype="SKILL"),
                    _card("Defend_R", ctype="SKILL"),
                    _card("Bash", cost=2),
                ],
                "draw_pile": [
                    _card("Strike_R"),
                    _card("Strike_R"),
                    _card("Strike_R"),
                    _card("Defend_R", ctype="SKILL"),
                    _card("Defend_R", ctype="SKILL"),
                ],
                "discard_pile": [],
                "exhaust_pile": [],
                "limbo": [],
            },
        }
    }


def _new_bc():
    state = json.dumps(_combat_state())
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(state)
    bc = sts.BattleContext()
    bc.init_from_json(gc, state)
    return bc


def test_battle_searcher_returns_best_action():
    bc = _new_bc()
    searcher = sts.BattleSearcher(bc, simulation_count=1000)
    action = searcher.best_action
    assert action.action_type in VALID_ACTION_TYPES


def test_battle_searcher_default_simulation_count():
    bc = _new_bc()
    searcher = sts.BattleSearcher(bc)
    assert searcher.best_action.action_type in VALID_ACTION_TYPES


def test_battle_action_indices():
    bc = _new_bc()
    searcher = sts.BattleSearcher(bc, simulation_count=1000)
    action = searcher.best_action
    assert isinstance(action.source_idx, int)
    assert isinstance(action.target_idx, int)
    if action.action_type == "CARD":
        assert 0 <= action.source_idx < len(bc.hand)


def test_battle_action_desc():
    bc = _new_bc()
    searcher = sts.BattleSearcher(bc, simulation_count=1000)
    action = searcher.best_action
    desc = action.desc(bc)
    assert isinstance(desc, str)
    assert len(desc) > 0
    assert "INVALID" not in desc


def test_battle_action_sequence():
    bc = _new_bc()
    searcher = sts.BattleSearcher(bc, simulation_count=1000)
    seq = searcher.best_action_sequence
    assert len(seq) >= 1
    assert all(a.action_type in VALID_ACTION_TYPES for a in seq)
