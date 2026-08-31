"""验证 init_from_json 绑定：GameContext/BattleContext 加载 + 异常防御。"""
import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts


def _card(cid, upgrades=0, cost=1):
    return {
        "id": cid,
        "name": cid,
        "cost": cost,
        "upgrades": upgrades,
        "type": "ATTACK",
        "rarity": "BASIC",
    }


def _starter_deck():
    return (
        [_card("Strike_R") for _ in range(5)]
        + [_card("Defend_R") for _ in range(4)]
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
                    _card("Defend_R"),
                    _card("Defend_R"),
                    _card("Bash", cost=2),
                ],
                "draw_pile": [
                    _card("Strike_R"),
                    _card("Strike_R"),
                    _card("Strike_R"),
                    _card("Defend_R"),
                    _card("Defend_R"),
                ],
                "discard_pile": [],
                "exhaust_pile": [],
                "limbo": [],
            },
        }
    }


def _new_gc():
    return sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)


def test_game_context_init_from_json():
    gc = _new_gc()
    gc.init_from_json(json.dumps(_combat_state()))
    assert gc.floor_num == 1
    assert gc.act == 1
    assert gc.cur_hp == 72
    assert gc.max_hp == 80
    assert gc.gold == 99
    assert len(gc.deck) == 10
    assert len(gc.relics) == 1


def test_battle_context_init_from_json():
    state = json.dumps(_combat_state())
    gc = _new_gc()
    gc.init_from_json(state)

    bc = sts.BattleContext()
    bc.init_from_json(gc, state)

    assert bc.monsters_count == 1
    assert bc.monsters_alive_count == 1
    assert len(bc.hand) == 5
    assert bc.player.energy == 3
    assert bc.player.hp == 72
    assert not bc.is_terminal()


def test_missing_room_type_raises_runtime_error():
    bad = _combat_state()
    del bad["game_state"]["room_type"]
    gc = _new_gc()
    with pytest.raises(RuntimeError, match="init_from_json failed"):
        gc.init_from_json(json.dumps(bad))


def test_empty_string_raises_runtime_error():
    gc = _new_gc()
    with pytest.raises(RuntimeError, match="init_from_json failed"):
        gc.init_from_json("")


def test_malformed_json_raises_runtime_error():
    gc = _new_gc()
    with pytest.raises(RuntimeError, match="init_from_json failed"):
        gc.init_from_json("{not valid json")


def test_battle_context_empty_string_raises_runtime_error():
    gc = _new_gc()
    gc.init_from_json(json.dumps(_combat_state()))
    bc = sts.BattleContext()
    with pytest.raises(RuntimeError, match="init_from_json failed"):
        bc.init_from_json(gc, "")
