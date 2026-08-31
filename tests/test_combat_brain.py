"""CombatBrain（MCTS 战斗决策）测试。

注意：sts.BattleSearcher 由另一个代理在 C++ 侧暴露，可能尚未编译进
当前 .pyd。所有依赖 BattleSearcher 的测试在其缺失时 skip；
降级路径（decide_with_fallback）不依赖绑定，始终可测。
"""
import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts

from combat_brain import CombatBrain
from main_rule import combat_action

HAS_SEARCHER = hasattr(sts, "BattleSearcher")
needs_searcher = pytest.mark.skipif(
    not HAS_SEARCHER, reason="sts.BattleSearcher 绑定尚未编译（另一个代理在做）")


def _card(cid, upgrades=0, cost=1, ctype="ATTACK"):
    return {
        "id": cid,
        "name": cid,
        "cost": cost,
        "upgrades": upgrades,
        "type": ctype,
        "rarity": "BASIC",
        "is_playable": True,
    }


def _combat_state():
    return {
        "available_commands": ["play", "end"],
        "ready_for_command": True,
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
            "deck": (
                [_card("Strike_R") for _ in range(5)]
                + [_card("Defend_R", ctype="SKILL") for _ in range(4)]
                + [_card("Bash", cost=2)]
            ),
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
        },
    }


@pytest.fixture()
def gc():
    ctx = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    ctx.init_from_json(json.dumps(_combat_state()))
    return ctx


@needs_searcher
def test_decide_returns_valid_command(gc):
    """MCTS 绑定时：decide 返回合法的 PLAY/END 命令。"""
    brain = CombatBrain(simulation_count=200)  # 测试用小模拟次数
    cmd = brain.decide(_combat_state(), gc)
    assert cmd is not None
    assert cmd == "END" or re.fullmatch(r"PLAY \d+( \d+)?", cmd)


@needs_searcher
def test_decide_play_index_in_hand(gc):
    """若返回 PLAY，牌的下标（1-based）应在手牌范围内。"""
    state = _combat_state()
    brain = CombatBrain(simulation_count=200)
    cmd = brain.decide(state, gc)
    if cmd and cmd.startswith("PLAY"):
        idx = int(cmd.split()[1])
        hand_size = len(state["game_state"]["combat_state"]["hand"])
        assert 1 <= idx <= hand_size


def test_decide_without_binding_returns_none(gc):
    """BattleSearcher 缺失时 decide 返回 None（不抛异常）。"""
    if HAS_SEARCHER:
        pytest.skip("绑定已存在，此用例不适用")
    brain = CombatBrain()
    assert brain.decide(_combat_state(), gc) is None


def test_fallback_when_binding_missing(gc):
    """绑定缺失时 decide_with_fallback 回退规则，不崩。"""
    if HAS_SEARCHER:
        pytest.skip("绑定已存在，此用例不适用")
    brain = CombatBrain()
    cmd = brain.decide_with_fallback(_combat_state(), gc, combat_action)
    assert cmd is not None
    assert cmd == "END" or cmd.startswith("PLAY")


def test_fallback_on_garbage_state(gc):
    """战斗 JSON 损坏时：decide 返回 None，fallback 仍给出规则命令。"""
    bad = _combat_state()
    del bad["game_state"]["combat_state"]  # 没有战斗数据
    brain = CombatBrain()
    assert brain.decide(bad, gc) is None
    cmd = brain.decide_with_fallback(bad, gc, combat_action)
    assert cmd == "END"  # 规则：没手牌可出 → END


def test_fallback_fn_called_on_exception(gc):
    """decide 内部异常（如 gc 未 init）时回退规则且不抛。"""
    fresh_gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)  # 未 init_from_json
    brain = CombatBrain()
    cmd = brain.decide_with_fallback(_combat_state(), fresh_gc, combat_action)
    assert cmd == "END" or cmd.startswith("PLAY")
