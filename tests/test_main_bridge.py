"""main.SimBrain 桥接测试：真实游戏 JSON → 模拟器 → comm 命令。

用构造的完整 Communication Mod JSON 驱动：
- MAP 屏：模拟器枚举地图动作，Scorer 选一个，翻译成 CHOOSE <下标>
- 战斗屏：不走模拟器，启发式出 PLAY / END
- 翻译/灌入失败：回退规则决策，不崩
"""
import json

import pytest

from main import SimBrain


def _base_game_state():
    """完整 comm mod game_state（init_from_json 要求字段齐全，缺字段会崩）。"""
    return {
        "screen_type": "MAP",
        "screen_state": {},
        "seed": 12345,
        "class": "IRONCLAD",
        "ascension_level": 0,
        "act": 1,
        "floor": 1,
        "current_hp": 72,
        "max_hp": 80,
        "gold": 99,
        "room_type": "MonsterRoom",
        "act_boss": "The Guardian",
        "deck": [
            {"id": "Strike_R", "name": "Strike", "cost": 1, "upgrades": 0,
             "type": "ATTACK", "rarity": "BASIC"},
            {"id": "Defend_R", "name": "Defend", "cost": 1, "upgrades": 0,
             "type": "SKILL", "rarity": "BASIC"},
        ],
        "relics": [{"id": "Burning Blood", "name": "Burning Blood", "counter": -1}],
        "potions": [],
        "map": [],
    }


def _map_state():
    return {
        "available_commands": ["choose", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": _base_game_state(),
    }


def _combat_state(hand, energy=3):
    game = _base_game_state()
    game["screen_type"] = "NONE"  # comm mod 战斗时 screen_type 是 NONE
    game["combat_state"] = {
        "hand": hand,
        "monsters": [
            {"id": "JawWorm", "name": "Jaw Worm", "current_hp": 40,
             "max_hp": 44, "is_gone": False, "intent": "ATTACK"},
        ],
        "player": {"energy": energy, "block": 0},
        "turn": 1,
    }
    return {
        "available_commands": ["play", "end", "potion", "state"],
        "ready_for_command": True,
        "in_game": True,
        "game_state": game,
    }


@pytest.fixture(scope="module")
def brain():
    return SimBrain()


def test_map_screen_produces_choose(brain):
    """MAP 屏：模拟器路径必须产出合法的 CHOOSE <下标> 命令。"""
    state = _map_state()
    cmd = brain.decide(state)
    assert cmd.startswith("CHOOSE "), f"MAP 屏应出 CHOOSE，实际: {cmd}"
    # 下标必须在 comm 的 choose 可选范围内（本例 2 个节点）
    idx = int(cmd.split()[1])
    assert 0 <= idx < 2


def test_combat_play_first_playable(brain):
    """战斗屏：有能出的攻击牌 → PLAY <hand_1based> <monster_0based>。"""
    state = _combat_state([
        {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
         "is_playable": True},
        {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL",
         "is_playable": True},
    ])
    cmd = brain.decide(state)
    assert cmd == "PLAY 1 0", f"实际: {cmd}"


def test_combat_end_when_nothing_playable(brain):
    """战斗屏：都出不了 → END。"""
    state = _combat_state([
        {"id": "Strike_R", "name": "Strike", "cost": 3, "type": "ATTACK",
         "is_playable": True},
    ], energy=1)
    assert brain.decide(state) == "END"


def test_fallback_when_sim_decide_raises(brain, monkeypatch):
    """模拟器路径抛异常 → 回退规则决策，不崩。"""
    def _boom(self, state, cmds):
        raise RuntimeError("simulated init_from_json failure")

    monkeypatch.setattr(SimBrain, "_sim_decide", _boom)
    cmd = brain.decide(_map_state())
    # 规则 bot 在 MAP 屏的兜底就是 CHOOSE 0
    assert cmd == "CHOOSE 0", f"实际: {cmd}"


def test_fallback_when_checkpoint_missing():
    """checkpoint 缺失 → scorer=None，宏观决策走规则降级。"""
    b = SimBrain(checkpoint_path="checkpoints/__nonexistent__.pt")
    assert b.scorer is None
    assert b.decide(_map_state()) == "CHOOSE 0"


def test_main_menu_start(brain):
    """主菜单：START IRONCLAD。"""
    state = {
        "available_commands": ["start", "state"],
        "ready_for_command": True,
        "in_game": False,
    }
    assert brain.decide(state) == "START IRONCLAD"


def test_not_ready_returns_state(brain):
    """ready_for_command=False → STATE（配合主循环 sleep 反自旋）。"""
    state = _map_state()
    state["ready_for_command"] = False
    assert brain.decide(state) == "STATE"
