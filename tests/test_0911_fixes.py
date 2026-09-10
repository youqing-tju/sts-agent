# -*- coding: utf-8 -*-
"""2026-09-11 凌晨四项修复的回归测试。

F1 虚弱/别名映射（models/power_alias.py）
F2 药水保命规则（SimBrain._desperate_potion_cmd / _natural_rule_cmd）
F3 MCTS 药水翻译（CombatBrain._to_comm → "POTION Use <slot>"）
F4 选牌打分（decide() 的 CARD_REWARD 分支）
"""
import json

import pytest


@pytest.fixture(scope="module")
def brain():
    """与 test_main_bridge.py 一致的 SimBrain fixture（默认构造 = 纯规则路径）。"""
    from main import SimBrain
    return SimBrain()


# ---------- F1：power id 别名 ----------

def test_power_alias_weakened_to_weak():
    """Weakened→Weak / Anger→Angry：不替换的话模拟器静默丢弃（真机 23 帧虚弱
    从未灌入 → MCTS 伤害全多算 25%）。"""
    from models.power_alias import alias_power_ids

    p = {"game_state": {"combat_state": {
        "player": {"powers": [{"id": "Weakened", "amount": 2},
                              {"id": "Strength", "amount": 3}]},
        "monsters": [{"powers": [{"id": "Anger", "amount": 1},
                                 {"id": "Artifact", "amount": 2}]},
                     {"powers": []}]}}}
    alias_power_ids(p)
    cs = p["game_state"]["combat_state"]
    assert cs["player"]["powers"][0]["id"] == "Weak"
    assert cs["player"]["powers"][1]["id"] == "Strength"
    assert cs["monsters"][0]["powers"][0]["id"] == "Angry"
    assert cs["monsters"][0]["powers"][1]["id"] == "Artifact"


def test_power_alias_never_raises():
    """别名替换绝不对异常输入抛出（失败安全）。"""
    from models.power_alias import alias_power_ids
    for bad in (None, {}, {"game_state": None},
                {"game_state": {"combat_state": {"player": None, "monsters": None}}},
                {"game_state": {"combat_state": {"player": {"powers": "x"}}}}):
        alias_power_ids(bad)   # 不抛即可


# ---------- F2：药水保命 ----------

def _combat_potion_state(hp, max_hp, potions, avail=("play", "end", "potion",
                                                     "key", "click", "wait", "state")):
    return {"available_commands": list(avail),
            "ready_for_command": True, "in_game": True,
            "game_state": {"screen_type": "NONE", "screen_name": "NONE",
                           "floor": 3, "act": 1, "current_hp": hp,
                           "max_hp": max_hp, "gold": 50, "seed": 123,
                           "potions": potions,
                           "combat_state": {"turn": 2, "hand": [], "monsters": []}}}


def test_desperate_potion_low_hp_defensive(brain):
    """低血（≤35%）+ 防御向药水 → POTION Use 0（真机血案：带 3 瓶药被打死）。"""
    st = _combat_potion_state(20, 80, [
        {"id": "Fire Potion", "can_use": True, "requires_target": False},
        {"id": "Block Potion", "can_use": True, "requires_target": False},
    ])
    got = brain._natural_rule_cmd(st)
    assert got == "POTION Use 1", f"应优先喝防御向药水（槽 1），实际 {got}"


def test_desperate_potion_low_hp_any_usable(brain):
    """低血 + 只有进攻药水 → 也喝（进攻也是保命），选第一个可用槽。"""
    st = _combat_potion_state(20, 80, [
        {"id": "Fire Potion", "can_use": True, "requires_target": False},
    ])
    assert brain._natural_rule_cmd(st) == "POTION Use 0"


def test_no_potion_when_healthy(brain):
    """血线健康（>35%）不乱喝药水。"""
    st = _combat_potion_state(70, 80, [
        {"id": "Block Potion", "can_use": True, "requires_target": False},
    ])
    got = brain._natural_rule_cmd(st)
    assert not str(got).startswith("POTION"), f"健康时不应喝药水: {got}"


def test_no_potion_when_none_usable(brain):
    """低血但药水不可用（can_use=false）/ 需要目标 → 不发 POTION。"""
    st = _combat_potion_state(20, 80, [
        {"id": "Block Potion", "can_use": False, "requires_target": False},
        {"id": "FearPotion", "can_use": True, "requires_target": True},
    ])
    got = brain._natural_rule_cmd(st)
    assert not str(got).startswith("POTION"), f"无可用药水不应发 POTION: {got}"


# ---------- F3：MCTS 药水翻译 ----------

def test_potion_translation_use():
    """POTION 动作 → "POTION Use <slot>"（官方语法含 Use 字样）。"""
    from combat_brain import CombatBrain

    class _Act:
        action_type = "POTION"
        source_idx = 1
        target_idx = 0

    st = {"game_state": {"potions": [
        None,
        {"id": "Block Potion", "can_use": True, "requires_target": False},
    ]}}
    assert CombatBrain._to_comm(_Act(), st) == "POTION Use 1"


def test_potion_translation_rejects_unusable():
    """can_use=false 的槽位不翻译；requires_target 的药水暂不翻译（都回退规则）。"""
    from combat_brain import CombatBrain

    class _Act:
        action_type = "POTION"
        source_idx = 0
        target_idx = 0

    st_no = {"game_state": {"potions": [{"id": "Block Potion", "can_use": False,
                                         "requires_target": False}]}}
    assert CombatBrain._to_comm(_Act(), st_no) is None
    st_tgt = {"game_state": {"potions": [{"id": "FearPotion", "can_use": True,
                                          "requires_target": True}]}}
    assert CombatBrain._to_comm(_Act(), st_tgt) is None


# ---------- F4：选牌打分 ----------

def _card_reward_state(cards, deck=None):
    return {"available_commands": ["choose", "skip", "key", "click", "wait", "state"],
            "ready_for_command": True, "in_game": True,
            "game_state": {"screen_type": "CARD_REWARD", "screen_name": "NONE",
                           "floor": 3, "act": 1, "current_hp": 70, "max_hp": 80,
                           "gold": 100, "seed": 456, "deck": deck or [],
                           "screen_state": {"cards": cards},
                           "choice_list": [c.get("id") for c in cards]}}


def test_card_reward_picks_best(brain):
    """选牌用 card_buy_score 打分选最高——不再"一律第一张"。

    真机帧形状：screen_state.cards 带 id/rarity/type。Shrug It Off（UNCOMMON +
    8 格挡）分数应高于 Wild Strike / Havoc，即使它排在第 2 位也要选它。
    """
    st = _card_reward_state([
        {"id": "Wild Strike", "rarity": "COMMON", "type": "ATTACK"},
        {"id": "Shrug It Off", "rarity": "UNCOMMON", "type": "SKILL"},
        {"id": "Havoc", "rarity": "COMMON", "type": "SKILL"},
    ])
    cmd = brain.decide(st)
    assert cmd == "CHOOSE 1", f"应选打分最高的第 2 张（Shrug It Off），实际 {cmd}"


def test_card_reward_fallback_on_empty_cards(brain):
    """cards 为空（异常输入）不崩，退回保守路径（CHOOSE 0 / STATE / SKIP 均可）。

    真实 CARD_REWARD 帧 cards 恒非空（3 张卡）；本用例只保证异常输入不崩、
    仍能给出一个合法命令（实测走规则兜底给 SKIP = 跳过这个坏掉的屏）。
    """
    st = _card_reward_state([])
    cmd = brain.decide(st)
    assert cmd in ("CHOOSE 0", "STATE", "SKIP"), f"空 cards 应退回保守路径，实际 {cmd}"
