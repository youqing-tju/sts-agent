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


# ---------- 药水栏满换药（真机 01:52 血案） ----------

def _full_bar_reward_state(choice_list, potions):
    return {"available_commands": ["choose", "potion", "proceed", "key",
                                   "click", "wait", "state"],
            "ready_for_command": True, "in_game": True,
            "game_state": {"screen_type": "COMBAT_REWARD", "screen_name": "NONE",
                           "floor": 8, "act": 2, "current_hp": 59, "max_hp": 80,
                           "gold": 100, "seed": 7,
                           "choice_list": list(choice_list),
                           "screen_state": {"rewards": []},
                           "potions": potions}}


def test_potion_full_bar_swap_discards_low_value(brain):
    """满栏 + 奖励含药水 → 先丢【低价值】那瓶腾位（绝不直接点 potion）。

    真机 01:52：满栏点 potion → 游戏弹"替换选择"界面 → Mod 不推帧 → 死锁 27 分钟。
    """
    brain._pot_discard_sent = False
    brain._pot_full_skip = False
    st = _full_bar_reward_state(["potion", "card"], [
        {"id": "Block Potion", "can_use": False, "requires_target": False},
        {"id": "Fruit Juice", "can_use": False, "requires_target": False},
        {"id": "Fire Potion", "can_use": False, "requires_target": False}])
    cmd = brain.decide(st)
    assert cmd == "POTION Discard 1", f"应丢低价值的 Fruit Juice（槽1），实际 {cmd}"


def test_potion_full_bar_giveup_after_discard(brain):
    """discard 发出后栏仍满（被拒/无效）→ 放弃换药：领非药水项，绝不重复发。"""
    brain._pot_discard_sent = False
    brain._pot_full_skip = False
    st = _full_bar_reward_state(["potion", "card"], [
        {"id": "Block Potion", "can_use": False, "requires_target": False},
        {"id": "Fire Potion", "can_use": False, "requires_target": False},
        {"id": "SpeedPotion", "can_use": False, "requires_target": False}])
    c1 = brain.decide(st)
    assert c1.startswith("POTION Discard"), f"第一次应先丢一瓶，实际 {c1}"
    c2 = brain.decide(st)          # 栏仍满（discard 假设无效）
    assert not c2.startswith("POTION"), f"不该重复发 discard，实际 {c2}"
    assert c2 == "CHOOSE 1", f"应领非药水奖励（card），实际 {c2}"


def test_potion_not_full_no_interception(brain):
    """栏不满（2/3）→ 不拦截，照常领取（第一次调用后应清状态并返回非 discard 命令）。"""
    brain._pot_discard_sent = False
    brain._pot_full_skip = False
    st = _full_bar_reward_state(["potion", "card"], [
        {"id": "Block Potion", "can_use": False, "requires_target": False},
        {"id": "Fire Potion", "can_use": False, "requires_target": False},
        {"id": "Potion Slot", "can_use": False, "requires_target": False}])
    cmd = brain.decide(st)
    assert not str(cmd).startswith("POTION Discard"), f"栏没满不应丢药，实际 {cmd}"


# ---------- MCTS 目标翻译（真机"总是打最左边"） ----------

def test_target_translates_sim_index_to_real_array():
    """MCTS 的 target 是"活怪序列"下标——必须翻译回真机完整数组下标。

    根因（真机实测）：initFromJson 只灌活怪（真机 4 只含 3 尸体 → 模拟器 1 只），
    索引空间不同。旧版丢弃模拟器目标、总发"真机第一个活怪"——用户观察：
    "总是从左边打起，Boss 进斩杀线也优先打爪牙"。
    核心回归：真机 [死, 活A, 死, 活B]，模拟器想打"第 2 只活怪"（target 1）→
    必须打真机下标 3（活B），而不是旧版的 1（活A）。
    """
    from combat_brain import CombatBrain

    class _Act:
        action_type = "CARD"
        source_idx = 1
        target_idx = 1

    st = {"game_state": {"combat_state": {
        "hand": [{"id": "X"}, {"id": "Strike_R"}],
        "monsters": [{"is_gone": True, "current_hp": 0},
                     {"is_gone": False, "current_hp": 20},
                     {"is_gone": True, "current_hp": 0},
                     {"is_gone": False, "current_hp": 5}],
        "player": {"energy": 2}}}}
    assert CombatBrain._to_comm(_Act(), st) == "PLAY 2 3", \
        "必须把'活怪序列 target 1'翻译成真机下标 3（活B）"

    class _Act0(_Act):
        target_idx = 0

    assert CombatBrain._to_comm(_Act0(), st) == "PLAY 2 1", \
        "target 0 → 真机第一个活怪（下标 1）"


def test_target_falls_back_on_out_of_range():
    """模拟器 target 越界 → 回退真机第一个活怪（不越界、不崩）。"""
    from combat_brain import CombatBrain

    class _Act:
        action_type = "CARD"
        source_idx = 1
        target_idx = 9

    st = {"game_state": {"combat_state": {
        "hand": [{"id": "X"}, {"id": "Strike_R"}],
        "monsters": [{"is_gone": True, "current_hp": 0},
                     {"is_gone": False, "current_hp": 7}],
        "player": {"energy": 1}}}}
    assert CombatBrain._to_comm(_Act(), st) == "PLAY 2 1"


def test_target_all_dead_returns_none():
    """全死 → None（交调用方走规则，绝不发指向尸体的命令）。"""
    from combat_brain import CombatBrain

    class _Act:
        action_type = "CARD"
        source_idx = 1
        target_idx = 0

    st = {"game_state": {"combat_state": {
        "hand": [{"id": "X"}, {"id": "Strike_R"}],
        "monsters": [{"is_gone": True}, {"is_gone": True}],
        "player": {"energy": 1}}}}
    assert CombatBrain._to_comm(_Act(), st) is None


# ---------- 篝火与升级屏（安全网之前交给宏决策） ----------

def _rest_state(hp, max_hp, done=False):
    return {"available_commands": (["proceed", "key", "click", "wait", "state"]
                                   if done else
                                   ["choose", "key", "click", "wait", "state"]),
            "ready_for_command": True, "in_game": True,
            "game_state": {"screen_type": "REST", "floor": 12, "act": 2,
                           "current_hp": hp, "max_hp": max_hp, "gold": 100,
                           "seed": 1,
                           "choice_list": None if done else ["rest", "smith"],
                           "screen_state": {"has_rested": done,
                                            "rest_options": [] if done else ["rest", "smith"]}}}


def test_rest_full_hp_chooses_smith(brain):
    """满血篝火必须锻造（smith），不是睡觉。

    用户反馈："火堆满血也要喜欢睡觉，几乎没敲过牌"。
    根因：REST 屏 choice_list 非空 → 安全网一律 CHOOSE 0（=rest=休息）。
    """
    cmd = brain.decide(_rest_state(80, 80))
    assert cmd == "CHOOSE 1", f"满血应锻造（smith 下标1），实际 {cmd}"


def test_rest_low_hp_chooses_rest(brain):
    """低血篝火休息（<40%）。"""
    cmd = brain.decide(_rest_state(20, 80))
    assert cmd == "CHOOSE 0", f"低血应休息（rest 下标0），实际 {cmd}"


def test_rest_done_proceeds(brain):
    """休息/锻造完成后 PROCEED 离开。"""
    assert brain.decide(_rest_state(90, 90, done=True)) == "PROCEED"


def test_grid_upgrade_picks_high_value_card(brain):
    """升级屏选高价值卡（非打击、高费优先）——真机帧形状。"""
    st = {"available_commands": ["choose", "key", "click", "wait", "state"],
          "ready_for_command": True, "in_game": True,
          "game_state": {"screen_type": "GRID", "floor": 12, "act": 2,
                         "current_hp": 60, "max_hp": 80, "seed": 1,
                         "choice_list": ["打击", "打击", "防御", "痛击"],
                         "screen_state": {
                             "for_purge": False, "for_upgrade": True,
                             "cards": [{"id": "Strike_R", "cost": 1},
                                       {"id": "Strike_R", "cost": 1},
                                       {"id": "Defend_R", "cost": 1},
                                       {"id": "Bash", "cost": 2}]}}}
    cmd = brain.decide(st)
    assert cmd == "CHOOSE 3", f"应升级痛击（Bash，高费非打击），实际 {cmd}"


def test_grid_confirm_frame(brain):
    """GRID 确认帧 → CONFIRM。"""
    st = {"available_commands": ["confirm", "cancel", "key", "click", "wait", "state"],
          "ready_for_command": True, "in_game": True,
          "game_state": {"screen_type": "GRID", "floor": 12, "act": 2,
                         "current_hp": 60, "max_hp": 80, "seed": 1,
                         "choice_list": None,
                         "screen_state": {"for_purge": False, "for_upgrade": True,
                                          "selected_cards": [3], "cards": []}}}
    assert brain.decide(st) == "CONFIRM"
