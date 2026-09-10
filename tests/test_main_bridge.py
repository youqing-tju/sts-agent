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


@pytest.fixture
def brain_unlocked(brain):
    """解锁首场战斗闸门的 brain。"""
    brain._combat_unlocked = True
    return brain


def test_map_screen_produces_choose(brain):
    """MAP 屏：模拟器路径必须产出合法的 CHOOSE <下标> 命令。"""
    state = _map_state()
    cmd = brain.decide(state)
    assert cmd.startswith("CHOOSE "), f"MAP 屏应出 CHOOSE，实际: {cmd}"
    # 下标必须在 comm 的 choose 可选范围内（本例 2 个节点）
    idx = int(cmd.split()[1])
    assert 0 <= idx < 2


def test_combat_play_first_playable(brain_unlocked):
    """战斗屏：有能出的攻击牌 → PLAY <hand_1based> <monster_0based>。"""
    state = _combat_state([
        {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
         "is_playable": True},
        {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL",
         "is_playable": True},
    ])
    cmd = brain_unlocked.decide(state)
    assert cmd == "PLAY 1 0", f"实际: {cmd}"


def test_combat_end_when_nothing_playable(brain_unlocked):
    """战斗屏：都出不了 → END。"""
    state = _combat_state([
        {"id": "Strike_R", "name": "Strike", "cost": 3, "type": "ATTACK",
         "is_playable": True},
    ], energy=1)
    assert brain_unlocked.decide(state) == "END"


def test_fallback_when_sim_decide_raises(brain_unlocked, monkeypatch):
    """模拟器路径抛异常 → 回退规则决策，不崩。"""
    def _boom(self, state, cmds):
        raise RuntimeError("simulated init_from_json failure")

    monkeypatch.setattr(SimBrain, "_sim_decide", _boom)
    cmd = brain_unlocked.decide(_map_state())
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


# ---------- 本次真机卡死回归（PLAY 目标语义 / 熔断 / 循环检测） ----------

def test_card_needs_target_fix1():
    """_card_needs_target：技能/群攻不需要目标，单体攻击需要，真实字段优先。"""
    from combat_brain import _card_needs_target

    def st(hand):
        return {"game_state": {"combat_state": {"hand": hand}}}

    # (a) 真实状态字段 has_target 优先
    assert _card_needs_target(st([{"has_target": True, "name": "Defend",
                                    "type": "SKILL"}]), 0) is True
    assert _card_needs_target(st([{"has_target": False, "name": "Defend",
                                    "type": "SKILL"}]), 0) is False
    # (b) 本地卡牌知识兜底
    assert _card_needs_target(st([{"name": "Defend", "type": "SKILL"}]), 0) is False
    assert _card_needs_target(st([{"name": "Strike", "type": "ATTACK"}]), 0) is True
    assert _card_needs_target(st([{"name": "Cleave", "type": "ATTACK"}]), 0) is False  # AOE
    assert _card_needs_target(st([{"name": "Inflame", "type": "POWER"}]), 0) is False
    # (c) 越界/缺手牌 → 默认需要（靠 error 帧纠错）
    assert _card_needs_target(st([{"name": "Strike"}]), 5) is True


def test_to_comm_no_target_for_skill_fix1():
    """_to_comm：技能/POWER 不带目标下标，单体攻击带，群体攻击不带。"""
    from combat_brain import CombatBrain

    class _Act:
        action_type = "CARD"
        source_idx = 0
        target_idx = 0

    assert CombatBrain._to_comm(
        _Act(), {"game_state": {"combat_state": {"hand": [{"name": "Defend",
                                                           "type": "SKILL"}]}}}) == "PLAY 1"
    assert CombatBrain._to_comm(
        _Act(), {"game_state": {"combat_state": {"hand": [{"name": "Strike",
                                                           "type": "ATTACK"}]}}}) == "PLAY 1 0"
    assert CombatBrain._to_comm(
        _Act(), {"game_state": {"combat_state": {"hand": [{"name": "Cleave",
                                                           "type": "ATTACK"}]}}}) == "PLAY 1"


def test_target_rejection_and_loop_detected_fix2_4_5(brain, monkeypatch):
    """真实序列回放：PLAY 1 0 被拒 → 换 PLAY 1 重发；交替循环在 5 次内判死。

    对应 22:32 真机死循环（259 条 "Selected card cannot be played with the
    selected target" + 210 次 PLAY 1 0 一步没走出去）。
    """
    from main import LoopDetector, _fingerprint

    # 模拟 ISMCTS 每次都选 hand[0]（一张被标记为「需要目标」的牌）→ PLAY 1 0
    hand = [{"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL",
             "has_target": True, "is_playable": True}]
    combat = _combat_state(hand)
    err = {"ready_for_command": True,
           "error": "Selected card cannot be played with the selected target."}

    monkeypatch.setattr(brain.combat_brain, "decide_with_fallback",
                        lambda s, gc, fb: "PLAY 1 0")

    guard = LoopDetector()
    issued = []
    stalled = False
    for _ in range(10):
        c = brain.decide(combat)
        issued.append(c)
        if guard.observe(_fingerprint(combat), has_game_state=True) >= guard.limit:
            stalled = True
            break
        ce = brain.decide(err)
        issued.append(ce)
        if guard.observe(_fingerprint(err), has_game_state=False) >= guard.limit:
            stalled = True
            break

    # 断言 1：第一次拒绝后，下一条命令不再带目标索引
    assert issued[1] == "PLAY 1", f"错误帧后命令应去掉目标下标: {issued}"
    # 断言 2：交替序列（COMBAT→error 交替、命令恒 PLAY 1 0）在 5 次内触发循环检测
    assert stalled, f"循环检测应在 5 次内触发，issued={issued}"
    # 断言 3：不会出现连续 5 次以上完全相同的命令
    max_run = run = 1
    for a, b in zip(issued, issued[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    assert max_run < 5, f"出现连续相同命令过长: {issued}"


def test_reject_streak_fuse_ignores_interleaved_state_fix3(brain, monkeypatch):
    """熔断计数：只在 error 帧自增、只在真实状态清零，不被自己回的 STATE 清零。

    复现旧 bug：被拒后插 STATE 把 _last_cmd 覆盖成 STATE，计数永远加不上。
    """
    err = {"ready_for_command": True, "error": "some other game error"}
    # 其它错误：连续 3 次 → 强制 END
    out = [brain.decide(err) for _ in range(3)]
    assert out[-1] == "END", f"连续 3 次被拒不相关错误应强制 END: {out}"
    # 收到真实状态后清零，再被拒不应立即 END
    assert brain.decide(_map_state()).startswith("CHOOSE ")
    assert brain.decide(err) != "END", "真实状态后应已清零熔断计数"
