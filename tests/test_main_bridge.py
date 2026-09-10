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


# ---------- 22:32 主因回归：禁止用模拟器假状态做决策 ----------

def test_real_sequence_no_simulator_on_fake_state(brain):
    """复现 22:32 主因序列：菜单 → 错误帧洪流 → 真实战斗 → 错误帧。

    1) 菜单 → START IRONCLAD
    2) 10 条 error 帧（无 game_state）→ 全 STATE，绝不出现 PLAY/CHOOSE/END
       （旧版正是在这种帧上拿模拟器假对局状态输出命令，导致死循环）
    3) 完整真实战斗状态 → 才允许模拟器命令；技能不带目标、攻击带目标
    4) 紧接错误帧 → 立刻退回 STATE，不重复上一条命令、不重跑模拟器
    """
    brain._last_cmd = ""
    brain._sim_fresh = False
    brain._reject_streak = 0

    # 1) 主菜单
    menu = {"available_commands": ["start"], "ready_for_command": True,
            "in_game": False}
    assert brain.decide(menu) == "START IRONCLAD"

    # 2) 10 条 error 帧（无 game_state）
    err = {"ready_for_command": True,
           "error": "Selected card cannot be played with the selected target."}
    err_out = [brain.decide(err) for _ in range(10)]
    assert all(c == "STATE" for c in err_out), f"错误帧不应产出模拟器命令: {err_out}"
    assert not any(c.startswith("PLAY") for c in err_out)
    assert not any(c.startswith("CHOOSE") for c in err_out)
    assert not any(c == "END" for c in err_out)

    # 3) 完整真实战斗状态（hand 一张技能 + 一张攻击）
    combat = _combat_state([
        {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL",
         "is_playable": True},
        {"id": "Strike_R", "name": "Strike", "cost": 1, "type": "ATTACK",
         "is_playable": True},
    ])
    c = brain.decide(combat)
    assert c != "STATE" and c.startswith("PLAY"), f"真实战斗状态应允许模拟器命令: {c}"
    src = int(c.split()[1]) - 1
    has_target = len(c.split()) >= 3
    hand = combat["game_state"]["combat_state"]["hand"]
    if has_target:
        assert hand[src].get("type", "").upper() == "ATTACK", c
    else:
        assert hand[src].get("type", "").upper() in ("SKILL", "POWER"), c

    # 4) 紧接错误帧：退回 STATE，不重复上一条命令
    c2 = brain.decide({"ready_for_command": True, "error": "some other game error"})
    assert c2 == "STATE", f"错误帧后应退回 STATE 而非重发: {c2}"
    assert c2 != c


def test_real_combat_skill_has_no_target_fixB(brain):
    """真实战斗单技能手牌：PLAY 不带目标下标（验证 (a) 真实状态/本地知识判定）。"""
    c = brain.decide(_combat_state([
        {"id": "Defend_R", "name": "Defend", "cost": 1, "type": "SKILL",
         "is_playable": True},
    ]))
    assert c == "PLAY 1", f"技能牌不应带目标: {c}"


def test_incomplete_state_never_drives_simulator(brain):
    """（Task 2 旧断言，Task 3 已改写）除 seed 外字段齐全 → P1 占位 seed 灌入成功，
    给出可执行命令而非 STATE；模拟器新鲜度被置真。

    注：这是 Task 3 P1 修复后的新正确行为——旧版"永远回 STATE"正是要修的
    静默死锁 failure mode。此处用 _base_game_state 构造"只缺 seed"的完整状态
    （deck/map/relics 等齐全），与实测 init_from_json 占位 seed 成功条件一致。

    隔离 gc：模块级 brain 的 gc 可能被前一个战斗测试里的 ISMCTS 改动到
    半模拟状态，重灌不同屏会失败；此处重置成干净上下文，专测 P1 占位 seed。
    """
    import slaythespire as sts
    brain.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    brain._sim_fresh = False
    brain._last_cmd = ""
    # 基于完整 game_state，只去掉 seed（模拟 Mod 没发 seed 的真实场景）
    gs = _base_game_state()
    gs.pop("seed", None)
    incomplete = {
        "available_commands": ["choose", "state"], "ready_for_command": True,
        "in_game": True,
        "game_state": gs,
    }
    cmd = brain.decide(incomplete)
    assert cmd != "STATE", f"缺 seed 状态不应空转: {cmd}"
    # 该状态无 choice_list，会走模拟器 RL 路径（P1 占位 seed 灌入成功才会新鲜、
    # 才会进入 _sim_decide 产出真实 CHOOSE），而非回 STATE 空转。
    assert cmd.startswith("CHOOSE "), f"P1 应让模拟器产出可执行命令: {cmd}"


# ---------- Task 3 安全网：真实状态必须能推进，禁止空转 ----------

def test_missing_seed_real_state_executable_not_state_fixC(brain):
    """(a) 完整战斗状态缺 seed + ready → 必须可执行而非 STATE（P0/P1 安全网）。

    对应本次新增 failure mode：_has_complete_game_state 要求 seed，缺 seed 时
    _sim_fresh 永远 False → 战斗分支都回 STATE → 静默死锁。P1 占位 seed 灌入
    使模拟器新鲜（ISMCTS 能跑）；P0 保证即使兜底也不空转。

    隔离 gc：避免被前一个战斗测试的 ISMCTS 改动影响（见同名注释）。
    """
    import slaythespire as sts
    brain.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    brain._sim_fresh = False
    brain._last_cmd = ""
    # 完整 game_state（deck/map/relics 齐全）只去掉 seed + 加 combat_state
    gs = _base_game_state()
    gs.pop("seed", None)
    gs["screen_type"] = "NONE"
    gs["combat_state"] = {
        "hand": [{"id": "Strike_R", "name": "Strike", "cost": 1,
                  "type": "ATTACK", "is_playable": True},
                 {"id": "Defend_R", "name": "Defend", "cost": 1,
                  "type": "SKILL", "is_playable": True}],
        "monsters": [{"id": "JawWorm", "name": "Jaw Worm", "current_hp": 40,
                      "max_hp": 44, "is_gone": False, "intent": "ATTACK"}],
        "player": {"energy": 3, "block": 0}, "turn": 1,
    }
    st = {
        "available_commands": ["play", "end", "potion", "state"],
        "ready_for_command": True, "in_game": True, "game_state": gs,
    }
    cmd = brain.decide(st)
    assert cmd != "STATE", f"缺 seed 战斗状态不应空转: {cmd}"
    assert cmd.startswith("PLAY") or cmd == "END", f"应可执行: {cmd}"


def test_no_available_commands_state_not_play_choose_fixC(brain):
    """(b) 20 个无 available_commands 的真实状态：允许 STATE（无命令可发），
    但绝不出现 PLAY / CHOOSE（避免越界/假命令）。"""
    brain._sim_fresh = False
    brain._last_cmd = ""
    screens = ["MAP", "EVENT", "REST", "SHOP_ROOM", "COMBAT_REWARD",
               "CHEST", "GRID"]
    for i in range(20):
        sc = screens[i % len(screens)]
        st = {
            "available_commands": [], "ready_for_command": True, "in_game": True,
            # 既缺 seed 也缺 gold → 不是"只缺 seed"，模拟器不新鲜，走真实状态规则
            "game_state": {"screen_type": sc, "floor": i + 1, "act": 1,
                           "current_hp": 70, "max_hp": 80,
                           "room_type": "MonsterRoom"},
        }
        cmd = brain.decide(st)
        assert cmd == "STATE", f"无可用命令应回 STATE，实际 {cmd} (screen={sc})"
        assert not cmd.startswith("PLAY"), cmd
        assert not cmd.startswith("CHOOSE"), cmd
        assert brain._sim_fresh is False, f"不完整状态不应新鲜 (screen={sc})"


def test_missing_seed_choice_list_all_choose0_fixC(brain):
    """(c) 20 个缺 seed 但 choice_list 非空的真实状态：全部 CHOOSE 0（P0 安全网）。

    CHOOSE 0 对事件/篝火/地图/选卡永远合法且推进游戏；无论 P1 占位 seed 是否让
    模拟器新鲜，选项屏都优先走真实状态规则，绝不回 STATE 空转。
    """
    brain._sim_fresh = False
    brain._last_cmd = ""
    screens = ["EVENT", "REST", "MAP", "CARD_SELECT", "GRID", "COMBAT_REWARD"]
    for i in range(20):
        sc = screens[i % len(screens)]
        st = {
            "available_commands": ["choose", "state"], "ready_for_command": True,
            "in_game": True,
            "game_state": {"screen_type": sc, "floor": i + 1, "act": 1,
                           "current_hp": 70, "max_hp": 80, "gold": 99 + i,
                           "choice_list": [{"x": 0}, {"x": 1}, {"x": 2}]},
        }
        cmd = brain.decide(st)
        assert cmd == "CHOOSE 0", \
            f"缺 seed+choice_list 应 CHOOSE 0，实际 {cmd} (screen={sc})"
