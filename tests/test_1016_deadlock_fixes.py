# -*- coding: utf-8 -*-
"""2026-10-16 真机死循环两项修复的回归测试。

背景（run `1016_234015`，卡死 10 分钟）：

1. **奖励屏 A↔B 死循环**
   `CARD_REWARD` 发 `SKIP` 后，`COMBAT_REWARD` 的 `rewards` 里**卡牌项依然存在**
   （真机帧：`rewards=[{"reward_type":"CARD"}]`）→ 规则版 `macro_action` 又发
   `CHOOSE 0` 把卡牌屏点开 → 与 `SKIP` 构成无限振荡（实测连续 **325 次**）。
   修复：本层已跳过卡牌奖励（`_card_skip_floor == floor`）且只剩卡牌项 → 只能 `PROCEED`。

2. **卡死守卫对二周期振荡完全失效**
   守卫三信号（连续相同指纹 / 同一真实状态重复 / 同一命令连续输出）**全部要求"连续"重复**，
   被 A→B→A→B 逐帧重置 → 三条同时失效（事件流里 `stall=0 / stall_giveup=0`，10 分钟无人发现）。
   修复：新增**信号 4 —— 短周期振荡检测**（周期 2~4）。
"""
import pytest


@pytest.fixture(scope="module")
def brain():
    from main import SimBrain
    return SimBrain()


def _card_reward_state(cards, floor=3):
    return {"available_commands": ["choose", "skip", "key", "click", "wait", "state"],
            "ready_for_command": True, "in_game": True,
            "game_state": {"screen_type": "CARD_REWARD", "floor": floor, "act": 1,
                           "current_hp": 70, "max_hp": 80, "gold": 100, "seed": 5,
                           "deck": [],
                           "choice_list": [c["id"] for c in cards],
                           "screen_state": {"cards": cards}}}


def _combat_reward_state(rewards, floor=3):
    """真机卡死帧同构：COMBAT_REWARD 且 screen_state.rewards 给出剩余奖励项。"""
    return {"available_commands": ["choose", "potion", "proceed",
                                   "key", "click", "wait", "state"],
            "ready_for_command": True, "in_game": True,
            "game_state": {"screen_type": "COMBAT_REWARD", "floor": floor, "act": 1,
                           "current_hp": 70, "max_hp": 80, "gold": 100, "seed": 5,
                           "deck": [], "choice_list": [],
                           "screen_state": {"rewards": rewards}}}


_LOW_CARDS = [{"id": "Clash", "rarity": "COMMON", "type": "ATTACK"},
              {"id": "Perfected Strike", "rarity": "COMMON", "type": "ATTACK"},
              {"id": "Twin Strike", "rarity": "COMMON", "type": "ATTACK"}]


# --------------------------------------------------------------------------
# 1. 奖励屏死循环
# --------------------------------------------------------------------------
def test_card_skip_then_combat_reward_proceeds(brain):
    """🔴 核心回归：跳过选牌后，只剩卡牌项的 COMBAT_REWARD **必须 PROCEED**。

    修复前返回 `CHOOSE 0` → 重新点开卡牌屏 → 与 SKIP 无限振荡。
    """
    brain._card_skip_floor = None
    assert brain.decide(_card_reward_state(_LOW_CARDS)) == "SKIP"
    assert brain._card_skip_floor == 3, "SKIP 时必须记下楼层，否则后续无法判断"

    st = _combat_reward_state([{"reward_type": "CARD"}], floor=3)
    assert brain.decide(st) == "PROCEED"


def test_card_skip_does_not_leak_to_next_floor(brain):
    """别的楼层不该被"已跳过"影响（否则会白跳一层的卡牌奖励）。"""
    brain._card_skip_floor = 3
    st = _combat_reward_state([{"reward_type": "CARD"}], floor=4)
    assert brain.decide(st) == "CHOOSE 0"


def test_combat_reward_still_chooses_when_not_skipped(brain):
    """未跳过时行为不变：照旧 `CHOOSE 0` 去点开卡牌奖励（不能把选牌机会丢光）。"""
    brain._card_skip_floor = None
    st = _combat_reward_state([{"reward_type": "CARD"}], floor=3)
    assert brain.decide(st) == "CHOOSE 0"


def test_combat_reward_with_other_rewards_still_chooses(brain):
    """奖励里还有金币/药水/遗物时，即便已跳过卡牌也必须继续 `CHOOSE 0` 领取。"""
    brain._card_skip_floor = 3
    st = _combat_reward_state([{"reward_type": "GOLD"}, {"reward_type": "CARD"}], floor=3)
    assert brain.decide(st) == "CHOOSE 0"


def test_card_picked_does_not_mark_skip(brain):
    """正常拿牌**不是**跳过 —— 不该触发 PROCEED 逻辑（否则会漏拿奖励）。"""
    brain._card_skip_floor = None
    good = [{"id": "Wild Strike", "rarity": "COMMON", "type": "ATTACK"},
            {"id": "Shrug It Off", "rarity": "UNCOMMON", "type": "SKILL"},
            {"id": "Havoc", "rarity": "COMMON", "type": "SKILL"}]
    cmd = brain.decide(_card_reward_state(good))
    assert cmd.startswith("CHOOSE"), cmd
    assert brain._card_skip_floor is None


# --------------------------------------------------------------------------
# 2. 卡死守卫：短周期振荡
# --------------------------------------------------------------------------
def _fp(screen, floor=3):
    return (screen, floor, 1, 70, 100, "MonsterRoom", None, None, None, True)


def test_cyclic_period_detects_two_cycle():
    from main import _cyclic_period
    a, b = ("CARD_REWARD", "SKIP"), ("COMBAT_REWARD", "CHOOSE 0")
    hist = [a, b, a, b, a, b]
    assert _cyclic_period(hist) == 2


def test_cyclic_period_detects_three_cycle():
    from main import _cyclic_period
    a, b, c = (1, "x"), (2, "y"), (3, "z")
    assert _cyclic_period([a, b, c, a, b, c]) == 3


def test_cyclic_period_none_on_advancing_sequence():
    from main import _cyclic_period
    assert _cyclic_period([(1, "a"), (2, "b"), (3, "c"), (4, "d")]) == 0


def test_guard_fires_on_two_cycle():
    """🔴 核心回归：A↔B 交替必须触发守卫（修复前三信号全部被交替重置 → 永不触发）。"""
    from main import LoopDetector as CommandGuard
    g = CommandGuard(limit=4, hard_limit=10)
    a, b = _fp("CARD_REWARD"), _fp("COMBAT_REWARD")
    sev = 0
    for i in range(12):
        if i % 2 == 0:
            sev = g.observe(a, has_game_state=True, cmd="SKIP")
        else:
            sev = g.observe(b, has_game_state=True, cmd="CHOOSE 0")
    assert sev >= g.limit, "二周期振荡未被守卫发现（severity=%d）" % sev
    assert g.cycle_period == 2


def test_guard_cycle_counter_resets_after_progress():
    """打破周期后计数必须归零（否则正常推进会被误判成卡死）。"""
    from main import LoopDetector as CommandGuard
    g = CommandGuard(limit=3, hard_limit=10)
    a, b = _fp("CARD_REWARD"), _fp("COMBAT_REWARD")
    for i in range(4):
        g.observe(a if i % 2 == 0 else b, has_game_state=True,
                  cmd="SKIP" if i % 2 == 0 else "CHOOSE 0")
    # 正常推进几帧（指纹与命令都在变）
    g.observe(_fp("MAP"), has_game_state=True, cmd="CHOOSE 1")
    g.observe(_fp("COMBAT", floor=4), has_game_state=True, cmd="PLAY 1 0")
    assert g.cycle_period == 0


def test_guard_ignores_state_frames_for_cycle():
    """STATE 只是"请求重发状态"，不参与周期计数（否则主菜单等待会被误判）。"""
    from main import LoopDetector as CommandGuard
    g = CommandGuard(limit=3, hard_limit=10)
    for i in range(10):
        g.observe(_fp("NONE"), has_game_state=False, cmd="STATE")
    assert g.cycle_period == 0
