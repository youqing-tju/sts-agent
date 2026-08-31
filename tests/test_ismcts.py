"""ISMCTS 投票战斗决策测试（combat_brain v2）。

绑定依赖说明：
- sts.BattleSearcher 已编译 → needs_searcher 用例运行
- sts.BattleContext.randomize_rng（C++ 侧另一个代理在加）未编译时，
  真实重洗的用例 skip；投票循环本身用 FakeBC（no-op randomize_rng）测试，
  不依赖该绑定。
"""
import json
import logging
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import slaythespire as sts

import combat_brain
from combat_brain import CombatBrain
from tests.test_combat_brain import _combat_state

HAS_SEARCHER = hasattr(sts, "BattleSearcher")
HAS_RANDOMIZE = hasattr(sts.BattleContext(), "randomize_rng")

needs_searcher = pytest.mark.skipif(
    not HAS_SEARCHER, reason="sts.BattleSearcher 绑定未编译")
needs_randomize = pytest.mark.skipif(
    not HAS_RANDOMIZE, reason="sts.BattleContext.randomize_rng 绑定未编译（V2-1 进行中）")

_CMD_RE = re.compile(r"END|PLAY \d+( \d+)?|POTION USE .*")

# 测试用小预算，别把 CI 拖死
SIMS = 50
K = 5


@pytest.fixture()
def gc():
    ctx = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    ctx.init_from_json(json.dumps(_combat_state()))
    return ctx


@needs_searcher
@needs_randomize
def test_decide_reproducible(gc):
    """同一状态两次 decide 结果一致（base_seed 由状态派生，防决策抖动）。"""
    brain = CombatBrain(simulation_count=SIMS, determinizations=K)
    cmd1 = brain.decide(_combat_state(), gc)
    brain2 = CombatBrain(simulation_count=SIMS, determinizations=K)
    cmd2 = brain2.decide(_combat_state(), gc)
    assert cmd1 is not None
    assert cmd1 == cmd2


@needs_searcher
@needs_randomize
def test_k10_returns_legal_command(gc):
    """K=10 投票返回合法命令（PLAY/END），且得票总数 == K。"""
    brain = CombatBrain(simulation_count=SIMS, determinizations=10)
    cmd = brain.decide(_combat_state(), gc)
    assert cmd is not None
    assert _CMD_RE.fullmatch(cmd), f"非法命令: {cmd!r}"
    assert brain.last_votes is not None
    assert sum(brain.last_votes.values()) == 10


@needs_searcher
def test_fallback_to_v1_when_randomize_missing(gc, monkeypatch):
    """randomize_rng 缺失时降级 v1 单次搜索，不崩、返回合法命令。"""
    monkeypatch.setattr(
        CombatBrain, "_supports_determinization", staticmethod(lambda bc: False))
    brain = CombatBrain(simulation_count=SIMS, determinizations=K)
    cmd = brain.decide(_combat_state(), gc)
    assert cmd is not None
    assert _CMD_RE.fullmatch(cmd), f"非法命令: {cmd!r}"
    assert brain.last_votes is None  # 没走 ISMCTS 路径


@needs_searcher
def test_vote_distribution_keys_legal(gc, monkeypatch, caplog):
    """投票循环跑通：得票分布的键合法、总票数 == K、日志打印分布。

    用 FakeBC（no-op randomize_rng）让 ISMCTS 路径在 V2-1 绑定编译前即可测：
    所有未来相同 → 预期全票投给同一候选。
    """
    class FakeBC(sts.BattleContext):
        def randomize_rng(self, seed):
            pass  # no-op：所有平行未来相同

    monkeypatch.setattr(combat_brain.sts, "BattleContext", FakeBC)
    brain = CombatBrain(simulation_count=SIMS, determinizations=K)
    with caplog.at_level(logging.INFO, logger="sts-agent.combat"):
        cmd = brain.decide(_combat_state(), gc)

    assert cmd is not None
    assert _CMD_RE.fullmatch(cmd), f"非法命令: {cmd!r}"

    votes = brain.last_votes
    assert votes is not None and len(votes) >= 1
    assert sum(votes.values()) == K
    for key in votes:
        action_type, identity = key
        assert action_type in ("CARD", "END_TURN", "POTION")
        assert isinstance(identity, str) and identity
    # no-op 随机化 → 所有未来应投同一候选
    assert len(votes) == 1
    # 日志里有得票分布
    assert any("ISMCTS votes" in r.message for r in caplog.records)
