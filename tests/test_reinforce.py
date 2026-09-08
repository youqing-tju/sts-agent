import math
import random

import torch

from env.base_env import Candidate, Obs, StepResult
from models.scorer import Scorer
from training.reinforce import ReinforceTrainer


def _rand_vec(n):
    return [random.gauss(0, 1) for _ in range(n)]


def _fake_episode(n_steps=3, n_cands=4, final_floor=10, win=False):
    traj = [(_rand_vec(468), [_rand_vec(443) for _ in range(n_cands)], 1)
            for _ in range(n_steps)]
    return (traj, final_floor, win)


class FakeEnv:
    """固定步数的假环境：3 个宏观决策点后终局（输）。"""

    def __init__(self, n_steps=3, n_cands=3):
        self._left = n_steps
        self._n_cands = n_cands

    def observe(self):
        return Obs(vec=_rand_vec(468), hp=50, max_hp=80, floor=3,
                   gold=0, deck=[], relics=[], screen="MAP_SCREEN")

    def candidates(self):
        if self._left <= 0:
            return []
        return [Candidate(idx=i, kind="MAP_NODE", desc_vec=_rand_vec(443),
                          label=f"node_{i}")
                for i in range(self._n_cands)]

    def step(self, idx):
        self._left -= 1
        done = self._left <= 0
        outcome = "PLAYER_LOSS" if done else "UNDECIDED"
        return StepResult(obs=self.observe(), done=done,
                          info={"hp_delta": -5, "outcome": outcome})


def test_update_returns_scalar_loss_and_finite_stats():
    torch.manual_seed(0)
    random.seed(0)
    tr = ReinforceTrainer(Scorer(), lr=1e-2)
    stats = tr.update([_fake_episode()])
    assert isinstance(stats["loss"], float) and math.isfinite(stats["loss"])
    assert math.isfinite(stats["entropy"])
    assert stats["mean_floor"] == 10.0
    assert stats["win_rate"] == 0.0


def test_update_actually_changes_params():
    torch.manual_seed(0)
    random.seed(0)
    scorer = Scorer()
    tr = ReinforceTrainer(scorer, lr=1e-2)
    before = [p.detach().clone() for p in scorer.parameters()]
    tr.update([_fake_episode(), _fake_episode(final_floor=20, win=True)])
    after = list(scorer.parameters())
    assert any(not torch.allclose(b, a) for b, a in zip(before, after))


def test_entropy_positive_and_baseline_moves():
    torch.manual_seed(0)
    random.seed(0)
    tr = ReinforceTrainer(Scorer(), lr=1e-2)
    stats = tr.update([_fake_episode(n_cands=3)])
    assert stats["entropy"] > 0
    # G = 10 + 0 = 10，更新后 baseline 应移向 10
    assert tr.baseline > 0


def test_collect_episode_fake_env():
    torch.manual_seed(0)
    random.seed(0)
    tr = ReinforceTrainer(Scorer(), lr=1e-2)
    traj, final_floor, win = tr.collect_episode(FakeEnv())
    assert len(traj) == 3
    for obs_vec, cand_vecs, pos in traj:
        assert len(obs_vec) == 468
        assert len(cand_vecs) == 3 and len(cand_vecs[0]) == 443
        assert 0 <= pos < 3
    assert isinstance(final_floor, int)
    assert win is False
    # 收集到的 episode 可直接用于 update（链路闭环）
    stats = tr.update([(traj, final_floor, win)])
    assert math.isfinite(stats["loss"])


def test_win_bonus_used_in_return():
    """G = final_floor + win_bonus * win：首局后 baseline == G。"""
    torch.manual_seed(0)
    random.seed(0)
    ep = _fake_episode(final_floor=10, win=True)
    tr_default = ReinforceTrainer(Scorer(), lr=1e-2)
    tr_default.update([ep])
    assert tr_default.baseline == 10.0 + 50.0  # 默认 win_bonus=50，向后兼容
    tr_200 = ReinforceTrainer(Scorer(), lr=1e-2, win_bonus=200)
    tr_200.update([ep])
    assert tr_200.baseline == 10.0 + 200.0


def test_collect_episode_with_policy_fn():
    torch.manual_seed(0)
    random.seed(0)
    tr = ReinforceTrainer(Scorer())
    traj, _, _ = tr.collect_episode(FakeEnv(), policy_fn=lambda obs, cands: 0)
    assert all(pos == 0 for _, _, pos in traj)


def test_lightspeed_link_two_games():
    """真实环境链路：mcts_sim=50 跑 2 局，验证 collect→update 通路。"""
    from env.lightspeed_env import LightspeedEnv

    torch.manual_seed(0)
    tr = ReinforceTrainer(Scorer(), lr=1e-3, mcts_sim=50)
    env = LightspeedEnv(mcts_simulations=50)
    for seed in (111, 222):
        env.reset(seed)
        ep = tr.collect_episode(env)
        stats = tr.update([ep])
        assert math.isfinite(stats["loss"])
        assert stats["entropy"] > 0
