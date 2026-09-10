import math
import random

import torch

from env.base_env import Candidate, Obs, StepResult
from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer, ValueNet
from training.ppo import PPOTrainer


def _rand_vec(n):
    return [random.gauss(0, 1) for _ in range(n)]


def _fake_episode(n_steps=3, n_cands=4, final_floor=10, win=False, logp_old=-1.5):
    traj = [(_rand_vec(OBS_DIM), [_rand_vec(DESC_VEC_DIM) for _ in range(n_cands)], 1, logp_old)
            for _ in range(n_steps)]
    return (traj, final_floor, win)


class FakeEnv:
    """固定步数的假环境：3 个宏观决策点后终局（输）。"""

    def __init__(self, n_steps=3, n_cands=3):
        self._left = n_steps
        self._n_cands = n_cands

    def observe(self):
        return Obs(vec=_rand_vec(OBS_DIM), hp=50, max_hp=80, floor=3,
                   gold=0, deck=[], relics=[], screen="MAP_SCREEN")

    def candidates(self):
        if self._left <= 0:
            return []
        return [Candidate(idx=i, kind="MAP_NODE", desc_vec=_rand_vec(DESC_VEC_DIM),
                          label=f"node_{i}")
                for i in range(self._n_cands)]

    def step(self, idx):
        self._left -= 1
        done = self._left <= 0
        outcome = "PLAYER_LOSS" if done else "UNDECIDED"
        return StepResult(obs=self.observe(), done=done,
                          info={"hp_delta": -5, "outcome": outcome})


def _make_trainer(**kw):
    torch.manual_seed(0)
    scorer = Scorer()
    value_net = ValueNet()
    return PPOTrainer(scorer, value_net, **kw), scorer, value_net


def test_value_net_shapes():
    torch.manual_seed(0)
    v = ValueNet()
    assert v(torch.randn(477)).shape == ()
    assert v(torch.randn(7, 477)).shape == (7,)


def test_collect_episode_stores_logp_old():
    random.seed(0)
    tr, _, _ = _make_trainer()
    traj, final_floor, win = tr.collect_episode(FakeEnv())
    assert len(traj) == 3
    for obs_vec, cand_vecs, pos, logp_old in traj:
        assert len(obs_vec) == 477
        assert len(cand_vecs) == 3 and len(cand_vecs[0]) == DESC_VEC_DIM
        assert 0 <= pos < 3
        assert isinstance(logp_old, float)
        assert math.isfinite(logp_old) and logp_old < 0  # log 概率恒负
    assert isinstance(final_floor, int)
    assert win is False


def test_collect_episode_with_seed_resets_env():
    random.seed(0)
    tr, _, _ = _make_trainer()

    class SeedEnv(FakeEnv):
        def __init__(self):
            super().__init__()
            self.seed_seen = None

        def reset(self, seed):
            self.seed_seen = seed

    env = SeedEnv()
    tr.collect_episode(env, seed=42)
    assert env.seed_seen == 42


def test_update_changes_both_networks():
    random.seed(0)
    tr, scorer, value_net = _make_trainer(lr=1e-2, epochs=2)
    s_before = [p.detach().clone() for p in scorer.parameters()]
    v_before = [p.detach().clone() for p in value_net.parameters()]
    stats = tr.update([_fake_episode(), _fake_episode(final_floor=25, win=True)])
    assert any(not torch.allclose(b, a)
               for b, a in zip(s_before, scorer.parameters()))
    assert any(not torch.allclose(b, a)
               for b, a in zip(v_before, value_net.parameters()))
    for k in ("loss", "policy_loss", "value_loss", "mean_floor",
              "win_rate", "entropy"):
        assert k in stats and math.isfinite(stats[k])
    assert stats["mean_floor"] == 17.5
    assert stats["win_rate"] == 0.5


def test_clip_keeps_loss_finite_when_logp_shifted():
    """logp_old 与当前策略差距巨大时，loss 有限不爆炸。"""
    random.seed(0)
    tr, _, _ = _make_trainer(lr=1e-2, epochs=2)
    # logp_old = -30（旧策略几乎不选该动作），当前网络重新前向 logp ≈ O(1)
    ep = _fake_episode(n_steps=5, logp_old=-30.0)
    stats = tr.update([ep, _fake_episode(n_steps=5, final_floor=30, win=True,
                                         logp_old=-30.0)])
    for k in ("loss", "policy_loss", "value_loss", "entropy"):
        assert math.isfinite(stats[k]), f"{k} 爆炸: {stats[k]}"


def test_clip_bounds_policy_loss_when_ratio_shrinks():
    """logp_old ≈ 0（ratio → 0 方向）时 clip 双符号封顶，policy_loss 有界。"""
    random.seed(0)
    tr, _, _ = _make_trainer(lr=1e-2, epochs=2)
    eps = [_fake_episode(n_steps=5, logp_old=-1e-6),
           _fake_episode(n_steps=5, final_floor=30, win=True, logp_old=-1e-6)]
    stats = tr.update(eps)
    assert math.isfinite(stats["policy_loss"])
    assert abs(stats["policy_loss"]) < 10


def test_train_batches_updates():
    """batch 语义：多局合并 update，且不足 batch 的尾部也能更新。"""
    random.seed(0)
    tr, _, _ = _make_trainer(lr=1e-2, batch_games=2, epochs=1)
    eps = [tr.collect_episode(FakeEnv()) for _ in range(3)]
    stats = tr.update(eps[:2])
    assert math.isfinite(stats["loss"])
    stats2 = tr.update(eps[2:])
    assert math.isfinite(stats2["loss"])


def test_lightspeed_link_two_games():
    """真实环境链路：mcts_sim=50 跑 2 局后 update 一次不崩。"""
    from env.lightspeed_env import LightspeedEnv

    torch.manual_seed(0)
    tr, _, _ = _make_trainer(mcts_sim=50, epochs=1)
    env = LightspeedEnv(mcts_simulations=50)
    eps = []
    for seed in (111, 222):
        env.reset(seed)
        eps.append(tr.collect_episode(env))
    stats = tr.update(eps)
    assert math.isfinite(stats["loss"])
    assert stats["entropy"] > 0
