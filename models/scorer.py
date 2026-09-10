"""Scorer：逐候选打分网络；ValueNet：状态价值网络。

score_i = MLP(obs ⊕ candidate_i)，输出每个候选的 logit，
外部取 softmax 采样（见 training/reinforce.py、training/ppo.py）。
V(s) = MLP(obs)，PPO 的 critic（见 training/ppo.py）。
"""
import torch
import torch.nn as nn

from .candidate_encoder import DESC_VEC_DIM


class Scorer(nn.Module):
    def __init__(self, obs_dim=477, cand_dim=DESC_VEC_DIM, hidden=(128, 128)):
        # cand_dim 默认跟随候选编码器当前维度（367→443 追加 KB 段后=443）；
        # 旧权重不兼容、需重训，故默认即新维度，避免链路维度错配。
        super().__init__()
        # 存下维度元数据：并行训练的 worker 进程需要据此重建同构 scorer
        self.obs_dim = obs_dim
        self.cand_dim = cand_dim
        self.hidden = tuple(hidden)
        layers = []
        d = obs_dim + cand_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs, cands):
        """obs: (obs_dim,) 或 (batch, obs_dim)；cands: (n_cands, cand_dim)。

        返回 (n_cands,) 或 (batch, n_cands) 的 logit。
        """
        single = obs.dim() == 1
        if single:
            obs = obs.unsqueeze(0)
        b, n = obs.shape[0], cands.shape[0]
        obs_e = obs.unsqueeze(1).expand(b, n, obs.shape[-1])
        cand_e = cands.unsqueeze(0).expand(b, n, cands.shape[-1])
        out = self.mlp(torch.cat([obs_e, cand_e], dim=-1)).squeeze(-1)
        return out.squeeze(0) if single else out


class ValueNet(nn.Module):
    """状态价值网络：V(s) = MLP(obs)，PPO 的 critic。"""

    def __init__(self, obs_dim=477, hidden=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs):
        """obs: (obs_dim,) 或 (batch, obs_dim)。返回标量或 (batch,) 的 V(s)。"""
        return self.mlp(obs).squeeze(-1)
