"""Scorer：逐候选打分网络。

score_i = MLP(obs ⊕ candidate_i)，输出每个候选的 logit，
外部取 softmax 采样（见 training/reinforce.py）。
"""
import torch
import torch.nn as nn


class Scorer(nn.Module):
    def __init__(self, obs_dim=412, cand_dim=368, hidden=(128, 128)):
        super().__init__()
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
