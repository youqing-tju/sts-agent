"""REINFORCE 训练循环。

- collect_episode：用当前 Scorer 玩一局。每个宏观决策点把
  (obs.vec, [cand.desc_vec...]) 过 scorer → softmax 采样（保留探索），
  轨迹记录 (obs_vec, cand_vecs, chosen_pos)。
- 奖励纯结果：G = final_floor + win_bonus * win，无过程奖励。
- update：对轨迹重新前向（带梯度）计算 log_prob/entropy，
  loss = -mean_episodes sum_t log_prob_t * (G - baseline) - entropy_coef * mean(entropy)，
  baseline 为最近 100 局 G 的滑动平均。
- train：主循环，每局 JSONL 日志（RunLogger），定期 eval_fn 评估。

【懒加载】LightspeedEnv / RunLogger 只在 train 内部导入：单测用 FakeEnv，
不需要加载 slaythespire .pyd。
"""
import random
from collections import deque

import torch
import torch.nn.functional as F

# 防御性步数上限：与 evaluation/evaluate.py 一致
_MAX_STEPS = 500

_WIN_OUTCOME = "PLAYER_VICTORY"


class ReinforceTrainer:
    def __init__(self, scorer, lr=1e-3, gamma=0.99, entropy_coef=0.01,
                 mcts_sim=8000, device="cpu", log_dir=None, win_bonus=50):
        self.scorer = scorer.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(self.scorer.parameters(), lr=lr)
        self.gamma = gamma  # 保留：当前奖励为终局一次性 G，未做逐步折扣
        self.entropy_coef = entropy_coef
        self.mcts_sim = mcts_sim
        self.win_bonus = win_bonus
        self.log_dir = log_dir
        self._g_hist = deque(maxlen=100)

    @property
    def baseline(self):
        """最近 100 局 G 的滑动平均（无历史时为 0）。"""
        return sum(self._g_hist) / len(self._g_hist) if self._g_hist else 0.0

    def _scores(self, obs_vec, cand_vecs):
        obs = torch.tensor(obs_vec, dtype=torch.float32, device=self.device)
        cands = torch.tensor(cand_vecs, dtype=torch.float32, device=self.device)
        return self.scorer(obs, cands)

    def collect_episode(self, env, policy_fn=None, logger=None):
        """从 env 当前状态（已 reset）玩到终局。

        policy_fn(obs, candidates) -> 候选列表位置，非空时替代采样（调试/对照用）。
        返回 (trajectory, final_floor, win)；trajectory 元素为
        (obs_vec, cand_vecs, chosen_pos)。
        """
        traj = []
        obs = env.observe()
        done, steps, res = False, 0, None
        while not done and steps < _MAX_STEPS:
            cands = env.candidates()
            if not cands:
                res = env.step(-1)  # 防御性分支；正常流程由 env 自动推进战斗
            else:
                cand_vecs = [c.desc_vec for c in cands]
                if policy_fn is not None:
                    pos = policy_fn(obs, cands)
                else:
                    with torch.no_grad():
                        probs = F.softmax(self._scores(obs.vec, cand_vecs), dim=-1)
                    pos = int(torch.multinomial(probs, 1).item())
                traj.append((list(obs.vec), cand_vecs, pos))
                if logger:
                    logger.log_decision(
                        floor=obs.floor, screen=obs.screen,
                        candidates=[c.label for c in cands],
                        chosen=cands[pos].label, hp=obs.hp)
                res = env.step(cands[pos].idx)
            obs, done = res.obs, res.done
            steps += 1
        final_floor = obs.floor
        win = bool(res and res.info.get("outcome") == _WIN_OUTCOME)
        if logger:
            logger.finish(final_floor=final_floor, win=win)
        return traj, final_floor, win

    def update(self, episodes):
        """对一批 episode 做一次 REINFORCE 更新，返回统计 dict。"""
        base = self.baseline
        loss_sum, ent_sum, n_ep = None, 0.0, 0
        floors, wins, gs = [], [], []
        for traj, final_floor, win in episodes:
            g = float(final_floor) + float(self.win_bonus) * float(win)
            gs.append(g)
            floors.append(float(final_floor))
            wins.append(float(win))
            adv = g - base
            ep_loss = 0.0
            for obs_vec, cand_vecs, pos in traj:
                logp = F.log_softmax(self._scores(obs_vec, cand_vecs), dim=-1)
                ent = -(logp.exp() * logp).sum()
                ep_loss = ep_loss - logp[pos] * adv
                ent_sum += float(ent.detach())
            loss_sum = ep_loss if loss_sum is None else loss_sum + ep_loss
            n_ep += 1
        if n_ep == 0:
            raise ValueError("episodes 为空")
        mean_ent = ent_sum / max(sum(len(t) for t, _, _ in episodes), 1)
        loss = loss_sum / n_ep - self.entropy_coef * mean_ent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self._g_hist.extend(gs)  # adv 用旧 baseline 计算后再更新历史
        return {
            "loss": float(loss.detach()),
            "mean_floor": sum(floors) / n_ep,
            "win_rate": sum(wins) / n_ep,
            "entropy": mean_ent,
        }

    def train(self, n_games, eval_every=500, eval_fn=None, log_fn=None):
        """主循环：collect_episode → update，定期 eval_fn(self, game) 评估。

        eval_fn(self, game_idx) 的返回值记入 history[i]["eval"]；
        log_fn(rec) 每局回调一次。返回训练历史 list[dict]。
        """
        from env.lightspeed_env import LightspeedEnv
        from logging.run_logger import RunLogger

        env = LightspeedEnv(mcts_simulations=self.mcts_sim)
        history = []
        for i in range(1, n_games + 1):
            seed = random.randrange(2**31)
            env.reset(seed)
            logger = RunLogger(self.log_dir, seed) if self.log_dir else None
            ep = self.collect_episode(env, logger=logger)
            stats = self.update([ep])
            rec = {"game": i, **stats}
            if eval_fn is not None and i % eval_every == 0:
                rec["eval"] = eval_fn(self, i)
            history.append(rec)
            if log_fn is not None:
                log_fn(rec)
        return history
