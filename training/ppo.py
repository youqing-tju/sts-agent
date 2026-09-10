"""PPO 训练循环（V3，设计见 docs/plans/2026-09-01-ppo-v3-design.md）。

与 ReinforceTrainer 的差异：
- Critic（ValueNet）替代滑动平均 baseline：A_t = G - V(s_t)，batch 内标准化；
- 收集时每步额外存 logp_old，update 时 K 个 epoch 重算 logp_new，
  ratio = exp(logp_new - logp_old)，clip eps=0.2；
- 批量更新：攒满 batch_games 局才 update 一次。

奖励纯终局：G = final_floor + win_bonus * win（gamma=1，每步共享同一 G）。
loss = policy_loss + 0.5 * value_loss - entropy_coef * mean(entropy)。

【懒加载】LightspeedEnv / RunLogger 只在 train 内部导入：单测用 FakeEnv，
不需要加载 slaythespire .pyd。
"""
import random

import torch
import torch.nn.functional as F

# 防御性步数上限：与 evaluation/evaluate.py 一致
_MAX_STEPS = 500

_WIN_OUTCOME = "PLAYER_VICTORY"


class PPOTrainer:
    def __init__(self, scorer, value_net, lr=3e-4, clip_eps=0.2,
                 entropy_coef=0.01, win_bonus=50, mcts_sim=2000,
                 epochs=4, batch_games=16, device="cpu", log_dir=None):
        self.scorer = scorer.to(device)
        self.value_net = value_net.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(self.scorer.parameters()) + list(self.value_net.parameters()),
            lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.win_bonus = win_bonus
        self.mcts_sim = mcts_sim
        self.epochs = epochs
        self.batch_games = batch_games
        self.log_dir = log_dir

    def _scores(self, obs_vec, cand_vecs):
        obs = torch.tensor(obs_vec, dtype=torch.float32, device=self.device)
        cands = torch.tensor(cand_vecs, dtype=torch.float32, device=self.device)
        return self.scorer(obs, cands)

    def _value(self, obs_vec):
        obs = torch.tensor(obs_vec, dtype=torch.float32, device=self.device)
        return self.value_net(obs)

    def collect_episode(self, env, seed=None, logger=None):
        """玩一局，返回 (trajectory, final_floor, win)。

        seed 非空时先 env.reset(seed)，否则从 env 当前状态继续。
        trajectory 元素为 (obs_vec, cand_vecs, chosen_pos, logp_old)，
        logp_old 为采样时策略下所选候选的 log 概率（detached）。
        """
        return _run_episode(seed, self.scorer, env, logger=logger,
                            device=self.device)


    def update(self, episodes):
        """对一批 episode 做 K 个 epoch 的 PPO 更新，返回统计 dict。"""
        if not episodes:
            raise ValueError("episodes 为空")

        # ---- 展平 batch：每步一条样本 ----
        steps = []  # (obs_vec, cand_vecs, pos, logp_old, g)
        floors, wins = [], []
        for traj, final_floor, win in episodes:
            g = float(final_floor) + float(self.win_bonus) * float(win)
            floors.append(float(final_floor))
            wins.append(float(win))
            for obs_vec, cand_vecs, pos, logp_old in traj:
                steps.append((obs_vec, cand_vecs, pos, logp_old, g))
        if not steps:
            raise ValueError("episodes 全部没有决策点")

        # ---- advantage：A_t = G - V(s_t)，用更新前的 critic，batch 内标准化 ----
        with torch.no_grad():
            advs = torch.tensor(
                [g - float(self._value(obs_vec))
                 for obs_vec, _, _, _, g in steps],
                dtype=torch.float32, device=self.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        logp_old_t = torch.tensor([s[3] for s in steps],
                                  dtype=torch.float32, device=self.device)
        g_t = torch.tensor([s[4] for s in steps],
                           dtype=torch.float32, device=self.device)

        stats = {}
        for _ in range(self.epochs):
            logp_news, ents, vs = [], [], []
            for (obs_vec, cand_vecs, pos, _, _), adv in zip(steps, advs):
                logp = F.log_softmax(self._scores(obs_vec, cand_vecs), dim=-1)
                logp_news.append(logp[pos])
                ents.append(-(logp.exp() * logp).sum())
                vs.append(self._value(obs_vec))
            logp_new_t = torch.stack(logp_news)
            ent_t = torch.stack(ents)
            v_t = torch.stack(vs)

            ratio = (logp_new_t - logp_old_t).exp()
            surr1 = ratio * advs
            surr2 = torch.clamp(ratio, 1 - self.clip_eps,
                                1 + self.clip_eps) * advs
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(v_t, g_t)
            entropy = ent_t.mean()
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            stats = {
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_loss": float(value_loss.detach()),
                "mean_floor": sum(floors) / len(floors),
                "win_rate": sum(wins) / len(wins),
                "entropy": float(entropy.detach()),
            }
        return stats

    def train(self, n_games, log_fn=None, log_dir=None, save_fn=None,
              workers=1):
        """主循环：攒满 batch_games 局 → update → 清空，直到 n_games。

        每局随机种子、每局一个 RunLogger JSONL（log_dir 非空时）；
        每次 update 产生一条 history 记录并回调 log_fn(rec)，
        字段保持 game/loss/mean_floor/win_rate/entropy（另含
        policy_loss/value_loss），与 REINFORCE 曲线分析兼容。
        尾部不足一个 batch 的局也会做最后一次 update。
        save_fn(i)：每次 update 后回调，用于定期保存 checkpoint（防中断丢进度）。

        workers > 1 时用多进程并行收集：游戏模拟（C++ MCTS）是单线程 CPU
        瓶颈，串行只吃满 1 核；并行后 CPU 利用率可接近 100%。并行路径不写
        RunLogger JSONL（子进程无法安全共享 logger 对象）。
        """
        from concurrent.futures import ProcessPoolExecutor
        from env.lightspeed_env import LightspeedEnv
        from logging.run_logger import RunLogger

        log_dir = log_dir if log_dir is not None else self.log_dir
        env = LightspeedEnv(mcts_simulations=self.mcts_sim)
        history = []
        parallel = bool(workers and workers > 1)

        if parallel:
            cfg = {
                "obs_dim": getattr(self.scorer, "obs_dim", 468),
                "cand_dim": getattr(self.scorer, "cand_dim", 443),
                "hidden": tuple(getattr(self.scorer, "hidden", (512, 256))),
                "mcts_sim": self.mcts_sim,
            }
            seeds_all = [random.randrange(2 ** 31) for _ in range(n_games)]
            nw = max(1, min(int(workers), self.batch_games))
            pos = 0
            # 持久进程池：进程启动需 import torch+slaythespire（数秒），
            # 若每批重建，启动开销会完全抵消并行收益。
            with ProcessPoolExecutor(max_workers=nw) as pool:
                while pos < n_games:
                    chunk = seeds_all[pos:pos + self.batch_games]
                    pos += len(chunk)
                    # worker 用 CPU 推理，权重必须先搬到 CPU
                    state = {k: v.detach().to("cpu").clone()
                             for k, v in self.scorer.state_dict().items()}
                    tasks = [(s, state, cfg) for s in chunk]
                    batch = list(pool.map(_collect_worker, tasks))
                    stats = self.update(batch)
                    rec = {"game": pos, **stats}
                    history.append(rec)
                    if log_fn is not None:
                        log_fn(rec)
                    if save_fn is not None:
                        save_fn(pos)
            return history

        batch = []
        for i in range(1, n_games + 1):
            seed = random.randrange(2**31)
            logger = RunLogger(log_dir, seed) if log_dir else None
            batch.append(self.collect_episode(env, seed=seed, logger=logger))
            if len(batch) < self.batch_games and i < n_games:
                continue
            stats = self.update(batch)
            batch = []
            rec = {"game": i, **stats}
            history.append(rec)
            if log_fn is not None:
                log_fn(rec)
            if save_fn is not None:
                save_fn(i)
        return history


def _run_episode(seed, scorer, env, logger=None, device="cpu"):
    """用给定 scorer 跑一局游戏（模块级，主进程与子进程共用）。

    抽成模块级函数是为了并行：子进程用 spawn 启动，无法继承主进程对象，
    需要能独立重建 scorer + env 来跑一局。
    """
    if seed is not None:
        env.reset(seed)
    traj = []
    obs = env.observe()
    done, steps, res = False, 0, None
    while not done and steps < _MAX_STEPS:
        cands = env.candidates()
        if not cands:
            res = env.step(-1)  # 防御性分支；正常流程由 env 自动推进战斗
        else:
            cand_vecs = [c.desc_vec for c in cands]
            with torch.no_grad():
                obs_t = torch.tensor(obs.vec, dtype=torch.float32, device=device)
                cands_t = torch.tensor(cand_vecs, dtype=torch.float32,
                                       device=device)
                logp = F.log_softmax(scorer(obs_t, cands_t), dim=-1)
                pos = int(torch.multinomial(logp.exp(), 1).item())
            traj.append((list(obs.vec), cand_vecs, pos,
                         float(logp[pos])))
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


def _collect_worker(args):
    """子进程 worker：重建 scorer + env 跑一局，返回 (traj, floor, win)。

    scorer 固定放 CPU —— 每个子进程建 CUDA context 约 300~500MB，
    16 进程会顶爆 8GB 显存；而 (1024,512) 的 CPU 推理单次仅约 1ms，
    相对游戏模拟（约 2s/局）可忽略。
    """
    import torch as _torch
    from env.lightspeed_env import LightspeedEnv
    from models.scorer import Scorer

    seed, state, cfg = args
    scorer = Scorer(obs_dim=cfg["obs_dim"], cand_dim=cfg["cand_dim"],
                    hidden=tuple(cfg["hidden"]))
    scorer.load_state_dict(state)
    scorer.eval()
    env = LightspeedEnv(mcts_simulations=cfg["mcts_sim"])
    return _run_episode(seed, scorer, env, device="cpu")
