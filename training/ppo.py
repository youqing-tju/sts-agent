"""PPO 训练循环（V3，设计见 docs/plans/2026-09-01-ppo-v3-design.md）。

与 ReinforceTrainer 的差异：
- Critic（ValueNet）替代滑动平均 baseline：A_t = G - V(s_t)，batch 内标准化；
- 收集时每步额外存 logp_old，update 时 K 个 epoch 重算 logp_new，
  ratio = exp(logp_new - logp_old)，clip eps=0.2；
- 批量更新：攒满 batch_games 局才 update 一次。

奖励纯终局：G = final_floor + win_bonus * win
            + upgrade_bonus * upgrade_count + relic_bonus * relic_count
（gamma=1，每步共享同一 G）。upgrade_count / relic_count 为终局牌组
升级卡数 / 遗物数，作为终局加成引导 agent 为长期收益忍受短期风险
（诊断见 logs/deck-strength-report.md：通关组升级卡 6.07 vs 3.63、
遗物 10.82 vs 9.58，而稀有卡/牌组厚度无差异）。两 bonus 默认 0，
完全向后兼容——不放任何规则，纯粹从奖励信号学。
loss = policy_loss + 0.5 * value_loss - entropy_coef * mean(entropy)。

【懒加载】LightspeedEnv / RunLogger 只在 train 内部导入：单测用 FakeEnv，
不需要加载 slaythespire .pyd。
"""
import random

import numpy as np

import torch
import torch.nn.functional as F

from models.candidate_encoder import DESC_VEC_DIM

# 防御性步数上限：与 evaluation/evaluate.py 一致
_MAX_STEPS = 500

_WIN_OUTCOME = "PLAYER_VICTORY"


def _count_upgraded(deck):
    """牌组里升级卡数。

    元素为 C++ 绑定的 Card 对象（实测，见 _tmp_probe_deck_fields.py）：
    升级字段是 int 型的 `upgrade_count`（>0 即升级），同时也有 bool 的
    `upgraded`。两字段等价（实测每个种子计数一致）。为稳妥用兼容写法：
    优先 upgrade_count>0，否则回退 upgraded 布尔。
    """
    n = 0
    for c in deck:
        uc = getattr(c, "upgrade_count", None)
        if isinstance(uc, int) and uc > 0:
            n += 1
        elif getattr(c, "upgraded", False):
            n += 1
    return n


def _count_relics(relics):
    """遗物数：relics 是可迭代对象，元素为 Relic（含 .id.name）。"""
    return len(list(relics))


class PPOTrainer:
    def __init__(self, scorer, value_net, lr=3e-4, clip_eps=0.2,
                 entropy_coef=0.01, win_bonus=50, mcts_sim=2000,
                 epochs=4, batch_games=16, device="cpu", log_dir=None,
                 upgrade_bonus=0.0, relic_bonus=0.0,
                 combat_solver="mcts", beam_width=4, beam_depth=8,
                 beam_max_nodes=3000,
                 route_lanes=True, lane_budget=None, width_portfolio=False,
                 weights=None, value_mode="static", rollout_max_steps=400):
        self.scorer = scorer.to(device)
        self.value_net = value_net.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            list(self.scorer.parameters()) + list(self.value_net.parameters()),
            lr=lr)
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.win_bonus = win_bonus
        self.upgrade_bonus = upgrade_bonus
        self.relic_bonus = relic_bonus
        self.mcts_sim = mcts_sim
        # 战斗求解器："mcts"（C++ Agent 自动打，默认）| "beam"（Python Beam Search 接管）
        self.combat_solver = combat_solver
        self.beam_width = beam_width
        self.beam_depth = beam_depth
        self.beam_max_nodes = beam_max_nodes
        # P0-②：Beam 高级机制开关（透传到 env 与 worker 子进程）
        self.route_lanes = route_lanes
        self.lane_budget = lane_budget
        self.width_portfolio = width_portfolio
        self.weights = weights          # 启发式权重覆盖（dict）；须透传到 worker 子进程
        # 节点估值方式："static"（默认）| "rollout"（打到终局再打分，含终局信息）
        self.value_mode = value_mode
        self.rollout_max_steps = rollout_max_steps
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
        """玩一局，返回 (trajectory, final_floor, win, upgrade_count, relic_count)。

        seed 非空时先 env.reset(seed)，否则从 env 当前状态继续。
        trajectory 元素为 (obs_vec, cand_vecs, chosen_pos, logp_old)，
        logp_old 为采样时策略下所选候选的 log 概率（detached）。
        upgrade_count / relic_count 为终局牌组升级卡数 / 遗物数（int）。
        """
        return _run_episode(seed, self.scorer, env, logger=logger,
                            device=self.device)


    def update(self, episodes):
        """对一批 episode 做 K 个 epoch 的 PPO 更新，返回统计 dict。

        【为什么写成批量张量】原实现是「每个 epoch 内逐决策点循环」：
        每步各发一次 _scores + _value 前向，一个 batch 要发
        epochs × T × 2 次微小 kernel。实测 8.5 ms/步（batch 16/64 完全线性），
        使主进程成为整个训练的关键路径（主进程 ~0.86 s/局 > worker 并行后的
        ~0.31 s/局），14 个 worker 大半时间空转、CPU 只有 11%。
        现在把「步 × 候选」压成一个 (T, C) 矩阵一次前向，数学完全等价
        （候选补齐位用 mask 屏蔽，熵项只对有效位求和）。
        """
        if not episodes:
            raise ValueError("episodes 为空")

        # ---- 展平 batch：每步一条样本（批量装箱，不做逐条 Python 循环）----
        obs_list, cand_lists, pos_list, logp_list, g_list = [], [], [], [], []
        floors, wins = [], []
        for traj, final_floor, win, upgrade_count, relic_count in episodes:
            g = (float(final_floor)
                 + float(self.win_bonus) * float(win)
                 + float(self.upgrade_bonus) * float(upgrade_count)
                 + float(self.relic_bonus) * float(relic_count))
            floors.append(float(final_floor))
            wins.append(float(win))
            for obs_vec, cand_vecs, pos, logp_old in traj:
                obs_list.append(obs_vec)
                cand_lists.append(cand_vecs)
                pos_list.append(pos)
                logp_list.append(logp_old)
                g_list.append(g)
        if not obs_list:
            raise ValueError("episodes 全部没有决策点")

        dev = self.device
        T = len(obs_list)
        C = max(len(c) for c in cand_lists)
        obs_t = torch.as_tensor(np.asarray(obs_list, dtype=np.float32), device=dev)
        # 候选补齐到等宽 (T, C, cand_dim)；mask 标出有效位
        cand_t = torch.zeros(T, C, self.scorer.cand_dim, device=dev)
        mask = torch.zeros(T, C, dtype=torch.bool, device=dev)
        for i, cl in enumerate(cand_lists):
            n = len(cl)
            cand_t[i, :n] = torch.as_tensor(
                np.asarray(cl, dtype=np.float32), device=dev)
            mask[i, :n] = True
        pos_t = torch.as_tensor(pos_list, dtype=torch.long, device=dev)
        logp_old_t = torch.as_tensor(logp_list, dtype=torch.float32, device=dev)
        g_t = torch.as_tensor(g_list, dtype=torch.float32, device=dev)

        # ---- advantage：A_t = G - V(s_t)，用更新前的 critic，batch 内标准化 ----
        with torch.no_grad():
            advs = g_t - self.value_net(obs_t)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        # 补齐位用「有限大负数」而不是 -inf：-inf 会让 p*logp 变成 0×(-inf)=nan，
        # 而 torch.where 的反向传播会把该 nan 泄漏成 NaN 梯度 → 权重被污染 →
        # 采样时报 "probability tensor contains inf/nan"。
        # 用 -1e4 时 exp(-1e4) 在 float32 下精确下溢为 0，既无 nan 也不影响熵项。
        NEG_BIG = -1e4
        stats = {}
        for _ in range(self.epochs):
            logits = self.scorer.forward_padded(obs_t, cand_t).masked_fill(~mask, NEG_BIG)
            logp_all = F.log_softmax(logits, dim=-1)
            logp_new_t = logp_all.gather(1, pos_t.unsqueeze(1)).squeeze(1)

            p = logp_all.exp()                      # 补齐位精确为 0
            ent_t = -(p * logp_all).sum(dim=-1)     # 0 × (-1e4) = -0.0，不污染求和

            v_t = self.value_net(obs_t)

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
        env = LightspeedEnv(mcts_simulations=self.mcts_sim,
                            combat_solver=self.combat_solver,
                            beam_width=self.beam_width,
                            beam_depth=self.beam_depth,
                            beam_max_nodes=self.beam_max_nodes,
                            route_lanes=self.route_lanes,
                            lane_budget=self.lane_budget,
                            width_portfolio=self.width_portfolio,
                            weights=self.weights,
                            value_mode=self.value_mode,
                            rollout_max_steps=self.rollout_max_steps)
        history = []
        parallel = bool(workers and workers > 1)

        if parallel:
            cfg = {
                "obs_dim": getattr(self.scorer, "obs_dim", 477),
                "cand_dim": getattr(self.scorer, "cand_dim", DESC_VEC_DIM),
                "hidden": tuple(getattr(self.scorer, "hidden", (512, 256))),
                "mcts_sim": self.mcts_sim,
                "combat_solver": self.combat_solver,
                "beam_width": self.beam_width,
                "beam_depth": self.beam_depth,
                "beam_max_nodes": self.beam_max_nodes,
                "route_lanes": self.route_lanes,
                "lane_budget": self.lane_budget,
                "width_portfolio": self.width_portfolio,
                "weights": self.weights,
                "value_mode": self.value_mode,
                "rollout_max_steps": self.rollout_max_steps,
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
    # 终局牌组/遗物计数（奖励塑形加成用；返回 int，多进程可 pickle）。
    # 真实环境 env.gc 必存在；测试用 FakeEnv 可能无 gc，缺省记 0 不报错。
    gc = getattr(env, "gc", None)
    if gc is not None:
        upgrade_count = _count_upgraded(gc.deck)
        relic_count = _count_relics(gc.relics)
    else:
        upgrade_count = 0
        relic_count = 0
    if logger:
        logger.finish(final_floor=final_floor, win=win)
    return traj, final_floor, win, upgrade_count, relic_count


def _collect_worker(args):
    """子进程 worker：重建 scorer + env 跑一局，返回 (traj, floor, win)。

    scorer 固定放 CPU —— 每个子进程建 CUDA context 约 300~500MB，
    16 进程会顶爆 8GB 显存；而 (1024,512) 的 CPU 推理单次仅约 1ms，
    相对游戏模拟（约 2s/局）可忽略。

    关键：必须把 torch 线程数压到 1。torch 默认每进程开满物理核数（16）线程，
    16 个 worker × 16 线程 = 256 线程抢 32 个硬件线程（8× 超订），
    上下文切换会吃掉绝大部分算力（实测 CPU 利用率从应有的 ~100% 掉到 7%）。
    """
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch as _torch
    _torch.set_num_threads(1)
    from env.lightspeed_env import LightspeedEnv
    from models.scorer import Scorer

    seed, state, cfg = args
    scorer = Scorer(obs_dim=cfg["obs_dim"], cand_dim=cfg["cand_dim"],
                    hidden=tuple(cfg["hidden"]))
    scorer.load_state_dict(state)
    scorer.eval()
    env = LightspeedEnv(mcts_simulations=cfg["mcts_sim"],
                        combat_solver=cfg.get("combat_solver", "mcts"),
                        beam_width=cfg.get("beam_width", 4),
                        beam_depth=cfg.get("beam_depth", 8),
                        beam_max_nodes=cfg.get("beam_max_nodes", 3000),
                        route_lanes=cfg.get("route_lanes", True),
                        lane_budget=cfg.get("lane_budget"),
                        width_portfolio=cfg.get("width_portfolio", False),
                        weights=cfg.get("weights"),
                        value_mode=cfg.get("value_mode", "static"),
                        rollout_max_steps=cfg.get("rollout_max_steps", 400))
    return _run_episode(seed, scorer, env, device="cpu")
