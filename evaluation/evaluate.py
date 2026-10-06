"""评估器：固定留出种子 + ProcessPoolExecutor 并行。

- make_eval_seeds：母种子 20260831 采样，可复现、互不相同；训练侧不得使用。
- run_one：单局评估（建 LightspeedEnv、跑完整局、可选 JSONL 日志）。
- evaluate：多进程并行。Windows 下 spawn 启动，worker 函数与 policy 实例
  都必须顶层可序列化。
- CLI：python -m evaluation.evaluate --seeds 100 --workers 8 --mcts-sim 200

【进程模型】C++ NNInterface 单进程只能观测一局（见 env/lightspeed_env.py
docstring），因此 max_tasks_per_child=1：一进程一局，每局都有真实 NN vec，
同时规避单例绑定后的 access violation。
【懒加载】slaythespire / RunLogger 只在 run_one 内部导入：父进程（CLI/汇总）
与纯函数单测都不需要加载 .pyd。
"""
import argparse
import random
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor

# 维度默认值直接引用编码器常量，避免散落硬编码在改维度时漏改
# （这两个模块都是纯 Python，不加载 torch/.pyd，不违反上面的懒加载约定）
from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM

# 留出种子母种子：固定，训练侧禁止使用本函数输出的种子
_MASTER_SEED = 20260831

# 防御性步数上限：与 tests/test_lightspeed_env.py 的全局随机冒烟一致
_MAX_STEPS = 500

_WIN_OUTCOME = "PLAYER_VICTORY"


def make_eval_seeds(n, master=_MASTER_SEED):
    """采 n 个留出评估种子：可复现、互不相同。"""
    return random.Random(master).sample(range(2**31), n)


class RandomPolicy:
    """内置随机策略：harness 自测/对照用。顶层类，可序列化；不依赖 agents 包。"""

    def choose(self, obs, candidates):
        return random.choice(candidates).idx


def run_one(seed, policy, mcts_sim=200, log_dir=None, env_kwargs=None):
    """用 policy 打完整局，返回 {"final_floor": int, "win": bool}。

    log_dir 非空时用 RunLogger 记录每个宏观决策
    （floor/screen/candidates 的 label 列表/chosen label/hp）与终局。
    """
    from env.lightspeed_env import LightspeedEnv

    env = LightspeedEnv(mcts_simulations=mcts_sim, **(env_kwargs or {}))
    obs = env.reset(seed)
    logger = None
    if log_dir:
        from logging.run_logger import RunLogger
        logger = RunLogger(log_dir=log_dir, seed=seed)

    done, steps, res = False, 0, None
    while not done and steps < _MAX_STEPS:
        cands = env.candidates()
        if cands:
            idx = policy.choose(obs, cands)
            if logger:
                chosen = next((c.label for c in cands if c.idx == idx), str(idx))
                logger.log_decision(floor=obs.floor, screen=obs.screen,
                                    candidates=[c.label for c in cands],
                                    chosen=chosen, hp=obs.hp)
        else:
            idx = -1  # 战斗内部（防御性分支；正常流程由 env 自动推进）
        res = env.step(idx)
        obs, done = res.obs, res.done
        steps += 1

    outcome = res.info.get("outcome", "") if res else ""
    final_floor = obs.floor
    win = outcome == _WIN_OUTCOME
    if logger:
        logger.finish(final_floor=final_floor, win=win)
    return {"final_floor": final_floor, "win": win}


def _run_one_worker(args):
    """ProcessPoolExecutor worker：顶层函数，参数全部可序列化。"""
    seed, policy, mcts_sim, log_dir, env_kwargs = args
    return run_one(seed, policy, mcts_sim=mcts_sim, log_dir=log_dir,
                   env_kwargs=env_kwargs)


def evaluate(policy, n_seeds, workers, mcts_sim=200, log_dir=None, env_kwargs=None):
    """在 n_seeds 个留出种子上并行评估 policy，返回与种子顺序一致的结果列表。

    policy 实例会随任务 pickle 到子进程，必须顶层可序列化。
    一进程一局（max_tasks_per_child=1），见模块 docstring 的进程模型说明。
    """
    seeds = make_eval_seeds(n_seeds)
    args = [(s, policy, mcts_sim, log_dir, env_kwargs) for s in seeds]
    with ProcessPoolExecutor(max_workers=workers,
                             max_tasks_per_child=1) as pool:
        return list(pool.map(_run_one_worker, args))


def summarize(results):
    """汇总 run_one 结果：avg_floor / std_floor（总体标准差）/ win_rate / n。"""
    floors = [r["final_floor"] for r in results]
    wins = [r["win"] for r in results]
    return {
        "avg_floor": statistics.fmean(floors),
        "std_floor": statistics.pstdev(floors) if len(floors) > 1 else 0.0,
        "win_rate": statistics.fmean(wins),
        "n": len(results),
    }


def _make_policy(name):
    """按名字构造策略。worker 子进程内惰性 import，保持父进程干净。"""
    if name == "random":
        return RandomPolicy()
    if name == "rule":
        from agents.rule_policy import RulePolicy
        return RulePolicy()
    raise SystemExit(f"未知策略: {name!r}（可选: random / rule）")


class GreedyScorerPolicy:
    """载入训练好的 Scorer 权重，逐候选打分取 argmax（不采样）。

    hidden=(512, 256) 需与训练时一致；obs_dim/cand_dim 因表征版本而异
    （默认取当前编码器常量，加载旧版本权重时需显式传入）。
    """

    def __init__(self, path, obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=(512, 256)):
        import torch
        from models.scorer import Scorer
        self.path = path
        self.obs_dim = obs_dim
        self.cand_dim = cand_dim
        self.hidden = tuple(hidden)
        self.scorer = Scorer(obs_dim=obs_dim, cand_dim=cand_dim, hidden=self.hidden)
        self.scorer.load_state_dict(torch.load(path, weights_only=True))
        self.scorer.eval()

    def choose(self, obs, candidates):
        import torch
        with torch.no_grad():
            obs_t = torch.tensor(obs.vec, dtype=torch.float32)
            cands_t = torch.tensor([c.desc_vec for c in candidates],
                                   dtype=torch.float32)
            scores = self.scorer(obs_t, cands_t)
        return candidates[int(scores.argmax().item())].idx


def main(argv=None):
    p = argparse.ArgumentParser(prog="evaluation.evaluate",
                                description="在固定留出种子上并行评估策略")
    p.add_argument("--seeds", type=int, default=100, help="评估种子数")
    p.add_argument("--workers", type=int, default=8, help="并行进程数")
    p.add_argument("--mcts-sim", type=int, default=200, help="战斗 MCTS 预算")
    p.add_argument("--policy", default="random", help="策略名（v1 内置 random）")
    p.add_argument("--checkpoint", default=None,
                   help="Scorer 权重路径；设定后用该权重的 greedy 策略评估")
    p.add_argument("--obs-dim", type=int, default=OBS_DIM, help="观测维度（旧版权重需显式指定）")
    p.add_argument("--cand-dim", type=int, default=DESC_VEC_DIM, help="候选维度（旧版权重需显式指定）")
    p.add_argument("--hidden", default="512,256",
                   help="Scorer 隐藏层，逗号分隔。PPO=512,256；REINFORCE=128,128")
    p.add_argument("--log-dir", default=None, help="可选：JSONL 日志目录")
    p.add_argument("--combat-solver", choices=["mcts", "beam"], default="mcts",
                   help="战斗求解器（与训练侧同名参数一致）")
    p.add_argument("--beam-width", type=int, default=2, help="Beam 束宽")
    p.add_argument("--beam-depth", type=int, default=6, help="Beam 最大深度")
    p.add_argument("--beam-max-nodes", type=int, default=300, help="Beam 单次最大展开节点数")
    args = p.parse_args(argv)

    if args.checkpoint:
        hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
        policy = GreedyScorerPolicy(args.checkpoint, obs_dim=args.obs_dim,
                                    cand_dim=args.cand_dim, hidden=hidden)
        pname = args.checkpoint
    else:
        policy = _make_policy(args.policy)
        pname = args.policy
    env_kwargs = {}
    if args.combat_solver == "beam":
        env_kwargs = {"combat_solver": "beam", "beam_width": args.beam_width,
                      "beam_depth": args.beam_depth,
                      "beam_max_nodes": args.beam_max_nodes}
    results = evaluate(policy, n_seeds=args.seeds, workers=args.workers,
                       mcts_sim=args.mcts_sim, log_dir=args.log_dir,
                       env_kwargs=env_kwargs)
    s = summarize(results)
    print(f"policy={pname} n={s['n']} mcts_sim={args.mcts_sim}")
    print(f"avg_floor={s['avg_floor']:.1f} std_floor={s['std_floor']:.1f} "
          f"win_rate={s['win_rate']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
