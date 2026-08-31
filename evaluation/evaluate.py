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


def run_one(seed, policy, mcts_sim=200, log_dir=None):
    """用 policy 打完整局，返回 {"final_floor": int, "win": bool}。

    log_dir 非空时用 RunLogger 记录每个宏观决策
    （floor/screen/candidates 的 label 列表/chosen label/hp）与终局。
    """
    from env.lightspeed_env import LightspeedEnv

    env = LightspeedEnv(mcts_simulations=mcts_sim)
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
    seed, policy, mcts_sim, log_dir = args
    return run_one(seed, policy, mcts_sim=mcts_sim, log_dir=log_dir)


def evaluate(policy, n_seeds, workers, mcts_sim=200, log_dir=None):
    """在 n_seeds 个留出种子上并行评估 policy，返回与种子顺序一致的结果列表。

    policy 实例会随任务 pickle 到子进程，必须顶层可序列化。
    一进程一局（max_tasks_per_child=1），见模块 docstring 的进程模型说明。
    """
    seeds = make_eval_seeds(n_seeds)
    args = [(s, policy, mcts_sim, log_dir) for s in seeds]
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
    """按名字构造策略。rule 策略由其它任务并行实现，此处不 import agents.rule_policy。"""
    if name == "random":
        return RandomPolicy()
    raise SystemExit(
        f"未知或暂未接入的策略: {name!r}（v1 内置 random；"
        f"rule 策略实现完成后由 Task 8 接入）"
    )


def main(argv=None):
    p = argparse.ArgumentParser(prog="evaluation.evaluate",
                                description="在固定留出种子上并行评估策略")
    p.add_argument("--seeds", type=int, default=100, help="评估种子数")
    p.add_argument("--workers", type=int, default=8, help="并行进程数")
    p.add_argument("--mcts-sim", type=int, default=200, help="战斗 MCTS 预算")
    p.add_argument("--policy", default="random", help="策略名（v1 内置 random）")
    p.add_argument("--log-dir", default=None, help="可选：JSONL 日志目录")
    args = p.parse_args(argv)

    policy = _make_policy(args.policy)
    results = evaluate(policy, n_seeds=args.seeds, workers=args.workers,
                       mcts_sim=args.mcts_sim, log_dir=args.log_dir)
    s = summarize(results)
    print(f"policy={args.policy} n={s['n']} mcts_sim={args.mcts_sim}")
    print(f"avg_floor={s['avg_floor']:.1f} std_floor={s['std_floor']:.1f} "
          f"win_rate={s['win_rate']:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
