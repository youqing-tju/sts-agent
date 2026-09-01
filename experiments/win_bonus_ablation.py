"""win bonus 消融实验：win_bonus=50 vs 200 对照。

从 checkpoints/long.pt 热启动，两组用相同初始权重各训 N 局
（mcts_sim=2000，lr=1e-3），随后各在 100 个留出种子上评估
（greedy 策略，MCTS@5000，多进程并行），输出对比表。

用法（项目根目录）:
    .venv/Scripts/python experiments/win_bonus_ablation.py
    .venv/Scripts/python experiments/win_bonus_ablation.py --games 2000 --eval-seeds 100

完整实验较慢（2000 局 x 2 组 + 200 局评估），常规回归测试不执行本脚本。
"""
import argparse
import copy
import json
import os
import sys

# 允许从项目根直接以脚本方式运行（python experiments/xxx.py）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from evaluation.evaluate import evaluate, summarize
from models.scorer import Scorer
from training.reinforce import ReinforceTrainer
from training.run_training import _GreedyPolicy


def run_group(win_bonus, init_state, args):
    """单组：热启动 -> 训练 -> 留出种子评估，返回汇总 dict。"""
    scorer = Scorer(obs_dim=412, cand_dim=408, hidden=(128, 128))
    scorer.load_state_dict(copy.deepcopy(init_state))

    log_dir = os.path.join(args.log_dir, f"win_bonus_{win_bonus:g}")
    os.makedirs(log_dir, exist_ok=True)
    trainer = ReinforceTrainer(scorer, lr=args.lr,
                               mcts_sim=args.train_mcts_sim,
                               log_dir=log_dir, win_bonus=win_bonus)

    curve_path = os.path.join(log_dir, "training_curve.jsonl")

    def log_fn(rec):
        with open(curve_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[win_bonus={win_bonus:g}] training {args.games} games "
          f"(mcts_sim={args.train_mcts_sim}, lr={args.lr}) ...")
    trainer.train(n_games=args.games, log_fn=log_fn)

    ckpt = os.path.join(args.checkpoint_dir, f"win_bonus_{win_bonus:g}.pt")
    torch.save(scorer.state_dict(), ckpt)
    print(f"[win_bonus={win_bonus:g}] checkpoint saved: {ckpt}")

    print(f"[win_bonus={win_bonus:g}] evaluating on {args.eval_seeds} held-out "
          f"seeds (mcts_sim={args.eval_mcts_sim}, workers={args.workers}) ...")
    results = evaluate(_GreedyPolicy(scorer), n_seeds=args.eval_seeds,
                       workers=args.workers, mcts_sim=args.eval_mcts_sim)
    summary = summarize(results)
    summary["win_bonus"] = win_bonus
    summary["checkpoint"] = ckpt
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(
        description="REINFORCE win bonus 消融：50 vs 200 对照实验")
    p.add_argument("--init-from", default="checkpoints/long.pt",
                   help="热启动 checkpoint")
    p.add_argument("--games", type=int, default=2000, help="每组训练局数")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-mcts-sim", type=int, default=2000)
    p.add_argument("--eval-seeds", type=int, default=100)
    p.add_argument("--eval-mcts-sim", type=int, default=5000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--win-bonuses", type=float, nargs="+", default=[50, 200])
    p.add_argument("--log-dir", default="logs/win-bonus-ablation")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    args = p.parse_args(argv)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    init_state = torch.load(args.init_from, weights_only=True)
    print(f"Warm start from {args.init_from}; groups: {args.win_bonuses}")

    summaries = [run_group(wb, init_state, args) for wb in args.win_bonuses]

    print("\n===== win bonus ablation (greedy, held-out seeds) =====")
    print(f"{'win_bonus':>10} {'avg_floor':>10} {'std_floor':>10} "
          f"{'win_rate':>10} {'n':>6}")
    for s in summaries:
        print(f"{s['win_bonus']:>10g} {s['avg_floor']:>10.1f} "
              f"{s['std_floor']:>10.1f} {s['win_rate']:>9.2%} {s['n']:>6}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
