"""正式训练入口：REINFORCE / PPO on sts_lightspeed。

用法:
    .venv/Scripts/python -m training.run_training --games 3000 --mcts-sim 8000
    .venv/Scripts/python -m training.run_training --algo ppo --games 5000 --mcts-sim 2000
"""

import argparse
import json
import os
import time

from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer
from training.reinforce import ReinforceTrainer


def _auto_device():
    """自动选择训练设备：有 CUDA 用 cuda，否则 cpu。"""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["reinforce", "ppo"], default="reinforce")
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--mcts-sim", type=int, default=8000)
    p.add_argument("--lr", type=float, default=None,
                   help="学习率；默认 reinforce=1e-3, ppo=3e-4")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-dir", default="logs/m2-training")
    p.add_argument("--checkpoint", default="checkpoints/scorer.pt")
    p.add_argument("--init-from", default=None,
                   help="从已有 checkpoint 继续训练（仅 reinforce 支持与 (128,128) 兼容的 ckpt）")
    p.add_argument("--win-bonus", type=float, default=50,
                   help="通关奖励：G = final_floor + win_bonus * win")
    p.add_argument("--seed", type=int, default=None,
                   help="随机种子。设定后 random/torch 同时播种，两次运行会走完全"
                        "相同的对局序列与网络初始化，可做配对 A/B（唯一变量是表征）")
    p.add_argument("--device", type=str, default=None,
                   help="训练设备；默认 auto：有 CUDA 用 cuda，否则 cpu")
    p.add_argument("--workers", type=int, default=None,
                   help="并行收集进程数（游戏模拟是单线程 CPU 瓶颈）；"
                        "默认 auto = min(16, CPU 核数)；1 表示串行")
    p.add_argument("--hidden", default="512,256",
                   help="Scorer 隐藏层，逗号分隔。V6=(512,256)；"
                        "V7 试过 (1024,512) 但实测无增益")
    args = p.parse_args()

    workers = args.workers
    if workers is None:
        workers = min(16, os.cpu_count() or 1)

    if args.seed is not None:
        import random as _random
        _random.seed(args.seed)
        import torch as _torch
        _torch.manual_seed(args.seed)

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.algo == "ppo":
        from models.scorer import ValueNet
        from training.ppo import PPOTrainer
        hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
        scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=hidden)
        value_net = ValueNet(obs_dim=OBS_DIM)
        lr = args.lr if args.lr is not None else 3e-4
        if args.init_from and os.path.exists(args.init_from):
            import torch
            scorer.load_state_dict(torch.load(args.init_from, weights_only=True))
            value_path = args.init_from + ".value.pt"
            if os.path.exists(value_path):
                value_net.load_state_dict(torch.load(value_path, weights_only=True))
            print(f"Resume from {args.init_from}")
        elif args.init_from:
            print("警告：--init-from 指定的 checkpoint 不存在，从零训练")
        trainer = PPOTrainer(scorer, value_net, lr=lr, mcts_sim=args.mcts_sim,
                             log_dir=args.log_dir, win_bonus=args.win_bonus,
                             device=args.device or _auto_device())
    else:
        scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=(128, 128))
        lr = args.lr if args.lr is not None else 1e-3
        if args.init_from:
            import torch
            scorer.load_state_dict(torch.load(args.init_from, weights_only=True))
            print(f"Warm start from {args.init_from}")
        trainer = ReinforceTrainer(scorer, lr=lr, mcts_sim=args.mcts_sim,
                                   log_dir=args.log_dir, win_bonus=args.win_bonus)

    # 训练曲线追加写入 JSONL，供监控/复盘
    curve_path = os.path.join(args.log_dir, "training_curve.jsonl")

    def log_fn(rec):
        with open(curve_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def eval_fn(trainer_ref, game_idx):
        # 简单自评：用当前策略 greedy 跑 3 个留出种子
        from evaluation.evaluate import run_one, make_eval_seeds
        import torch
        seeds = make_eval_seeds(100)[:3]
        floors = []
        for s in seeds:
            r = run_one(s, _GreedyPolicy(trainer_ref.scorer), mcts_sim=2000)
            floors.append(r["final_floor"])
        return {"eval_games": len(seeds), "eval_avg_floor": sum(floors) / len(floors)}

    print(f"Training: algo={args.algo} {args.games} games, mcts_sim={args.mcts_sim}, lr={lr}")
    t0 = time.time()
    if args.algo == "ppo":
        # 定期保存 checkpoint（每 ~400 局一次），防止中断丢失进度
        import torch as _torch
        def _save_fn(game_idx):
            _torch.save(scorer.state_dict(), args.checkpoint)
            _torch.save(value_net.state_dict(), args.checkpoint + ".value.pt")
        hist = trainer.train(n_games=args.games, log_fn=log_fn,
                             save_fn=(lambda i: _save_fn(i) if i % 400 == 0 else None),
                             workers=workers)
    else:
        hist = trainer.train(n_games=args.games, eval_every=args.eval_every,
                             eval_fn=eval_fn, log_fn=log_fn)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed / args.games:.1f}s/game)")

    import torch
    torch.save(scorer.state_dict(), args.checkpoint)
    if args.algo == "ppo":
        # value net 存 sidecar（评估只需要 scorer，保持 checkpoint 格式兼容）
        torch.save(value_net.state_dict(), args.checkpoint + ".value.pt")
    print(f"Checkpoint saved: {args.checkpoint}")

    # 最终汇总
    floors = [h["mean_floor"] for h in hist]
    n = len(floors)
    print(f"Last 100 games avg floor: {sum(floors[-100:]) / min(100, n):.1f}")


class _GreedyPolicy:
    """评估用：取 scorer 最高分候选（不采样）。"""
    def __init__(self, scorer):
        self.scorer = scorer

    def choose(self, obs, candidates):
        import torch
        with torch.no_grad():
            obs_t = torch.tensor(obs.vec, dtype=torch.float32)
            cands_t = torch.tensor([c.desc_vec for c in candidates],
                                   dtype=torch.float32)
            scores = self.scorer(obs_t, cands_t)
        chosen_pos = int(scores.argmax().item())
        return candidates[chosen_pos].idx


if __name__ == "__main__":
    main()
