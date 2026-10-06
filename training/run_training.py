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
    p.add_argument("--upgrade-bonus", type=float, default=0.0,
                   help="奖励塑形：终局升级卡数加成（每升级卡 +N 分）。"
                        "默认 0（完全向后兼容）。诊断见 logs/deck-strength-report.md")
    p.add_argument("--relic-bonus", type=float, default=0.0,
                   help="奖励塑形：终局遗物数加成（每遗物 +N 分）。"
                        "默认 0（完全向后兼容）")
    p.add_argument("--save-every-n", type=int, default=0,
                   help="每 N 局额外保存带局号的独立快照（不覆盖主 checkpoint）。"
                        "训练非单调、峰值常在中前期，用它避免丢峰值")
    p.add_argument("--entropy-coef", type=float, default=0.01,
                   help="PPO 熵正则系数（越大探索越强）。默认 0.01；"
                        "策略过早收敛/训练曲线平台化时可提到 0.03~0.05")
    p.add_argument("--seed", type=int, default=None,
                   help="随机种子。设定后 random/torch 同时播种，两次运行会走完全"
                        "相同的对局序列与网络初始化，可做配对 A/B（唯一变量是表征）")
    p.add_argument("--device", type=str, default=None,
                   help="训练设备；默认 auto：有 CUDA 用 cuda，否则 cpu")
    p.add_argument("--workers", type=int, default=None,
                   help="并行收集进程数（游戏模拟是单线程 CPU 瓶颈）；"
                        "默认 auto = min(16, CPU 核数)；1 表示串行")
    p.add_argument("--batch-games", type=int, default=16,
                   help="每攒满多少局做一次 PPO update；进程池大小被它封顶"
                        "（nw = min(workers, batch_games)），所以要用 >16 worker "
                        "必须同时抬高本值")
    p.add_argument("--combat-solver", choices=["mcts", "beam"], default="mcts",
                   help="战斗求解器：mcts（C++ Agent 自动打，默认）| beam（Python Beam Search 接管）")
    p.add_argument("--beam-width", type=int, default=4, help="Beam 束宽（仅 --combat-solver beam）")
    p.add_argument("--beam-depth", type=int, default=8, help="Beam 最大搜索深度")
    p.add_argument("--beam-max-nodes", type=int, default=3000, help="Beam 单次求解最大展开节点数")
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
                             batch_games=args.batch_games,
                             entropy_coef=args.entropy_coef,
                             upgrade_bonus=args.upgrade_bonus,
                             relic_bonus=args.relic_bonus,
                             combat_solver=args.combat_solver,
                             beam_width=args.beam_width,
                             beam_depth=args.beam_depth,
                             beam_max_nodes=args.beam_max_nodes,
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
            # 训练是「非单调」的（实测峰值常出现在中前期，见 logs/v10-final-report.md），
            # 若只覆盖单个文件会丢失沿途最好状态。--save-every-n N 额外保留带局号的
            # 独立快照，便于事后对多个 checkpoint 各自评估、择优。
            _n = getattr(args, "save_every_n", 0)
            if _n and game_idx > 0 and game_idx % _n == 0:
                snap = (args.checkpoint[:-3] + "_g%d.pt" % game_idx
                        if args.checkpoint.endswith(".pt")
                        else args.checkpoint + "_g%d.pt" % game_idx)
                _torch.save(scorer.state_dict(), snap)
                _torch.save(value_net.state_dict(), snap + ".value.pt")
                print("[snapshot] %s" % snap)
        # save_every_n 同时决定「主 checkpoint + 带局号快照」的保存间隔；未指定时沿用 400。
        # 注意坑：若外层仍固定按 400 调用 _save_fn，而只在内部判 i % save_every_n == 0，
        # 那么当 save_every_n 不是 400 的倍数时（如 1000），i 永远取不到 1000 这类值，
        # 快照间隔会被静默放大到 lcm(400, n)（1000 → 实际 2000）。故间隔必须在外层统一。
        _save_mod = args.save_every_n if args.save_every_n else 400
        hist = trainer.train(n_games=args.games, log_fn=log_fn,
                             save_fn=(lambda i: _save_fn(i) if i % _save_mod == 0 else None),
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
