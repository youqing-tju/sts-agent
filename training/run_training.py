"""正式训练入口：REINFORCE on sts_lightspeed。

用法:
    .venv/Scripts/python -m training.run_training --games 3000 --mcts-sim 8000
"""

import argparse
import json
import os
import time

from models.scorer import Scorer
from training.reinforce import ReinforceTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=3000)
    p.add_argument("--mcts-sim", type=int, default=8000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-dir", default="logs/m2-training")
    p.add_argument("--checkpoint", default="checkpoints/scorer.pt")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    scorer = Scorer(obs_dim=412, cand_dim=368, hidden=(128, 128))
    trainer = ReinforceTrainer(scorer, lr=args.lr, mcts_sim=args.mcts_sim,
                               log_dir=args.log_dir)

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

    print(f"Training: {args.games} games, mcts_sim={args.mcts_sim}, lr={args.lr}")
    t0 = time.time()
    hist = trainer.train(n_games=args.games, eval_every=args.eval_every,
                         eval_fn=eval_fn, log_fn=log_fn)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s ({elapsed / args.games:.1f}s/game)")

    import torch
    torch.save(scorer.state_dict(), args.checkpoint)
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
