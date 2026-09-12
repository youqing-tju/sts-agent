"""BC 预热（后训练 / SFT）：用人类 .run 样本对 Scorer 做监督学习。

对标 LLM 三阶段训练：
    预训练   = 表征 + 知识库（obs/cand 编码，已由 V10 表征完成）
    **后训练 = 本脚本（行为克隆，模仿人类专家的选牌决策）**
    RL 微调  = training/run_training.py（PPO / REINFORCE，超越专家）

产出 `checkpoints/v10_bc.pt`，作为 V10 RL 训练的起点（避免从随机策略开始）。

用法：
    .venv/Scripts/python.exe -m training.bc_warmup --data data/human_bc/card_reward.npz
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer


def _load(path):
    d = np.load(path, allow_pickle=False)
    obs, cands = d["obs"], d["cands"]
    labels, mask = d["labels"], d["mask"]
    meta = [json.loads(x) for x in d["meta"]]
    return obs, cands, labels, mask, meta


def _split_by_file(meta, val_frac, seed):
    """按 .run 文件分组划分——同一局的样本不能跨 train/val（否则泄漏）。"""
    files = sorted({m["file"] for m in meta})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(files))
    n_val = max(1, int(round(len(files) * val_frac)))
    val_files = {files[i] for i in perm[:n_val]}
    is_val = np.array([m["file"] in val_files for m in meta])
    return ~is_val, is_val, len(val_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/human_bc/card_reward.npz")
    ap.add_argument("--out", default="checkpoints/v10_bc.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", default="512,256")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = args.data if os.path.isabs(args.data) else os.path.join(_ROOT, args.data)
    obs, cands, labels, mask, meta = _load(path)
    # 防御：标签必须落在候选矩阵宽度内（越界会让 cross_entropy 触发 CUDA assert）
    valid = labels < cands.shape[1]
    if not valid.all():
        print(f"[warn] 过滤 {int((~valid).sum())} 条越界标签样本")
        obs, cands, labels, mask = obs[valid], cands[valid], labels[valid], mask[valid]
        meta = [m for m, v in zip(meta, valid) if v]
    n = len(obs)
    print(f"[data] {path}")
    print(f"[data] 样本 {n} 条，obs {obs.shape[1:]}，候选 {cands.shape[1:]}"
          f"（DESC_VEC_DIM={DESC_VEC_DIM}，OBS_DIM={OBS_DIM}）")
    if obs.shape[1] != OBS_DIM or cands.shape[2] != DESC_VEC_DIM:
        raise SystemExit(f"维度不匹配：obs {obs.shape[1]} vs {OBS_DIM}，"
                         f"cand {cands.shape[2]} vs {DESC_VEC_DIM}")

    tr, va, n_val_files = _split_by_file(meta, args.val_frac, args.seed)
    print(f"[split] train {tr.sum()} / val {va.sum()}（val 局数 {n_val_files}）")
    # 人类真实候选数分布（用于判断"随机基线"）
    n_cands = mask.sum(axis=1)
    print(f"[stats] 候选数分布: {np.bincount(n_cands.astype(int)).tolist()}"
          f"（索引=候选数）")

    hidden = tuple(int(x) for x in args.hidden.split(","))
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=hidden).to(dev)
    opt = torch.optim.Adam(scorer.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    O = torch.tensor(obs, dtype=torch.float32, device=dev)
    C = torch.tensor(cands, dtype=torch.float32, device=dev)
    L = torch.tensor(labels, dtype=torch.long, device=dev)
    M = torch.tensor(mask, dtype=torch.float32, device=dev)
    idx_tr = torch.tensor(np.where(tr)[0], device=dev)
    idx_va = torch.tensor(np.where(va)[0], device=dev)

    def evaluate(idx):
        scorer.eval()
        correct = total = 0
        with torch.no_grad():
            for i in range(0, len(idx), args.batch):
                b = idx[i:i + args.batch]
                s = scorer.forward_padded(O[b], C[b]).masked_fill(M[b] == 0, -1e9)
                pred = s.argmax(dim=1)
                correct += int((pred == L[b]).sum())
                total += len(b)
        return correct / max(1, total)

    # 随机基线：每次随机挑一个真实候选
    rng = np.random.default_rng(args.seed)
    rnd_correct = sum(1 for i in range(n)
                      if rng.integers(0, int(n_cands[i])) == labels[i])
    print(f"[baseline] 随机选 = {rnd_correct / n * 100:.1f}%")

    best_acc, best_state, best_ep = -1.0, None, -1
    hist = []
    for ep in range(1, args.epochs + 1):
        scorer.train()
        perm = idx_tr[torch.randperm(len(idx_tr), device=dev)]
        tot_loss = 0.0
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            s = scorer.forward_padded(O[b], C[b]).masked_fill(M[b] == 0, -1e9)
            loss = F.cross_entropy(s, L[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.detach()) * len(b)
        sched.step()
        tr_acc = evaluate(idx_tr)
        va_acc = evaluate(idx_va)
        hist.append({"epoch": ep, "loss": tot_loss / max(1, len(perm)),
                     "train_acc": tr_acc, "val_acc": va_acc,
                     "lr": sched.get_last_lr()[0]})
        if va_acc > best_acc:
            best_acc, best_ep = va_acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f"  epoch {ep:>3} loss={tot_loss / max(1, len(perm)):.4f} "
                  f"train={tr_acc * 100:.1f}% val={va_acc * 100:.1f}%")

    print(f"\n[result] 最佳 val 一致率 = {best_acc * 100:.1f}% (epoch {best_ep})"
          f"｜随机基线 {rnd_correct / n * 100:.1f}%")
    out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(best_state, out)
    json.dump({"history": hist, "best_val_acc": best_acc, "best_epoch": best_ep,
               "n_samples": int(n), "n_val_files": n_val_files,
               "random_baseline": rnd_correct / n},
              open(out + ".history.json", "w", encoding="utf-8"), indent=1)
    print(f"[save] {out}（+ .history.json）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
