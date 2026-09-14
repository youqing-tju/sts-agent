"""Self-play 专家数据生成（胜局过滤 / rejection sampling）。

用当前最强模型 E1（`checkpoints/v10_scratch.pt`）以**采样**（非 argmax）策略
跑很多局，只保留「导向胜利」的决策轨迹，作为后续 BC 预热 + RL 微调的专家数据。

背景（关键）：之前用人类 .run（A20 采集）做 BC 预热是**负迁移**——
A20 保守打法在 A0 反而拖累（实测 BC 起点 27.53 层 / 6.7% 通关率，
而从零训练 46.30 层 / 56.7%）。故本轮改用自有模拟器在 A0 场景生成专家数据。

    输出 npz 与 `training/human_data.py` 的 BC 数据**完全兼容**，
    `training/bc_warmup.py` 可直接复用（无需改动）：
        obs    : (N, 477)    float32
        cands  : (N, 8, 369) float32   (MAX_CANDS=8，不足补零)
    labels : (N,)       int64      (候选列表中被选中的下标 pos)
    mask   : (N, 5)     float32    (真实候选=1，padding=0)
    meta   : (N,)       object     (每行一个 JSON 字符串)

用法：
    .venv/Scripts/python.exe -m training.self_play_data --games 10 --out data/self_play/e1_wins.npz
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

from models.obs_encoder import OBS_DIM
from models.candidate_encoder import DESC_VEC_DIM
from models.scorer import Scorer
from env.lightspeed_env import LightspeedEnv

# 维度常量全部从编码器取，不硬编码（与 human_data / bc_warmup 同源）
MAX_CANDS = 8             # 候选宽度（原 5 → 8：救回 REWARDS/SHOP/CARD_SELECT 多候选屏）
_MAX_STEPS = 500          # 防御性步数上限，与 ppo.py / evaluate.py 一致
_WIN_OUTCOME = "PLAYER_VICTORY"
_CKPT = "checkpoints/v10_scratch.pt"


def _run_episode(seed, scorer, env, device="cpu", temperature=1.0):
    """用采样策略跑一局，返回 (samples, final_floor, win, skipped)。

    samples   : 通过的（候选数 ≤ MAX_CANDS）决策点列表，每项为 dict
                {"obs","cands"(已补齐到 MAX_CANDS×DESC),"label","mask","screen",
                 "seed","floor"}
    win       : 是否通关
    final_floor : 终局楼层
    skipped   : {screen: 被跳过的样本数}（候选数 > MAX_CANDS 的屏）

    逻辑照抄 training/ppo.py 的 _run_episode：cand_vecs 直接取
    env.candidates() 返回的 Candidate.desc_vec（不再重新编码）。不同点：
    - 每个决策点都把样本存下来（供 BC 用）；
    - 候选数超过 MAX_CANDS 时**不存储该样本但照常推进游戏**（否则 label 越界、
      实测会触发 CUDA assert），并在 skipped 中计数；
    - 额外记录 screen 供各屏分布统计。
    - temperature：采样温度（T=1 即原始 PPO 采样；T<1 更尖锐、贴近 greedy）。
    """
    env.reset(seed)
    samples, skipped = [], {}
    obs = env.observe()
    done, steps, res = False, 0, None
    while not done and steps < _MAX_STEPS:
        cands = env.candidates()
        if not cands:
            res = env.step(-1)        # 防御性分支；正常由 env 自动推进战斗
        else:
            cand_vecs = [c.desc_vec for c in cands]
            with torch.no_grad():
                obs_t = torch.tensor(obs.vec, dtype=torch.float32, device=device)
                cands_t = torch.tensor(cand_vecs, dtype=torch.float32, device=device)
                # 温度：logits / T 后再 softmax；T=1 等价于原始采样
                logp = F.log_softmax(scorer(obs_t, cands_t) / max(temperature, 1e-3),
                                     dim=-1)
                pos = int(torch.multinomial(logp.exp(), 1).item())
            n = len(cands)
            if n <= MAX_CANDS:
                pad = [0.0] * DESC_VEC_DIM
                vecs = list(cand_vecs) + [list(pad)] * (MAX_CANDS - n)
                mask = [1.0] * n + [0.0] * (MAX_CANDS - n)
                samples.append({
                    "obs": np.asarray(obs.vec, dtype=np.float32),
                    "cands": np.asarray(vecs, dtype=np.float32),
                    "label": pos,
                    "mask": np.asarray(mask, dtype=np.float32),
                    "screen": obs.screen,
                    "seed": seed,
                    "floor": int(obs.floor),
                })
            else:
                # 候选过多（如 SHOP 十几个商品）：跳过该样本，避免 label 越界
                skipped[obs.screen] = skipped.get(obs.screen, 0) + 1
            res = env.step(cands[pos].idx)
        obs, done = res.obs, res.done
        steps += 1
    final_floor = obs.floor
    win = bool(res and res.info.get("outcome") == _WIN_OUTCOME)
    return samples, final_floor, win, skipped


def _parse_keep(spec):
    """解析 --keep：wins | floor>=N | all。返回 (mode, floor)。"""
    if spec == "wins":
        return "wins", None
    if spec == "all":
        return "all", None
    if spec.startswith("floor>="):
        return "floor", int(spec.split(">=")[1])
    raise argparse.ArgumentTypeError(f"--keep 取值非法: {spec!r}")


def _keep_game(mode, final_floor, win, floor):
    if mode == "all":
        return True
    if mode == "wins":
        return bool(win)
    if mode == "floor":
        return final_floor >= floor
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default="data/self_play/e1_wins.npz")
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--mcts-sim", type=int, default=8000)
    ap.add_argument("--keep", default="wins",
                    help="wins（默认，只留通关局）| floor>=N（终局楼层≥N）| all")
    ap.add_argument("--min-floor", type=int, default=None,
                    help="保留 final_floor>=N 的局（与 --keep 并存；设置后以它为准）")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="采样温度：logits/T 后 softmax（T=1 即原始采样，T<1 更尖锐）")
    ap.add_argument("--workers", type=int, default=1,
                    help="并行 worker 数（本轮固定单进程，勿改）")
    args = ap.parse_args()

    keep_mode, keep_floor = _parse_keep(args.keep)
    if args.workers != 1:
        print("[warn] --workers 必须为 1（后台训练占满 CPU），已强制单进程")
        args.workers = 1

    # ---- 加载 E1 权重 ----
    device = "cpu"
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=(512, 256))
    ckpt = os.path.join(_ROOT, _CKPT)
    if not os.path.exists(ckpt):
        raise SystemExit(f"找不到 E1 权重: {ckpt}")
    state = torch.load(ckpt, weights_only=True, map_location=device)
    scorer.load_state_dict(state)
    scorer.eval()
    # 单进程 + 采样推理，压线程避免与后台训练抢核
    torch.set_num_threads(1)
    print(f"[load] E1 {ckpt}  obs_dim={OBS_DIM} cand_dim={DESC_VEC_DIM} "
          f"hidden=(512,256)")

    env = LightspeedEnv(mcts_simulations=args.mcts_sim)

    obs_list, cands_list, labels_list, mask_list, meta_list = [], [], [], [], []
    game_report, screen_counts, skipped_total = [], {}, {}
    n_games = n_wins = 0
    n_discarded = 0

    for i in range(args.games):
        seed = args.seed_base + i
        samples, final_floor, win, skipped = _run_episode(
            seed, scorer, env, device=device, temperature=args.temperature)
        n_games += 1
        if win:
            n_wins += 1
        game_report.append({"seed": seed, "final_floor": int(final_floor), "win": bool(win)})
        for k, v in skipped.items():
            skipped_total[k] = skipped_total.get(k, 0) + v
        # 过滤：--min-floor 优先于 --keep；否则用 --keep 语义
        if args.min_floor is not None:
            keep = final_floor >= args.min_floor
            effective_keep = f"min-floor>={args.min_floor}"
        else:
            keep = _keep_game(keep_mode, final_floor, win, keep_floor)
            effective_keep = args.keep
        if not keep:
            n_discarded += 1
            continue
        for s in samples:
            obs_list.append(s["obs"])
            cands_list.append(s["cands"])
            labels_list.append(s["label"])
            mask_list.append(s["mask"])
            meta = {"seed": int(s["seed"]), "floor": int(s["floor"]),
                    "screen": s["screen"], "win": bool(win),
                    "final_floor": int(final_floor),
                    # file 供 bc_warmup._split_by_file 按局分组（防泄漏），
                    # 每局一个独立 group
                    "file": f"selfplay/{seed}"}
            meta_list.append(meta)
            screen_counts[s["screen"]] = screen_counts.get(s["screen"], 0) + 1

    # ---- 装箱 ----
    if obs_list:
        OBS = np.stack(obs_list)
        CANDS = np.stack(cands_list)
        LABELS = np.asarray(labels_list, dtype=np.int64)
        MASK = np.stack(mask_list)
        META = np.array([json.dumps(m) for m in meta_list])
    else:
        OBS = np.zeros((0, OBS_DIM), dtype=np.float32)
        CANDS = np.zeros((0, MAX_CANDS, DESC_VEC_DIM), dtype=np.float32)
        LABELS = np.zeros((0,), dtype=np.int64)
        MASK = np.zeros((0, MAX_CANDS), dtype=np.float32)
        META = np.array([], dtype=object)

    # ---- labels 有效性自查（必须报告）----
    if len(LABELS):
        n_real = MASK.sum(axis=1)
        valid = (LABELS < n_real) & (MASK[np.arange(len(LABELS)), LABELS] == 1.0)
        n_bad = int((~valid).sum())
    else:
        valid = np.array([], dtype=bool)
        n_bad = 0

    # ---- 报告 ----
    win_rate = (n_wins / n_games * 100) if n_games else 0.0
    print("\n================ self-play 数据报告 ================")
    print(f"[games] 总局数={n_games}  通关局数={n_wins}  通关率={win_rate:.1f}%")
    print(f"[keep = {effective_keep}] 保留局数={n_games - n_discarded}  丢弃局数={n_discarded}")
    # 按楼层分档分布（帮 team-lead 决定 --min-floor 取值）
    b = {"<20": 0, "20-29": 0, "30-39": 0, "40-49": 0}
    for g in game_report:
        ff = g["final_floor"]
        if ff < 20:
            b["<20"] += 1
        elif ff < 30:
            b["20-29"] += 1
        elif ff < 40:
            b["30-39"] += 1
        elif ff < 50:
            b["40-49"] += 1
    print(f"[floor-buckets] <20:{b['<20']} | 20-29:{b['20-29']} | "
          f"30-39:{b['30-39']} | 40-49:{b['40-49']} | 通关:{n_wins}")
    print(f"[samples] 保留决策点样本数={(len(LABELS))}")
    print(f"[shape] obs {OBS.shape} / cands {CANDS.shape} / labels {LABELS.shape} / mask {MASK.shape}")
    print(f"[label-check] 越界/无效 label = {n_bad} 条 / 共 {len(LABELS)} 条 "
          f"({ 'OK ✓' if n_bad == 0 else '!! FAIL'})")
    if screen_counts:
        top = sorted(screen_counts.items(), key=lambda kv: -kv[1])
        dist = ", ".join(f"{k}={v}" for k, v in top)
        print(f"[screen] 各屏分布（top）→ {dist}")
    else:
        print("[screen] 无保留样本")
    if skipped_total:
        sk = ", ".join(f"{k}={v}" for k, v in sorted(skipped_total.items(), key=lambda kv: -kv[1]))
        print(f"[skipped] 候选数>MAX_CANDS 被跳过的样本: {sk}")
    # 每局元数据汇总（含被丢弃的局）
    wins_floors = [g["final_floor"] for g in game_report if g["win"]]
    if wins_floors:
        print(f"[per-game] 通关局 final_floor min/mean/max = "
              f"{min(wins_floors)}/{sum(wins_floors)/len(wins_floors):.1f}/{max(wins_floors)}")
    disc = [g for g in game_report if not (
        (final_floor >= args.min_floor) if args.min_floor is not None
        else _keep_game(keep_mode, g["final_floor"], g["win"], keep_floor))]
    if disc:
        df = [g["final_floor"] for g in disc]
        print(f"[per-game] 丢弃局 final_floor min/mean/max = "
              f"{min(df)}/{sum(df)/len(df):.1f}/{max(df)}")

    # ---- 落盘 ----
    out = os.path.join(_ROOT, args.out)
    if n_bad == 0:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, obs=OBS, cands=CANDS, labels=LABELS,
                            mask=MASK, meta=META)
        size_mb = os.path.getsize(out) / 1e6
        print(f"[save] {out}  ({size_mb:.2f} MB)")
    else:
        print(f"[save] 跳过落盘：label 自检失败（{n_bad} 条无效），请排查")
        return 1
    print("===================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
