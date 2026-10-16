# -*- coding: utf-8 -*-
"""宏观指标提取 —— handoff §3「Beam 宽度 ↔ 抓牌质量」对照实验所需。

**为什么需要这个**：比较「不同 `beam_width` 训练的 RL」时**不能只看楼层** ——
战斗执行强度会直接改变楼层，从而掩盖"抓牌策略"本身的差异。
要看**宏观指标**：通关局的牌组规模、遗物数、到达后期时的资源。

⚠️ **局限（已实测）**：`Obs.deck` 是 `CardId.name` 列表（如 `STRIKE_RED`），
**不含升级信息** → 「升级数」这个指标目前**拿不到**，
除非扩展 `Obs` 的字段（会牵动 `obs_vec` 编码与权重形状，暂不做）。
本工具提供：`floor / win / deck 大小 / relics 数 / gold` + **各 Act 进入点快照**。

用法（务必在项目根运行）：
  .venv/Scripts/python.exe scripts/probes/_tmp_macro_metrics.py \
      --checkpoint checkpoints/v14_defbeam_2k.pt --seeds 100 --workers 6 \
      --beam-weights-preset defensive --out logs/macro-v14.json
"""
import argparse
import json
import os
import statistics as st
import sys
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: Act 边界（按楼层）；实测楼层会超过 50（heart 战），所以第三段放宽
_ACT_RANGES = ((1, 1, 16), (2, 17, 33), (3, 34, 999))


def _act_of(floor: int) -> int:
    for a, lo, hi in _ACT_RANGES:
        if lo <= floor <= hi:
            return a
    return 0


def _run_one(seed, policy, mcts_sim, env_kwargs):
    """跑完整一局，返回终局指标 + 各 Act 进入点的快照。"""
    from env.lightspeed_env import LightspeedEnv
    from evaluation.evaluate import _WIN_OUTCOME

    env = LightspeedEnv(mcts_simulations=mcts_sim, **(env_kwargs or {}))
    obs = env.reset(seed)
    act_snap = {}
    done, steps, res, cur_act = False, 0, None, 0
    while not done and steps < 2000:
        a = _act_of(obs.floor)
        if a > cur_act:                      # 刚进入新 Act → 记快照
            cur_act = a
            act_snap[str(a)] = {"floor": obs.floor, "hp": obs.hp, "gold": obs.gold,
                                "deck": len(obs.deck), "relics": len(obs.relics)}
        cands = env.candidates()
        idx = policy.choose(obs, cands) if cands else -1
        res = env.step(idx)
        obs, done = res.obs, res.done
        steps += 1

    outcome = res.info.get("outcome", "") if res else ""
    return {"seed": seed, "final_floor": obs.floor, "win": outcome == _WIN_OUTCOME,
            "deck": len(obs.deck), "relics": len(obs.relics), "gold": obs.gold,
            "hp": obs.hp, "act_snap": act_snap, "steps": steps}


def _worker(args):
    """ProcessPoolExecutor worker —— 必须是模块级顶层函数（可 pickle）。"""
    return _run_one(*args)


def _mean(xs):
    return round(st.mean(xs), 2) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--mcts-sim", type=int, default=20000)
    ap.add_argument("--beam-width", type=int, default=2)
    ap.add_argument("--beam-depth", type=int, default=6)
    ap.add_argument("--beam-max-nodes", type=int, default=300)
    ap.add_argument("--beam-weights-preset", default=None)
    ap.add_argument("--beam-value-mode", choices=["static", "rollout"], default="static",
                    help="节点估值方式（rollout = 打到终局再打分，含终局信息）")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from evaluation.evaluate import GreedyScorerPolicy, make_eval_seeds

    policy = GreedyScorerPolicy(a.checkpoint)
    env_kwargs = {"combat_solver": "beam", "beam_width": a.beam_width,
                  "beam_depth": a.beam_depth, "beam_max_nodes": a.beam_max_nodes,
                  "value_mode": a.beam_value_mode}
    if a.beam_weights_preset:
        from combat_beam import PRESETS
        env_kwargs["weights"] = PRESETS[a.beam_weights_preset]

    seeds = make_eval_seeds(a.seeds)
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows = list(pool.map(_worker, [(s, policy, a.mcts_sim, env_kwargs) for s in seeds]))

    wins = [r for r in rows if r["win"]]
    act3 = [r for r in rows if "3" in r["act_snap"]]
    print("checkpoint=%s  preset=%s  n=%d" % (a.checkpoint, a.beam_weights_preset, len(rows)))
    print("win_rate=%.1f%%  avg_floor=%.2f  (SE≈%.1f pp)"
          % (100 * len(wins) / len(rows), _mean([r["final_floor"] for r in rows]),
             100 * (len(wins) / len(rows) * (1 - len(wins) / len(rows)) / len(rows)) ** 0.5))
    print("终局 deck 均值=%.1f  relics 均值=%.1f  gold 均值=%.0f"
          % (_mean([r["deck"] for r in rows]), _mean([r["relics"] for r in rows]),
             _mean([r["gold"] for r in rows])))
    print("\n⭐ 宏观指标（通关局，n=%d）—— 「抓牌质量」的主指标" % len(wins))
    if wins:
        print("   deck 均值=%.1f   relics 均值=%.1f   gold 均值=%.0f"
              % (_mean([r["deck"] for r in wins]), _mean([r["relics"] for r in wins]),
                 _mean([r["gold"] for r in wins])))
    print("   到达 Act3 的局数=%d (%.1f%%)" % (len(act3), 100 * len(act3) / len(rows)))
    if act3:
        for act in ("1", "2", "3"):
            sub = [r["act_snap"][act] for r in act3 if act in r["act_snap"]]
            if sub:
                print("   Act%s 进入点: deck=%.1f relics=%.1f gold=%.0f hp=%.1f (n=%d)"
                      % (act, _mean([s["deck"] for s in sub]), _mean([s["relics"] for s in sub]),
                         _mean([s["gold"] for s in sub]), _mean([s["hp"] for s in sub]), len(sub)))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        print("\n已写入 %s" % a.out)


if __name__ == "__main__":
    main()
