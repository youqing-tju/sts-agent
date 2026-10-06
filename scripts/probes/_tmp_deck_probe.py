# -*- coding: utf-8 -*-
"""牌组强度探针（deck-strength 任务）。

只读分析：仅新增本脚本，不改任何生产代码。单进程。

目标：检验「牌组强度不够」假说——宏观决策没为 Act3 储备足够强的牌组。
方法：v10_seg3.pt + LightspeedEnv(MCTS=20000) + greedy 跑 N 局
      （种子 make_eval_seeds(200)[:N]）。
      每局在「刚进入 Act3」的第一个宏观决策点（env.gc.act == 3 且上次<3）
      记录牌组强度指标；终局记录 win / final_floor。
分组：A=通关局 / B=死在 Act3 局（final_floor 在 34-50）。
"""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import slaythespire as sts
from env.lightspeed_env import LightspeedEnv
from models.candidate_encoder import DESC_VEC_DIM
from models.obs_encoder import OBS_DIM
from models.scorer import Scorer
from training.run_training import _GreedyPolicy
from evaluation.evaluate import make_eval_seeds
from models.card_data import get_base_cost

CKPT = "checkpoints/v10_seg3.pt"
MCTS = 20000
HIDDEN = (512, 256)
N = int(os.environ.get("N_GAMES", "3"))
OUT = os.path.join(ROOT, "logs", "deck_strength_raw.json")

PROBE = os.environ.get("PROBE", "0") == "1"  # 打印字段结构而非完整跑


def card_cost(c):
    """卡牌费用：绑定层 cost 有效则用，否则回退查 base_cost 字典。"""
    cost = getattr(c, "cost", 0)
    try:
        cost = float(cost)
    except (TypeError, ValueError):
        cost = 0.0
    if cost and cost > 0:
        return cost
    return get_base_cost(c.id.name)


def snap_deck(gc):
    """进入 Act3 时的牌组强度指标。"""
    deck = list(gc.deck)
    n_cards = len(deck)
    n_rare = sum(1 for c in deck if getattr(c.rarity, "name", str(c.rarity)) == "RARE")
    n_up = sum(1 for c in deck if getattr(c, "upgraded", False))
    costs = [card_cost(c) for c in deck]
    avg_cost = (sum(costs) / n_cards) if n_cards else 0.0
    n_relics = len(list(gc.relics))
    n_potions = gc.potion_count
    hp = gc.cur_hp
    max_hp = gc.max_hp
    return {
        "n_cards": n_cards,
        "n_rare": n_rare,
        "n_upgraded": n_up,
        "avg_cost": round(avg_cost, 3),
        "n_relics": n_relics,
        "n_potions": n_potions,
        "hp": hp,
        "max_hp": max_hp,
        # 诊断用原始列表
        "deck_names": [c.id.name for c in deck],
        "rare_names": [c.id.name for c in deck
                       if getattr(c.rarity, "name", str(c.rarity)) == "RARE"],
        "up_names": [c.id.name for c in deck if getattr(c, "upgraded", False)],
    }


def main():
    t0 = time.time()
    print(f"[init] 载入 {CKPT}  seeds={make_eval_seeds(200)[:N]}", flush=True)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=HIDDEN)
    scorer.load_state_dict(sd)
    scorer.eval()
    policy = _GreedyPolicy(scorer)

    env = LightspeedEnv(mcts_simulations=MCTS)
    seeds = make_eval_seeds(200)[:N]
    games = []
    probe_printed = False

    for gi, seed in enumerate(seeds):
        obs = env.reset(seed)
        done = False
        steps = 0
        recorded_act3 = False
        act3_snap = None
        while not done and steps < 500:
            gc = env.gc
            # 刚进入 Act3 的第一个宏观决策点
            if not recorded_act3 and gc.act == 3:
                act3_snap = snap_deck(gc)
                recorded_act3 = True
                if PROBE and not probe_printed:
                    # 打印字段结构用于核对
                    sample_card = next(iter(gc.deck), None)
                    sample_relic = next(iter(gc.relics), None)
                    print("=== PROBE deck card fields ===", flush=True)
                    if sample_card is not None:
                        print("card.dir:", [a for a in dir(sample_card)
                              if not a.startswith("__")][:60], flush=True)
                        print("card.id.name:", sample_card.id.name, flush=True)
                        print("card.rarity:", repr(sample_card.rarity),
                              "-> name:", getattr(sample_card.rarity, "name", "?"), flush=True)
                        print("card.upgraded:", getattr(sample_card, "upgraded", "?"), flush=True)
                        print("card.cost:", getattr(sample_card, "cost", "?"), flush=True)
                        print("card.type:", getattr(sample_card.type, "name", "?"), flush=True)
                    print("=== PROBE relic fields ===", flush=True)
                    if sample_relic is not None:
                        print("relic.dir:", [a for a in dir(sample_relic)
                              if not a.startswith("__")][:60], flush=True)
                        print("relic.id.name:", sample_relic.id.name, flush=True)
                    print("=== PROBE first act3 snap ===", flush=True)
                    print(json.dumps(act3_snap, ensure_ascii=False, indent=1), flush=True)
                    probe_printed = True
            cands = env.candidates()
            idx = policy.choose(obs, cands) if cands else -1
            res = env.step(idx)
            obs, done = res.obs, res.done
            steps += 1

        gc = env.gc
        win = (gc.outcome == sts.GameOutcome.PLAYER_VICTORY)
        rec = {
            "seed": seed,
            "win": bool(win),
            "outcome": gc.outcome.name,
            "final_act": int(gc.act),
            "final_floor": int(obs.floor),
            "reached_act3": recorded_act3,
            "act3_snap": act3_snap,
        }
        games.append(rec)
        print(f"  [{gi+1}/{N}] seed={seed} win={win} act={gc.act} "
              f"floor={obs.floor} reachedA3={recorded_act3} "
              f"cards={act3_snap['n_cards'] if act3_snap else '-'} "
              f"rare={act3_snap['n_rare'] if act3_snap else '-'} "
              f"up={act3_snap['n_upgraded'] if act3_snap else '-'} "
              f"relics={act3_snap['n_relics'] if act3_snap else '-'} "
              f"({time.time()-t0:.0f}s)", flush=True)

    os.makedirs("logs", exist_ok=True)
    payload = {"ckpt": CKPT, "mcts": MCTS, "n": N, "seeds": seeds, "games": games}
    json.dump(payload, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    wins = sum(1 for g in games if g["win"])
    print(f"\n[done] {N} 局：通关 {wins}/{N}，耗时 {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
