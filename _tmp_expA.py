"""实验 A：模拟器训练侧 SKIP 候选真实存在性。
跑多局模拟器，遇 REWARDS/选牌屏 dump 全部候选 kind/label/bits，统计构成。
单进程；KMP_DUPLICATE_LIB_OK=TRUE。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import slaythespire as sts
from models.candidate_encoder import decode_action_bits
from env.lightspeed_env import LightspeedEnv

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
SIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
SEED0 = 1000
MAX_STEPS = 600

env = LightspeedEnv(mcts_simulations=SIMS)

rewards_screens = 0          # 出现选牌(REWARDS)屏次数
total_cands = 0
kind_counter = {}            # 所有屏候选 kind 计数
skip_bits_list = []          # SKIP 候选的 bits 采样
skip_in_rewards = 0          # REWARDS 屏里含 SKIP 的次数
rewards_card_cands = 0       # REWARDS 屏里卡牌候选数
rewards_skip_cands = 0       # REWARDS 屏里 SKIP 候选数
samples = []                 # 保存若干 REWARDS 屏的候选明细

for g in range(N_GAMES):
    seed = SEED0 + g
    try:
        env.reset(seed)
    except Exception as e:
        print(f"[warn] reset seed={seed} failed: {e}", file=sys.stderr)
        continue
    steps = 0
    while steps < MAX_STEPS:
        cands = env.candidates()
        if not cands:
            break
        screen = env.gc.screen_state.name
        acts = env._actions
        for a in acts:
            d = decode_action_bits(a.bits)
            kind = env._candidate_kind(screen, d)
            kind_counter[kind] = kind_counter.get(kind, 0) + 1
            total_cands += 1
            if screen == "REWARDS" and kind == "SKIP":
                if len(skip_bits_list) < 5:
                    skip_bits_list.append(a.bits)
        if screen == "REWARDS":
            rewards_screens += 1
            # 该屏候选明细
            detail = {"screen": screen, "candidates": []}
            n_card = n_skip = 0
            for a in acts:
                d = decode_action_bits(a.bits)
                kind = env._candidate_kind(screen, d)
                if kind == "SKIP":
                    n_skip += 1
                    rewards_skip_cands += 1
                elif kind == "CARD_REWARD":
                    n_card += 1
                    rewards_card_cands += 1
                detail["candidates"].append({
                    "kind": kind, "label": a.describe(env.gc) or f"0x{a.bits:08x}",
                    "bits": a.bits, "decoded": d,
                })
            if n_skip > 0:
                skip_in_rewards += 1
            if len(samples) < 8:
                samples.append(detail)
        # 推进：随机选一个候选
        pick = random.randrange(len(acts))
        res = env.step(pick)
        steps += 1
        if res.done:
            break
    # print progress
    print(f"[game {g}] seed={seed} steps={steps} rewards_screens_so_far={rewards_screens}",
          file=sys.stderr)

print("=== EXPERIMENT A SUMMARY ===")
print("games:", N_GAMES)
print("rewards_screens(REWARDS选牌屏次数):", rewards_screens)
print("REWARDS屏含SKIP的次数:", skip_in_rewards, "/", rewards_screens)
print("REWARDS屏卡牌候选总数:", rewards_card_cands, " SKIP候选总数:", rewards_skip_cands)
print("全部屏候选kind计数:", json.dumps(kind_counter, ensure_ascii=False))
print("SKIP bits采样:", skip_bits_list)
print("SKIP bits decode:", [decode_action_bits(b) for b in skip_bits_list])

out = {
    "experiment": "A",
    "n_games": N_GAMES,
    "rewards_screens": rewards_screens,
    "rewards_screens_with_skip": skip_in_rewards,
    "rewards_card_candidates": rewards_card_cands,
    "rewards_skip_candidates": rewards_skip_cands,
    "kind_counter": kind_counter,
    "skip_bits_samples": skip_bits_list,
    "skip_bits_decode": [decode_action_bits(b) for b in skip_bits_list],
    "samples": samples,
}
with open("logs/rl_skip_pref_probe.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("wrote logs/rl_skip_pref_probe.json")
