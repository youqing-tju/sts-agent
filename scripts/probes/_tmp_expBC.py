"""实验 B + C：V9 对 SKIP 的打分 & V9 卡牌偏好画像。
加载 V9 一次，复用单例 nn + 单个 GameContext，单进程。
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import slaythespire as sts
from models.candidate_encoder import (
    DESC_VEC_DIM, CardInfo, decode_action_bits, encode_candidate, KIND_LIST,
)
from models.card_data import CARD_ARCHETYPE, CARD_EFFECTS, normalize_card_name, EFFECT_KEYS
from models.card_index import CARD_INDEX, CARD_INDEX_MAP
from models.obs_encoder import OBS_DIM, encode_deck_stats, enhance_observation
from models.scorer import Scorer
from models.live_candidates import build_card_reward_candidates, _card_info_from_json
from models.candidate_encoder import DEAD_FIELDS
from env.base_env import Candidate
from env.lightspeed_env import LightspeedEnv
from models.power_alias import alias_power_ids

CKPT = "checkpoints/ppo_v9_10000.pt"
SKIP_BITS = (6 << 27)  # reward_type=6

# ---------- 加载 scorer ----------
scorer = Scorer(obs_dim=OBS_DIM, cand_dim=DESC_VEC_DIM, hidden=(512, 256))
sd = torch.load(CKPT, weights_only=True)
scorer.load_state_dict(sd)
scorer.eval()
print(f"[ok] scorer loaded obs={OBS_DIM} cand={DESC_VEC_DIM}", file=sys.stderr)

nni = sts.getNNInterface()
gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)


def sanitize(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    return obj


def init_gc(frame):
    payload = sanitize(frame)
    gs = payload.get("game_state") or {}
    if not isinstance(gs, dict):
        return False
    if gs.get("seed") is None:
        gs = dict(gs); gs["seed"] = 0; payload = dict(payload); payload["game_state"] = gs
    try:
        alias_power_ids(payload)
        gc.init_from_json(json.dumps(payload))
        return True
    except Exception:
        return False


def obs_of(g):
    raw = list(nni.getObservation(g))
    obs = enhance_observation(raw, g, card_index=CARD_INDEX,
                              arch_table=CARD_ARCHETYPE, card_effects=CARD_EFFECTS)
    deck_arch = encode_deck_stats(raw, card_index=CARD_INDEX,
                                  arch_table=CARD_ARCHETYPE, card_effects=CARD_EFFECTS)[:5]
    return obs, deck_arch


def score_specs(specs, deck_arch, obs):
    vecs = []
    for sp in specs:
        proto = Candidate(idx=0, kind=sp["kind"], desc_vec=[], label=sp["label"])
        vecs.append(encode_candidate(
            proto, card=sp.get("card"), room=sp.get("room"), bits=sp.get("bits"),
            deck_arch=deck_arch, event_key=sp.get("event_key"),
            event_option_idx=sp.get("event_option_idx")))
    with torch.no_grad():
        logits = scorer(torch.tensor(obs, dtype=torch.float32),
                        torch.tensor(vecs, dtype=torch.float32))
    return [float(x) for x in logits]


# ===== 实验 A 步骤2：SKIP 向量 vs 卡牌向量 差异 =====
def spec(kind, card=None, bits=None):
    return {"idx": 0, "select_idx": 0, "kind": kind,
            "label": card.name if card else kind, "card": card, "bits": bits}

skip_vec = encode_candidate(Candidate(idx=0, kind="SKIP", desc_vec=[], label="SKIP"),
                             card=None, bits=SKIP_BITS, deck_arch=[0,0,0,0,0])
strike_card = CardInfo(name="BASH", card_type="ATTACK", rarity="BASIC", cost=2, upgraded=False)
strike_vec = encode_candidate(Candidate(idx=0, kind="CARD_REWARD", desc_vec=[], label="Bash"),
                              card=strike_card, bits=0, deck_arch=[0,0,0,0,0])
seg_off = [(0,10,"kind"),(10,22,"card_attr"),(22,132,"card_onehot"),(132,170,"semantic"),
           (170,205,"cross"),(205,215,"room"),(215,223,"numeric"),(223,271,"lookahead"),
           (271,295,"potion_kb"),(295,327,"relic_kb"),(327,347,"event_kb")]
def nonzero_segs(vec):
    out = {}
    for a,b,name in seg_off:
        s = sum(1 for v in vec[a:b] if v != 0.0)
        if s: out[name] = s
    return out
vec_diff = {"skip_nonzero_segs": nonzero_segs(skip_vec),
            "card_nonzero_segs": nonzero_segs(strike_vec),
            "skip_kind_index": KIND_LIST.index("SKIP"),
            "skip_numeric_reward_type_norm": skip_vec[215+3]}


# ===== 实验 B：真机 CARD_REWARD 帧 + SKIP 候选打分 =====
STATE_FILES = sorted(glob.glob("logs/realtime/0911_*_states.jsonl"))
frames = []
for fp in STATE_FILES:
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                try:
                    obj = json.loads(line.encode("utf-8","surrogatepass").decode("utf-8","replace"))
                except Exception:
                    continue
            if not isinstance(obj, dict):
                continue
            gs = obj.get("game_state") or {}
            if isinstance(gs, dict) and (gs.get("screen_type") or "").upper() == "CARD_REWARD":
                frames.append(obj)
print(f"[ok] collected {len(frames)} CARD_REWARD frames", file=sys.stderr)

B = {"frames": [], "skip_rank_dist": {}, "skip_is_first": 0, "n": 0,
     "skip_mean_rank": 0.0, "skip_scores": [], "weak_frames_skip_rank": []}
rank_sum = 0
n_weak = 0
weak_rank_sum = 0
for fr in frames:
    if not init_gc(fr):
        continue
    obs, deck_arch = obs_of(gc)
    gs = fr.get("game_state", {})
    card_specs = build_card_reward_candidates(gs)
    if not card_specs:
        continue
    specs = list(card_specs)
    specs.append(spec("SKIP", card=None, bits=SKIP_BITS))
    scores = score_specs(specs, deck_arch, obs)
    # 排名（0=最高分）
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    rank_of_skip = order.index(len(scores) - 1)  # SKIP 是最后一个 spec
    B["skip_rank_dist"][str(rank_of_skip)] = B["skip_rank_dist"].get(str(rank_of_skip), 0) + 1
    rank_sum += rank_of_skip + 1  # 1-based
    B["skip_scores"].append(scores[-1])
    if rank_of_skip == 0:
        B["skip_is_first"] += 1
    # KB 该跳代理规则：三张全是 common (rarity_score==0)
    all_common = all((sp.get("card").rarity == "BASIC" or sp.get("card").rarity == "COMMON")
                     for sp in card_specs)
    if all_common:
        n_weak += 1
        weak_rank_sum += rank_of_skip + 1
        B["weak_frames_skip_rank"].append(rank_of_skip + 1)
    B["n"] += 1
    B["frames"].append({
        "n_cards": len(card_specs),
        "card_scores": scores[:-1],
        "skip_score": scores[-1],
        "skip_rank_0based": rank_of_skip,
        "all_common": all_common,
        "labels": [sp["label"] for sp in card_specs],
    })
    if B["n"] >= 60:
        break
B["skip_mean_rank"] = round(rank_sum / B["n"], 3) if B["n"] else None
B["skip_is_first_frac"] = round(B["skip_is_first"] / B["n"], 3) if B["n"] else None
B["weak_n"] = n_weak
B["weak_skip_mean_rank"] = round(weak_rank_sum / n_weak, 3) if n_weak else None
print(f"[B] n={B['n']} skip_first_frac={B['skip_is_first_frac']} mean_rank={B['skip_mean_rank']}",
      file=sys.stderr)


# ===== 实验 C：V9 卡牌偏好画像 =====
CARDS = {
    "力量": ["Inflame", "Spot Weakness", "Demon Form", "Limit Break"],
    "防战": ["Barricade", "Entrench", "Body Slam", "Juggernaut", "Feel No Pain", "Second Wind", "Shrug It Off", "True Grit"],
    "大攻击": ["Bludgeon", "Heavy Blade", "Whirlwind", "Reaper", "Carnage"],
    "debuff": ["Bash", "Thunderclap", "Clothesline", "Shockwave", "Uppercut", "Intimidate"],
    "功能": ["Offering", "Battle Trance", "Corruption", "Dark Embrace", "Burning Pact"],
    "基准": ["Strike", "Defend", "Pommel Strike"],
}

def card_to_info(name):
    # 候选名 → CARD_INDEX_MAP 规范名（大写+下划线），保证卡牌精确 one-hot 命中
    canon = name.replace(" ", "_").upper()
    if canon not in CARD_INDEX_MAP:
        canon = normalize_card_name(name).upper()
    k = normalize_card_name(name)
    e = CARD_EFFECTS.get(k, {})
    ctype = "ATTACK" if e.get("is_attack") else ("POWER" if e.get("is_power") else "SKILL")
    rarity = {0.0: "COMMON", 0.5: "UNCOMMON", 1.0: "RARE"}.get(e.get("rarity_score"), "COMMON")
    cost = e.get("base_cost", 1)
    return CardInfo(name=canon, card_type=ctype, rarity=rarity, cost=float(cost), upgraded=False)

# 基线 (a) 开局
env = LightspeedEnv(mcts_simulations=5)
env.reset(2000)
obs_a, deck_a = obs_of(env.gc)
# 基线 (b) 中局：找一个 floor>=10 非战斗真机帧
obs_b = None; deck_b = None
for fp in STATE_FILES:
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                try:
                    obj = json.loads(line.encode("utf-8","surrogatepass").decode("utf-8","replace"))
                except Exception:
                    continue
            if not isinstance(obj, dict):
                continue
            gs = obj.get("game_state") or {}
            st = (gs.get("screen_type") or "") if isinstance(gs, dict) else ""
            if st not in ("MAP", "REST", "SHOP_ROOM", "EVENT"):
                continue
            if init_gc(obj):
                if gc.floor_num >= 10 and gc.screen_state.name not in ("COMBAT", "BATTLE"):
                    obs_b, deck_b = obs_of(gc)
                    break
        if obs_b is not None:
            break
print(f"[C] baseline_b floor={gc.floor_num} screen={gc.screen_state.name}", file=sys.stderr)

C = {"baselines": {}, "groups": {}, "per_card": {}, "top10": {}, "bottom10": {}}
for bname, (obs, deck) in (("opening", (obs_a, deck_a)), ("midgame", (obs_b, deck_b))):
    if obs is None:
        continue
    scores = {}
    for grp, names in CARDS.items():
        for nm in names:
            info = card_to_info(nm)
            sp = spec("CARD_REWARD", card=info, bits=0)
            sc = score_specs([sp], deck, obs)[0]
            scores[nm] = sc
    C["baselines"][bname] = {"scores": scores}
    # 分组平均
    grp_avg = {}
    for grp, names in CARDS.items():
        grp_avg[grp] = round(sum(scores[n] for n in names) / len(names), 3)
    C["groups"][bname] = grp_avg
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    C["top10"][bname] = ranked[:10]
    C["bottom10"][bname] = ranked[-10:]
    C["per_card"][bname] = scores


# ===== 字段缺失/不可区分清单（基于 EFFECT_KEYS / DEAD_FIELDS）=====
field_analysis = {
    "EFFECT_KEYS": EFFECT_KEYS,
    "DEAD_FIELDS": list(DEAD_FIELDS),
    "defensive_cards_fields": {},
}
def_card_defs = {
    "Barricade": "格挡不清空(block_retain)", "Entrench": "格挡翻倍(block_multiplier)",
    "Body Slam": "格挡转伤害(block_to_damage)", "Juggernaut": "获得格挡时造成伤害(block_on_gain_trigger)",
    "Feel No Pain": "消耗触发格挡(on_exhaust_trigger)", "Second Wind": "消耗得格挡(exhaust_payoff_scaling)",
    "Shrug It Off": "出牌得格挡(base_block)", "True Grit": "出牌得格挡+消耗(base_block/exhausts_others)",
}
for nm, desc in def_card_defs.items():
    k = normalize_card_name(nm)
    e = CARD_EFFECTS.get(k, {})
    field_analysis["defensive_cards_fields"][nm] = {"desc": desc, "effect_fields": e}

out = {"experiment": "BC", "vec_diff": vec_diff,
       "B": {k: v for k, v in B.items() if k != "frames"} | {"frames_detail": B["frames"]},
       "C": C, "field_analysis": field_analysis}
with open("logs/rl_skip_pref_probe.json", "r", encoding="utf-8") as f:
    A = json.load(f)
A.update(out)
with open("logs/rl_skip_pref_probe.json", "w", encoding="utf-8") as f:
    json.dump(A, f, ensure_ascii=False, indent=2)
print("[ok] wrote logs/rl_skip_pref_probe.json", file=sys.stderr)
# 打印关键摘要
print("VEC_DIFF:", json.dumps(vec_diff, ensure_ascii=False))
print("B skip_first_frac:", B["skip_is_first_frac"], "mean_rank:", B["skip_mean_rank"],
      "weak_mean_rank:", B["weak_skip_mean_rank"])
for bname in C["groups"]:
    print(f"C groups[{bname}]:", C["groups"][bname])
    print(f"C top3[{bname}]:", C["top10"][bname][:3])
    print(f"C bottom3[{bname}]:", C["bottom10"][bname][-3:])
