# -*- coding: utf-8 -*-
"""MCTS 预算扫描实验：复现真机 ISMCTS 战斗决策，扫描 SIMS ∈ {2000,8000,50000}
（前 3 帧额外 200000），并做 seed 稳定性测试。落盘 logs/mcts_budget_probe.json。

单进程；不改生产代码；临时脚本根目录 _tmp_mcts_probe.py。
"""
import os, sys, json, time, glob, math
from collections import OrderedDict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import slaythespire as sts
from models.power_alias import alias_power_ids
import combat_brain as cb_mod


def _sanitize(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj


SIMS_LEVELS = [2000, 8000, 50000]
BIG_SIMS = 200000
N_FRAMES_TARGET = 30
STABILITY_FRAMES = 10
DETERMINIZATIONS = 16

OUT_JSON = os.path.join(ROOT, "logs", "mcts_budget_probe.json")
OUT_JSON_TMP = OUT_JSON + ".tmp"


# ----------------------------------------------------------------------------
# 1. 采样真机战斗帧
# ----------------------------------------------------------------------------
def hand_key(cs):
    hand = cs.get("hand") or []
    # 用 (name,cost,upgrades) 去重，而非 uuid（uuid 每帧唯一，无法连续去重）
    return (cs.get("turn"), tuple((h.get("id"), h.get("cost"), h.get("upgrades", 0))
                                  for h in hand if isinstance(h, dict)))


def is_combat_frame(st):
    gs = st.get("game_state") or {}
    cs = gs.get("combat_state") or {}
    hand = cs.get("hand") or []
    return gs.get("screen_type") == "NONE" and bool(cs) and bool(hand)


def collect_frames():
    files = sorted(glob.glob(os.path.join(ROOT, "logs", "realtime", "0911_*_states.jsonl")))
    cands = []
    for fn in files:
        seen = set()
        with open(fn, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    st = json.loads(line)
                except Exception:
                    continue
                if not is_combat_frame(st):
                    continue
                cs = st["game_state"]["combat_state"]
                k = hand_key(cs)
                if k in seen:
                    continue
                seen.add(k)
                cands.append((fn, idx, st))
    return files, cands


def choose_frames(cands):
    # 多样性分桶：energy 档(1/2/3/其他) × 多怪? × 有可打1费牌?
    buckets = {}
    picked = []
    picked_keys = set()
    # 先给每个候选打多样性特征（需要 bc 才能知 energy/playable，但为轻量先用
    # combat_state 里的 cost 粗判）。energy 在 frame 里常为 None，故用 simulator
    # 计算；这里只做初选，最终 energy 以 simulator 为准。
    def coarse_bucket(st):
        cs = st["game_state"]["combat_state"]
        hand = cs.get("hand") or []
        costs = [h.get("cost") for h in hand if isinstance(h, dict)]
        has1 = any(c == 1 for c in costs)
        mc = len(cs.get("monsters") or [])
        return (("multi" if mc > 1 else "single"), ("h1" if has1 else "noh1"))
    # round-robin 取桶，尽量覆盖
    ORDER = [("single", "h1"), ("single", "noh1"), ("multi", "h1"), ("multi", "noh1")]
    # 按桶归组
    by_bucket = {}
    for item in cands:
        b = coarse_bucket(item[2])
        by_bucket.setdefault(b, []).append(item)
    ptrs = {b: 0 for b in by_bucket}
    # 轮转取，直到取满或所有桶取尽
    guard = 0
    while len(picked) < N_FRAMES_TARGET:
        progressed = False
        for b in ORDER:
            if b not in by_bucket:
                continue
            arr = by_bucket[b]
            i = ptrs[b]
            if i < len(arr):
                fn, idx, st = arr[i]
                key = (fn, idx)
                if key not in picked_keys:
                    picked.append((fn, idx, st))
                    picked_keys.add(key)
                    progressed = True
                ptrs[b] += 1
                if len(picked) >= N_FRAMES_TARGET:
                    break
        guard += 1
        if not progressed or guard > 10000:
            break
    # 若多样性覆盖不足，补充分桶前的剩余候选
    for item in cands:
        if len(picked) >= N_FRAMES_TARGET:
            break
        fn, idx, st = item
        key = (fn, idx)
        if key not in picked_keys:
            picked.append(item)
            picked_keys.add(key)
    return picked


# ----------------------------------------------------------------------------
# 2. 灌入与搜索辅助
# ----------------------------------------------------------------------------
def build_bc(clean):
    """构造 gc + bc0（已 init_from_json）。返回 (gc, bc0) 或 (None,None) 失败。"""
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(json.dumps(clean))
    bc0 = sts.BattleContext()
    bc0.init_from_json(gc, json.dumps(clean))
    return gc, bc0


def frame_meta(bc0, st):
    player = bc0.player
    energy = getattr(player, "energy", None)
    hp = getattr(player, "hp", None)
    if hp is None:
        cs = (st.get("game_state") or {}).get("combat_state") or {}
        hp = ((cs.get("player") or {}).get("hp"))
    monster_count = len(bc0.monsters) if hasattr(bc0, "monsters") else \
        len((st["game_state"]["combat_state"].get("monsters") or []))
    hand = []
    any_playable = False
    for c in bc0.hand:
        cid = c.id.name if hasattr(c.id, "name") else str(c.id)
        try:
            playable = bool(c.is_playable())
        except Exception:
            playable = False
        cost = getattr(c, "cost_for_turn", None)
        hand.append({"id": cid, "cost": cost, "playable": playable})
        if playable:
            any_playable = True
    return {
        "energy": energy, "hp": hp, "monster_count": monster_count,
        "hand": hand, "any_playable": any_playable,
    }


def ismcts_with_offset(state, gc, bc0, sim_count, determinizations, seed_offset):
    """复刻 CombatBrain._ismcts_decide，但 base_seed 加 seed_offset。"""
    base_seed = cb_mod.CombatBrain._base_seed(state, gc) + seed_offset
    searcher_cls = getattr(sts, "BattleSearcher")
    votes = OrderedDict()
    winners = {}
    for k in range(determinizations):
        bc_k = sts.BattleContext(bc0)
        bc_k.randomize_rng(base_seed + k)
        action = searcher_cls(bc_k, sim_count).best_action
        if callable(action):
            action = action()
        if action is None:
            continue
        key = cb_mod.CombatBrain._vote_key(action, bc0)
        votes[key] = votes.get(key, 0) + 1
        winners.setdefault(key, action)
    if not votes:
        return None, {}
    winner_key = max(votes, key=lambda kk: votes[kk])
    return winner_key, dict(votes)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    import logging
    logging.disable(logging.CRITICAL)  # 压住 CombatBrain 的 INFO 日志

    files, cands = collect_frames()
    print(f"[info] 候选战斗帧去重后共 {len(cands)} 条，来源文件 {len(files)} 个",
          flush=True)
    picked = choose_frames(cands)
    print(f"[info] 选定 {len(picked)} 帧做预算扫描", flush=True)

    cb = cb_mod.CombatBrain(simulation_count=50000, determinizations=DETERMINIZATIONS)

    # 加载已有结果以便断点续跑
    results = []
    done_keys = set()
    if os.path.exists(OUT_JSON):
        try:
            with open(OUT_JSON, encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                done_keys.add((r["file"], r["frame_idx"]))
            print(f"[info] 续跑：已存在 {len(results)} 条结果", flush=True)
        except Exception:
            results = []

    file_short = lambda fn: os.path.basename(fn)

    for fi, (fn, idx, st) in enumerate(picked):
        key = (file_short(fn), idx)
        if key in done_keys:
            print(f"[skip] frame {fi+1}/{len(picked)} {key}", flush=True)
            continue
        t0 = time.time()
        clean = _sanitize(st)
        alias_power_ids(clean)
        try:
            gc, bc0 = build_bc(clean)
        except Exception as e:
            print(f"[fail-init] frame {fi+1} {key}: {e!r}", flush=True)
            results.append({"file": file_short(fn), "frame_idx": idx,
                            "error": "init_from_json failed",
                            "cmd_by_sims": {}, "votes": {}, "ms": {},
                            "end_with_playable": None})
            _dump(results)
            continue

        meta = frame_meta(bc0, st)
        turn = (st["game_state"]["combat_state"].get("turn"))
        cmd_by_sims = {}
        ms = {}
        votes = {}
        end_with_playable = None
        failed = False
        for sims in SIMS_LEVELS:
            try:
                gc2, _ = build_bc(clean)  # 每次 decide 前重灌 gc（安全）
                cb.sim_count = sims
                t1 = time.time()
                cmd = cb.decide(st, gc2)
                dt = (time.time() - t1) * 1000.0
                cmd_by_sims[sims] = cmd
                ms[sims] = round(dt, 1)
                if sims == 50000:
                    votes = {str(k): v for k, v in (cb.last_votes or {}).items()}
            except Exception as e:
                print(f"[fail-decide] frame {fi+1} sims={sims}: {e!r}", flush=True)
                cmd_by_sims[sims] = f"<err:{e!r}>"
                failed = True
            # 前 3 帧额外跑 200000
            if fi < 3 and sims == 50000:
                try:
                    gc2, _ = build_bc(clean)
                    cb.sim_count = BIG_SIMS
                    t1 = time.time()
                    cmd = cb.decide(st, gc2)
                    dt = (time.time() - t1) * 1000.0
                    cmd_by_sims[BIG_SIMS] = cmd
                    ms[BIG_SIMS] = round(dt, 1)
                except Exception as e:
                    cmd_by_sims[BIG_SIMS] = f"<err:{e!r}>"
        # 判定 END 且有可打牌（用 50000 档命令）
        cmd5 = cmd_by_sims.get(50000)
        if cmd5 == "END" and meta["any_playable"]:
            end_with_playable = True
        elif cmd5 is not None:
            end_with_playable = False

        # 决策稳定性（前 10 帧，SIMS=50000，base_seed 偏移）
        stability = None
        if fi < STABILITY_FRAMES and not failed:
            try:
                gc_s, bc0_s = build_bc(clean)
                winners = []
                for j in range(5):
                    wk, _ = ismcts_with_offset(st, gc_s, bc0_s, 50000,
                                               DETERMINIZATIONS, j * 10000)
                    winners.append(str(wk))
                consistent = len(set(winners)) == 1
                stability = {"winners": winners, "consistent": consistent,
                             "agreement": 1.0 if consistent else
                             (max(winners.count(w) for w in set(winners)) / len(winners))}
            except Exception as e:
                stability = {"error": f"{e!r}"}
                print(f"[fail-stability] frame {fi+1}: {e!r}", flush=True)

        rec = {
            "file": file_short(fn),
            "frame_idx": idx,
            "turn": turn,
            "energy": meta["energy"],
            "hp": meta["hp"],
            "monster_count": meta["monster_count"],
            "hand": meta["hand"],
            "any_playable": meta["any_playable"],
            "cmd_by_sims": cmd_by_sims,
            "end_with_playable": end_with_playable,
            "votes": votes,
            "ms": ms,
            "stability": stability,
            "elapsed_s": round(time.time() - t0, 1),
        }
        results.append(rec)
        done_keys.add(key)
        _dump(results)
        cmd_str = {str(k): v for k, v in cmd_by_sims.items()}
        print(f"[done] frame {fi+1}/{len(picked)} {key} turn={turn} "
              f"E={meta['energy']} hp={meta['hp']} mc={meta['monster_count']} "
              f"cmd={cmd_str} endPlay={end_with_playable} "
              f"stab={stability.get('consistent') if isinstance(stability,dict) else None}",
              flush=True)

    _dump(results)
    print(f"[complete] 共 {len(results)} 帧结果已写入 {OUT_JSON}", flush=True)


def _dump(results):
    with open(OUT_JSON_TMP, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    if os.path.exists(OUT_JSON):
        os.remove(OUT_JSON)
    os.rename(OUT_JSON_TMP, OUT_JSON)


if __name__ == "__main__":
    main()
