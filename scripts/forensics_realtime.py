#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sts-agent 真机日志取证（只读解析）。
输入：logs/realtime/<run>_decisions.jsonl / _states.jsonl，及根目录 sts_agent_debug.log
输出：logs/realtime_forensics.json（原始聚合）+ 在 main 中写 markdown 由另一脚本完成。
本脚本只读取日志，不修改任何代码或日志。
"""
import json, ast, os, re, glob
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(ROOT, "logs", "realtime")
DEBUG = os.path.join(ROOT, "sts_agent_debug.log")

RUNS = ["0911_013551", "0911_022209", "0911_025542"]

# ---- debuff 判定：描述里含「易伤」(vulnerable) 或「虚弱」(weak) 的牌即视为 debuff 牌 ----
DEBUFF_RE = re.compile(r"易伤|虚弱|Weak|Vulnerable", re.IGNORECASE)
# 候选命名集合（按实际出现的 id 为准，做交叉校验）
NAMED_DEBUFF_IDS = {"Bash", "Thunderclap", "Clothesline", "Shockwave",
                    "Uppercut", "Intimidate", "Blind"}


def is_debuff(card):
    desc = card.get("description") or ""
    name = card.get("name") or ""
    cid = card.get("id") or ""
    if DEBUFF_RE.search(desc) or DEBUFF_RE.search(name):
        return True
    if cid in NAMED_DEBUFF_IDS:
        return True
    return False


def load_frames(path):
    """返回 [(tgid, turn_val, energy, hand_cards)]，hand_cards=[{id,name,type,cost,playable}]"""
    frames = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            gs = d.get("game_state") or {}
            cs = gs.get("combat_state")
            if not cs:
                continue
            hand = []
            for c in cs.get("hand", []):
                hand.append({
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "type": c.get("type"),
                    "cost": c.get("cost"),
                    "playable": c.get("is_playable", True),
                })
            frames.append((cs.get("turn"), cs.get("player", {}).get("energy"), hand))
    # 赋 tgid：连续相等 turn 为同一战斗回合实例
    out = []
    tgid = -1
    prev = None
    for turn, energy, hand in frames:
        if prev is None or turn != prev:
            tgid += 1
            prev = turn
        out.append((tgid, turn, energy, hand))
    return out


def load_decisions(path):
    """返回战斗动作序列 [{cmd,kind,degraded,fb,n,ts}]"""
    acts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            c = d.get("cmd") or ""
            if c == "END" or c.startswith("PLAY"):
                acts.append({
                    "cmd": c, "kind": d.get("kind"), "degraded": d.get("degraded"),
                    "fb": d.get("total_fallbacks"), "n": d.get("n"), "ts": d.get("ts"),
                })
    return acts


def resolve_play_card(acts, frames):
    """贪心把每个决策对齐到一帧，得到：(tgid, turn_val, card_or_None for PLAY, energy_for_END)。
    frames 不自带 ts，按“读状态→出命令”的实时顺序用帧指针近似对齐。"""
    fi = 0
    res = []
    for a in acts:
        c = a["cmd"]
        if c.startswith("PLAY"):
            # PLAY {idx(1-based)} [target]
            parts = c.split()
            idx = int(parts[1]) - 1  # 0-based
            # 前进到第一帧手牌足够长
            while fi < len(frames) and len(frames[fi][3]) <= idx:
                fi += 1
            if fi >= len(frames):
                fi = len(frames) - 1
            tgid, turn, energy, hand = frames[fi]
            card = hand[idx] if 0 <= idx < len(hand) else None
            res.append({"a": a, "tgid": tgid, "turn": turn, "card": card, "energy": energy})
            fi += 1  # 近似：出牌后读到新帧
        else:  # END
            if fi >= len(frames):
                fi = len(frames) - 1
            tgid, turn, energy, hand = frames[min(fi, len(frames) - 1)]
            res.append({"a": a, "tgid": tgid, "turn": turn, "card": None, "energy": energy, "hand": hand})
            fi += 1
    return res


def analyze_run(run):
    df = os.path.join(RT, f"{run}_decisions.jsonl")
    sf = os.path.join(RT, f"{run}_states.jsonl")
    frames = load_frames(sf)
    acts = load_decisions(df)
    resolved = resolve_play_card(acts, frames)

    # (c) 命令来源：kind/degraded 分布
    kind_deg = Counter()
    end_kind_deg = Counter()
    end_wasted_by_deg = Counter()   # ('-|True' / 'COMBAT|False') -> 留能量可打牌的 END 数
    play_n = end_n = 0
    for a in acts:
        kind_deg[(a["kind"], a["degraded"])] += 1
        if a["cmd"] == "END":
            end_n += 1
            end_kind_deg[(a["kind"], a["degraded"])] += 1
        else:
            play_n += 1

    # (b) 每个 END 命令对应帧的 energy + 手牌
    end_wasted = 0       # energy>=1 且手牌存在 cost<=energy 的可打牌
    end_allunplay = 0    # energy>=1 但手牌全不可打
    end_no_energy = 0    # energy<=0
    for r in resolved:
        if r["a"]["cmd"] != "END":
            continue
        e = r["energy"]
        hand = r.get("hand") or []
        if e is None:
            e = 0
        playable_left = [c for c in hand
                         if (c.get("playable") is not False)
                         and isinstance(c.get("cost"), int) and 0 <= c["cost"] <= e]
        if e >= 1 and playable_left:
            end_wasted += 1
            end_wasted_by_deg[(r["a"]["kind"], r["a"]["degraded"])] += 1
        elif e >= 1:
            end_allunplay += 1
        else:
            end_no_energy += 1

    # (b) 帧侧（稳健，不依赖决策对齐）：每个战斗回合实例末帧 energy + 手牌
    frame_turn_wasted = 0
    frame_turn_total = 0
    for tgid, turn, energy, hand in frames:
        # 只看每个 tgid 的“末帧”（同 tgid 最后出现的那帧）
        pass
    # 取每个 tgid 的最后一条
    last_by_tgid = {}
    for tgid, turn, energy, hand in frames:
        last_by_tgid[tgid] = (turn, energy, hand)
    for tgid, (turn, energy, hand) in last_by_tgid.items():
        frame_turn_total += 1
        if energy is None:
            energy = 0
        playable_left = [c for c in hand
                         if (c.get("playable") is not False)
                         and isinstance(c.get("cost"), int) and 0 <= c["cost"] <= energy]
        if energy >= 1 and playable_left:
            frame_turn_wasted += 1

    # (2) 卡名分布 + debuff 顺序
    card_counter = Counter()
    debuff_id_counter = Counter()     # 实际被判定为 debuff 的卡 id
    # 按 tgid 聚合每个回合的打出序列
    turns = defaultdict(list)  # tgid -> [card_dict_or_None ...]（仅 PLAY）
    for r in resolved:
        if r["a"]["cmd"].startswith("PLAY"):
            turns[r["tgid"]].append(r["card"])
    debuff_positions = Counter()      # debuff 第几个打出
    turns_with_both = 0               # 同回合既打攻击又打 debuff
    both_turn_examples = []
    bash_vs_attack = []               # 手牌同时有 Bash 与直接攻击时先打谁
    bash_first_n = 0
    bash_after_n = 0
    play_seq_examples = []
    for tgid, seq in turns.items():
        seq = [c for c in seq if c]  # 去掉解析失败
        if not seq:
            continue
        for c in seq:
            card_counter[c.get("id")] += 1
            if is_debuff(c):
                debuff_id_counter[c.get("id")] += 1
        has_atk = any(c.get("type") == "ATTACK" for c in seq)
        has_db = any(is_debuff(c) for c in seq)
        if has_atk and has_db:
            turns_with_both += 1
            # debuff 位置（1-based）
            for pos, c in enumerate(seq, 1):
                if is_debuff(c):
                    debuff_positions[pos] += 1
            # 记录前若干实例
            if len(both_turn_examples) < 12:
                both_turn_examples.append({
                    "tgid": tgid,
                    "turn": frames_turn_of(resolved, tgid),
                    "seq": [c.get("id") for c in seq],
                    "debuff_ids": [c.get("id") for c in seq if is_debuff(c)],
                })
        # Bash vs 直接攻击（同回合）
        if any(c.get("id") == "Bash" for c in seq) and any(
                c.get("type") == "ATTACK" and c.get("id") != "Bash" for c in seq):
            bash_idx = next((i for i, c in enumerate(seq) if c.get("id") == "Bash"), None)
            atk_idx = next((i for i, c in enumerate(seq)
                            if c.get("type") == "ATTACK" and c.get("id") != "Bash"), None)
            first_bash = bash_idx < atk_idx if (bash_idx is not None and atk_idx is not None) else None
            if first_bash is True:
                bash_first_n += 1
            elif first_bash is False:
                bash_after_n += 1
            bash_vs_attack.append({
                "tgid": tgid, "turn": frames_turn_of(resolved, tgid),
                "seq": [c.get("id") for c in seq],
                "first_bash": first_bash,
            })
        # 完整打出顺序实例（前若干个回合）
        if len(play_seq_examples) < 15:
            play_seq_examples.append({
                "tgid": tgid, "turn": frames_turn_of(resolved, tgid),
                "seq": [c.get("id") for c in seq],
            })

    # 中位数 debuff 位置
    items = []
    for pos, cnt in debuff_positions.items():
        items += [pos] * cnt
    if items:
        items.sort()
        mid = (len(items) - 1) / 2.0
        if mid == int(mid):
            median_pos = items[int(mid)]
        else:
            median_pos = (items[int(mid)] + items[int(mid) + 1]) / 2.0
    else:
        median_pos = None

    return {
        "run": run,
        "n_combat_acts": len(acts),
        "n_play": play_n,
        "n_end": end_n,
        "kind_degraded": {f"{k[0]}|{k[1]}": v for k, v in kind_deg.items()},
        "end_kind_degraded": {f"{k[0]}|{k[1]}": v for k, v in end_kind_deg.items()},
        "end_wasted_by_degraded": {f"{k[0]}|{k[1]}": v for k, v in end_wasted_by_deg.items()},
        # (b) END 命令侧
        "end_energy_wasted_playable": end_wasted,
        "end_energy_all_unplayable": end_allunplay,
        "end_no_energy": end_no_energy,
        # (b) 帧侧
        "frame_turns_total": frame_turn_total,
        "frame_turns_wasted_playable": frame_turn_wasted,
        # (2)
        "card_top": card_counter.most_common(20),
        "debuff_ids_detected": dict(debuff_id_counter.most_common()),
        "turns_with_both_atk_debuff": turns_with_both,
        "debuff_order_positions": dict(sorted(debuff_positions.items())),
        "debuff_median_position": median_pos,
        "bash_first_n": bash_first_n,
        "bash_after_n": bash_after_n,
        "both_turn_examples": both_turn_examples,
        "bash_vs_attack": bash_vs_attack[:15],
        "play_seq_examples": play_seq_examples,
    }


def frames_turn_of(resolved, tgid):
    for r in resolved:
        if r["tgid"] == tgid:
            return r["turn"]
    return None


# ---------------- debug log: ISMCTS votes + 回退 ----------------
def analyze_debug_log():
    total = 0
    with_end = 0
    end_winner = 0
    end_votes = 0
    card_votes = 0
    # 回退/翻译失败相关
    kw = Counter()
    keywords = ["cannot be played", "selected card", "fallback", "退化成 END",
                "回退规则", "MCTS 战斗异常", "翻译", "target"]
    with open(DEBUG, encoding="utf-8", errors="replace") as f:
        for line in f:
            low = line.lower()
            for k in keywords:
                if k.lower() in low:
                    kw[k] += 1
            if "ISMCTS votes" in line:
                m = re.search(r"ISMCTS votes:\s*(\{.*\})", line)
                if not m:
                    continue
                try:
                    d = ast.literal_eval(m.group(1))
                except Exception:
                    continue
                total += 1
                ev = d.get(("END_TURN", "END_TURN"), 0)
                if ev > 0:
                    with_end += 1
                    end_votes += ev
                cv = sum(v for kk, v in d.items() if kk[0] == "CARD")
                card_votes += cv
                # winner
                if d:
                    winner = max(d.items(), key=lambda x: x[1])[0]
                    if winner == ("END_TURN", "END_TURN"):
                        end_winner += 1
    return {
        "votes_total": total,
        "votes_with_END_TURN": with_end,
        "votes_END_TURN_winner": end_winner,
        "end_turn_vote_share_pct": round(100.0 * end_votes / (end_votes + card_votes), 1) if (end_votes + card_votes) else 0.0,
        "keyword_counts": dict(kw),
    }


def main():
    out = {"runs": {}, "debug_log": analyze_debug_log()}
    for run in RUNS:
        out["runs"][run] = analyze_run(run)
    with open(os.path.join(ROOT, "logs", "realtime_forensics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 打印关键摘要
    print("=== DEBUG LOG ===")
    print(json.dumps(out["debug_log"], ensure_ascii=False, indent=2))
    for run, r in out["runs"].items():
        print(f"\n=== {run} ===")
        print("END(命令侧) wasted_playable=%d all_unplay=%d no_energy=%d / total_end=%d"
              % (r["end_energy_wasted_playable"], r["end_energy_all_unplayable"],
                 r["end_no_energy"], r["n_end"]))
        print("帧侧 turns_total=%d wasted_playable=%d" % (r["frame_turns_total"], r["frame_turns_wasted_playable"]))
        print("END kind|degraded:", r["end_kind_degraded"])
        print("end_wasted_by_degraded:", r["end_wasted_by_degraded"])
        print("debuff_ids_detected:", r["debuff_ids_detected"])
        print("both_atk_debuff turns=%d debuff_positions=%s median=%s"
              % (r["turns_with_both_atk_debuff"], r["debuff_order_positions"], r["debuff_median_position"]))
        print("bash_first=%d bash_after=%d" % (r["bash_first_n"], r["bash_after_n"]))
        print("card_top10:", r["card_top"][:10])


if __name__ == "__main__":
    main()
