# -*- coding: utf-8 -*-
"""Part 1: 在已有 30 帧上补测 10000 / 20000 两档 MCTS，统计与 50000 档决策一致率。
复用 _tmp_mcts_probe.py 的采样/灌入逻辑，单进程；不改生产代码。
落盘 logs/mcts_budget_tuning.json（Part1 段）。
"""
import os, sys, json, time, glob
from collections import OrderedDict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import slaythespire as sts
from models.power_alias import alias_power_ids
import combat_brain as cb_mod

NEW_SIMS = [10000, 20000]
DETERMINIZATIONS = 16
OUT_JSON = os.path.join(ROOT, "logs", "mcts_budget_tuning.json")
PROBE_JSON = os.path.join(ROOT, "logs", "mcts_budget_probe.json")


def _sanitize(obj):
    if isinstance(obj, str):
        return obj.encode("utf-8", "replace").decode("utf-8")
    if isinstance(obj, list):
        return [_sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    return obj


def load_frame_state(fn_short, frame_idx):
    """从 realtime JSONL 重新读取某一行完整状态。"""
    full = os.path.join(ROOT, "logs", "realtime", fn_short)
    with open(full, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == frame_idx:
                line = line.strip()
                return json.loads(line) if line else None
    return None


def build_gc(st):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
    gc.init_from_json(json.dumps(st))
    return gc


def main():
    import logging
    logging.disable(logging.CRITICAL)

    probe = json.load(open(PROBE_JSON, encoding="utf-8"))
    cb = cb_mod.CombatBrain(simulation_count=50000, determinizations=DETERMINIZATIONS)

    # 载入已有的 Part1 中途结果（断点续跑）
    existing = {}
    if os.path.exists(OUT_JSON):
        try:
            blob = json.load(open(OUT_JSON, encoding="utf-8"))
            existing = blob.get("part1_frames", {})
        except Exception:
            existing = {}

    records = []
    for r in probe:
        fn = r["file"]
        idx = r["frame_idx"]
        ref500 = r.get("cmd_by_sims", {}).get("50000")
        key = f"{fn}#{idx}"
        if key in existing:
            rec = existing[key]
            rec["ref_50000"] = ref500
            records.append(rec)
            continue
        st = load_frame_state(fn, idx)
        if st is None:
            records.append({"file": fn, "frame_idx": idx, "ref_50000": ref500,
                            "error": "frame not found", "cmds": {}})
            continue
        clean = _sanitize(st)
        alias_power_ids(clean)
        cmds = {}
        ms = {}
        ok = True
        for sims in NEW_SIMS:
            try:
                gc2 = build_gc(clean)
                cb.sim_count = sims
                t1 = time.time()
                cmd = cb.decide(st, gc2)
                dt = (time.time() - t1) * 1000.0
                cmds[sims] = cmd
                ms[sims] = round(dt, 1)
            except Exception as e:
                cmds[sims] = f"<err:{e!r}>"
                ok = False
        rec = {"file": fn, "frame_idx": idx, "ref_50000": ref500,
               "cmds": cmds, "ms": ms, "error": None if ok else "decide-error"}
        records.append(rec)
        # 实时落盘 Part1
        _dump(records)
        print(f"[done] {key} ref500={ref500} cmds={cmds}", flush=True)

    # 统计一致率
    def agg(sims):
        num = den = 0
        none_pairs = 0
        for rec in records:
            ref = rec.get("ref_50000")
            got = rec.get("cmds", {}).get(sims)
            if ref is None or got is None:
                none_pairs += 1
                continue
            den += 1
            if ref == got:
                num += 1
        return num, den, none_pairs

    summary = {}
    for sims in NEW_SIMS:
        num, den, none_pairs = agg(sims)
        summary[sims] = {
            "agree": num, "denom": den,
            "agree_rate": (num / den) if den else None,
            "none_pairs": none_pairs,
        }
    print("=== Part1 agreement vs 50000 ===")
    for sims, s in summary.items():
        print(f"  {sims}: {s['agree']}/{s['denom']} = "
              f"{s['agree_rate'] if s['agree_rate'] is not None else 'NA'} "
              f"(none_pairs={s['none_pairs']})")

    blob = {}
    if os.path.exists(OUT_JSON):
        try:
            blob = json.load(open(OUT_JSON, encoding="utf-8"))
        except Exception:
            blob = {}
    blob["part1_frames"] = {f"{r['file']}#{r['frame_idx']}": r for r in records}
    blob["part1_summary"] = {str(k): v for k, v in summary.items()}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    print(f"[complete] Part1 写入 {OUT_JSON}")


def _dump(records):
    blob = {}
    if os.path.exists(OUT_JSON):
        try:
            blob = json.load(open(OUT_JSON, encoding="utf-8"))
        except Exception:
            blob = {}
    blob["part1_frames"] = {f"{r['file']}#{r['frame_idx']}": r for r in records}
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(blob, f, ensure_ascii=False, indent=1)
    if os.path.exists(OUT_JSON):
        os.remove(OUT_JSON)
    os.rename(tmp, OUT_JSON)


if __name__ == "__main__":
    main()
