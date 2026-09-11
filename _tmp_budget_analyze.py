# -*- coding: utf-8 -*-
"""分析 Part1：拼出 预算×{一致率 vs 50000, 每帧耗时} 的完整表。
一致率定义：在非 None 帧上，命令串 == 50000 参考命令串 的比例。
"""
import json, statistics

probe = json.load(open('logs/mcts_budget_probe.json'))
blob = json.load(open('logs/mcts_budget_tuning.json'))
p1 = blob['part1_frames']

# 收集每帧各档命令（统一到 dict[str,cmd]）
frame_cmds = {}
for r in probe:
    key = f"{r['file']}#{r['frame_idx']}"
    frame_cmds[key] = {str(k): v for k, v in r.get('cmd_by_sims', {}).items()}
    frame_cmds[key]['file'] = r['file']
    frame_cmds[key]['idx'] = r['frame_idx']
    frame_cmds[key]['ms'] = {str(k): v for k, v in r.get('ms', {}).items()}
for key, rec in p1.items():
    if key not in frame_cmds:
        frame_cmds[key] = {}
        frame_cmds[key]['file'] = rec['file']
        frame_cmds[key]['idx'] = rec['frame_idx']
    frame_cmds[key]['10000'] = rec.get('cmds', {}).get('10000')
    frame_cmds[key]['20000'] = rec.get('cmds', {}).get('20000')
    rec_ms = {str(k): v for k, v in rec.get('ms', {}).items()}
    frame_cmds[key].setdefault('ms', {})
    frame_cmds[key]['ms'].update(rec_ms)

ref = '50000'
print("=== 每帧各档命令 ===")
for key in sorted(frame_cmds):
    fc = frame_cmds[key]
    row = {lvl: fc.get(lvl) for lvl in ['2000', '8000', '10000', '20000', '50000']}
    print(f"{key:45s} " + "  ".join(f"{lvl}={str(row[lvl])[:10]}" for lvl in ['2000','8000','10000','20000','50000']))

def agree_rate(lvl):
    num = den = 0
    for fc in frame_cmds.values():
        ca = fc.get(lvl)
        cb = fc.get(ref)
        if ca is None or cb is None:
            continue
        den += 1
        if ca == cb:
            num += 1
    return num, den, (num / den if den else None)

print("\n=== 一致率 vs 50000 ===")
for lvl in ['2000', '8000', '10000', '20000', '50000']:
    n, d, rate = agree_rate(lvl)
    print(f"  {lvl}: {n}/{d} = {rate:.3f}" if rate is not None else f"  {lvl}: n/a")

print("\n=== 每帧耗时 ms (mean/max over present) ===")
lvl_ms = {lvl: [] for lvl in ['2000', '8000', '10000', '20000', '50000']}
for fc in frame_cmds.values():
    for lvl in lvl_ms:
        m = fc.get('ms', {}).get(lvl)
        if m is not None:
            lvl_ms[lvl].append(m)
for lvl in ['2000', '8000', '10000', '20000', '50000']:
    ms = lvl_ms[lvl]
    if ms:
        print(f"  {lvl}: mean={statistics.fmean(ms):.1f} max={max(ms):.1f} n={len(ms)}")
