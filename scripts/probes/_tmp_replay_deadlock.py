# -*- coding: utf-8 -*-
"""回放真机死循环帧，验证 2026-10-16 两项修复是否真的能解套。

用 `logs/realtime/1016_234015_states.jsonl`（真机卡死那次的 L1 状态流）：
  ① 对每一帧跑**修复后的** `SimBrain.decide()`，看命令序列；
  ② 同时把 (指纹, 命令) 喂给 `LoopDetector`，看它第几帧报警。

期望：
  - 修复前：CARD_REWARD→SKIP 与 COMBAT_REWARD→CHOOSE 0 无限交替，守卫**永不报警**；
  - 修复后：只剩卡牌项且本层已跳过 → COMBAT_REWARD→**PROCEED**，循环被打断；
    即便逻辑再次退化，守卫也会在 ~8 帧内报 stall。

用法：
    .venv/Scripts/python.exe scripts/probes/_tmp_replay_deadlock.py [run 前缀]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:                          # noqa: BLE001
                    pass
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "1016_234015"
    path = os.path.join(ROOT, "logs", "realtime", run + "_states.jsonl")
    frames = load(path)
    print("读入 %d 帧: %s" % (len(frames), path))

    from main import SimBrain, LoopDetector, _fingerprint

    brain = SimBrain()                 # 默认构造 = 规则路径
    guard = LoopDetector(limit=8, hard_limit=20)

    # 只回放"卡死尾段"：从第一次出现 COMBAT_REWARD(仅 CARD) 开始
    start = None
    for i, s in enumerate(frames):
        gs = s.get("game_state") or {}
        if (gs.get("screen_type") or "").upper() == "COMBAT_REWARD":
            rws = ((gs.get("screen_state") or {}).get("rewards") or [])
            if rws and all(str(r.get("reward_type", "")).upper() == "CARD" for r in rws):
                start = i
                break
    if start is None:
        print("没找到『COMBAT_REWARD 仅剩卡牌项』的帧，换 run 试试")
        return
    tail = frames[start:]
    print("从第 %d 帧开始回放 %d 帧（卡死尾段）\n" % (start, len(tail)))

    fired_at = None
    cmds = []
    for k, s in enumerate(tail[:40]):
        gs = s.get("game_state") or {}
        scr = (gs.get("screen_type") or "").upper()
        try:
            cmd = brain.decide(s)
        except Exception as e:                             # noqa: BLE001
            cmd = "ERR:%s" % type(e).__name__
        cmds.append(cmd)
        sev = guard.observe(_fingerprint(s), has_game_state=bool(s.get("game_state")),
                            cmd=cmd)
        if fired_at is None and sev >= guard.limit:
            fired_at = k
        if k < 14 or (fired_at is not None and k == fired_at):
            print("  #%-3d %-14s -> %-10s  守卫 severity=%d %s"
                  % (k, scr, cmd, sev, "  ← 报警" if sev >= guard.limit else ""))

    print()
    from collections import Counter
    print("命令分布:", dict(Counter(cmds)))
    if "PROCEED" in cmds:
        print("✅ 修复生效：COMBAT_REWARD 已改为 PROCEED（不再重开卡牌屏）")
    else:
        print("❌ 没出现 PROCEED —— 修复未生效，需复查")
    if fired_at is not None:
        print("✅ 守卫生效：第 %d 帧报警（即便逻辑再退化也能解套）" % fired_at)
    else:
        print("⚠️ 守卫未报警（若命令已推进则属正常）")


if __name__ == "__main__":
    main()
