# -*- coding: utf-8 -*-
"""真机 run 报告（只读解析 `logs/realtime/`，不改任何日志）。

回答实跑之后最该看的四个问题：
  1. **跑通了吗** —— 局数、终局楼层、胜负
  2. **决策降级率** —— `degraded` / `total_fallbacks`（规则兜底是**红线**：偏高说明模型/桥接有问题）
  3. **卡死过吗** —— `stall` / `stall_giveup` 事件（防卡死守卫触发次数）
  4. **战斗层在实际打什么** —— `kind` 分布与战斗决策占比

用法：
    .venv/Scripts/python.exe scripts/realtime_report.py                 # 最新的 run
    .venv/Scripts/python.exe scripts/realtime_report.py 1016_231045     # 指定 run 前缀
    .venv/Scripts/python.exe scripts/realtime_report.py --list          # 列出所有 run
"""
import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RT = os.path.join(ROOT, "logs", "realtime")


def _runs():
    out = {}
    for p in glob.glob(os.path.join(RT, "*_decisions.jsonl")):
        base = os.path.basename(p)
        prefix = base[: -len("_decisions.jsonl")]
        out[prefix] = p
    return out


def _read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:                              # noqa: BLE001
                continue
    return rows


def report(prefix):
    dec = _read_jsonl(os.path.join(RT, prefix + "_decisions.jsonl"))
    ev = _read_jsonl(os.path.join(RT, prefix + "_events.jsonl"))
    st_path = os.path.join(RT, prefix + "_states.jsonl")
    n_states = sum(1 for _ in open(st_path, encoding="utf-8")) if os.path.exists(st_path) else 0

    print("=" * 68)
    print("run %s   决策 %d 条 | 事件 %d 条 | 状态 %d 帧" % (prefix, len(dec), len(ev), n_states))
    print("=" * 68)
    if not dec:
        print("（无决策记录 —— 这个 run 没真正玩起来，或刚启动就被关掉）")
        return

    floors = [d.get("floor") or 0 for d in dec]
    print("楼层: 起 %s → 终 %s   （最高 %s）" % (floors[0], floors[-1], max(floors)))
    kinds = Counter(d.get("kind") for d in dec)
    print("决策类型:", dict(kinds))
    combat = [d for d in dec if d.get("kind") == "COMBAT"]
    print("战斗决策: %d 条（占 %.1f%%）"
          % (len(combat), 100.0 * len(combat) / len(dec)))

    deg = sum(1 for d in dec if d.get("degraded"))
    fb = max((d.get("total_fallbacks") or 0) for d in dec)
    print("🔴 降级决策: %d / %d (%.2f%%)   累计 fallback = %d"
          % (deg, len(dec), 100.0 * deg / len(dec), fb))

    evk = Counter(e.get("kind") for e in ev)
    print("事件类型:", dict(evk))
    n_stall = evk.get("stall", 0)
    n_giveup = evk.get("stall_giveup", 0)
    print("卡死守卫: stall=%d  stall_giveup=%d%s"
          % (n_stall, n_giveup, "  ⚠️ 触发过放弃" if n_giveup else ""))

    # 终局状态（若 states 有记录）
    if os.path.exists(st_path):
        last = None
        with open(st_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)
                    except Exception:                      # noqa: BLE001
                        pass
        if last:
            gs = last.get("game_state") or {}
            screen = (gs.get("screen_type") or gs.get("screen_state") or "?")
            deck = gs.get("deck") or []
            relics = gs.get("relics") or []
            print("终局状态: screen=%s floor=%s act=%s hp=%s/%s gold=%s"
                  % (screen, gs.get("floor"), gs.get("act"),
                     gs.get("current_hp"), gs.get("max_hp"), gs.get("gold")))
            print("         牌组 %d 张 | 遗物 %d 个"
                  % (len(deck) if isinstance(deck, list) else -1,
                     len(relics) if isinstance(relics, list) else -1))

    # 时长
    ts = [d.get("ts") for d in dec if d.get("ts")]
    if len(ts) >= 2:
        dur = ts[-1] - ts[0]
        print("时长: %.1f 分钟  （决策频率 %.2f 次/秒）" % (dur / 60.0, len(ts) / dur))
    print()

    # 降级的具体样本（若有）
    bad = [d for d in dec if d.get("degraded")][:5]
    if bad:
        print("降级样本（前 5 条）:")
        for d in bad:
            print("  n=%s floor=%s screen=%s cmd=%s scores=%s"
                  % (d.get("n"), d.get("floor"), d.get("screen"), d.get("cmd"),
                     str(d.get("scores"))[:60]))


def main():
    args = sys.argv[1:]
    runs = _runs()
    if not runs:
        print("logs/realtime/ 下没有 *_decisions.jsonl")
        return
    if "--list" in args:
        for k in sorted(runs):
            print("  %s" % k)
        return
    if args and not args[0].startswith("--"):
        prefix = args[0]
        if prefix not in runs:
            cand = [k for k in runs if k.startswith(prefix)]
            if not cand:
                print("找不到 run: %s（用 --list 看可用的）" % prefix)
                return
            prefix = sorted(cand)[-1]
    else:
        prefix = sorted(runs)[-1]
    report(prefix)


if __name__ == "__main__":
    main()
