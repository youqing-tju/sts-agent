# -*- coding: utf-8 -*-
"""端到端等价性 + 提速验证：新版（整数 id）vs 旧版（字符串 id）的 BeamSolver。

判据：**对每个状态，两版 solve() 返回的动作序列必须逐字相同**。
（因为只改了状态表示的编码方式，去重语义不变 → 搜索结果必须完全一致。）

用法：.venv/Scripts/python.exe scripts/probes/_tmp_beam_equivalence.py
"""
import importlib.util
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import combat_beam as cb_new  # noqa: E402
from tests.test_combat_beam import _collect_states  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


OLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_tmp_combat_beam_old.py")

BEAM_KW = dict(beam_width=3, max_depth=6, max_nodes=400)


def main():
    cb_old = _load(OLD_PATH, "combat_beam_old")

    states = _collect_states(120)
    open_states = [s for s in states if not s.is_terminal()]
    print("收集到状态 %d 个（非终局 %d 个）" % (len(states), len(open_states)))

    n = min(25, len(open_states))
    sample = open_states[:n]

    # 预热
    cb_old.fingerprint(sample[0])
    cb_new.fingerprint(sample[0])

    mismatches = []
    t_old = t_new = 0.0
    for i, s in enumerate(sample):
        t0 = time.perf_counter()
        plan_old = cb_old.BeamSolver(**BEAM_KW).solve(s)
        t_old += time.perf_counter() - t0

        t0 = time.perf_counter()
        plan_new = cb_new.BeamSolver(**BEAM_KW).solve(s)
        t_new += time.perf_counter() - t0

        if plan_old != plan_new:
            mismatches.append((i, plan_old, plan_new))

    print("\n对比 %d 个状态：" % len(sample))
    if mismatches:
        print("🔴 有 %d 个状态的方案不同：" % len(mismatches))
        for i, po, pn in mismatches[:3]:
            print("   [%d] old=%s" % (i, po))
            print("        new=%s" % pn)
    else:
        print("✅ 全部 %d 个状态的 solve() 方案**逐字相同** → 语义等价" % len(sample))

    print("\n耗时（solve 总时长，含搜索全过程）：")
    print("   旧版（字符串指纹）: %7.2f s   （%.1f ms/次）" % (t_old, t_old / len(sample) * 1000))
    print("   新版（整数  指纹）: %7.2f s   （%.1f ms/次）" % (t_new, t_new / len(sample) * 1000))
    if t_new > 0:
        print("   → 端到端 **提速 %.1f×**" % (t_old / t_new))

    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
