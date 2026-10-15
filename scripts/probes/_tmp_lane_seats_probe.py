# -*- coding: utf-8 -*-
"""探针：在**生产配置**下检查泳道必保席是否真的在占位（而非空转）。

背景：`_select_beam` 只在 `len(expanded) > width` 时才启用泳道。若生产环境
（W=4 / depth=6 / max_nodes=300）每层展开出的节点数常常 ≤ W，泳道就是空转。

用法：.venv/Scripts/python.exe scripts/probes/_tmp_lane_seats_probe.py [width] [seeds]
"""
import collections
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import combat_beam  # noqa: E402

R = collections.defaultdict(list)
_orig = combat_beam.BeamSolver.solve


def _patched(self, bc):
    plan = _orig(self, bc)
    st = self.last_stats or {}
    key = "portfolio" if self.width_portfolio else "single"
    R[key].append((st.get("lane_seats", 0), st.get("nodes", 0),
                   st.get("plan_len", 0), st.get("won", False)))
    return plan


combat_beam.BeamSolver.solve = _patched


def main():
    width = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    from evaluation.evaluate import _make_policy, run_one

    policy = _make_policy("random")
    env_kwargs = {"combat_solver": "beam", "beam_width": width, "beam_depth": 6,
                  "beam_max_nodes": 300}
    for s in range(seeds):
        res = run_one(1000 + s, policy, mcts_sim=20000, env_kwargs=env_kwargs)
        print("  seed %d -> floor=%s win=%s" % (1000 + s, res.get("final_floor"), res.get("win")))

    rows = R["single"]
    if not rows:
        print("（没有 solve 调用记录）")
        return
    seats = [r[0] for r in rows]
    nz = sum(1 for x in seats if x > 0)
    print("\nsolve 调用 %d 次（%d 局 × 每局多场战斗 × 多决策步）" % (len(rows), seeds))
    print("  lane_seats 分布：", dict(collections.Counter(seats)))
    print("  占位比例（seats>0）： %d / %d = %.1f%%" % (nz, len(rows), 100.0 * nz / len(rows)))
    print("  平均 nodes/次：%.1f  平均 seats/次：%.2f" % (
        sum(r[1] for r in rows) / len(rows), sum(seats) / len(seats)))


if __name__ == "__main__":
    main()
