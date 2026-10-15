# -*- coding: utf-8 -*-
"""Beam Search 战斗求解器 —— 仿照 CombatSolver 的确定性搜索范式，用于塔一铁甲战士。

## 为什么写这个

我们的战斗层用 MCTS（`BattleScumSearcher2`，C++，UCT + 随机 rollout）。但 STS 战斗
**基本是确定性的**（抽牌堆顺序已知、敌人意图可见）——对确定性问题用随机采样 rollout
等于浪费采样，这可能是"改评估函数 / 加预算"等一系列改进都无效的深层原因。

CombatSolver（github.com/Torch1230/CombatSolver，MIT，STS2 战斗路线求解器）用完全不同的
范式：**确定性分层 Beam Search + 字段级状态指纹去重 + 支配剪枝**。本模块把它复刻到
我们的 STS1 / Ironclad / A0 场景。

## 与 CombatSolver 的组件对应

| CombatSolver（C#，STS2）                    | 本模块（Python，STS1）                        |
|--------------------------------------------|----------------------------------------------|
| `ICombatPredictionForkableState.Fork`      | `sts.BattleContext(bc)` 拷贝构造（实测 0.001ms）|
| `Expand`（Expansion.cs:22）                 | `_expand()`：枚举 `is_valid` 动作并 `execute` |
| `StateFingerprint` / `BuildStateKey`       | `fingerprint()` 字段级元组                    |
| `Transpositions`（转置表）                   | `_seen`（**同层**去重保留最优）               |
| `BeamRetentionPolicy.RankBest`（:786）      | `_rank()`：按 `evaluate` 降序取 top-W         |
| `BeamRankScore`（Ranking.cs:848）           | `evaluate()` 启发式                           |
| 回合层预算（Phases.cs:1500-1522）            | `max_depth` / `max_nodes`                     |

## 已知限制（P1，故意保留）

1. **Power 读不到**：C++ 绑定只暴露 `strength / dexterity / focus / artifact`，
   Vulnerable / Weak / Frail / Poison 等**读不到** → 指纹会漏判（"有易伤"与"无易伤"
   可能算出同一指纹）、启发式缺该特征。**待绑定补齐后再提升精度**
   （但先验证算法本身是否有效，这才是 P1 的目的）。
2. **药水动作跳过**：绑定未暴露战斗内药水索引 → 只解"出牌 + 结束回合"。
3. **只做精确去重**，未实现 CombatSolver 的六维支配关系（TranspositionLabel）。

## 用法

```python
import json, slaythespire as sts
from combat_beam import BeamSolver

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
gc.init_from_json(json.dumps(game_state))
bc = sts.BattleContext(); bc.init_from_json(gc, json.dumps(game_state))

solver = BeamSolver(beam_width=8, max_depth=24, max_nodes=20000)
plan = solver.solve(bc)            # -> [('CARD', 0, 0), ('CARD', 1, 0), ('END_TURN', 0, 0), ...]
print(solver.last_stats)           # {'nodes':…, 'ms':…, 'outcome':…, 'score':…}
```
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import slaythespire as sts

# --------------------------------------------------------------------------
# 评估权重（量级参考 CombatSolver 的 SolverWeights.cs：HP = 30_000）
# --------------------------------------------------------------------------
_W_HP = 30_000          # 玩家生命（最高权重，同 CombatSolver 的 projectedHp×30000）
_W_BLOCK = 18_000       # 格挡：有效但不如此刻的血（会过期）
_W_ENERGY = 9_000       # 未花掉的能量 = 潜在资源
_W_ENEMY_HP = 12_000    # 敌人剩余血量 = 还需输出的伤害量
_W_INTENT = 6_000       # 敌人意图伤害 = 下回合威胁
_W_STR = 15_000         # 力量（本场持续收益）
_W_DEX = 12_000         # 敏捷
_W_HAND = 2_500         # 手牌 = 选择空间
_W_DRAW = 500           # 牌库余量（很弱）

_WIN_BASE = 1e9
_LOSS_BASE = -1e9
_TURN_PENALTY = 500.0   # 胜利时回合越少越好


# --------------------------------------------------------------------------
# 状态指纹：字段级元组（对应 CombatSolver 的 StateFingerprintBuilder）
# --------------------------------------------------------------------------
def _intent_damage(m, bc: "sts.BattleContext") -> int:
    """怪物当前意图的**总**伤害。

    ⚠️ 绑定签名是 `intent_damage(bc) -> (base_damage, attack_count)`（需要传 bc！），
    所以总伤害 = 两者相乘（如 (11, 1) → 11；(5, 3) → 15）。
    """
    d, n = m.intent_damage(bc)
    return int(d) * int(n)


def fingerprint(bc: "sts.BattleContext") -> tuple:
    """把战斗状态压成可哈希的元组。⚠️ 不含 Power（绑定未暴露），见模块 docstring 限制 1。

    ⚠️ pybind11 绑定里**属性与方法混用**（实测清单）：
        - `Monster.intent_damage()` 是**方法**（取属性会得到含内存地址的 bound method
          → 指纹每次不同，去重彻底失效）
        - `BattleContext.is_terminal()` 同样是方法
        - `Card.is_playable()` / `Card.requires_target()` 也是方法
        - Player 的字段则**全是属性**

    ⚡ 性能（2026-10-15 优化，P0-①）：
        手牌与牌堆**必须用 `c.id.value`（整数），不要用 `c.id.name`（字符串）**。
        实测 `CardId.name` 单次访问 **≈ 444 µs**（内部按名线性查表），而 `.value` 仅
        **≈ 1.2 µs** —— **快 358×**。原实现里这两行占了 fingerprint 总耗时的 **98%**
        （2553 µs → 约 22 µs，**提速 ≈ 115×**）。
        等价性已验证：371 个 CardId 常量的 `value` 与 `name` **一一对应**
        （`scripts/probes/_tmp_cardid_value_bijection.py`），故去重语义完全不变。
        ⚠️ 指纹里的整数不参与对外展示；要人工阅读请单独打印 `c.id.name`。
    """
    p = bc.player
    return (
        bc.turn,
        bc.input_state,
        p.hp, p.block, p.energy,
        p.strength, p.dexterity, p.focus, p.artifact,
        # 手牌：顺序无关（排序），含费用与升级态
        tuple(sorted((c.id.value, c.cost_for_turn, bool(c.upgraded)) for c in bc.hand)),
        # 抽牌堆：**顺序重要**（决定后续抽到什么）
        tuple(c.id.value for c in bc.draw_pile),
        tuple(sorted(c.id.value for c in bc.discard_pile)),
        tuple(sorted(c.id.value for c in bc.exhaust_pile)),
        tuple((m.name, m.hp, m.block, _intent_damage(m, bc), bool(m.is_alive))
              for m in bc.monsters),
    )


# --------------------------------------------------------------------------
# 动作枚举（对应 Expansion.cs 的 Expand）
# --------------------------------------------------------------------------
def legal_actions(bc: "sts.BattleContext") -> list[tuple[str, int, int]]:
    """枚举当前状态下所有合法动作，返回 [(type_name, idx1, idx2), ...]。

    用 `is_valid` 过滤 —— 这与 C++ 内部 `getLegalActions` 同源，保证不漏不错。
    药水暂不枚举（见模块 docstring 限制 2）。
    """
    out: list[tuple[str, int, int]] = []
    n_mon = len(bc.monsters)
    for i, card in enumerate(bc.hand):
        # 需要有目标的牌：对每个存活目标试一次；否则只需 idx2=0
        if card.requires_target():
            targets = range(n_mon) if n_mon else (0,)
        else:
            targets = (0,)
        for t in targets:
            if sts.BattleAction(sts.ActionType.CARD, i, t).is_valid(bc):
                out.append(("CARD", i, t))
    if sts.BattleAction(sts.ActionType.END_TURN, 0, 0).is_valid(bc):
        out.append(("END_TURN", 0, 0))
    return out


#: 动作类型名 → sts.ActionType。
#: **对外公开**：`env/lightspeed_env.py` 的 Beam 战斗循环（_run_combat_beam）要用它
#: 把 BeamSolver 返回的 `(type_name, idx1, idx2)` 还原成可执行动作。
ACTION_TYPE_BY_NAME = {
    "CARD": sts.ActionType.CARD,
    "POTION": sts.ActionType.POTION,
    "END_TURN": sts.ActionType.END_TURN,
}
_ACTION_TYPE = ACTION_TYPE_BY_NAME   # 兼容旧名（内部使用）


def _make(act: tuple[str, int, int]) -> "sts.BattleAction":
    t, i, j = act
    return sts.BattleAction(ACTION_TYPE_BY_NAME[t], i, j)


# --------------------------------------------------------------------------
# 启发式评估（对应 Ranking.cs 的 BeamRankScore）
# --------------------------------------------------------------------------
def evaluate(bc: "sts.BattleContext") -> float:
    """给非终局状态打分。分数越高越好。**启发式质量决定 Beam 的成败。**"""
    if bc.is_terminal():
        P = bc.player
        if bc.outcome == sts.BattleOutcome.PLAYER_VICTORY:
            # 赢：血越多、回合越少越好（同 CombatSolver 的 100×(35 + HP − 回合×0.01)）
            return _WIN_BASE + P.hp * _W_HP - bc.turn * _TURN_PENALTY
        # 输：仍偏爱"多消耗了敌人"的局面（给失败分支一点梯度）
        enemy_hp = sum(m.hp for m in bc.monsters if m.is_alive)
        return _LOSS_BASE - enemy_hp * _W_ENEMY_HP

    P = bc.player
    score = 0.0
    score += P.hp * _W_HP
    score += P.block * _W_BLOCK
    score += P.energy * _W_ENERGY
    score += P.strength * _W_STR
    score += P.dexterity * _W_DEX
    for m in bc.monsters:
        if m.is_alive:
            score -= m.hp * _W_ENEMY_HP
            score -= _intent_damage(m, bc) * _W_INTENT
            score -= m.block * 4_000
    score += len(bc.hand) * _W_HAND
    score += len(bc.draw_pile) * _W_DRAW
    return score


# --------------------------------------------------------------------------
# 节点与主循环
# --------------------------------------------------------------------------
@dataclass
class _Node:
    bc: "sts.BattleContext"
    path: tuple = ()          # ((type, i, j), ...)
    score: float = 0.0
    depth: int = 0


class BeamSolver:
    """确定性 Beam Search 战斗求解器。

    Args:
        beam_width: 每层保留的节点数（CombatSolver 默认档位 45~135，我们先用小值验证）
        max_depth:  最大动作步数（含 END_TURN；一步 = 一个动作）
        max_nodes:  全局展开上限（对应 CombatSolver 的 MaxExpandedNodes）
        dedup:      True = 同层指纹去重保最优；False = 不去重（对照用）
        seed:       保留参数（本求解器完全确定性，不使用 RNG）
    """

    def __init__(self, beam_width: int = 8, max_depth: int = 24,
                 max_nodes: int = 20_000, dedup: bool = True, seed: int = 0):
        self.beam_width = int(beam_width)
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.dedup = bool(dedup)
        self.last_stats: dict = {}

    def solve(self, bc: "sts.BattleContext") -> list[tuple[str, int, int]]:
        """主循环（对应 CombatBeamSolver.Phases.cs:1479 的分层推进）。

        返回：从当前状态出发的最优动作序列。
        说明：全程在**分叉副本**上操作，不修改传入的 `bc`。
        """
        t0 = time.perf_counter()
        root = _Node(bc=sts.BattleContext(bc), path=(), score=evaluate(bc), depth=0)
        beam: list[_Node] = [root]
        best_terminal: _Node | None = None
        # ⚠️ best_open **不能**以 root 初始化：root 是"尚未行动"的局面（能量/手牌未消耗、
        # 血未掉），而 evaluate 里 `hp×30000` 权重最大 —— 任何一次行动后分数都会低于它。
        # 若以 root 为初值，则所有后代都赢不过 root → chosen 永远是 root → solve 返回空方案
        # → 调用方拿不到动作而死锁（实测：energy=0 只能 END_TURN 的帧上 100% 复现）。
        best_open: _Node | None = None
        nodes = 0

        for _ in range(self.max_depth):
            if not beam or nodes >= self.max_nodes:
                break
            expanded: list[_Node] = []
            for node in beam:
                for act in legal_actions(node.bc):
                    if nodes >= self.max_nodes:
                        break
                    child_bc = sts.BattleContext(node.bc)     # Fork
                    try:
                        _make(act).execute(child_bc)
                    except Exception:
                        continue                              # 非法/不支持的动作直接跳过
                    nodes += 1
                    child = _Node(bc=child_bc, path=node.path + (act,),
                                  score=0.0, depth=node.depth + 1)
                    child.score = evaluate(child_bc)
                    if child_bc.is_terminal():
                        if (child_bc.outcome == sts.BattleOutcome.PLAYER_VICTORY
                                and (best_terminal is None or child.score > best_terminal.score)):
                            best_terminal = child
                    else:
                        expanded.append(child)
                        if best_open is None or child.score > best_open.score:
                            best_open = child

            if best_terminal is not None:
                # 已经找到胜利路线：若本层没有更好的，可提前收（对应 CombatSolver 的
                # acceptableBattleHpLossReached → 立即排空前沿）
                if not expanded:
                    break
                top = max(c.score for c in expanded) if expanded else 0
                if best_terminal.score >= top:
                    break

            if self.dedup:
                best_by_fp: dict[tuple, _Node] = {}
                for c in expanded:
                    fp = fingerprint(c.bc)
                    cur = best_by_fp.get(fp)
                    if cur is None or c.score > cur.score:
                        best_by_fp[fp] = c
                expanded = list(best_by_fp.values())

            expanded.sort(key=lambda n: -n.score)
            beam = expanded[:self.beam_width]

        chosen = best_terminal if best_terminal is not None else best_open
        if chosen is None:
            # 极端兜底：本层一个可展开节点都没有（例如全部动作都会立即终局且非胜利）。
            # 至少要返回一个合法动作，否则调用方会「无动作」死锁。
            acts = legal_actions(root.bc)
            if not acts:
                self.last_stats = {"nodes": nodes, "ms": (time.perf_counter() - t0) * 1000.0,
                                   "depth": 0, "score": float("nan"), "won": False,
                                   "outcome": str(bc.outcome), "plan_len": 0, "dead_end": True}
                return []
            chosen = _Node(bc=root.bc, path=(acts[0],), score=float("nan"), depth=1)
        self.last_stats = {
            "nodes": nodes,
            "ms": (time.perf_counter() - t0) * 1000.0,
            "depth": chosen.depth,
            "score": chosen.score,
            "won": bool(best_terminal is not None),
            "outcome": str(chosen.bc.outcome),
            "plan_len": len(chosen.path),
        }
        return list(chosen.path)

    def decide(self, bc: "sts.BattleContext") -> tuple[str, int, int] | None:
        """只取第一个动作（真机部署时每步调一次）。"""
        plan = self.solve(bc)
        return plan[0] if plan else None


__all__ = ["BeamSolver", "fingerprint", "legal_actions", "evaluate", "ACTION_TYPE_BY_NAME"]
