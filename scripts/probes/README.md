# scripts/probes/ —— 历史探针脚本集合

这里存放项目开发过程中一次性使用的**历史探针 / 分析脚本**（原散落在仓库根目录，
命名前缀 `_tmp_*` / `tmp_probe_event*`）。它们用于：

- 探测 `slaythespire` 扩展与 `GameContext` 的字段/API（`*_api_probe*.py`、`probe_frame.py`）
- 统计与诊断运行结果（`_tmp_deck_report.py`、`_tmp_mcts_analyze.py`、`_tmp_hypo_report.py`）
- 训练/评估实验的临时入口（`_tmp_eval_v10.py`、`_tmp_expA.py`、`_tmp_budget_*.py`）

> **这些不是生产代码。** 生产模块（`main.py`、`combat_beam.py`、`combat_brain.py`、
> `shop_brain.py`、`main_rule.py`、`verify_env.py`）仍在仓库根目录，不要与本目录混淆。

## 运行方式

**务必以「项目根」为当前工作目录**运行：

```bash
# 从项目根（含 .git 的那一级）执行
.venv/Scripts/python scripts/probes/_tmp_beam_vs_mcts.py
```

多数脚本内部依赖相对路径（`logs/`、`checkpoints/`、`data/`），并假设
CWD = 项目根。若在别处运行，路径会失配。

## 关于 `__file__` 定位

早期脚本用：

```python
_HERE = os.path.dirname(os.path.abspath(__file__))   # 迁移前 __file__ 就在项目根，成立
```

归位到 `scripts/probes/` 后，`_HERE` 会指向本目录而非项目根，该类脚本会失效。
**不打算逐个回修**（数量多、多数已过期）。

若某个旧脚本仍需运行且报「找不到 logs/ 或 checkpoints/」，请按下面写法自行修正：

```python
def _find_project_root(start):
    d = os.path.dirname(os.path.abspath(start))
    while not os.path.isdir(os.path.join(d, ".git")):
        parent = os.path.dirname(d)
        if parent == d:
            return os.path.dirname(os.path.abspath(start))
        d = parent
    return d

_HERE = _find_project_root(__file__)   # 现在指向项目根
```

## 已完成根定位改造的脚本

以下 3 个是最近仍在使用的脚本，已改为「从 `__file__` 向上查找含 `.git` 的目录」，
在任意 CWD 下均可正确指向项目根的 `logs/` 与 `checkpoints/`：

| 脚本 | 用途 |
|---|---|
| `_tmp_shaping_causal.py` | V11 奖励塑形因果诊断（E1 vs V11 牌组指标分层） |
| `_tmp_beam_vs_mcts.py` | Beam Search vs MCTS 同帧连续对局对比 |
| `_tmp_eval_v11.py` | V11 vs E1 固定种子配对评估 |

同目录下的 `*.log` / `*.txt` 是这些探针的历史输出，一并从根目录迁来归档。

## 2026-10-15 新增（Beam 性能优化 P0-① / 高级机制 P0-② 相关）

| 脚本 | 用途 | 依赖 |
|---|---|---|
| `_tmp_beam_fingerprint_probe.py` | 分解 `fingerprint()` 各段耗时，定位到 `CardId.name`（444 µs/次） | 无 |
| `_tmp_cardid_value_bijection.py` | 验证 371 个 `CardId` 的 `value` ↔ `name` 一一映射（换整数 id 的前提） | 无 |
| `_tmp_beam_equivalence.py` | 新旧指纹实现下 `solve()` 方案逐字对比 + 端到端提速 | 需先把旧版导出：`git show <commit>:combat_beam.py > scripts/probes/_tmp_combat_beam_old.py`（**该副本不入库**） |
| `_tmp_lane_seats_probe.py` | 统计泳道必保席在生产配置下的占位率（防止机制空转） | 无 |

结论与数据见 `logs/beam-fingerprint-optimization.md`。

## 2026-10-16 新增（启发式权重调参 P0）

| 脚本 | 用途 |
|---|---|
| `_tmp_weight_sweep.py` | 扫描 `Weights` 的 14 个单维覆盖配置，**配对比较**（配对 t + McNemar）vs baseline |

> 之所以能用 100 局做配对检验：本项目评估是**确定性**的（同种子同配置两次运行结果逐项相同），
> 所以每个种子可配对，灵敏度远高于两组独立比例检验。
> 结果与判读见 `logs/weight-tuning-report.md`。

| 脚本 | 用途 |
|---|---|
| `_tmp_macro_metrics.py` | **宏观指标提取**：各 Act 进入点快照 + 通关局牌组/遗物 —— 「抓牌质量」主指标（handoff §3 对照实验所需）|
| `_tmp_beam_realtime_probe.py` | **真机可用性验证**：用 L1 真实战斗日志按**真机路径**构造 bc 并跑 Beam，统计成功率与单步耗时 |

## 2026-10-16 晚 新增（Beam 终局估值 —— `value_mode="rollout"`）

背景：`logs/sts1-solver-comparison-report.md` 查出 Beam 的 `evaluate()` **没有终局信息**，
这正是「更宽/更深/更大预算全部变差」的根本解释。故给 Beam 补上 rollout 终局估值。

| 脚本 | 用途 |
|---|---|
| `_tmp_terminal_value_cost.py` | **成本与信息量探针**：量化「Python 贪心 rollout 到终局」（≈1.0 ms/次、14.3 步）与「C++ `BattleSearcher` 当价值函数」（sims=30 → 0.23 ms）两条路的代价 |
| `_tmp_paired_compare.py` | **配对比较分析器**（复用）：吃两个以上宏观指标 JSON，输出 Δfloor / 配对 t / McNemar b,c / p / delta 分布 / 通关局宏观指标。用法：`_tmp_paired_compare.py <baseline.json> <变体.json> [...] [--n=100]` |

结论与数据见 `logs/beam-terminal-value-report.md`。

### ⚠️ 测试 fixture 的重要修正（同批）

`tests/test_combat_beam.py` 的 `_bc()` 原先用 `bc.init_from_gc(gc)`，而 `gc` 是
`init_from_json` 载入的 —— **这条路读未初始化内存（UB）**：实测 20 次里 **19 次怪物列表为空**、
1 次得到垃圾值 `(1, 999)`，导致测试**随机跳过/随机通过**（同一份代码三次运行 34/35/36 passed）。
已改为真机同源的 `bc.init_from_json(gc, json)`（20/20 稳定，怪物 = `JAW_WORM 42hp`）。
⚠️ 训练环境（`env/lightspeed_env.py`）用 `init_from_gc` 是**合法的**（那里的 `gc` 是引擎
自建的真实 `GameContext`），**不要顺手改**。
