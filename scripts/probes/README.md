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
