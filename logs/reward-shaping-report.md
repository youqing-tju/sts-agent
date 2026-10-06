# 奖励塑形报告（reward-shaping）

把「升级卡数 / 遗物数」作为**终局加成**加进 PPO 回报，引导 agent 为长期收益
忍受短期风险。纯奖励信号驱动，**无任何规则兜底**。

## 1. env.gc.deck / env.gc.relics 实测元素结构

探针脚本 `_tmp_probe_deck_fields.py`（跑 1 局）实测：

- `env.gc.deck`：C++ 绑定的 `Card` 对象列表（**不是 dict**）。
  - `c.id.name`：卡名（str）
  - **升级字段 = `c.upgrade_count`（int，`>0` 即升级）**；另有 `c.upgraded`（bool）。
    两字段等价（多种子实测计数完全一致）。`upgrades` 字段不存在。
  - `upgrade_count = sum(1 for c in deck if c.upgrade_count > 0 or c.upgraded)`
- `env.gc.relics`：`Relic` 对象列表；`relic_count = len(list(gc.relics))`。
  - `r.id.name`：遗物名（str）

## 2. 四处改动（diff 摘要）

文件 `training/ppo.py`：
- L9-15：更新回报公式注释，写出两个新 bonus。
- L36-58：新增模块级 `_count_upgraded(deck)` / `_count_relics(relics)` 辅助。
- L31-43 `__init__`：加 `upgrade_bonus=0.0`、`relic_bonus=0.0`（默认 0，向后兼容）。
- L115-122 `update()`：解包新 5 元组，G 公式加
  `+ upgrade_bonus*upgrade_count + relic_bonus*relic_count`。
- L298-316 `_run_episode`：局末读 `env.gc.deck/relics`，返回
  `(traj, final_floor, win, upgrade_count, relic_count)`（int，多进程可 pickle；
  FakeEnv 无 `gc` 时记 0）。`_collect_worker` 返回值自动透传。

文件 `training/run_training.py`：
- L42-47：CLI 加 `--upgrade-bonus` / `--relic-bonus`（默认 0.0）。
- L101-105：透传给 `PPOTrainer`。

> `ReinforceTrainer` 未改：其回报在 `reinforce.py:93` 独立写（单进程、用自己的
> `collect_episode`/`update`，不走 ppo 的 `_run_episode`）。本次**只支持 PPO**，
> 不为对称而乱改；REINFORCE 如需同款塑形是后续独立任务。

## 3. 冒烟测试 + bonus 生效证据

- `py_compile training/ppo.py training/run_training.py`：✅ 通过。
- 冒烟（开 bonus）：`--algo ppo --games 20 --mcts-sim 2000 --workers 4
  --upgrade-bonus 1.0 --relic-bonus 1.0` → 7s 跑通，无报错。✅
- 回归（关 bonus）：同参数 `--upgrade-bonus 0 --relic-bonus 0` → 7s 跑通；
  bonus=0 时公式退化为原 `floor + win_bonus*win`，G 数值与原行为完全一致。✅

**bonus 生效证据**（`_tmp_bonus_evidence.py`，5 局随机种子）：
```
 seed floor win  up rel  G(noB)  G(b1.0)    Δ
   11    16   0   0   4    16.0     20.0   4.0 OK
   22    16   0   2   1    16.0     19.0   3.0 OK
   33    23   0   2   2    23.0     27.0   4.0 OK
   44    30   0   6   7    30.0     43.0  13.0 OK
   55     6   0   0   2     6.0      8.0   2.0 OK
```
`G(b1.0) = floor + 50*win + 1*upgrade_count + 1*relic_count`，
`Δ = upgrade_count + relic_count` 逐局吻合 → bonus 确已注入回报。

- pytest：`249 passed, 2 skipped`。原 `test_ppo.py` 因 3 元组契约需同步更新
  为新 5 元组（并加 1 条奖励塑形测试）；属 API 变更的必要维护，非破坏。

## 4. 遗留风险

1. **诱导无脑升级 / 无脑打精英**：bonus>0 会让 agent 偏向篝火锻造、抢精英遗物，
   可能提高 Act1/2 死亡率（为长期升级/遗物承担战斗风险）。权重已按诊断差距保守取
   `1.0/1.0`（约 +2.4 / +1.2 分），远小于基线楼层(~45)/通关(50)，应温和。
2. **回报尺度变化影响 advantage 标准化**：G 上限抬升约 +3~+13 分，PPO 用
   `A = G - V(s)` 后 batch 内标准化，critic 会自适应；但初期 critic 偏差可能放大
   advantage，需关注训练前几百局的 entropy/win_rate 曲线。
3. **`upgrade_count` vs `upgraded`**：当前两字段等价，绑定层若改 `upgrade_count`
   语义（如允许 >1 次升级）计数口径会变，已用 `>0` 阈值规避。
4. **REINFORCE 不受影响**：本次仅 PPO；若日后要统一需单独改 `reinforce.py`。
