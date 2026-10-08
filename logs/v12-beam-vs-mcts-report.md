# Beam Search vs MCTS 战斗求解器 —— 同训练量对照实验

> 目的：回答"**把战斗层从 MCTS 换成确定性 Beam Search，是否更好**"。
> 这是"复刻 CombatSolver"这条技术路线的第一次**受控对比**。
>
> 设计：两个模型**只差一个变量 —— 战斗求解器**；训练量、超参、种子、评估口径全部相同。

---

## 1. 实验设计

| 项 | Beam 组 | MCTS 组 |
|---|---|---|
| 训练入口 | `training/run_training.py --algo ppo` | 同 |
| 局数 | 500 | 500 |
| 训练用战斗求解器 | **`--combat-solver beam`** | **`--combat-solver mcts`** |
| Beam 参数 | `--beam-width 2 --beam-depth 6 --beam-max-nodes 300` | —（不适用） |
| 其它 | `--mcts-sim 8000 --workers 8 --batch-games 10 --eval-every 250 --save-every-n 250` | 同 |
| 权重 | `checkpoints/v12_beam_500.pt` | `checkpoints/v12_mcts_500.pt` |

**评估口径**（两组完全一致）：`evaluation/evaluate.py --seeds 100 --mcts-sim 20000`，
**greedy** 策略（argmax），同一套 100 固定种子（`make_eval_seeds(100)`）；
各自用**与自己训练时相同的战斗求解器**打分（即"训练怎么打，评估就怎么打"）。

---

## 2. 结果

| 模型 | 战斗求解器 | **平均楼层** | **通关率** | 楼层标准差 | 训练期 floor（采样式） |
|---|---|---|---|---|---|
| `v12_beam_500` | **Beam W=2** | 31.6 | **24.0%** | 15.5 | 18.1 |
| `v12_mcts_500` | MCTS | 32.4 | **8.0%** | 11.1 | 28.8 |

**通关率显著性检验**（两比例 z 检验，n=100 每组）：
- 合并比例 `p̄ = (24+8)/200 = 0.16`
- `SE = √(0.16 × 0.84 × (1/100 + 1/100)) = 0.0518`
- `z = (0.24 − 0.08) / 0.0518 ≈ **3.09**`，**p ≈ 0.002** → **显著**

---

## 3. 判读

### 3.1 核心结论：楼层相当，但**通关率差 3 倍**

**平均楼层** 31.6 vs 32.4（差 0.8，在噪声内）；**通关率** 24% vs 8%（**z≈3.09，p≈0.002，显著**）。

两者不是矛盾，而是**同一个现象的两面**：

| | Beam | MCTS |
|---|---|---|
| 楼层标准差 | **15.5（大）** | 11.1（小） |
| 通关率 | **24%（高）** | 8%（低） |
| 性格 | **高风险高回报**：要么走得远、要么死得早 | 稳定但**到不了顶** |

### 3.2 与既有假说一致

这支持了我们最初的判断（选择复刻 CombatSolver 的动机）：
**STS 战斗基本是确定性的，用随机 rollout 去逼近它是在浪费采样。**

- **Beam 能搜出"必胜路线"**：早先在真机战斗帧上实测过 —— 它给出 12 步完整路线并判定必胜
  （而 MCTS 在同一帧上，不同预算档位给出**自相矛盾**的动作：2000 档 `PLAY 4`、8000 档 `PLAY 1`）。
- **MCTS 的随机 rollout 在关键战斗（尤其 Boss）容易失手**：它的方差来自采样，
  而 Boss 战的容错窗口很窄 —— 一次 rollout 运气不好就可能选出致命动作。
- 代价：Beam **慢 100~200 倍**（训练 7.8s/局 vs MCTS 约 1s/局），且稳定性更差（部分局面
  W=2 的束太窄搜不透）。

### 3.3 一个反直觉的观察：训练期曲线会**严重低估** Beam 模型

| | Beam 组 | MCTS 组 |
|---|---|---|
| 训练末期 floor（**采样**式探索） | 18.1 | 28.8 |
| 评估 floor（**greedy**） | **31.6** | 32.4 |
| **差距** | **13.5 层** | 3.6 层 |

训练曲线里 Beam 组"看起来差 10 层"，但 greedy 评估下两组几乎持平 ——
**因为 Beam 组策略尚未收敛（熵 0.77），采样时噪声极大**。

> 📌 **可复用的教训：训练期 floor 是"采样式"的，会系统性低估尚未收敛的策略。
> 跨模型比较必须用同口径的 greedy 评估，不能看训练曲线。**

---

## 4. 局限（勿过度解读）

1. **只有 500 局训练量**（E1 基线是 7984 局）→ 反映的是**早期阶段**，**不代表最终水平**
2. Beam 用的是**最小预算 W=2/depth=6/max=300**（不是标准档 W=4）→ **未做参数调优**
3. 两组评估是**独立运行**（虽用同一套固定种子，但未做逐局严格配对比较）
4. 通关率 24% / 8% 的分母只有 100，**置信区间较宽**

---

## 5. 下一步（按价值排序）

| # | 事项 | 预期收益 |
|---|---|---|
| **1** | **Beam 性能优化**：热点已定位 —— `combat_beam.fingerprint()` 占单节点 **98%** 耗时（0.5ms vs 裸"拷贝+执行"0.01ms），瓶颈是大量 pybind 对象访问与字符串 `id.name` | 预计 **2~3× 提速** → 训练时间从 12-13h 降到 4-5h，**且对以后所有 Beam 训练都有效** |
| **2** | **Beam 训到 8000 局**再评估，与 E1（44.40 层 / 51.0%）对比 | 回答"Beam 的**最终**上限是否更高" |
| 3 | 试 `W=3~4`（更大束宽） | 可能改善 Beam 的方差，但更慢 |

---

## 附：复现命令

```bash
# 训练（--combat-solver 是唯一变量）
python -m training.run_training --algo ppo --games 500 --mcts-sim 8000 \
  --combat-solver beam --beam-width 2 --beam-depth 6 --beam-max-nodes 300 \
  --workers 8 --batch-games 10 --eval-every 250 --save-every-n 250 \
  --checkpoint checkpoints/v12_beam_500.pt --log-dir logs/v12-beam-500
#   （MCTS 组：把 --combat-solver 换成 mcts 并去掉 beam-* 参数）

# 评估（两组同口径）
python -m evaluation.evaluate --checkpoint checkpoints/v12_beam_500.pt \
  --seeds 100 --workers 8 --mcts-sim 20000 --combat-solver beam \
  --beam-width 2 --beam-depth 6 --beam-max-nodes 300
```

**相关产物**：`combat_beam.py`（求解器）、`env/lightspeed_env.py`（`combat_solver` 开关）、
`training/ppo.py` + `training/run_training.py`（训练侧接线）、`evaluation/evaluate.py`（评估侧接线）。
