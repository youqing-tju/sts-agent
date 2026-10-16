# v13（defensive 训练）vs v12（default 训练）对比报告

> 2026-10-16｜验证「用更强的战斗执行训练 RL，模型是否更好」
> **一句话结论**：**权重改进本身被独立复现**（两个模型换上 `defensive` 都 +9.5 pp）；
> 但**用 defensive 训练并没有让模型更好**（v13 ≈ v12，差异不显著）——
> 500 局训练量远未收敛，训练层面的差异被噪声掩盖。

---

## 1. 实验设计（2×2）

| 因子 | 水平 |
|---|---|
| **训练权重** | `default`（CombatSolver 原量级） vs `defensive`（本轮调出的最优） |
| **评估权重** | `default` vs `defensive` |

- 两个模型均为 **500 局 PPO**、`W=2 / depth=6 / nodes=300`、`mcts-sim 8000`
- 评估：**200 个固定留出种子、greedy、6 workers、`mcts-sim 20000`**
- ⚠️ 评估种子**是嵌套的**（`seeds(200)[:100] == seeds(100)` 已验证）→ 不同 n 之间也可配对

## 2. 结果

| 模型 | 评估 @ default | 评估 @ defensive | Δ（换评分） |
|---|---|---|---|
| **v12**（`v12_beam_500.pt`，500 局 @ default） | 29.41 层 / 21.5% | **32.0 层 / 31.0%** | +2.59 层 / **+9.5 pp** |
| **v13**（`v13_defbeam.pt`，500 局 @ defensive） | 29.6 层 / 17.5% | **33.0 层 / 27.0%** | +3.40 层 / **+9.5 pp** |

**换训练权重的差异（固定评估口径）**：

| 评估口径 | v12 | v13 | Δ |
|---|---|---|---|
| @ default | 29.41 层 / 21.5% | 29.6 层 / 17.5% | +0.19 层 / **−4.0 pp** |
| @ defensive | 32.0 层 / 31.0% | 33.0 层 / 27.0% | +1.00 层 / **−4.0 pp** |

## 3. 判读

### 3.1 ✅ 权重改进被**独立复现**（本报告最有价值的结论）

两个**不同模型**换上 `defensive` 后，通关率**都提高 9.5 pp**（21.5→31.0、17.5→27.0），
方向与幅度高度一致 —— 这比单一模型的 A/B 更强地支持了
`logs/weight-tuning-report.md` 的结论：**组合权重改进是真实的、可迁移的**。

### 3.2 ❌ 训练层面的差异**不显著**（v13 ≈ v12）

- 楼层：v13 略高（+0.19 / +1.00 层）
- 通关率：v13 略**低**（−4.0 pp，两处一致）

200 局下通关率的标准误约 **3.2 pp**，4.0 pp 的差异 ≈ **1.25 SE** → **不显著**。
（楼层 +1 层在 std≈15 的分布下更不显著。）

**"楼层略升、通关率略降"** 的组合值得注意但不是结论：可能意味着 v13 更能推进到后期、
但 Boss 战力不足；也可能纯粹是噪声。

### 3.3 为什么看不出来 —— 训练远未收敛

对比两者的训练曲线（末 10 条记录）：

| | v12 | v13 |
|---|---|---|
| 训练期 `mean_floor` | 15.9 ~ 18.1 | **22.4 ~ 36.9** |
| `win_rate` | 0.00 ~ 0.02 | 0.10 ~ 0.40 |
| `value_loss` | ≈ 121（末条） | **1000 ~ 1700**（高波动） |

- ⭐ **v13 训练期楼层显著高于 v12** → **defensive 权重确实透传到了 worker 子进程并生效** ✅
  （这是对透传链路的有效端到端验证）
- ⚠️ 但 v13 的 `value_loss` 仍在 1000+ 剧烈波动、`win_rate` 在 0.1~0.4 之间跳动
  → **500 局远未收敛**（参考 E1 基线用了 **7984 局**）

**结论**：训练层面的差异需要**更大的训练量**才能测出来，当前实验**不足以证伪**。

## 4. 局限

1. **单次训练**：v12 / v13 未用同一随机种子 → 训练随机性与权重效应混杂。
   要严格控制需「同 seed × 多组」，成本高。
2. **训练量不足**：500 局（E1 用 7984 局）。
3. 评估的 200 局对通关率的分辨力约 ±3 pp（95% CI ≈ ±6 pp）。

## 5. 结论与下一步

### ✅ 可以直接采用

**推理与部署时使用 `--beam-weights-preset defensive`** —— 收益已被两个独立模型复现（各 +9.5 pp），
**不需要重新训练**即可拿到。

### ⏭ 下一步

| 优先级 | 事项 | 理由 |
|---|---|---|
| **P0** | **扩大训练量**（2000 → 8000 局） | 500 局明显未收敛；训练层面的一切结论都需先解决这一点 |
| P1 | handoff §3 的核心实验：**Beam 宽度 ↔ 抓牌质量** | 需要先有收敛的训练量；"变弱"的杠杆在宽度（W=1 尚未测） |
| P2 | 同 seed 对照训练 | 若要严格控制训练随机性 |

## 6. 复现

```bash
cd "C:/Youqing-Vault/Embodied-AI/07-projects/sts-agent"
export KMP_DUPLICATE_LIB_OK=TRUE

# 训练（500 局 @ defensive，实测 300 s / 4 workers）
.venv/Scripts/python.exe -m training.run_training --algo ppo --games 500 \
  --mcts-sim 8000 --combat-solver beam --beam-width 2 --beam-depth 6 --beam-max-nodes 300 \
  --workers 4 --batch-games 10 --eval-every 250 --save-every-n 250 \
  --beam-weights-preset defensive \
  --checkpoint checkpoints/v13_defbeam.pt --log-dir logs/v13-defbeam

# 2×2 评估
C="--seeds 200 --workers 6 --mcts-sim 20000 --combat-solver beam --beam-width 2 --beam-depth 6 --beam-max-nodes 300"
.venv/Scripts/python.exe -m evaluation.evaluate --checkpoint checkpoints/v12_beam_500.pt $C --beam-weights-preset defensive
.venv/Scripts/python.exe -m evaluation.evaluate --checkpoint checkpoints/v13_defbeam.pt  $C
.venv/Scripts/python.exe -m evaluation.evaluate --checkpoint checkpoints/v13_defbeam.pt  $C --beam-weights-preset defensive
```

> ⭐ **附带成果**：训练速度从优化前的约 **31 s/局** 降到 **0.6 s/局**（500 局 300 秒）——
> 这是 P0-① 指纹优化的直接收益，也让「8000 局」这样的大训练量第一次变得可行（约 80 分钟）。
