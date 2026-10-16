# v15 entropy_coef 扫描报告 —— 负结果：探索不是瓶颈

> 2026-10-16｜动机见 `logs/v14-training-report.md` §6「下一步 P0」
> **一句话结论**：提高 `entropy_coef` **没有带来改进** —— 0.02 与 0.01 通关率**完全相同**、
> McNemar **33/33 完全平手**；0.05 反而**趋于更差**（−7.5 pp，不显著）。
> **「探索不足」不是当前平台期的瓶颈，`entropy_coef` 保持默认 0.01。**

---

## 1. 动机

`v14`（2000 局）训练曲线显示：**1000–1500 局后进入平台**，同时
`entropy` 从 0.89 单调降到 **0.50**。据此提出假设 —— **探索不足**是平台期原因，
于是扫 `--entropy-coef` ∈ {0.02, 0.05}（基线 0.01），各 2000 局。

## 2. 训练

| 变体 | entropy_coef | 训练耗时 | 0.5→ | 末 100 局采样楼层 |
|---|---|---|---|---|
| v14（基线） | 0.01 | 990 s | 0.5 s/局 | 35.2 |
| v15_ec002 | 0.02 | 932 s | 0.5 s/局 | 32.9 |
| v15_ec005 | 0.05 | 1323 s | **0.7 s/局** | 31.1 |

> 采样楼层随 entropy 单调下降 —— 但这是**采样**指标，探索越多动作越随机，
> **不能用来比较策略质量**，必须看 greedy 评估。

## 3. 评估（200 局 @ defensive，同 200 个固定留出种子，greedy）

| 模型 | avg_floor | **通关率** | 胜/局 | 通关局 deck | 通关局 relics | Act3 进入时 hp |
|---|---|---|---|---|---|---|
| **v14**（ec=0.01） | 36.84 | **53.5%** | 107/200 | 31.1 | 4.5 | 73.5 |
| v15_ec002（ec=0.02） | **38.92** | **53.5%** | 107/200 | **24.6** | 6.0 | 73.8 |
| v15_ec005（ec=0.05） | 34.01 | 46.0% | 92/200 | 29.7 | 7.3 | **55.0** |

### 配对检验（vs v14，同种子）

| 变体 | Δfloor | 配对 t | McNemar b/c |
|---|---|---|---|
| v15_ec002 | +2.09 | +1.62（不显著） | **33 / 33（完全平手）** |
| v15_ec005 | −2.83 | −1.65（不显著） | 39 / 54（更差） |

## 4. 判读

1. ❌ **假设被否定**：把探索提高一倍（0.01 → 0.02）**既没变好也没变坏** ——
   通关率一模一样（107/200），逐种子配对 **33 赢 / 33 输**，教科书式的"无差异"。
2. 🔴 **0.05 明确趋于更差**：通关率 −7.5 pp、Act3 进入时血量 73.5 → **55.0**、
   `gold` 304（vs 44）—— 血量和消费管理明显变糟。虽然未达显著（z≈1.5），
   但方向一致（楼层 −2.83、McNemar 39/54），**没有理由采用**。
3. 📌 **有趣的副产物**：`ec=0.02` 拿到了**更精简的通关牌组**（24.6 张 vs 31.1）与
   **更高的平均楼层**（38.92 vs 36.84），但**通关率完全相同**。
   → 「牌组更小」并不自动等价于「通关率更高」，两者不是同一个指标
   （与 `logs/v14-training-report.md` §3 的结论一起看：thin deck 是**好策略的特征**之一，
   但把它当唯一目标会走偏）。

## 5. 结论与下一步

### ✅ 结论

- **`--entropy-coef` 保持默认 0.01**；无需改动任何训练脚本。
- **平台期的原因不在探索** → 排除了一条假设，这也是有价值的进展。

### ⏭ 下一步（重新排序）

| 优先级 | 事项 | 理由 |
|---|---|---|
| **P0** | **handoff §3 核心实验：Beam 宽度 ↔ 抓牌质量** | 项目最初的方法论设想，**至今未验证**；现在条件成熟（`_tmp_macro_metrics.py` 宏观指标工具 + 稳定基准） |
| P1 | 训练量 4000 局 | 500→2000 提升巨大，但曲线显示已近平台；值得确认上限 |
| P1 | 表征 / 网络容量 / 学习率 | 若宽度实验无果，下一组超参杠杆 |
| P2 | 上云 | 训练已提速 60×（0.5 s/局 → 2000 局 17 分钟），**单机也够用**，优先级下调 |

## 6. 复现

```bash
cd "C:/Youqing-Vault/Embodied-AI/07-projects/sts-agent"
export KMP_DUPLICATE_LIB_OK=TRUE

# 训练（各 2000 局）
for EC in 0.02 0.05; do
  .venv/Scripts/python.exe -m training.run_training --algo ppo --games 2000 \
    --mcts-sim 8000 --combat-solver beam --beam-width 2 --beam-depth 6 --beam-max-nodes 300 \
    --workers 6 --batch-games 10 --eval-every 1000 --save-every-n 1000 \
    --entropy-coef $EC --beam-weights-preset defensive \
    --checkpoint checkpoints/v15_ec0${EC#0.}.pt --log-dir logs/v15-ec$EC
done

# 评估（含宏观指标）
for ck in v14_defbeam_2k v15_ec002 v15_ec005; do
  .venv/Scripts/python.exe scripts/probes/_tmp_macro_metrics.py \
    --checkpoint checkpoints/$ck.pt --seeds 200 --workers 6 \
    --beam-weights-preset defensive --out logs/macro-$ck.json
done
```
