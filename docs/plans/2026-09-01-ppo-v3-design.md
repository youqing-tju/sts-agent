# V3：PPO + 网络扩容 设计文档

> 2026-09-01 · 状态：已批准（用户选 A：先隔离变量，候选编码不动）

## 1. 动机

消融实验结论（win_bonus 50 vs 200 无差异，续训原地踏步）排除了奖励权重假说。
剩余嫌疑按优先级：REINFORCE 方差过高 > 网络容量不足（116k 参数）。
V3 一次换两个变量，但都属于"学习算法层"，与表征层（368 维编码）正交，归因仍然清晰。

## 2. PPO 适配我们的场景

我们的 MDP 特点：候选动作集逐点变化（variable action set）、奖励纯终局（G = final_floor + win_bonus×win，中间奖励全 0）。

设计取舍：
- **不要 GAE**：中间奖励全 0 且 gamma=1 时，每个决策点的优势就是 A_t = G − V(s_t)，GAE 退化成同一式，没必要引入
- **Critic 替代滑动平均 baseline**：V(s) 是独立 MLP（obs 412 → 256 → 1），比 REINFORCE 的移动平均更好地拟合"当前局势的期望回报"
- **候选打分不变**：Scorer 结构不动（obs ⊕ cand → logit → softmax 采样），只是 hidden 从 (128,128) 扩到 (512,256)
- **PPO clip**：ratio = exp(logp_new − logp_old)，clip ε=0.2，K=4 epochs，batch = 16 局一更新

超参：
- lr = 3e-4（PPO 比 REINFORCE 敏感，比 REINFORCE 的 1e-3 低）
- entropy_coef = 0.01（沿用）
- win_bonus = 50（沿用，消融已证明权重不敏感）
- mcts_sim 训练 2000 / 评估 5000（与 baseline 可比）

## 3. 网络扩容

| 组件 | v2 | v3 |
|---|---|---|
| Scorer hidden | (128, 128) | (512, 256) |
| Scorer 参数量 | 116k | ~530k |
| Value net | 无（滑动平均） | 412 → 256 → 1（~106k） |

注意：hidden 变了 → **无法从 long.pt 热启动**（形状不兼容），v3 从零训练。这是诚实对照的代价。

## 4. 实现任务拆分

- **V3-1（子代理）**：`models/scorer.py` 加 ValueNet；`training/ppo.py` 新增 PPOTrainer（collect_batch → 存 (obs_vec, cand_vecs, chosen_idx, logp_old) → update 里重算 logp/V、clip、entropy）+ 测试
- **V3-2（我自己）**：`training/run_training.py` 加 `--algo {reinforce,ppo}` 参数；从零启动 5000 局训练
- **V3-3（我自己）**：100 留出种子评估，与 REINFORCE（31.6/3.0%）同条件对比，写进 M4 报告

## 5. 验收标准

- tests/ 全绿（含 PPO 的 clip/value/reproducibility 用例）
- v3 训练 5000 局后 100 种子评估：avg_floor > 31.6 或 win_rate > 3.0% 视为有效突破；无突破也是有效结论（说明瓶颈在表征层，转向编码增强）
