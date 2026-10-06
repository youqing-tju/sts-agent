# 战斗终局评估函数改进报告（combat-eval）

## 1. 当前 evaluateEndState 完整公式（BattleScumSearcher2.cpp:407-451）

```
potionScore = potionCount * 4
if outcome == PLAYER_VICTORY:            # :410-414
    return 100 * (35 + curHp + potionScore - turn*0.01) - 0.05*energyWasted
else:  # PLAYER_DEFEAT（isTerminalState 只放行 VICTORY/DEFEAT） :416-450
    bossDamageProgress = (1 - getNonMinionMonsterCurHpRatio(bc)) * 300
    alivePenalty       = monstersAlive * -20
    killProgress       = (monsterCount - monstersAlive) * 150
    energyPenalty = energyWasted * -0.2 * (couldHaveSpikers ? 0 : 1)
    return -100 + bossDamageProgress + alivePenalty + killProgress
           + energyPenalty + cardsDrawn*0.03 + potionScore/2 + turn*0.2
```

**失败（玩家死亡）分支 · 旧公式**（改前，:416-425）：
`(1 - getNonMinionMonsterCurHpRatio(bc))*10 + (monstersAlive*-1) + energyPenalty + cardsDrawn*0.03 + potionScore/2 + turn*0.2`
→ 返回**很小的正值（约 0~15）**，且 boss 损血项仅 *10、存活怪项仅 -1，梯度≈无。

## 2. 缺陷诊断 + 候选方案

**缺陷**
- 失败分支是小正值且梯度近乎为零：Boss 战（DONU_AND_DECA / AWAKENED_ONE / TIME_EATER）随机 playout 多死于终局，旧评分无法区分"差一点赢"与"崩盘"，MCTS 在难局里没有指向击杀的梯度 → 满血进 Boss 仍阵亡（5/9 局，纯战术问题）。
- 胜利分支 HP 权重 100/HP，回合惩罚仅 turn*0.01（*100 后=1/回合）形同虚设：鼓励保守留血、拖回合，正合 snowball Boss 下风。

**候选**
1. 【推荐】失败分支加"指向击杀的稠密进度信号"（boss 已损血 + 已杀怪 + 存活怪惩罚），负基线保证仍远低于胜利。低风险、直接针对 Boss 打法瓶颈。
2. 提高胜利分支回合惩罚（0.01→0.05）鼓励速杀。风险：HP 权重 100 倍主导，系数需 ~0.5-5 才有感，过大会冒进送死。
3. 压缩胜利分支 HP 权重。风险：普通战斗变冒进、死亡率上升。

**推荐**：方案 1（保守、只动失败分支线性项，不动搜索算法/胜利分支）。

## 3. 实际改动（diff 摘要）
- 失败分支：负基线 `-100`；新增 `bossDamageProgress*300`、`alivePenalty -20/怪`、`killProgress +150/杀怪`；保留 energyPenalty/cardsDrawn/potionScore/turn 项。最大失败分仍 << 最小胜利分（~3600），搜索永远优先活。
- 胜利分支：**未改动**（HP 主导使回合系数无效，强行调大风险高，故不动）。

## 4. 编译结果
- 成功。产物 `third_party/sts_lightspeed/build/slaythespire.cp313-win_amd64.pyd`（1,956,352 B，Sep 14 20:38）。
- 已备份旧 pyd：`slaythespire.cp313-win_amd64.pyd.bak_0914`；已拷到项目根（拷贝前确认无 python 占用）。
- 关键坑：`cl.exe` 读 INCLUDE/LIB 需原生 `C:/...` 路径（不是 `/c/...`），`MSYS_NO_PATHCONV=1` 下必须用 Windows 盘符形式，否则 `cstdint` 找不到。

## 5. 验证
**单帧**（logs/realtime/0911_113720_lastboss.json，新 pyd）：Boss 帧（玩家 71 血）搜索 8000 次跑完，best_action=Whirlwind+（1220 次访问，avg_eval 226），序列 28 步合理；root 统计分散合理。✅

**10 局 A/B**（v10_seg3.pt + LightspeedEnv(mcts=8000) + greedy，种子 = make_eval_seeds(200)[:10]）：

| seed | OLD | NEW | | seed | OLD | NEW |
|---|---|---|---|---|---|---|
| 856258841 | 51 | 51 | | 1894911376 | 50 | 50 |
| 809721117 | 33 | **51** | | 1751539841 | 51 | 51 |
| 1270809863 | 33 | **50** | | 365587230 | 51 | 51 |
| 1953131477 | **45** | 33 | | 1576371696 | 51 | 51 |
| 335776950 | 51 | 51 | | 1362355165 | 51 | 51 |

- OLD: avg_floor 46.7，胜率 60%（6/10）
- NEW: avg_floor 49.0，胜率 70%（7/10）
- NEW 在 809721117/1270809863 由 33→51（活到 Boss），1953131477 由 45→33（1 处回退），其余持平。样本小（±10 层噪声），仅说明"无明显崩坏，略好"。

## 6. 遗留风险
- 失败分支梯度变强后，必死局里 MCTS 更偏好"差一点赢的死法"——方向正确；系数（300/150）若过强可能让某些局多拖回合、浪费药水，已用负基线+远低于胜利封顶，影响有限。
- 胜利分支未动：满血 Boss 死若源于"胜利分支过度奖励留血导致慢速送 snowball"，本改只缓解（靠失败分支引导），未根治；后续可评估方案 2/3。
- 仅 10 局 A/B，结论不显著；需 50+ 局确认。
