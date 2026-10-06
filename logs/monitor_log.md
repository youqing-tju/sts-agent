## 2026-08-31 19:35（基线）
- 游戏状态：运行中
- 通关：否
- 异常总数：42462（多为手动干预导致的 init_from_json null 字段错误）
- 全局决策分布：PLAY=11624, CHOOSE=15463, END=495, rule_fallback=411
- 最近 200 行：PLAY=38, rule_fallback=0, errors=18
- 备注：MCTS 战斗已接入（出牌有目标选择），RL 宏观决策正常，消融实验已停释放 CPU

## 2026-08-31 20:36
- 游戏状态：已停止（tasklist 无 java.exe）
- 通关：否
- 异常数：0（最近 200 行无 RuntimeError/Traceback/crash）
- 决策分布：PLAY=0, CHOOSE=2, END=0, rule_fallback=2
- 备注：日志尾部连续刷 screen=INVALID 警告（约每 125ms 一条），agent 仍在运行但读不到有效游戏画面，偶发 EVENT/MAP 画面均走 rule fallback，疑似游戏进程已退出、agent 空轮询。
- ALERT：游戏进程（java.exe）已停止，需人工确认是正常退出还是崩溃。
