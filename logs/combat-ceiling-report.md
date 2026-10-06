# 战斗层天花板探针报告（v10_seg3.pt · MCTS 20000 · greedy）

## 1. 方法与口径

- **模型/环境**：`checkpoints/v10_seg3.pt`（Scorer 477/369, 512/256, `weights_only=True`）；`LightspeedEnv(mcts_simulations=20000)` + greedy argmax；种子 `make_eval_seeds`。
- **界定一场战斗**：`env.step()` 内 `agent.playout(gc)` 一次打完整场战斗（实测每战=1 次 playout）。覆写 `_advance`：playout 前若 `screen==BATTLE` 记一场战，`hp_before`=进战前 `cur_hp`、playout 后 `hp_after`=出战 `cur_hp`；`lethal`=`outcome==PLAYER_LOSS`；遭遇名取 `gc.encounter.name`；Boss/Elite 用 `gc.cur_room.name` 区分。
- **样本**：50 局固定种子，共 1118 场战斗。

**3 局试探原始数据（口径确认：遭遇/floor/act/room/hp_before→after/loss）**：

- 第1局 seed=856258841 通关=True floor=51 hp=36（27 战，示例前6）：
    SMALL_SLIMES         f1 A1 MONSTER 80->80 loss=0
    CULTIST              f2 A1 MONSTER 80->80 loss=0
    TWO_LOUSE            f4 A1 MONSTER 80->73 loss=7
    GREMLIN_GANG         f5 A1 MONSTER 73->70 loss=3
    BLUE_SLAVER          f6 A1 MONSTER 70->63 loss=7
    LOTS_OF_SLIMES       f7 A1 MONSTER 63->63 loss=0
- 第2局 seed=809721117 通关=True floor=51 hp=23（27 战，示例前6）：
    TWO_LOUSE            f1 A1 MONSTER 80->74 loss=6
    SMALL_SLIMES         f4 A1 MONSTER 74->74 loss=0
    JAW_WORM             f5 A1 MONSTER 74->74 loss=0
    TWO_FUNGI_BEASTS     f7 A1 MONSTER 80->80 loss=0
    GREMLIN_NOB          f8 A1 ELITE   80->79 loss=1
    LARGE_SLIME          f10 A1 MONSTER 79->79 loss=0
- 第3局 seed=1270809863 通关=True floor=51 hp=3（21 战，示例前6）：
    TWO_LOUSE            f1 A1 MONSTER 80->78 loss=2
    CULTIST              f4 A1 MONSTER 80->80 loss=0
    JAW_WORM             f5 A1 MONSTER 80->79 loss=1
    TWO_FUNGI_BEASTS     f7 A1 MONSTER 80->79 loss=1
    BLUE_SLAVER          f8 A1 MONSTER 79->79 loss=0
    LOOTER               f10 A1 MONSTER 79->69 loss=10

**口径校验**：标准前 30 种子复现 **通关 21/30=70%**、Act1 死 0（=已知基线 70% / Act1 零死亡）→ 战斗切分正确。全 50 局含更难的后 20 种子（仅 25% 通关）故整体略低。

## 2. 四张统计表

**表2-1 死亡遭遇 TOP10**（致死次数 / 典型 floor·act / 平均失血）：
遭遇                       致死  floor act avg_loss
DONU_AND_DECA             4     50   3     82.5
AWAKENED_ONE              4     50   3     32.5
CHAMP                     2     33   2     64.5
AUTOMATON                 2     33   2     53.0
THREE_CULTIST             2     25   2     16.0
SPIRE_GROWTH              2     45   3     10.0
HEXAGHOST                 1     16   1     80.0
TIME_EATER                1     50   3     70.0
GIANT_HEAD                1     45   3     58.0
SLAVERS                   1     25   2     40.0

**表2-2 平均 HP 损失 TOP10（含未致死·隐性吃血点；n=出现次数）**：
遭遇                     avg_loss max_loss   n   致死
TIME_EATER                 47.6       70   7    Y
AUTOMATON                  46.2       95  14    Y
DONU_AND_DECA              41.8      107  16    Y
LAGAVULIN_EVENT            39.0       52   2     
HEXAGHOST                  37.4       80  16    Y
CHAMP                      34.5       80  10    Y
COLLECTOR                  29.7       68  19     
GIANT_HEAD                 29.2       58   5    Y
AWAKENED_ONE               28.7       57  12    Y
THE_GUARDIAN               25.4       78  16     

**表2-3 Boss 战统计**（胜率=到达且未致死；act=出现幕）：
act遭遇                       次数     胜率 avg_loss
  1HEXAGHOST                16   94%     37.4
  1THE_GUARDIAN             16  100%     25.4
  1SLIME_BOSS               16  100%     15.2
  2COLLECTOR                19  100%     29.7
  2AUTOMATON                14   86%     46.2
  2CHAMP                    10   80%     34.5
  3DONU_AND_DECA            16   75%     41.8
  3AWAKENED_ONE             12   67%     28.7
  3TIME_EATER                7   86%     47.6
  4THE_HEART                 1  100%     22.0

**表2-4 Act 死亡率复核**（全50局 vs 已知 0%/20%/23%/57%）：
- Act1 死 2/50 ｜ Act2 死 8/50 ｜ Act3 死 13/50 ｜ 通关 26/50 = 52% ｜ Boss 总胜率 89%（113/127）

## 3. 失败 Boss 战逐回合细节（复现=MCTS 20000 隔离重演该 Boss 战）

- **DONU_AND_DECA**（f50 A3，实败 hp 107→0）：
  - 我方HP关键帧：107@0 → 90@5 → 84@10 → 56@14 → 32@23 → 8@28 → 0@33（末 0/107）
  - 主敌HP关键帧：250@0 → 226@2 → 216@4 → 188@8 → 183@9 → 171@18 → 166@20 → 128@21 → 112@22 → 86@24 → 70@27 → 56@30 → 42@31（末 42）
  - 单回合最大掉血 28；34 回合 → 单回合爆发崩盘（复现阵亡）
- **AWAKENED_ONE**（f45 A3，实败 hp 40→0）：
  - 我方HP关键帧：40@0 → 30@5 → 40@9 → 22@13 → 21@20 → 20@26 → 18@33 → 17@39 → 5@45 → 0@53（末 0/40）
  - 主敌HP关键帧：49@0 → 44@1 → 38@2 → 31@9 → 25@12 → 5@15 → 0@16（末 0）
  - 单回合最大掉血 18；54 回合 → 逐步磨死（复现获胜）
- **CHAMP**（f33 A2，实败 hp 80→0）：
  - 我方HP关键帧：80@0 → 78@7 → 74@13 → 62@21 → 68@25 → 54@26 → 44@40 → 43@44 → 38@54 → 29@69 → 21@79（末 21/87）
  - 主敌HP关键帧：420@0 → 415@1 → 408@4 → 378@5 → 357@6 → 353@15 → 343@16 → 339@19 → 333@20 → 327@23 → 319@24 → 313@25 → 304@28 → 296@31 → 283@32 → 277@35 → 267@38 → 263@42 → 249@43 → 247@47 → 244@51 → 238@52 → 235@55 → 230@56 → 224@57 → 204@58 → 189@59 → 181@63 → 177@67 → 172@68 → 167@72 → 149@73 → 135@74 → 131@78 → 128@83 → 101@85 → 97@87 → 91@88 → 86@89 → 68@90 → 58@93 → 40@94 → 34@95 → 9@98 → 0@99（末 0）
  - 单回合最大掉血 14；100 回合 → 逐步磨死（复现获胜）

## 4. 结论

- **口径可信**：标准 30 种子复现 70% 通关 + Act1 零死亡，与已知基线一致；战斗切分无 bug。
- **瓶颈在 Act3 与 Boss 战**：Act3 死亡 13/50 最高；Act3 Boss 胜率最低（DONU_AND_DECA 75%、AWAKENED_ONE 67%），是全剧最致命的两场。
- **最致命遭遇 TOP3**（按致死次数）：`DONU_AND_DECA(4), AWAKENED_ONE(4), CHAMP(2)`；次为 AUTOMATON(2, Act2)；按平均失血则是 TIME_EATER(47.6)/AUTOMATON(46.2)/DONU_AND_DECA(41.8)。
- **失败形态**：以逐步磨死为主，但 DONU_AND_DECA 偶发单回合爆发崩盘（实测单回合 −28），是典型高地压局面的崩点。
- **可改进空间（明显）**：失败集中在 Boss 战，且复现显示 MCTS 在边缘局常能磨赢（AWAKENED_ONE/CHAMP 复现均胜），说明提升 MCTS 评估函数 / Boss 战预算 / 确定性即可把部分败局转为胜局。宏观决策已饱和（Act1 零死亡），应继续攻**战斗层**而非抠宏观。