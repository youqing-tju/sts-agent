"""事件选项知识库（EVENT_OPTION 候选的语义特征来源）。

背景与「成败前提」（见 sts-agent 任务书）
=======================================
杀戮尖塔 RL 宏观决策里，事件候选（kind=EVENT_OPTION）在 C++ 模拟器
（slaythespire 绑定）里的真实形态是：

- `GameAction.describe(gc)` 对 EVENT 类动作**恒返回空串**，因此
  `candidate.label` 必然回退成 ``action_0x<bits>``（见 env/lightspeed_env.py）。
- 但自 commit f332356 起，绑定**已暴露** `gc.get_event_info()` →
  ``{'event_id':..., 'event_name':..., 'option_indices':[...]}``，
  `event_name` 即为当前事件名（**显示名风格**，如 'The Cleric'、
  'The Ssssserpent'、'Living Wall'，可与本 KB 一级 key 直接对齐，
  无需枚举名→显示名别名映射）。
- 一个事件内多个选项的 `bits` 里 **idx1 = 选项序号**，且与
  `get_event_info().option_indices` 完全对应（已用 30 种子实测确认）。

结论（诚实优先，实测更新 2026-09）：
本 KB 以「事件规范名（显示名）」为一级 key、「选项序号」为二级 key，
数据源自调研报告第七章，正确且完整。新绑定已能传入 `event_key`
（= `event_name`）与 `event_option_idx`（= `bits.idx1`），因此
`get_event_option_features` 在事件名被本 KB 收录时**会真正非零**。

覆盖率现状（30 种子实测 15 个相异事件 + 后续补录）：
- 已收录：The Cleric / Living Wall / Wing Statue / The Ssssserpent /
  Shining Light / Golden Shrine / Scrap Ooze / Face Trader / The Divine Fountain /
  We Meet Again! / Wheel of Change（原 11 个），另补录 NEOW、Big Fish、
  Golden Idol、Note For Yourself、The Library（5 个，官方 Wiki 核验）。
- 已知仍未覆盖：Cursed Tome 等较冷门事件——若运行时遇到会返回 20 维
  全零（合法降级，非命名错配）。
- 2026-09 补录批次（官方 Wiki 逐条核验）：补齐 Act1 World of Goop；
  Act2 The Mausoleum / The Joust / Ancient Writing / Council of Ghosts /
  Vampires / Forgotten Altar / Pleading Vagrant；Act3 Falling / Sensory Stone /
  Winding Halls / Mind Bloom / Tomb of Lord Red Mask / Secret Portal。
  注：「Fountain of Cleansing（净化喷泉）」与已收录的 The Divine Fountain
  为同一事件（仅翻译/叫法不同：喝下移除全部诅咒卡、仅在持有诅咒时出现），
  故**不重复建条**，仅在 The Divine Fountain 处注明别名。
- Ancient Writing 任务清单曾误写为「The Ancient Writing / 获得 N'loth's Gift
  遗物」；N'loth's Gift 属于独立事件 N'loth，本事件实为 Back to Basics
  （移除一张卡 / 升级全部 Strike 与 Defend），已按 Wiki 实证编码。
- NEOW 选项内容动态（祝福池随机 + 选项数 2/4 随上一局是否打到 Boss 变化），
  仅 Fourth Blessing（替换初始遗物为 Boss 遗物）是确定性 RELIC，其余标 OTHER。

20 维布局（总数严格 20）
=======================
A. 选项类别 one-hot（12 维，恰好一个 1）
  [0]  删卡   REMOVE
  [1]  升级   UPGRADE
  [2]  金币   GOLD
  [3]  战斗   COMBAT
  [4]  诅咒   CURSE      （选项主题以诅咒为核心时置位）
  [5]  治疗   HEAL
  [6]  遗物   RELIC
  [7]  药水   POTION
  [8]  复制   DUPLICATE
  [9]  变化   TRANSMOG   （把卡随机变成其他卡）
  [10] 赌博   GAMBLE     （结果不确定/随机）
  [11] 其他   OTHER

B. 代价与收益（8 维，归一化，带符号用有符号归一化）
  [12] hp_cost     当前生命损失（仅损失，max(0,Δhp)/30，截断 [0,1]）
  [13] max_hp_cost 最大生命损失（max(0,Δmaxhp)/30，截断 [0,1]）
  [14] gold_gain   净金币变化（有符号：gain 正 / 花费负），/300，截断 [-1,1]
  [15] curse_risk  获得诅咒风险 0/1
  [16] card_removed 删卡数量 /3，截断 [0,1]
  [17] card_upgraded 升级数量 /3，截断 [0,1]
  [18] relic_gain  是否得遗物 0/1
  [19] potion_gain 是否得药水 0/1

注意：⭐ 推荐度**不进向量**（那是标签不是特征），单独由
`get_event_option_rating()` 提供，可作启发式对照或辅助监督信号。

本模块纯函数、不 import slaythespire，便于无 pyd 单测。
"""
import re

EVENT_KB_DIM = 20

# ---- A 段：类别 one-hot 顺序（固定）----
CATEGORY_LIST = [
    "REMOVE",     # 0 删卡
    "UPGRADE",    # 1 升级
    "GOLD",       # 2 金币
    "COMBAT",     # 3 战斗
    "CURSE",      # 4 诅咒
    "HEAL",       # 5 治疗
    "RELIC",      # 6 遗物
    "POTION",     # 7 药水
    "DUPLICATE",  # 8 复制
    "TRANSMOG",   # 9 变化
    "GAMBLE",     # 10 赌博
    "OTHER",      # 11 其他
]
_NCAT = len(CATEGORY_LIST)            # 12
_OFF_CAT = 0
_OFF_COST = _NCAT                     # 12
_COST_DIM = 8                         # 8 → 末尾 20

# 归一化分母
_HP_NORM = 30.0
_GOLD_NORM = 300.0
_CARD_NORM = 3.0

# ---- 数值归一化辅助 ----
def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _norm_hp(loss: float) -> float:
    return _clip(float(loss) / _HP_NORM, 0.0, 1.0) if loss and loss > 0 else 0.0


def _norm_gold(delta: float) -> float:
    return _clip(float(delta) / _GOLD_NORM, -1.0, 1.0)


def _norm_count(n: float) -> float:
    return _clip(float(n) / _CARD_NORM, 0.0, 1.0)


def _opt(cat, *, hp_cost=0.0, max_hp_cost=0.0, gold=0.0, curse=0,
         remove=0, upgrade=0, relic=0, potion=0, rating=0):
    """构造单个选项记录。cat: CATEGORY_LIST 中的类别名；rating: ⭐ 数(0-5)。"""
    return {
        "cat": cat,
        "hp_cost": float(hp_cost),
        "max_hp_cost": float(max_hp_cost),
        "gold": float(gold),
        "curse": int(bool(curse)),
        "remove": int(remove),
        "upgrade": int(upgrade),
        "relic": int(bool(relic)),
        "potion": int(bool(potion)),
        "rating": int(rating),
    }


# ---- EVENT_KB：一级 key = 事件规范名（Slay the Spire 事件 id），
#             value["options"] = 按选项序号(0-based)排列的选项记录列表 ----
EVENT_KB = {
    # ===================== 开局事件（每局必现）=====================
    "NEOW": {
        # 注：NEOW 祝福选项内容动态（从祝福池随机），且选项数随上一局
        # 是否打到 Boss 在 2/4 之间变化。此处按最常见的 4 选项模式编码：
        # 前三个祝福内容随机故标 OTHER，第四个（替换初始遗物为随机
        # Boss 遗物）是唯一确定性选项。
        "options": [
            _opt("OTHER", rating=3),   # First Blessing：牌库祝福（删卡/变化/升级/选卡/无色/稀有，随机）
            _opt("OTHER", rating=3),   # Second Blessing：奖励（+8最大生命/Neow's Lament/普通遗物/100金/3药水，随机）
            _opt("OTHER", rating=2),   # Third Blessing：劣势+更大奖励（有随机代价）
            _opt("RELIC", relic=1, rating=4),  # Fourth Blessing：替换初始遗物为随机 Boss 遗物（确定）
        ]
    },
    # ===================== Act 1 =====================
    "Big Fish": {
        "options": [
            _opt("HEAL", rating=3),            # [Banana] 治疗 1/3 最大生命
            _opt("HEAL", rating=4),            # [Donut] 最大生命 +5
            _opt("RELIC", relic=1, curse=1, rating=2),  # [Box] 随机遗物 + 诅咒「悔恨」
        ]
    },
    "Golden Idol": {
        "options": [
            _opt("RELIC", relic=1, rating=3),  # [Grab] 获得遗物「金神像」（后续选代价）
            _opt("OTHER", rating=3),           # [Leave] 离开
        ]
    },
    "The Cleric": {
        "options": [
            _opt("HEAL", gold=-75, rating=3),          # 治疗 35HP，花费 75 金币
            _opt("REMOVE", gold=-50, remove=1, rating=5),  # 删除一张卡，花费 50 金币
            _opt("OTHER", rating=1),                   # 离开
        ]
    },
    "Living Wall": {
        "options": [
            _opt("REMOVE", remove=1, rating=5),        # 删除一张卡
            _opt("UPGRADE", upgrade=2, rating=4),      # 升级两张卡
            _opt("TRANSMOG", rating=3),               # 变化一张卡
        ]
    },
    "Wing Statue": {
        "options": [
            _opt("REMOVE", hp_cost=7, remove=1, rating=5),  # 祈祷：删卡，失去 7 HP
            _opt("GOLD", gold=50, rating=2),           # 摧毁：获得 50 金币
            _opt("OTHER", rating=2),                   # 离开
        ]
    },
    "The Ssssserpent": {
        "options": [
            _opt("GOLD", gold=150, curse=1, rating=1),  # 同意：150 金币 + 诅咒「负罪感」
            _opt("OTHER", rating=5),                   # 离开
        ]
    },
    "Shining Light": {
        "options": [
            _opt("OTHER", rating=5),                   # 离开
            _opt("UPGRADE", hp_cost=15, upgrade=2, rating=2),  # 走进：失去 15 HP，升级两张卡
        ]
    },
    "Golden Shrine": {
        "options": [
            _opt("GOLD", gold=100, rating=4),          # 祈祷：获得 100 金币
            _opt("GOLD", gold=275, curse=1, rating=3),  # 亵渎：275 金币 + 诅咒「悔恨」
            _opt("OTHER", rating=4),                   # 离开
        ]
    },
    "Hypnotizing Colored Mushrooms": {
        "options": [
            _opt("HEAL", curse=1, rating=2),           # 吃：恢复 7 HP + 诅咒「寄生」
            _opt("COMBAT", relic=1, rating=4),         # 踩扁：战斗得遗物「奇怪蘑菇」
        ]
    },
    "Scrap Ooze": {
        "options": [
            _opt("RELIC", relic=1, rating=3),          # 伸手：50% 得遗物（失败损失 HP）
            _opt("OTHER", rating=4),                   # 离开
        ]
    },
    "Dead Adventurer": {
        "options": [
            _opt("GAMBLE", rating=3),                  # 搜寻：可能金币/遗物，怪物可能回归
            _opt("OTHER", rating=2),                   # 离开（报告未列，按常规事件补）
        ]
    },
    "World of Goop": {
        "options": [
            _opt("GOLD", gold=75, hp_cost=11, rating=4),   # [Gather Gold] +75 金币，失去 11 HP
            _opt("OTHER", rating=2),                        # [Leave It] 离开（Wiki：另损失 20-50 随机金币，此处不另编码）
        ]
    },
    # ===================== Act 2 =====================
    "The Library": {
        "options": [
            _opt("OTHER", rating=4),           # [Read] 从 20 张卡选 1 张加入牌组
            _opt("HEAL", rating=3),            # [Sleep] 治疗 33% 最大生命（A15+ 20%）
        ]
    },
    "Face Trader": {
        "options": [
            _opt("RELIC", relic=1, rating=4),          # 交易：随机 5 张脸遗物之一
            _opt("GOLD", gold=50, hp_cost=3, rating=2),  # 触碰：50 金币，失去 3 HP
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "The Nest": {
        "options": [
            _opt("GOLD", gold=99, rating=5),           # 抢了钱就跑：99 金币
            _opt("RELIC", relic=1, hp_cost=6, rating=3),  # 留在队伍中：遗物「仪式匕首」-6 HP
        ]
    },
    "Colosseum": {
        "options": [
            _opt("COMBAT", gold=100, relic=1, rating=5),  # 继续：战斗胜利得金币+遗物
            _opt("OTHER", rating=3),                   # 怂了：逃跑
        ]
    },
    "Ominous Forge": {
        "options": [
            _opt("UPGRADE", upgrade=1, rating=4),      # 锻造：升级一张卡
            _opt("RELIC", relic=1, curse=1, rating=2),  # 翻查：遗物「弯曲钳子」+ 诅咒「疼痛」
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "The Mausoleum": {
        "options": [
            _opt("RELIC", relic=1, curse=1, rating=3),  # [Open Coffin] 得遗物；50%(A15后100%)获得诅咒「Writhe」
            _opt("OTHER", rating=3),                    # [Leave] 离开
        ]
    },
    "The Joust": {
        "options": [
            _opt("GAMBLE", gold=-50, rating=3),         # [Murderer] 押 50 金，70% 赢 100 金（无离开选项）
            _opt("GAMBLE", gold=-50, rating=3),         # [Owner] 押 50 金，30% 赢 250 金
        ]
    },
    "Ancient Writing": {
        "options": [
            _opt("REMOVE", remove=1, rating=4),         # [Elegance] 从牌组移除一张卡
            _opt("UPGRADE", upgrade=3, rating=4),       # [Simplicity] 升级全部 Strike 与 Defend（数量截断至 1.0）
        ]
    },
    "Council of Ghosts": {
        "options": [
            _opt("OTHER", max_hp_cost=40, rating=3),    # [Accept] 得 5(3) 张 Apparition，失去 50% 最大生命（截断 1.0）
            _opt("OTHER", rating=2),                    # [Refuse] 拒绝
        ]
    },
    "Vampires": {
        "options": [
            _opt("OTHER", rating=4),                    # [Offer: Blood Vial] 交出 Blood Vial：移除全部 Strike，得 5 张 Bite（需持有）
            _opt("OTHER", max_hp_cost=40, rating=3),    # [Accept] 移除全部 Strike，得 5 张 Bite，失去 30% 最大生命（截断 1.0）
            _opt("OTHER", rating=3),                    # [Refuse] 拒绝
        ]
    },
    "Forgotten Altar": {
        "options": [
            _opt("RELIC", relic=1, rating=4),           # [Offer: Golden Idol] 交出 Golden Idol，得遗物「Bloody Idol」（需持有）
            _opt("OTHER", hp_cost=20, rating=3),        # [Sacrifice] +5 最大生命，失去 25% 当前生命（约 20）
            _opt("CURSE", curse=1, rating=1),           # [Desecrate] 获得诅咒「Decay」
        ]
    },
    "Pleading Vagrant": {
        "options": [
            _opt("RELIC", relic=1, gold=-85, rating=3),  # [Offer Gold] 花 85 金，得遗物
            _opt("RELIC", relic=1, curse=1, rating=2),   # [Rob] 得遗物，获得诅咒「Shame」
            _opt("OTHER", rating=3),                     # [Leave] 离开
        ]
    },
    # ===================== Act 3 =====================
    "Falling": {
        "options": [
            _opt("REMOVE", remove=1, rating=3),         # [Land] 失去一张随机 Skill（强制弃卡）
            _opt("REMOVE", remove=1, rating=3),         # [Channel] 失去一张随机 Power
            _opt("REMOVE", remove=1, rating=3),         # [Strike] 失去一张随机 Attack
        ]
    },
    "Sensory Stone": {
        "options": [
            _opt("OTHER", rating=3),                    # [Recall] 加 1 张无色卡
            _opt("OTHER", hp_cost=5, rating=3),         # [Recall] 加 2 张无色卡，失去 5 HP
            _opt("OTHER", hp_cost=10, rating=4),        # [Recall] 加 3 张无色卡，失去 10 HP
        ]
    },
    "Winding Halls": {
        "options": [
            _opt("OTHER", hp_cost=10, rating=3),        # [Embrace Madness] 得 2 张 Madness，失去 12.5% 最大生命（约 10）
            _opt("HEAL", curse=1, rating=3),            # [Focus] 获得诅咒「Writhe」，治疗 25% 最大生命
            _opt("OTHER", max_hp_cost=4, rating=2),     # [Retrace Your Steps] 失去 5% 最大生命（约 4）
        ]
    },
    "Mind Bloom": {
        "options": [
            _opt("COMBAT", gold=50, relic=1, rating=4),  # [I am War] 与 Act1 Boss 战斗，胜得稀有遗物 + 50 金
            _opt("UPGRADE", upgrade=3, relic=1, rating=4),  # [I am Awake] 升级全部卡 + 遗物「Mark of the Bloom」（无法再治疗）
            _opt("GOLD", gold=999, curse=1, rating=3),   # [I am Rich/Healthy] +999 金/2诅咒 或 回满血/1诅咒（按楼层二选一，同占 idx2）
        ]
    },
    "Tomb of Lord Red Mask": {
        "options": [
            _opt("RELIC", relic=1, gold=-200, rating=3),  # [Offer: X Gold] 失去全部金币，得遗物「Red Mask」（持有红面具时变为 +222 金，同占 idx0）
            _opt("OTHER", rating=3),                     # [Leave] 离开
        ]
    },
    "Secret Portal": {
        "options": [
            _opt("COMBAT", rating=3),                    # [Enter the Portal] 直接进入 Act3 Boss 战
            _opt("OTHER", rating=3),                     # [Leave] 离开
        ]
    },
    # ===================== 全 Act 通用神龛 =====================
    "Note For Yourself": {
        "options": [
            _opt("OTHER", rating=3),           # [Take and Give] 取回存储卡 + 存一张卡
            _opt("OTHER", rating=2),           # [Ignore] 无事发生
        ]
    },
    "Purifier": {
        "options": [
            _opt("REMOVE", remove=1, rating=5),        # 祈祷：删除一张卡
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "Upgrade Shrine": {
        "options": [
            _opt("UPGRADE", upgrade=1, rating=5),      # 祈祷：升级一张卡
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "Transmogrifier": {
        "options": [
            _opt("TRANSMOG", rating=3),                # 祈祷：变化一张卡
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "Bonfire Spirits": {
        "options": [
            _opt("REMOVE", remove=1, rating=5),        # 献上卡：删除一张卡（按稀有度给奖励）
        ]
    },
    "The Divine Fountain": {
        # 别名/翻译：Fountain of Cleansing（净化喷泉）。任务清单中的
        # "Fountain of Cleansing" 即本事件，故不再另建重复条目。
        "options": [
            _opt("REMOVE", remove=2, rating=5),        # 喝：删除所有可移除诅咒卡（按数量近似置 2）
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "Duplicator": {
        "options": [
            _opt("DUPLICATE", rating=5),               # 祈祷：复制一张卡
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "Lab": {
        "options": [
            _opt("POTION", potion=1, rating=5),        # 进入：获得 3 个随机药水
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "The Woman in Blue": {
        "options": [
            _opt("POTION", potion=1, rating=4),        # 购买：被迫购买 1 个药水
            _opt("OTHER", rating=2),                   # 离开
        ]
    },
    "Match and Keep": {
        "options": [
            _opt("GAMBLE", rating=2),                  # 玩游戏：配对游戏，可能加入诅咒
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "We Meet Again!": {
        "options": [
            _opt("RELIC", relic=1, gold=-300, rating=3),  # 交易遗物：支付 300 金币
            _opt("RELIC", relic=1, rating=4),          # 交易药水：支付 1 个药水
            _opt("RELIC", relic=1, remove=1, rating=4),  # 交易卡牌：牺牲一张牌
            _opt("OTHER", rating=3),                   # 离开
        ]
    },
    "Wheel of Change": {
        "options": [
            _opt("OTHER", rating=5),                   # 离开
            _opt("GAMBLE", rating=2),                  # 转动：随机结果
        ]
    },
    # ===================== 2026-09 补录：模拟器已实现但 KB 缺失的 9 个事件 =====================
    # 来源：third_party/sts_lightspeed/src/game/GameContext.cpp 各 case Event::XXX 分支
    # （C++ 选项序号即 bits.idx1 / get_event_info()["option_indices"] 的值）。
    # 数值以 C++ 真实分支为准；不确定的字段填保守值并标 `# 待核`。
    "Old Beggar": {
        # canAddEvent: gold>=75 才出现。idx0 花 75 金删 1 张卡；idx1 离开。
        "options": [
            _opt("REMOVE", gold=-75, remove=1, rating=4),  # 给钱：花 75 金，删除一张卡
            _opt("OTHER", rating=2),                       # 离开
        ]
    },
    "Cursed Tome": {
        # 多相位阅读事件（C++ 用 info.eventData 记录相位，idx 随阅读推进）。
        # 数值以 C++ 分支为准；相位损失 HP 与最终获得遗物的确切血量标 `# 待核`。
        "options": [
            _opt("OTHER", rating=2),                       # [0] 继续阅读（相位 0，暂无偿）
            _opt("OTHER", rating=3),                       # [1] 离开
            _opt("OTHER", hp_cost=1, rating=1),            # [2] 阅读：失去 phase(=1) HP   # 待核相位损失
            _opt("OTHER", hp_cost=2, rating=1),            # [3] 阅读：失去 phase(=2) HP   # 待核
            _opt("OTHER", hp_cost=3, rating=1),            # [4] 阅读：失去 phase(=3) HP   # 待核
            _opt("RELIC", hp_cost=12, relic=1, rating=4),  # [5] 读完：失去 10/15 HP + 随机遗物(Necronomicon/Enchiridion/Nilrys Codex)  # 待核 hp
            _opt("OTHER", hp_cost=3, rating=3),            # [6] 收尾：失去 3 HP + 结束
        ]
    },
    "Designer In-Spire": {
        # canAddEvent: act==2||3 且 gold>=75。6 选项定制商店式事件。
        "options": [
            _opt("UPGRADE", gold=-40, upgrade=1, rating=4),  # [0] 花 40 金，升级一张卡
            _opt("UPGRADE", upgrade=2, rating=4),            # [1] 升级两张随机卡
            _opt("REMOVE", gold=-60, remove=1, rating=4),    # [2] 花 60 金，删除一张卡
            _opt("TRANSMOG", gold=-60, rating=3),            # [3] 花 60 金，变化两张卡   # 变化 2 张
            _opt("REMOVE", gold=-90, remove=1, rating=3),    # [4] 花 90 金，删除一张卡（随后升级一张）
            _opt("OTHER", hp_cost=3, rating=3),              # [5] 失去 3 HP，结束
        ]
    },
    "Augmenter": {
        "options": [
            _opt("OTHER", rating=3),                     # [0] 获得卡 J.A.X.   # 待核(类别，得卡无专属类别)
            _opt("TRANSMOG", rating=3),                  # [1] 变化两张卡
            _opt("RELIC", relic=1, rating=4),            # [2] 获得遗物「突变力量」(Mutagenic Strength)
        ]
    },
    "Knowing Skull": {
        # canAddEvent: act==2 且 curHp>12。每次拜访 HP 损失随相位递增（info.hpAmountN++），
        # 确切起始值由运行时决定，下列 HP 用首访保守值并标 `# 待核`。
        "options": [
            _opt("GOLD", gold=90, hp_cost=7, rating=3),  # [0] 失去约 7 HP，获得 90 金   # 待核 hp
            _opt("OTHER", hp_cost=9, rating=3),          # [1] 失去约 9 HP，获得一张无色卡   # 待核 hp
            _opt("POTION", potion=1, hp_cost=11, rating=4),  # [2] 失去约 11 HP，获得一个药水   # 待核 hp
            _opt("OTHER", hp_cost=6, rating=3),          # [3] 失去 6 HP，结束
        ]
    },
    "Masked Bandits": {
        "options": [
            _opt("GOLD", gold=-300, rating=1),           # [0] 失去全部金币（金额不定，截断 -1.0）   # 待核(全额不定)
            _opt("COMBAT", gold=30, relic=1, rating=4),  # [1] 战斗：胜得 25-35 金 + 遗物「红面具」   # 待核 gold
        ]
    },
    "Mysterious Sphere": {
        "options": [
            _opt("COMBAT", gold=50, relic=1, rating=5),  # [0] 战斗：胜得 45-55 金 + 随机稀有遗物   # 待核 gold
            _opt("GOLD", gold=99, rating=3),             # [1] 获得 99 金（不利时 50）   # 待核 gold
        ]
    },
    "N'loth": {
        # canAddEvent: act==2 且 relics.size()>=2。牺牲一个遗物换「N'loth's Gift」。
        "options": [
            _opt("RELIC", relic=1, rating=4),            # [0] 牺牲遗物[relicIdx0]，获得 N'loth's Gift
            _opt("RELIC", relic=1, rating=4),            # [1] 牺牲遗物[relicIdx1]，获得 N'loth's Gift
            _opt("OTHER", rating=3),                     # [2] 离开
        ]
    },
    "The Moai Head": {
        # canAddEvent: curHp<=maxHp/2 或持有「金神像」。idx0 失去 12.5-18% 最大生命后回满；
        # idx1 得 333 金但失去持有的「金神像」遗物。
        "options": [
            _opt("HEAL", max_hp_cost=10, rating=4),      # [0] 失去约 12.5% 最大生命(截断)，回满血   # 待核(12.5-18% maxHp)
            _opt("GOLD", gold=333, rating=4),             # [1] 获得 333 金（失去持有的「金神像」遗物）   # 待核(失去遗物无负字段)
            _opt("OTHER", rating=3),                     # [2] 离开
        ]
    },
}


# ---- 事件名归一化索引（支持大小写/空格/标点变体查询）----
_NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_event_key(name: str) -> str:
    """把任意事件名变体归一化为小写、去非字母数字的连接串。

    例："The Cleric!" / "the  CLERIC" / "THE_CLERIC" → "thecleric"
    """
    if not name:
        return ""
    return _NONALNUM_RE.sub("", name.lower())


_NORM_INDEX = {normalize_event_key(k): k for k in EVENT_KB}

# ======================================================================
# ⚠️ 显示名别名映射（修复 C++ 绑定返回名与本 KB 规范名不一致，见 commit）
# ----------------------------------------------------------------------
# C++ 绑定的事件 display 名来自
# third_party/sts_lightspeed/include/constants/Events.h 的 eventGameNames[]：
#   "The Colosseum" / "Transmorgrifier"
# 而本 KB 早期按 Wiki 显示名收为 "Colosseum" / "Transmogrifier"（注意后者游戏
# 本就拼成少一个 o，以 C++ 为准）。两处不一致导致查表静默返回全零。这里
# **不改旧键名**，而是在查表时先查主索引再查别名表。别名经与主表相同的
# normalize_event_key 归一化（去非字母数字），确保 "The Colosseum" → thecolosseum
# → 命中 "Colosseum"；"Transmorgrifier" → transmorgrifier → 命中 "Transmogrifier"。
_EVENT_NAME_ALIASES = {
    # C++ eventGameNames[] 显示名   → 本 KB 规范名
    "The Colosseum":  "Colosseum",
    "Transmorgrifier": "Transmogrifier",  # 游戏本就拼成少个 o，以 C++ 为准
}
_EVENT_ALIAS_INDEX = {
    normalize_event_key(cpp_name): canonical
    for cpp_name, canonical in _EVENT_NAME_ALIASES.items()
}


def _resolve_key(event_key: str):
    if not event_key:
        return None
    k = normalize_event_key(event_key)
    key = _NORM_INDEX.get(k)
    if key is None:
        key = _EVENT_ALIAS_INDEX.get(k)
    return key


def get_event_option_features(event_key: str, option_idx: int = None,
                              label: str = None) -> list:
    """返回 20 维事件选项特征向量；未收录 / 越界返回 20 个 0.0（不抛异常）。

    event_key: 事件规范名（支持大小写/空格/标点变体，内部归一化查表）。
    option_idx: 选项序号（0-based）；None 或越界 → 全零。
    label: 预留参数（当前绑定下事件候选 label 恒为 ``action_0x<bits>``，
           无法用于解析事件，故忽略；保留以匹配调用约定）。
    """
    vec = [0.0] * EVENT_KB_DIM
    key = _resolve_key(event_key)
    if key is None or option_idx is None:
        return vec
    rec = EVENT_KB[key]
    if not (0 <= option_idx < len(rec["options"])):
        return vec
    opt = rec["options"][option_idx]

    # A 段：类别 one-hot（恰好一个 1）
    ci = CATEGORY_LIST.index(opt["cat"])
    vec[_OFF_CAT + ci] = 1.0

    # B 段：代价与收益
    vec[_OFF_COST + 0] = _norm_hp(opt["hp_cost"])
    vec[_OFF_COST + 1] = _norm_hp(opt["max_hp_cost"])
    vec[_OFF_COST + 2] = _norm_gold(opt["gold"])
    vec[_OFF_COST + 3] = float(opt["curse"])
    vec[_OFF_COST + 4] = _norm_count(opt["remove"])
    vec[_OFF_COST + 5] = _norm_count(opt["upgrade"])
    vec[_OFF_COST + 6] = float(opt["relic"])
    vec[_OFF_COST + 7] = float(opt["potion"])
    return vec


def get_event_option_rating(event_key: str, option_idx: int = None) -> int:
    """返回该事件选项的 ⭐ 推荐度（0-5）；未收录 / 越界返回 0。

    向量不含推荐度（标签非特征），此函数单独提供，可作启发式对照或辅助监督。
    """
    key = _resolve_key(event_key)
    if key is None or option_idx is None:
        return 0
    rec = EVENT_KB[key]
    if not (0 <= option_idx < len(rec["options"])):
        return 0
    return rec["options"][option_idx]["rating"]


def event_count() -> int:
    return len(EVENT_KB)


def option_count() -> int:
    return sum(len(v["options"]) for v in EVENT_KB.values())
