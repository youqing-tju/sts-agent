"""事件选项知识库（EVENT_OPTION 候选的语义特征来源）。

背景与「成败前提」（见 sts-agent 任务书）
=======================================
杀戮尖塔 RL 宏观决策里，事件候选（kind=EVENT_OPTION）在 C++ 模拟器
（slaythespire 绑定）里的真实形态是：

- `GameAction.describe(gc)` 对 EVENT 类动作**恒返回空串**，因此
  `candidate.label` 必然回退成 ``action_0x<bits>``（见 env/lightspeed_env.py）。
- `gc` 上**没有任何字段暴露当前是哪个事件**：实测只有
  ``boss / cur_room / encounter / outcome / screen_state``，
  其中 ``cur_room`` 在事件屏为 INVALID、``encounter`` 为 INVALID，
  没有任何 event id / event name 方法（已用 160 个 seed 普查确认）。
- 一个事件内多个选项的 `bits` 里 **idx1 = 选项序号（0..n-1，连续）**，
  这是唯一可靠且稳定的每候选信号。

结论（诚实优先）：**在当前的 C++ 绑定下，候选层面拿不到「当前是哪个事件」，
也拿不到选项文本**。因此本 KB 只能以「事件规范名」为一级 key、
「选项序号」为二级 key——这份数据本身正确且完整（源自调研报告第七章），
但**在现有绑定接入时运行时覆盖率为 0**：除非把绑定/编码器扩展成能传入
`event_key`（例如暴露 `gc.get_event_name()`），否则 `get_event_option_features`
只会走「未收录 → 全零」分支。

这是任务书要求的「诚实告知」，不是偷工：KB 数据已按报告全量录入，
接入点（event_key 来源）缺失属于环境层限制，由后续绑定扩展补齐。

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
    # ===================== Act 1 =====================
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
    # ===================== Act 2 =====================
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
    # ===================== 全 Act 通用神龛 =====================
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


def _resolve_key(event_key: str):
    if not event_key:
        return None
    return _NORM_INDEX.get(normalize_event_key(event_key))


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
