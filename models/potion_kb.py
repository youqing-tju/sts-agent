"""药水知识库（Potion Knowledge Base）。

把"药水候选"编码成 **24 维**语义特征向量，供 candidate_encoder 拼接进
443 维候选描述向量（candidate_encoder 已接入；本文件只提供纯查询接口）。

数据来源：《杀戮尖塔-铁甲战士-深度调研报告》第九章"药水系统与资源管理"
（已与官方 Wiki 交叉核验）。重点覆盖铁甲战士能拿到的常见药水。

======================================================================
⚠️ 重要接入说明（probe 结论，见 commit message / 项目报告）
----------------------------------------------------------------------
本作 C++ 绑定（slaythespire.cp313-win_amd64.pyd）的
`GameAction.describe(gc)` 对**所有**候选都返回空串，药水/遗物候选的
label 实际是 `action_0x<bits>`（只编码 reward_type + 槽位序号，**不含
药水/遗物名字**）；游戏状态也未暴露"当前奖励的具体药水/遗物名"
（只有 `gc.relics` 给已持有的遗物，`gc.potion_slots` 给玩家药水栏）。

因此调用方必须自行解析出药水**显示名**再传给 `get_potion_features(name)`
（显示名约定见下方 POTION_KB 的 key：Title Case，如 "Blood Potion"、
"Gambler's Brew"、"Fairy in a Bottle"）。`name` 经 `_norm_potion` 归一化
（小写 / 空格与连字符→下划线 / 去撇号），对大小写、空格、连字符变体
都鲁棒。若确实拿不到名字，可传 `""` 或任意未知串，返回 24 个 0。
======================================================================

24 维布局（顺序敏感，勿改；与报告第五章/第九章语义对齐）：
  [0,  8)  药水类别 one-hot：
            0 ATTACK 攻击 / 1 DEFENSE 防御 / 2 ENERGY 能量 / 3 DRAW 抽牌 /
            4 HEAL 治疗 / 5 STRENGTH 力量 / 6 STATUS 减益(虚弱/易伤) / 7 OTHER 其他
  [8,  9)  战斗内可用（combat_in，0/1）
  [9, 10)  战斗外可用（combat_out，0/1）
  [10,13)  稀有度 one-hot：10 COMMON / 11 UNCOMMON / 12 RARE
  [13,14)  综合数值强度分档 power_score（0~1）
  [14,24)  细分数值分档（均 0~1）：
            14 heal      治疗量    15 block    格挡/板甲
            16 damage    直接伤害  17 energy   能量
            18 draw      抽牌      19 strength 永久力量
            20 str_temp  临时力量  21 weak     虚弱施加
            22 vuln      易伤施加  23 thorns   荆棘
"""
import re

POTION_FEATURE_DIM = 24

# 类别顺序固定（与下方索引严格对应）
POTION_CATEGORIES = ["ATTACK", "DEFENSE", "ENERGY", "DRAW", "HEAL", "STRENGTH", "STATUS", "OTHER"]
POTION_RARITY = ["COMMON", "UNCOMMON", "RARE"]

# 细分数值字段名（仅供内部构造与测试引用，顺序即 [14,24) 的下标偏移）
_POTION_NUMERIC_FIELDS = [
    "heal", "block", "damage", "energy", "draw",
    "strength", "str_temp", "weak", "vuln", "thorns",
]


def _norm_potion(name: str) -> str:
    """显示名 → 归一化 key：小写；撇号直接删除；其余非字母数字（空格/连字符/下划线）
    统一折叠为单个下划线。

    这样 "Blood Potion" / "blood potion" / "BLOOD-POTION" / "Blood_Potion" 都映射到
    "blood_potion"，对大小写、空格、连字符、下划线变体都鲁棒。
    """
    n = (name or "").strip().lower()
    n = n.replace("'", "")
    n = re.sub(r"[^a-z0-9]+", "_", n)
    return n.strip("_")


# 每条：category / rarity / combat_in / combat_out / power / 10 个细分数值（见上）。
# 数值为 0~1 的经验分档（基础值，未升级；升级后更强，但 KB 只表达基础语义）。
_POTION_RAW = {
    # ---- 铁甲战士专属（报告 9.2）----
    "Blood Potion":      dict(category="HEAL",    rarity="COMMON",  ci=1, co=1, power=0.50, heal=0.50),
    "Elixir":            dict(category="OTHER",   rarity="UNCOMMON",ci=1, co=1, power=0.30),
    "Heart of Iron":     dict(category="DEFENSE", rarity="RARE",    ci=1, co=1, power=0.70, block=0.70),

    # ---- 通用药水·普通（报告 9.3）----
    "Block Potion":      dict(category="DEFENSE", rarity="COMMON",  ci=1, co=1, power=0.60, block=0.60),
    "Strength Potion":   dict(category="STRENGTH",rarity="COMMON",  ci=1, co=1, power=0.50, strength=0.50),
    "Energy Potion":     dict(category="ENERGY",  rarity="COMMON",  ci=1, co=1, power=0.50, energy=0.50),
    "Fire Potion":       dict(category="ATTACK",  rarity="COMMON",  ci=1, co=1, power=0.50, damage=0.50),
    "Explosive Potion":  dict(category="ATTACK",  rarity="COMMON",  ci=1, co=1, power=0.45, damage=0.40),
    "Flex Potion":       dict(category="STRENGTH",rarity="COMMON",  ci=1, co=1, power=0.45, str_temp=0.50),
    "Weak Potion":       dict(category="STATUS",  rarity="COMMON",  ci=1, co=1, power=0.40, weak=0.40),
    "Fear Potion":       dict(category="STATUS",  rarity="COMMON",  ci=1, co=1, power=0.45, vuln=0.40),
    "Swift Potion":      dict(category="DRAW",    rarity="COMMON",  ci=1, co=1, power=0.40, draw=0.40),
    # 以下 7 个为官方 Common 通用药水，首版 KB 遗漏（wiki 42 种全表核验补录）
    "Attack Potion":     dict(category="ATTACK",  rarity="COMMON",  ci=1, co=1, power=0.45, damage=0.30),  # 选1攻击牌0费打出
    "Dexterity Potion":  dict(category="DEFENSE", rarity="COMMON",  ci=1, co=1, power=0.50, block=0.40),   # +2敏捷
    "Speed Potion":      dict(category="DEFENSE", rarity="COMMON",  ci=1, co=1, power=0.45, block=0.50),   # +5临时敏捷
    "Power Potion":      dict(category="OTHER",   rarity="COMMON",  ci=1, co=1, power=0.50),                # 选1能力牌0费打出
    "Skill Potion":      dict(category="OTHER",   rarity="COMMON",  ci=1, co=1, power=0.40),                # 选1技能牌0费打出
    "Colorless Potion":  dict(category="OTHER",   rarity="COMMON",  ci=1, co=1, power=0.40),                # 选1无色牌0费打出
    "Blessing of the Forge": dict(category="OTHER", rarity="COMMON", ci=1, co=0, power=0.60),               # 升级手牌所有牌(本场战斗)

    # ---- 通用药水·罕见（报告 9.3）----
    "Liquid Bronze":     dict(category="DEFENSE", rarity="UNCOMMON",ci=1, co=1, power=0.40, block=0.30, thorns=0.40),
    "Regen Potion":      dict(category="HEAL",    rarity="UNCOMMON",ci=1, co=1, power=0.40, heal=0.30),
    "Essence of Steel":  dict(category="DEFENSE", rarity="UNCOMMON",ci=1, co=1, power=0.55, block=0.50),
    "Gambler's Brew":    dict(category="DRAW",    rarity="UNCOMMON",ci=1, co=1, power=0.40, draw=0.40),
    "Liquid Memories":   dict(category="DRAW",    rarity="UNCOMMON",ci=1, co=1, power=0.45, draw=0.45),
    "Duplication Potion":dict(category="OTHER",  rarity="UNCOMMON",ci=1, co=1, power=0.50),
    # 以下 2 个为官方 Uncommon 通用药水，首版 KB 遗漏
    "Ancient Potion":    dict(category="DEFENSE", rarity="UNCOMMON",ci=1, co=1, power=0.50, block=0.40),   # +1人工制品
    "Distilled Chaos":   dict(category="DRAW",    rarity="UNCOMMON",ci=1, co=1, power=0.50, draw=0.60),   # 打出牌堆顶3张
    "Smoke Bomb":        dict(category="OTHER",   rarity="RARE",    ci=1, co=0, power=0.30),              # 非boss战斗逃离（官方为 Rare，首版误标 Uncommon）

    # ---- 通用药水·稀有（报告 9.3，高价值）----
    "Cultist Potion":    dict(category="STRENGTH",rarity="RARE",    ci=1, co=1, power=0.85, strength=0.80),  # 力量流神级（仪式匕首每回合+1力量）
    "Fruit Juice":       dict(category="HEAL",    rarity="RARE",    ci=1, co=1, power=0.40, heal=0.25),
    "Fairy in a Bottle": dict(category="HEAL",    rarity="RARE",    ci=1, co=1, power=0.80, heal=0.80),       # 复活神药
    "Snecko Oil":        dict(category="DRAW",    rarity="RARE",    ci=1, co=1, power=0.55, draw=0.60),
    "Entropic Brew":     dict(category="OTHER",   rarity="RARE",    ci=1, co=1, power=0.40),
}


def _build_potion_kb():
    kb = {}
    for disp_name, raw in _POTION_RAW.items():
        kb[_norm_potion(disp_name)] = raw
    return kb


# 归一化 key → 原始条目；调用方也可读 POTION_KB 看收录了哪些
POTION_KB = _build_potion_kb()


def get_potion_features(name: str) -> list:
    """返回 24 维药水特征向量；未收录（含空名）返回 24 个 0.0，绝不抛异常。"""
    entry = POTION_KB.get(_norm_potion(name))
    if entry is None:
        return [0.0] * POTION_FEATURE_DIM

    vec = [0.0] * POTION_FEATURE_DIM
    # 0-7 类别 one-hot
    cat = entry.get("category")
    if cat in POTION_CATEGORIES:
        vec[POTION_CATEGORIES.index(cat)] = 1.0
    # 8-9 战斗内/外可用
    vec[8] = 1.0 if entry.get("ci") else 0.0
    vec[9] = 1.0 if entry.get("co") else 0.0
    # 10-12 稀有度 one-hot
    rar = entry.get("rarity")
    if rar in POTION_RARITY:
        vec[10 + POTION_RARITY.index(rar)] = 1.0
    # 13 综合强度
    vec[13] = float(entry.get("power", 0.0))
    # 14-23 细分数值
    for off, fld in enumerate(_POTION_NUMERIC_FIELDS):
        v = entry.get(fld)
        if v is not None:
            vec[14 + off] = float(v)
    return vec
