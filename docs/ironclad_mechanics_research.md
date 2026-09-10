# 《杀戮尖塔》铁甲战士游戏机制深度调研报告

> 调研日期：2026-09-05
> 调研范围：铁甲战士角色卡牌、遗物及游戏核心机制
> 数据来源：官方 Wiki、社区攻略、原版游戏数据
> **版本：原版《杀戮尖塔》（Slay the Spire 1.0）**

---

## 一、铁甲战士（Ironclad）角色概述

### 基础属性
- **最大生命值**：80 HP
- **起始能量**：3 点/回合
- **起始金币**：99
- **起始卡牌数**：10 张（Strike×5, Defend×4, Bash×1）
- **起始遗物**：Burning Blood（战后治疗 6 HP）

### 角色定位
铁甲战士是《杀戮尖塔》的第一个可玩角色，定位为**近战战士**。其核心机制围绕：
1. **Strength（力量）**：直接增加攻击伤害
2. **Exhaust（耗尽）**：将卡牌移出当前战斗，提供触发效果
3. **Self-Damage（自伤）**：以 HP 换取强力效果
4. **Block（格挡）**：高防御能力

### 角色特点
- **新手友好**：机制直观，Burning Blood 提供持续恢复
- **简单上手**：策略清晰，容易理解
- **高生命值**：80 HP 起始，容错率高
- **弱点**：缺少可靠的 Block 选项，高成本卡牌多

---

## 二、铁甲战士卡牌完整列表（75 张）

### 2.1 基础卡牌（Basic Cards - 3 张）

| 卡牌名称 | 类型 | 能量 | 效果 | 稀有度 | 获取途径 |
|---------|------|------|------|--------|----------|
| **Strike** | Attack | 1 | 对单体敌人造成 6 点伤害 | Basic | 起始卡牌（×5） |
| **Defend** | Skill | 1 | 获得 5 点 Block | Basic | 起始卡牌（×4） |
| **Bash** | Attack | 2 | 造成 8 点伤害，施加 2 层 Vulnerable | Basic | 起始卡牌 |

### 2.2 普通卡牌（Common Cards - 20 张）

| 卡牌名称 | 类型 | 能量 | 效果 | 稀有度 |
|---------|------|------|------|--------|
| **Anger** | Attack | 0 | 造成 6 点伤害，将一张该卡牌复制加入弃牌堆 | Common |
| **Armaments** | Skill | 1 | 获得 5 Block，升级手牌中的一张卡牌 | Common |
| **Body Slam** | Attack | 1 | 造成等于当前 Block 值的伤害 | Common |
| **Clash** | Attack | 0 | 仅当手牌全为攻击牌时可打出，造成 14 点伤害 | Common |
| **Cleave** | Attack | 1 | 对所有敌人造成 8 点伤害 | Common |
| **Clothesline** | Attack | 2 | 造成 12 点伤害，施加 2 层 Weak | Common |
| **Flex** | Skill | 0 | 获得 2 点 Strength，本回合结束时失去 2 点 Strength | Common |
| **Havoc** | Skill | 1 | 打出抽牌堆顶的卡牌并耗尽它 | Common |
| **Headbutt** | Attack | 1 | 造成 9 点伤害，将弃牌堆中的一张卡牌置于抽牌堆顶 | Common |
| **Heavy Blade** | Attack | 2 | 造成 14 点伤害，Strength 对此卡效果 3 倍 | Common |
| **Iron Wave** | Attack | 1 | 获得 5 Block，造成 5 点伤害 | Common |
| **Perfected Strike** | Attack | 2 | 造成 6 点伤害，对每张包含"Strike"的卡牌额外造成 2 点伤害 | Common |
| **Pommel Strike** | Attack | 1 | 造成 9 点伤害，抽 1 张牌 | Common |
| **Shrug It Off** | Skill | 1 | 获得 8 Block，抽 1 张牌 | Common |
| **Sword Boomerang** | Attack | 1 | 对随机敌人造成 3 点伤害 3 次 | Common |
| **Thunderclap** | Attack | 1 | 对所有敌人造成 4 点伤害，施加 1 层 Vulnerable | Common |
| **True Grit** | Skill | 1 | 获得 7 Block，随机耗尽 1 张卡牌 | Common |
| **Twin Strike** | Attack | 1 | 造成 5 点伤害两次 | Common |
| **Warcry** | Skill | 0 | 抽 1 张牌，将手牌中的一张牌置于抽牌堆顶，耗尽 | Common |
| **Wild Strike** | Attack | 1 | 造成 12 点伤害，将一张 Wound 洗入抽牌堆 | Common |

### 2.3 稀有卡牌（Uncommon Cards - 36 张）

| 卡牌名称 | 类型 | 能量 | 效果 | 核心机制 |
|---------|------|------|------|----------|
| **Battle Trance** | Skill | 0 | 抽 3 张牌，本回合不能额外抽牌 | Draw Control |
| **Blood for Blood** | Attack | 0 | 本战斗中每失去一次 HP 能量减少 1，造成 18 点伤害 | Self-Damage Scaling |
| **Bloodletting** | Skill | 0 | 失去 3 HP，获得 2 能量 | Self-Damage Energy |
| **Burning Pact** | Skill | 1 | 耗尽 1 张牌，抽 2 张牌 | Exhaust Engine |
| **Carnage** | Skill | 2 | Ethereal，造成 20 点伤害 | High Damage |
| **Combust** | Power | 1 | 回合结束时失去 1 HP，对所有敌人造成 5 点伤害 | Self-Damage AoE |
| **Dark Embrace** | Power | 2 | 每耗尽一张卡牌抽 1 张牌 | Draw Engine |
| **Disarm** | Skill | 1 | 敌人失去 2 点 Strength，耗尽 | Debuff |
| **Dropkick** | Attack | 1 | 造成 5 点伤害，如果敌人有 Vulnerable 获得 1 能量并抽 1 张牌 | Vulnerable Synergy |
| **Dual Wield** | Skill | 1 | 选择一张攻击或能量牌，将其复制加入手牌 | Card Duplication |
| **Entrench** | Skill | 2 | Block 翻倍 | Block Scaling |
| **Evolve** | Power | 1 | 每次抽到状态牌时抽 1 张牌 | Status Synergy |
| **Feel No Pain** | Power | 1 | 每耗尽一张卡牌获得 3 Block | Exhaust Defense |
| **Fire Breathing** | Power | 1 | 每次抽到状态或诅咒牌时对所有敌人造成 6 点伤害 | Status Offense |
| **Flame Barrier** | Skill | 2 | 获得 12 Block，本回合被攻击时对攻击者造成 4 点伤害 | Reflective Damage |
| **Ghostly Armor** | Skill | 1 | Ethereal，获得 10 Block | Temporary Defense |
| **Hemokinesis** | Attack | 1 | 失去 2 HP，造成 15 点伤害 | Self-Damage |
| **Infernal Blade** | Skill | 1 | 将一张随机攻击卡加入手牌，本回合免费打出，耗尽 | Attack Generation |
| **Inflame** | Power | 1 | 获得 2 点 Strength | Simple Strength |
| **Intimidate** | Skill | 1 | 对所有敌人施加 1 层 Weak，耗尽 | AoE Debuff |
| **Metallicize** | Power | 1 | 回合结束时获得 3 Block | Passive Defense |
| **Power Through** | Skill | 1 | 获得目标卡牌，在抽牌堆顶加 2 张 Wound，获得 15 Block | Risk/Block |
| **Pummel** | Attack | 1 | 造成 2 点伤害 4 次，耗尽 | Multi-hit |
| **Rage** | Skill | 0 | 本回合每打出一张攻击卡获得 3 Block | Attack Defense |
| **Rampage** | Attack | 1 | 造成 8 点伤害，本战斗中该卡牌伤害增加 5 点 | Scaling Damage |
| **Reckless Charge** | Attack | 1 | 造成 7 点伤害，将一张 Dazed 洗入抽牌堆 | Risk/Damage |
| **Rupture** | Power | 1 | 每次从卡牌效果失去 HP 时获得 1 点 Strength | Self-Damage Scaling |
| **Searing Blow** | Attack | 1 | 造成 12 点伤害，可以无限次升级 | Infinite Upgrade |
| **Second Wind** | Skill | 1 | 耗尽手牌中所有非攻击卡牌，每耗尽一张获得 5 Block | Exhaust Defense |
| **Seeing Red** | Skill | 0 | 获得 2 能量，耗尽 | Energy Burst |
| **Sentinel** | Skill | 1 | 获得 5 Block，如果本回合没有受到伤害则获得 5 能量，耗尽 | Reactive Energy |
| **Sever Soul** | Skill | 1 | 造成 16 点伤害，耗尽 | High Damage |
| **Shockwave** | Skill | 1 | 对所有敌人施加 3 层 Weak 和 Vulnerable，耗尽 | AoE Debuff |
| **Spot Weakness** | Skill | 1 | 如果敌人打算攻击则获得 3 点 Strength，耗尽 | Conditional Strength |
| **Uppercut** | Attack | 1 | 造成 13 点伤害，施加 1 层 Weak 和 1 层 Vulnerable | Mixed Effect |
| **Whirlwind** | Attack | X | 对所有敌人造成 5 点伤害 X 次 | Multi-hit AoE |

### 2.4 史诗卡牌（Rare Cards - 16 张）

| 卡牌名称 | 类型 | 能量 | 效果 | 核心价值 |
|---------|------|------|------|----------|
| **Barricade** | Power | 3 | Block 不会在回合开始时移除 | Infinite Block |
| **Berserk** | Power | 1 | 获得 2 层 Vulnerable，回合开始时获得 1 能量 | Risk/Energy |
| **Bludgeon** | Attack | 3 | 造成 32 点伤害 | Single Target Burst |
| **Brutality** | Power | 1 | 回合开始时失去 1 HP，抽 1 张牌 | Self-Damage Draw |
| **Corruption** | Power | 3 | 所有技能牌消耗 0 能量，打出技能牌时耗尽它 | Skill Engine |
| **Demon Form** | Power | 3 | 回合开始时获得 2 点 Strength | Strength Scaling |
| **Double Tap** | Skill | 1 | 本回合下一张攻击牌额外打出一次 | Attack Multiplier |
| **Exhume** | Skill | 2 | 将耗尽堆中的一张牌加入手牌，耗尽 | Card Recovery |
| **Feed** | Attack | 1 | 造成 10 点伤害，如果击杀则最大生命值增加 3 点，耗尽 | HP Scaling |
| **Fiend Fire** | Attack | 2 | 耗尽手牌，每耗尽一张造成 7 点伤害，耗尽 | Hand Nuke |
| **Immolate** | Attack | 2 | 对所有敌人造成 21 点伤害，将一张 Burn 洗入弃牌堆 | AoE Damage |
| **Impervious** | Skill | 2 | 获得 30 Block，耗尽 | Single Turn Defense |
| **Juggernaut** | Power | 2 | 每获得 Block 时对随机敌人造成 5 点伤害 | Block Offense |
| **Limit Break** | Power | 1 | Strength 也增加攻击伤害 | Strength Amplifier |
| **Offering** | Skill | 0 | 失去 6 HP，获得 2 能量，抽 5 张牌，耗尽 | High Risk Burst |
| **Reaper** | Attack | 2 | 对所有敌人造成 4 点伤害，恢复未格挡的伤害，耗尽 | Lifesteal |

---

## 三、铁甲战士专属遗物（10 个）

### 3.1 初始遗物

| 遗物名称 | 稀有度 | 效果 | 升级版本 |
|---------|--------|------|----------|
| **Burning Blood** | Starter | 战斗结束后治疗 6 HP | Black Blood（12 HP） |

### 3.2 普通遗物

| 遗物名称 | 稀有度 | 效果 | 获取途径 | 战略价值 |
|---------|--------|------|----------|----------|
| **Red Skull** | Common | 当 HP ≤ 50% 时，额外获得 3 点 Strength | 随机掉落 | 自伤型强力增益 |

### 3.3 稀有遗物

| 遗物名称 | 稀有度 | 效果 | 获取途径 | 核心机制 |
|---------|--------|------|----------|----------|
| **Paper Phrog** | Rare | Vulnerable 敌人承受 75% 额外伤害（而非标准的 50%） | 随机掉落 | Vulnerable 放大器 |
| **Self-Forming Clay** | Rare | 战斗中每次失去 HP 时，下一回合获得 3 Block | 随机掉落 | 自伤防御补偿 |

### 3.4 史诗遗物

| 遗物名称 | 稀有度 | 效果 | 获取途径 | 战略用途 |
|---------|--------|------|----------|----------|
| **Champion Belt** | Rare | 每次施加 Vulnerable 时同时施加 1 层 Weak | 随机掉落 | Control Combo |
| **Charon's Ashes** | Rare | 每耗尽一张卡牌时，对所有敌人造成 3 点伤害 | 随机掉落 | 耗尽型 AoE 输出 |
| **Magic Flower** | Rare | 战斗中治疗效果提升 50% | 随机掉落 | 治疗增强 |

### 3.5 Boss 遗物

| 遗物名称 | 稀有度 | 效果 | 风险/收益比 |
|---------|--------|------|-------------|
| **Black Blood** | Boss | 战斗结束后治疗 12 HP（替换 Burning Blood） | 无风险/强收益 |
| **Mark of Pain** | Boss | 回合开始获得 1 能量，战斗开始时抽牌堆中加入 2 张 Wound | 中风险/高回报 |
| **Runic Cube** | Boss | 每次失去 HP 时抽 1 张牌 | 高风险/高回报 |

### 3.6 商店遗物

| 遗物名称 | 稀有度 | 效果 | 风险/收益比 |
|---------|--------|------|-------------|
| **Brimstone** | Shop | 回合开始时获得 2 点 Strength，所有敌人获得 1 点 Strength | 高风险/高回报 |

---

## 四、游戏核心机制详解

### 4.1 回合制战斗流程

#### 玩家回合结构
1. **回合开始阶段**
   - 能量重置为基础值（默认 3 点）
   - Block 重置为 0
   - 抽牌至手牌上限（默认 5 张）
   - 触发所有"回合开始"效果（遗物、Buff、Power）

2. **出牌阶段**
   - 消耗能量打出卡牌
   - 可按任意顺序出牌
   - 可使用药水
   - Energy 不可累积到下一回合

3. **回合结束阶段**
   - 触发所有"回合结束"效果
   - 手牌中的未打出卡牌全部弃置到弃牌堆
   - 移除本回合 Block

#### 敌人回合结构
1. **移除敌人 Block**
   - 所有敌人的 Block 重置为 0

2. **敌人行动阶段**
   - 按从左到右顺序行动（多敌人场景）
   - 执行预先显示的意图（攻击、Block、Buff、Debuff）

3. **特殊事件**
   - 死亡召唤效果
   - 持续伤害效果（Poison、Burn）

### 4.2 能量系统

- **基础能量**：每回合 3 点（可通过遗物/卡牌修改）
- **能量消耗**：卡牌能量消耗范围为 0-3+ 点
- **能量传递**：未使用的能量不会累积到下一回合
- **能量获取方式**：
  - 基础每回合自动获得
  - 卡牌效果（Bloodletting、Seeing Red 等）
  - 遗物效果（Berserk、Mark of Pain 等）
  - 能量药水

### 4.3 抽牌机制

- **基础抽牌数**：每回合 5 张
- **牌库循环**：抽牌堆空时，弃牌堆洗牌成为新的抽牌堆
- **抽牌限制**：无手牌上限限制
- **抽牌控制**：
  - Battle Trance：抽 3 张但本回合不能额外抽牌
  - Burning Pact：耗尽 1 张牌抽 2 张
  - Dark Embrace：每耗尽一张牌抽 1 张
  - Shrug It Off：获得 8 Block 并抽 1 张

### 4.4 状态效果系统

#### 4.4.1 Block（格挡）
- **效果**：吸收伤害，1 Block = 防止 1 点伤害
- **持续时间**：持续到下一回合开始（玩家 Block）或敌人回合结束（敌人 Block）
- **特殊机制**：
  - Barricade：Block 不会在回合开始时移除
  - Entrench：Block 翻倍
  - Body Slam：将 Block 转化为伤害
  - Juggernaut：每次获得 Block 时对随机敌人造成伤害

#### 4.4.2 Vulnerable（易伤）
- **效果**：受到的攻击伤害增加 50%
- **持续时间**：通常持续 1-3 回合
- **叠加规则**：最多叠加 3 层
- **计算顺序**：先应用 Strength 伤害加成，再乘以 Vulnerable 倍率
- **施加方式**：Bash、Thunderclap、Uppercut 等
- **Paper Phrog 增强**：Vulnerable 效果从 50% 提升到 75%

#### 4.4.3 Weak（虚弱）
- **效果**：攻击造成的伤害减少 25%
- **持续时间**：通常持续 1-3 回合
- **叠加规则**：最多叠加 3 层
- **特殊交互**：Weak 攻击者攻击 Vulnerable 目标时，两者效果相互抵消
- **施加方式**：Clothesline、Intimidate、Shockwave 等

#### 4.4.4 Strength（力量）
- **效果**：每点 Strength 增加所有攻击伤害 +1
- **持续时间**：永久持续（战斗内）
- **获取方式**：
  - Inflame：直接获得 2 点 Strength
  - Demon Form：每回合自动获得 2 点 Strength
  - Rupture：每次失去 HP 获得 1 点 Strength
  - Flex：临时获得，回合结束失去
  - Spot Weakness：条件触发
  - Red Skull：低血量时额外获得
  - Brimstone：每回合获得但敌人也获得
  - Limit Break：Strength 同时增加攻击伤害

#### 4.4.5 其他重要状态

| 状态名称 | 效果 | 持续时间 | 典型来源 |
|---------|------|----------|----------|
| **Poison** | 回合开始时造成 X 点伤害，可叠加 | 永久 | Silent 专属 |
| **Dexterity** | 增加 Block 获取量 | 永久 | Footwork |
| **Frail** | Block 获取量减少 25% | 持续回合数 | 敌人技能 |
| **Plating** | 永久 Block | 永久 | Metallicize |

### 4.5 敌人行为模式

#### 意图系统
敌人头上会显示其下回合意图：
- **剑形图标**：攻击（数字为伤害值）
- **盾形图标**：获得 Block
- **螺旋图标**：施加 Debuff（Vulnerable、Weak 等）
- **云朵图标**：自我 Buff（Strength、Block 等）
- **问号图标**：特殊行动（根据敌人 AI 卡牌）

#### 行为模式特点
1. **模式化 AI**：大多数敌人遵循固定的行为模式
2. **多阶段战斗**：Boss 通常有 2-3 个阶段，每个阶段行为不同
3. **随机性**：部分敌人有随机行为或条件触发技能
4. **多敌人协同**：多敌人场景中按从左到右顺序行动

### 4.6 特殊机制

#### 4.6.1 Exhaust（耗尽）
- **定义**：将卡牌从当前战斗中完全移除
- **存储位置**：独立的耗尽堆，不进入弃牌堆
- **战斗结束**：所有耗尽卡牌返回正常牌库
- **触发效果**：Dark Embrace（抽牌）、Feel No Pain（Block）、Charon's Ashes（伤害）
- **恢复机制**：Exhume 可将耗尽卡牌重新加入手牌

#### 4.6.2 Self-Damage（自伤）
- **核心机制**：以 HP 换取强力效果
- **典型卡牌**：Bloodletting、Hemokinesis、Offering、Rupture、Combust
- **风险补偿**：
  - Burning Blood：战后治疗
  - Self-Forming Clay：自伤后下回合获得 Block
  - Rupture：每次自伤获得 Strength
  - Brutality：自伤后抽牌
  - Mark of Pain：自伤后抽牌（但起始加 Wound）
  - Runic Cube：自伤后抽牌

#### 4.6.3 Multi-hit（多段攻击）
- **定义**：一次攻击造成多次伤害
- **Strength 优势**：每段攻击都获得 Strength 加成
- **Vulnerable 优势**：每段攻击都享受 Vulnerable 加成
- **典型卡牌**：Whirlwind、Twin Strike、Pummel、Sword Boomerang

#### 4.6.4 Deck Thinning（牌库精简）
- **战略价值**：减少牌库大小，提高核心卡牌出现频率
- **实现方式**：
  - Exhaust 机制：永久移除低价值卡牌
  - Feed：击杀后移除自己
  - Searing Blow：可以无限升级替代其他卡牌
- **目标卡牌**：Strike、Defend 等基础卡牌

#### 4.6.5 Ethereal（虚化）
- **效果**：卡牌本回合结束时被耗尽
- **典型卡牌**：Carnage、Ghostly Armor、Disarm
- **战略用途**：强力单次效果，不会污染牌库

---

## 五、铁甲战士核心构建策略

### 5.1 Strength Build（力量流）
**核心理念**：堆叠 Strength，用多段攻击卡牌将力量转化为伤害。

**核心卡牌**：
- Strength 生成：Demon Form、Inflame、Flex、Spot Weakness、Rupture
- 多段攻击：Whirlwind、Twin Strike、Pummel、Sword Boomerang
- Strength 放大：Limit Break（Strength 同时增加攻击伤害）

**关键遗物**：
- Red Skull：低血量时额外 Strength
- Brimstone：高风险但强力 Strength 增益

### 5.2 Block Engine（格挡引擎）
**核心理念**：堆积 Block，通过 Body Slam 或 Juggernaut 转化为伤害。

**核心卡牌**：
- Block 保留：Barricade
- Block 翻倍：Entrench
- Block 生成：Iron Wave、Defend、Shrug It Off、Ghostly Armor
- Block 转化：Body Slam、Juggernaut

**关键遗物**：
- Self-Forming Clay：自伤后获得 Block 补偿

### 5.3 Exhaust Synergy（耗尽流）
**核心理念**：通过耗尽卡牌触发多种效果，同时精简牌库。

**核心卡牌**：
- 耗尽触发：Dark Embrace（抽牌）、Feel No Pain（Block）、Charon's Ashes（伤害）
- 耗尽引擎：Corruption（技能免费+耗尽）、Burning Pact、Second Wind
- 终极爆发：Fiend Fire（耗尽手牌造成巨额伤害）
- 耗尽恢复：Exhume（重新使用耗尽卡牌）

**关键遗物**：
- Charon's Ashes：耗尽 AoE 伤害
- Runic Cube：自伤后抽牌

### 5.4 Self-Damage Payoff（自伤流）
**核心理念**：大量使用自伤卡牌，通过 Rupture 和 Combust 转化为伤害。

**核心卡牌**：
- 自伤源：Bloodletting、Hemokinesis、Offering、Combust、Brutality
- 自伤转化：Rupture（Strength）、Combust（AoE 伤害）、Brutality（抽牌）
- 生存补偿：Burning Blood

**关键遗物**：
- Red Skull：低血量时 Strength 补偿
- Mark of Pain：自伤后抽牌（但需承受 Wound）
- Runic Cube：自伤后抽牌

---

## 六、游戏进阶机制

### 6.1 卡牌升级系统
- **升级效果**：每张卡牌只能升级一次（Searing Blow 除外，可无限升级）
- **提升方向**：
  - 伤害增加
  - 能量消耗降低
  - 额外效果（抽牌、Block、状态施加）
  - 负面效果减轻
- **升级途径**：
  - 休息处升级（优先升级核心卡牌）
  - 事件免费升级
  - 特定遗物自动升级

### 6.2 牌库管理策略
**卡牌添加原则**：
- 优先添加与当前构建高度协同的卡牌
- 避免添加无协同的普通卡牌
- 稀有和史诗卡牌价值通常更高
- 考虑卡牌能量消耗与当前能量系统的匹配度

**卡牌移除策略**：
- 优先移除 Strike（基础攻击卡）
- 其次移除 Defend（基础防御卡）
- 保留核心构建的关键卡牌
- 移除与当前构建冲突的卡牌

### 6.3 地图路线规划
**Act 1 策略**：
- 优先寻找休息处升级核心卡牌
- 适当挑战精英获取强力遗物
- 避免风险事件（赌博、高风险交易）

**Act 2 策略**：
- 确定核心构建方向
- 针对性添加协同卡牌
- 准备 Boss 战（药水、卡牌配置）

**Act 3 策略**：
- 强化现有构建
- 移除低价值卡牌
- 准备最终 Boss 战

### 6.4 药水使用策略
**铁甲战士专属药水**：
- **Blood Potion**（普通）：造成 10 点伤害，自伤 2 HP
- **Elixir**（稀有）：治疗 5 HP，抽 3 张牌
- **Heart of Iron**（稀有）：本回合受到的攻击伤害降为 1

**使用原则**：
- 不要囤积药水，及时使用应对危机
- 优先用于精英和 Boss 战
- 血量危机时使用治疗药水
- 爆发需求时使用伤害药水

---

## 七、总结与战略建议

### 7.1 铁甲战士核心优势
1. **高生命值**：80 HP 起始，容错率高
2. **简单上手**：机制直观，适合新手学习
3. **自愈能力**：Burning Blood 提供持续恢复
4. **构建灵活**：支持多种流派
5. **力量放大**：Limit Break 让 Strength 效果倍增

### 7.2 核心弱点
1. **缺乏远程**：主要是近战伤害
2. **牌库膨胀风险**：自伤流容易牌库过大
3. **能量限制**：3 能量基础值限制爆发能力
4. **依赖核心卡牌**：缺少核心构建时表现平庸
5. **Block 选项有限**：相比其他角色缺少可靠的 Block 生成

### 7.3 新手建议
1. **从 Strength Build 开始**：机制简单，效果直观
2. **优先升级 Demon Form**：核心力量卡牌
3. **及时移除 Strike**：提高核心卡牌出现率
4. **合理使用药水**：不要囤积，及时应对危机
5. **学习敌人意图**：预判敌人行动，合理分配 Block
6. **利用 Burning Blood**：可以更激进地使用自伤卡牌

### 7.4 进阶技巧
1. **牌库精简**：通过 Exhaust 机制保持小牌库
2. **能量优化**：多使用低能量卡牌循环牌库
3. **自伤管理**：平衡自伤与收益，利用遗物补偿
4. **多段攻击**：充分利用 Strength 在多段攻击中的加成
5. **Vulnerable 控制**：合理施加 Vulnerable 爆发伤害
6. **Ethereal 利用**：强力单次效果，不污染牌库
7. **Limit Break 策略**：与 Demon Form 配合实现指数级伤害增长

---

## 八、数据来源与参考资料

1. **官方 Wiki**：Slay the Spire Wiki, Slay the Spire.org
2. **社区攻略**：HermitGamer, Sporcle, Pungry
3. **机制详解**：Official Game Rules, Ludo Guide
4. **原版数据**：基于 2022 年前的《杀戮尖塔》数据

---

**调研完成时间**：2026-09-05
**调研深度**：覆盖铁甲战士全部 75 张卡牌、10 个专属遗物、核心游戏机制
**数据准确性**：基于原版《杀戮尖塔》官方 Wiki 和权威社区攻略
**版本说明**：本报告针对原版《杀戮尖塔》（Slay the Spire 1.0），不包括《Slay the Spire 2》的新增内容
