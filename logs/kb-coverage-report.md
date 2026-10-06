# KB 覆盖探针报告：367–442 段实测激活率

- 生成时间：2026-09-10T12:43:10
- 局数：40（MCTS@200，策略 RulePolicy）
- 候选向量维度：443
- 总候选采样数：6036，胜局：0

## 1. 候选 kind 分布

| kind | 次数 | 占比 |
|---|---:|---:|
| ACTION | 547 | 9.06% |
| BOSS_RELIC | 24 | 0.40% |
| CARD_REWARD | 1647 | 27.29% |
| EVENT_OPTION | 371 | 6.15% |
| MAP_NODE | 848 | 14.05% |
| REST_OPTION | 271 | 4.49% |
| REWARD | 905 | 14.99% |
| SHOP_ITEM | 401 | 6.64% |
| SKIP | 946 | 15.67% |
| TREASURE | 76 | 1.26% |

## 2. 药水 KB 段 [367,391)（24 维）激活率

- **全局激活率**：7.11%（即该段非零的候选占全部候选比例）

| kind | 候选数 | 该段非零数 | 激活率 |
|---|---:|---:|---:|
| ACTION | 547 | 0 | 0.00% |
| BOSS_RELIC | 24 | 0 | 0.00% |
| CARD_REWARD | 1647 | 0 | 0.00% |
| EVENT_OPTION | 371 | 0 | 0.00% |
| MAP_NODE | 848 | 0 | 0.00% |
| REST_OPTION | 271 | 0 | 0.00% |
| REWARD | 905 | 330 | 36.46% |
| SHOP_ITEM | 401 | 99 | 24.69% |
| SKIP | 946 | 0 | 0.00% |
| TREASURE | 76 | 0 | 0.00% |

## 3. 遗物 KB 段 [391,423)（32 维）激活率

- **全局激活率**：3.15%（即该段非零的候选占全部候选比例）

| kind | 候选数 | 该段非零数 | 激活率 |
|---|---:|---:|---:|
| ACTION | 547 | 0 | 0.00% |
| BOSS_RELIC | 24 | 18 | 75.00% |
| CARD_REWARD | 1647 | 0 | 0.00% |
| EVENT_OPTION | 371 | 0 | 0.00% |
| MAP_NODE | 848 | 0 | 0.00% |
| REST_OPTION | 271 | 0 | 0.00% |
| REWARD | 905 | 124 | 13.70% |
| SHOP_ITEM | 401 | 48 | 11.97% |
| SKIP | 946 | 0 | 0.00% |
| TREASURE | 76 | 0 | 0.00% |

## 4. 事件 KB 段 [423,443)（20 维）激活率

- **全局激活率**：5.65%（即该段非零的候选占全部候选比例）

| kind | 候选数 | 该段非零数 | 激活率 |
|---|---:|---:|---:|
| ACTION | 547 | 0 | 0.00% |
| BOSS_RELIC | 24 | 0 | 0.00% |
| CARD_REWARD | 1647 | 0 | 0.00% |
| EVENT_OPTION | 371 | 341 | 91.91% |
| MAP_NODE | 848 | 0 | 0.00% |
| REST_OPTION | 271 | 0 | 0.00% |
| REWARD | 905 | 0 | 0.00% |
| SHOP_ITEM | 401 | 0 | 0.00% |
| SKIP | 946 | 0 | 0.00% |
| TREASURE | 76 | 0 | 0.00% |

## 5. label 弃用率（label 以 `action_0x` 开头 = 描述回退，无可读名）

- **全局弃用率**：100.00%

| kind | 候选数 | 弃用数 | 弃用率 |
|---|---:|---:|---:|
| ACTION | 547 | 547 | 100.00% |
| BOSS_RELIC | 24 | 24 | 100.00% |
| CARD_REWARD | 1647 | 1647 | 100.00% |
| EVENT_OPTION | 371 | 371 | 100.00% |
| MAP_NODE | 848 | 848 | 100.00% |
| REST_OPTION | 271 | 271 | 100.00% |
| REWARD | 905 | 905 | 100.00% |
| SHOP_ITEM | 401 | 401 | 100.00% |
| SKIP | 946 | 946 | 100.00% |
| TREASURE | 76 | 76 | 100.00% |

## 6. label 哈希段 [271,367)（96 维）冗余度 / 是否唯一信号

- 在 **卡牌 one-hot 段 [22,132) 非零** 的候选里，label 哈希段也非零的比例：**1628/1628 = 100.00%**
- 在 **三个 KB 段全零 且 卡牌段全零** 的候选里（这些候选没有 KB/卡牌结构化信号），label 哈希段是否非零：**3448/3448 = 100.00%**；同时其 bits 数值段 [215,223) 非零的比例：**2919/3448 = 84.66%**
  - 注：label 全局 100% 弃用（见 §5），所有 label 都是 `action_0x<bits>` 回退串，因此 label 哈希段**只编码了 action bits**，与 [215,223) 数值段高度重叠，并非独立的语义信号。
  - 即便 label 哈希非零，这些候选也几乎都有 bits 数值段非零 → label 哈希主要是 bits 的冗余表达，删 271–367 段对模型影响很小。

## 7. 实际观测到的内容名全集（去重）与名字解析为 None 的次数

- **药水名（32 种）**：Ancient Potion, Attack Potion, Blessing Of The Forge, Block Potion, Blood Potion, Colorless Potion, Cultist Potion, Dexterity Potion, Distilled Chaos, Duplication Potion, Elixir Potion, Energy Potion, Essence Of Steel, Explosive Potion, Fairy Potion, Fear Potion, Fire Potion, Flex Potion, Fruit Juice, Gamblers Brew, Heart Of Iron, Liquid Bronze, Liquid Memories, Power Potion, Regen Potion, Skill Potion, Smoke Bomb, Snecko Oil, Speed Potion, Strength Potion, Swift Potion, Weak Potion
- **遗物名（80 种）**：Akabeko, Anchor, Ancient Tea Set, Art Of War, Astrolabe, Bag Of Preparation, Black Blood, Black Star, Blood Vial, Blue Candle, Bronze Scales, Calipers, Captains Wheel, Centennial Puzzle, Ceramic Fish, Champion Belt, Chemical X, Clockwork Souvenir, Coffee Dripper, Darkstone Periapt, Dollys Mirror, Dream Catcher, Eternal Feather, Frozen Egg, Fusion Hammer, Gremlin Horn, Hand Drill, Happy Flower, Horn Cleat, Kunai, Lantern, Lees Waffle, Magic Flower, Mark Of Pain, Matryoshka, Maw Bank, Meat On The Bone, Medical Kit, Mummified Hand, Nilrys Codex, Odd Mushroom, Oddly Smooth Stone, Omamori, Orange Pellets, Orichalcum, Orrery, Pantograph, Paper Phrog, Peace Pipe, Pear, Pen Nib, Potion Belt, Preserved Insect, Prismatic Shard, Red Skull, Regal Pillow, Runic Cube, Runic Pyramid, Self Forming Clay, Singing Bowl, Slavers Collar, Sling Of Courage, Snecko Eye, Sozu, Strange Spoon, Strike Dummy, Sundial, The Boot, The Courier, Thread And Needle, Tiny Chest, Tiny House, Toolbox, Toxic Egg, Toy Ornithopter, Unceasing Top, Vajra, Velvet Choker, War Paint, White Beast Statue
- **事件名（25 种）**：Big Fish, Cursed Tome, Dead Adventurer, Face Trader, Forgotten Altar, Golden Idol, Golden Shrine, Hypnotizing Colored Mushrooms, Living Wall, NEOW, Note For Yourself, Old Beggar, Ominous Forge, Purifier, Scrap Ooze, Shining Light, The Cleric, The Ssssserpent, The Woman in Blue, Transmorgrifier, Upgrade Shrine, We Meet Again!, Wheel of Change, Wing Statue, World of Goop
- 拿到名字但 KB 段仍全零（=名字未进 KB / 归一化失败）：药水 9 次，遗物 0 次，事件 30 次
- 名字为 None 但 KB 段却非零（异常，理论上不应发生）：药水 0 次，遗物 0 次
- **药水：拿到名字但 KB 里查不到（命名对齐缺口）**：Elixir Potion, Fairy Potion  ← C++ 显示名与 KB key 不一致
- **事件：拿到名字但 KB 里查不到**：Cursed Tome, Old Beggar, Transmorgrifier

### 名字解析为 None 的次数（本应带该 KB 名却拿不到 → 该段必然全零）

- 药水适用候选（REWARD/SHOP 的 reward_type==3）：共 438 个，其中 None **0** 个（0.00%）
- 遗物适用候选（REWARD/SHOP 的 reward_type==4 + BOSS_RELIC）：共 196 个，其中 None **6** 个（3.06%）
- 事件适用候选（EVENT_OPTION）：共 371 个，其中 None **0** 个（0.00%）

## 8. KB 文件条目命中情况（从未被查询到的条目数）

| KB | 条目总数 | 实际命中 | 从未查询 |
|---|---:|---:|---:|
| 药水 KB | 33 | 30 | 3 |
| 遗物 KB | 180 | 80 | 100 |
| 事件 KB | 43 | 22 | 21 |

> 注：从未查询 = KB 文件里定义的条目，在本次 40 局采样中没有任何候选命中它（可能因采样局数有限、或该遗物/事件在铁甲 A0 全解锁下低概率出现）。

## 9. 各 KB 段全局全零比例

- 药水段全零：**92.89%**
- 遗物段全零：**96.85%**
- 事件段全零：**94.35%**

## 10. 结论

### 三个 KB 段到底有没有用？
- 药水段全局激活率 **7.11%**，遗物段 **3.15%**，事件段 **5.65%**。
  - kind=REWARD：药水段 36.46%，遗物段 13.70%，事件段 0.00%。
  - kind=SHOP_ITEM：药水段 24.69%，遗物段 11.97%，事件段 0.00%。
  - kind=BOSS_RELIC：药水段 0.00%，遗物段 75.00%，事件段 0.00%。
  - kind=EVENT_OPTION：药水段 0.00%，遗物段 0.00%，事件段 91.91%。

### 问题根因：是编码器没拿到名字，还是 KB 内容不对？
- **编码器其实拿得到名字**：本应带 KB 名的候选里，名字解析为 None 的比例很低（药水 0.00%、遗物 3.06%、事件 0.00%；其中遗物 6 次 None 是 BOSS_RELIC/SHOP 里 idx 越界导致的边界情况）。所以「激活率低」不是因为编码器拿不到名字。
- **真正的缺口在 KB 内容/命名对齐**：明明拿到了名字却查不到 KB 条目（药水 9 次、遗物 0 次、事件 30 次）。典型如 C++ 显示名 ['Elixir Potion', 'Fairy Potion'] 与 KB key（如 Elixir / Fairy in a Bottle）不一致，归一化后仍 miss。
  → 修复方向：把 KB key 对齐到 C++ 实际返回的水药显示名（或反过来在编码侧做别名映射），而非怀疑编码器没拿到名字。

### label 哈希是否「唯一信号」？
- 在卡牌段非零的候选里，label 哈希同时非零的概率 100.00%；在三 KB 段+卡牌段全零的候选里，label 哈希是非零信号的比例 100.00%，而其中 bits 数值段 [215,223) 也非零的比例 84.66%。
  → 存在无 bits 信号的候选，label 哈希可能是其唯一非零信号，删除需谨慎。

### KB 条目覆盖
- 药水 KB 33 条命中 30 条；遗物 KB 180 条命中 80 条；事件 KB 43 条命中 22 条。未命中多为采样局数有限 + 低概率内容，不一定是 KB 缺陷。

### 采样局限（如实说明）
- 本次 40 局全部 PLAYER_LOSS（RulePolicy 较弱，多在 Act1–2 阵亡），因此 Act3 BOSS 遗物、后期高稀有内容、部分事件出现次数偏少，激活率偏向「前期内容」。若要看全量覆盖，建议用更强的策略（如训练好的 Scorer）或更多局数重测；本报告的「从未查询」条目数应理解为**40 局采样下的下界**，而非 KB 绝对无用的证明。
