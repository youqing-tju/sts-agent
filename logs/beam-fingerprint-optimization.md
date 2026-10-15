# Beam 状态指纹性能优化报告（P0-①）

> 2026-10-15｜对应 `docs/next-chat-handoff.md` §8 的 **P0-①**
> 结论：**微基准提速 125×，生产路径提速 6.1×，且搜索结果逐字不变。**

---

## 1. 结论速览

| 指标 | 优化前 | 优化后 | 提升 |
|---|---|---|---|
| `fingerprint()` 单次 | **2553 µs** | **20.4 µs** | **125×** |
| `BeamSolver.solve()` 单次（25 状态均值） | 52.9 ms | 1.2 ms | **45.4×** |
| **生产路径 8 局评估**（同种子 / 同配置） | **274 s** | **45 s** | **6.1×** |
| 8 局评估结果 | floor 26.9 / win 12.50% | floor 26.9 / win 12.50% | **完全一致** |

改动仅一行逻辑 ×4：**`c.id.name` → `c.id.value`**。

---

## 2. 问题

`docs/next-chat-handoff.md` §4 坑 1 记录：Beam 是纯 Python，单进程 W=2/depth=6/max-nodes=300
约 24 秒/局；性能热点定位在 `fingerprint()` 占**单节点 98%** 耗时。原计划「目标 2–3× 提速」。

## 3. 诊断

探针：`scripts/probes/_tmp_beam_fingerprint_probe.py`（分段的微基准）。
初始战斗状态（5 张手牌 / 5 张抽牌堆 / 1 敌人），N=2000 次：

| 分段 | 耗时 |
|---|---|
| `fingerprint()` 整体 | 2553 µs |
| `tuple(c.id.name for c in draw_pile)`（5 张） | 1289 µs |
| `hand: (name, cost, upgraded) 未排序`（5 张） | 1644 µs |
| `p.hp … p.artifact`（9 个属性） | **2.0 µs** |
| `monsters` 5 元组（含 intent） | **3.7 µs** |
| `BattleContext(bc)` 拷贝构造（Fork） | **1.2 µs** |

→ **手牌 + 牌堆两行就占 98%**；Player / Monster / 拷贝构造全都可忽略。
**瓶颈不在元组、不在排序、不在 pybind 对象拷贝，只在 `CardId.name` 这一个属性。**

逐字段细测（单张卡，N=20000）：

| 表达式 | 耗时 |
|---|---|
| `c.id`（对象本身） | 0.78 µs |
| **`c.id.value`** | **1.24 µs** |
| `c.id.name` | **444.14 µs** |
| `str(c.id)`（对照） | 同一量级 |

## 4. 根因

`slaythespire.CardId` 的 `.name` 属性**每次访问都做一次按名查找**（该枚举有 371 个成员），
单次 **≈ 444 µs**；而 `.value` 是直接返回整数枚举值，**≈ 1.24 µs** —— **差 358×**。

指纹里手牌 5 张 + 抽牌堆 5 张 = 10 次 `.name` → 约 4.4 ms…实测指纹 2.5 ms（有缓存/短路差异），
无论如何都在同一量级，足以解释 98% 的占比。

## 5. 修复

`combat_beam.py` 的 `fingerprint()`：

```python
# 手牌：顺序无关（排序），含费用与升级态
tuple(sorted((c.id.value, c.cost_for_turn, bool(c.upgraded)) for c in bc.hand)),
# 抽牌堆：**顺序重要**（决定后续抽到什么）
tuple(c.id.value for c in bc.draw_pile),
tuple(sorted(c.id.value for c in bc.discard_pile)),
tuple(sorted(c.id.value for c in bc.exhaust_pile)),
```

`monsters` 段的 `m.name` **保留不动**（实测 Monster 没有这个慢实现，整段仅 3.7 µs）。

## 6. 等价性验证（三层，缺一不可）

> ⚠️ 「快」不是理由，**等价**才是。指纹变了会破坏去重 → 搜索语义变了 → 训练结果不可比。

**① 枚举双射**（`scripts/probes/_tmp_cardid_value_bijection.py`）
371 个 CardId 常量：`value` 全部唯一 ✅、`name` 全部唯一 ✅、属性名与 `.name` 零不一致 ✅
→ `value ↔ name` 是一一映射。

**② 等价类划分一致**（`tests/test_combat_beam.py::test_fingerprint_equivalent_partition_matches_name_impl`）
从初始状态确定性展开 180 个可达状态，验证
`fp_new(a)==fp_new(b)` ⟺ `fp_old(a)==fp_old(b)`（**双向单射**）。
→ 去重语义**零影响**。

**③ 方案逐字相同 + 结果一致**（`scripts/probes/_tmp_beam_equivalence.py`）
- 25 个真实战斗状态，`solve()` 返回的动作序列**全部逐字相同**
- 8 局生产评估（同 checkpoint / 同种子 / 同配置）：`avg_floor=26.9`、`std_floor=15.3`、
  `win_rate=12.50%` —— **三项完全一致**

## 7. 性能数据（生产路径实测）

命令（`workers=1`，串行，便于比较）：

```bash
.venv/Scripts/python.exe -m evaluation.evaluate \
  --checkpoint checkpoints/v12_beam_500.pt --seeds 8 --workers 1 \
  --mcts-sim 20000 --combat-solver beam --beam-width 2 --beam-depth 6 --beam-max-nodes 300
```

| 版本 | 墙钟 | 单局 |
|---|---|---|
| 旧版（`c.id.name`） | **4 m 33.8 s** | 34.2 s |
| 新版（`c.id.value`） | **45.0 s** | 5.6 s |

> 端到端只有 6.1×（而非 solve 的 45×），因为固定开销不随优化下降：
> 进程启动 + torch 导入 ≈ 8 s；8 局宏观决策（`mcts_sim=20000`）约 32 s。
> 扣掉固定开销后，**纯 Beam 搜索部分 ≈ 234 s → 5.2 s（约 45×）**，与微基准吻合。

### 对云成本的直接意义（上云前必做的理由）

- 500 局单进程：34.2 s × 500 ≈ **4.8 h** → 5.6 s × 500 ≈ **47 min**
- 4 workers（实测并行效率约 50%）粗估：**≈ 2.4 h → ≈ 24 min**
  ⚠️ **外推值，需上云后实测确认**；但量级足以支撑「先优化再上云」。

## 8. 新增测试

`tests/test_combat_beam.py`（**新建，此前 Beam 求解器零测试覆盖**），16 个用例：

- `CardId.value ↔ name` 双射
- fingerprint：稳定性 / 拷贝不变性 / 可哈希 / 区分性（出牌后、血量变化）/ **等价类划分一致**
- legal_actions：含 END_TURN / 类型合法 / **能量耗尽后只剩 END_TURN**
- evaluate：有限浮点
- solve：**返回合法首动作** / **能量耗尽不返回空方案（死锁回归）** / **不改动传入 bc** /
  遵守 max_nodes 预算 / stats 字段完整
- **性能回归**：`fingerprint` 超过 300 µs 即失败（抓住退回字符串实现）

测试总数：**249 → 265 passed / 2 skipped**。

## 9. 顺带发现（新坑，已写入测试 docstring）

**`init_from_gc` 忽略 `combat_state.player` 下的 `energy` 与 `current_hp`：**

```python
p["energy"] = 0        # 改了 → bc.player.energy 仍是 3
p["current_hp"] = 60   # 改了 → bc.player.hp 仍是 72
```

- 血量要改**顶层** `game_state.current_hp` 才生效
- 能量改不动（`player.energy_per_turn` 也无效），**只能靠出牌耗尽**
- 写测试时若不清楚这点，会写出「断言 energy=0 只剩 END_TURN」这种永远为假的用例

## 10. 复现

```bash
cd "C:/Youqing-Vault/Embodied-AI/07-projects/sts-agent"
export KMP_DUPLICATE_LIB_OK=TRUE

.venv/Scripts/python.exe scripts/probes/_tmp_beam_fingerprint_probe.py    # 微基准分解
.venv/Scripts/python.exe scripts/probes/_tmp_cardid_value_bijection.py    # 双射验证
.venv/Scripts/python.exe scripts/probes/_tmp_beam_equivalence.py          # 新旧方案对比
.venv/Scripts/python.exe -m pytest tests/test_combat_beam.py -v          # 16 项
```

> `_tmp_beam_equivalence.py` 依赖 `_tmp_combat_beam_old.py`（用
> `git show <旧 commit>:combat_beam.py` 导出到同目录），该旧版副本**不入库**。
