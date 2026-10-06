"""lightspeed_env：把 sts_lightspeed C++ 模拟器封装成 StsEnv 统一接口。

设计要点（M1 宏观/战斗分离）：
- 宏观决策（选路/卡牌奖励/商店/篝火/事件/BOSS奖励）暴露为 candidates，交给外部 Policy 决策。
- 战斗内部不暴露：Agent 设 pause_on_noncombat=True，playout 自动打完战斗，
  在每个非战斗决策点暂停并把控制权交还 env；进/出战 HP 差记入 StepResult.info.hp_delta。

C++ 侧能力（commit 9e1e391，绑定名以 _probe_api.py 实测为准）：
- sts.Agent()：simulation_count_base / pause_on_noncombat / playout(gc)
- sts.GameContext(CharacterClass, seed, ascension)：
  outcome / screen_state（枚举，.name 取字符串）/ cur_hp / max_hp / gold / floor_num / act /
  deck（卡牌对象列表，.id 为枚举，.id.name 取可读字符串）/ relics（同理）
- sts.get_all_actions(gc) → [GameAction]；GameAction.bits / is_valid(gc) / execute(gc) / describe(gc)
  （EVENT 类动作 describe 可能返回空串，label 回退为 action_0x<bits>）
- sts.getNNInterface().getObservation(gc) → 412 维观测向量

【C++ 限制已修复】原 NNInterface 单例 bug（getNNInterface 重复调用堆损坏、
getObservation 只认首个 GameContext）已在绑定层修复：getNNInterface 改用
return_value_policy::reference（根因是 pybind11 默认对原始指针 take_ownership，
Python GC 会 delete 单例）。修复后单进程可连续观测多局游戏。
"""

import slaythespire as sts

from models.candidate_encoder import (
    DESC_VEC_DIM,
    CardInfo,
    _LOOKAHEAD_LAYERS,
    compute_map_lookahead,
    decode_action_bits,
    encode_candidate,
)
from models.card_data import CARD_ARCHETYPE, CARD_EFFECTS
from models.card_index import CARD_INDEX
from models.obs_encoder import (
    encode_deck_stats,
    encode_skip_deck_context,
    enhance_observation,
)

from .base_env import Obs, Candidate, StepResult, StsEnv

# 候选动作 desc_vec 维度：直接引用编码器常量，避免两处硬编码漂移
_DESC_VEC_DIM = DESC_VEC_DIM

# NN 观测维度
_NN_OBS_DIM = 477

# screen_state 枚举名 → Candidate.kind
_KIND_BY_SCREEN = {
    "MAP_SCREEN": "MAP_NODE",
    "REWARDS": "REWARD",
    "SHOP_ROOM": "SHOP_ITEM",
    "REST_ROOM": "REST_OPTION",
    "EVENT_SCREEN": "EVENT_OPTION",
    "BOSS_RELIC_REWARDS": "BOSS_RELIC",
}

# 防御性上限：playout 推进循环的最大迭代次数，防止异常状态下死循环
_MAX_ADVANCE_ROUNDS = 10000
_MAX_COMBAT_STEPS = 2000      # Beam 驱动战斗时的单场最大决策步数（防死循环）

# NNInterface 单例缓存（C++ 所有权 bug 已修复，缓存仅为减少调用开销）
_nni_singleton = None


class LightspeedEnv(StsEnv):
    def __init__(self, mcts_simulations: int = 200,
                 combat_solver: str = "mcts",
                 beam_width: int = 4,
                 beam_depth: int = 8,
                 beam_max_nodes: int = 3000):
        """combat_solver: `"mcts"`（默认，C++ Agent 自动打完战斗）| `"beam"`（Python Beam 接管）。

        beam_* 仅在 combat_solver="beam" 时生效。**训练用小预算**：
        Beam 单决策耗时 ≈ O(W × depth × 合法动作数)，W=4/depth=8 约 50~150ms；
        一局约 10~30 场战斗 × 3~10 决策 → 数秒/局（8 workers 下可接受）。
        """
        self._agent = sts.Agent()
        self._agent.simulation_count_base = mcts_simulations
        self._agent.pause_on_noncombat = True  # 非战斗决策点交还控制权（核心开关）
        self._combat_solver = combat_solver
        # Beam 模式：让 playout 在**战斗入口**也暂停（pause_on_combat），
        # 改由 Python 的 _run_combat_beam 驱动战斗循环。
        self._agent.pause_on_combat = (combat_solver == "beam")
        self._beam = None
        if combat_solver == "beam":
            from combat_beam import BeamSolver   # 延迟 import：避免无关场景加载
            self._beam = BeamSolver(beam_width=beam_width, max_depth=beam_depth,
                                    max_nodes=beam_max_nodes)
        self.gc = None
        self._actions = []  # candidates() 缓存的动作列表，step 按 idx 取用
        self._ascension = 0
        self._map_cache = None       # (seed, act) 对应的 SpireMap，供 MAP_NODE 房间类型查询
        self._map_cache_key = None

    def reset(self, seed: int) -> Obs:
        """建新局并推进到第一个宏观决策点。"""
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
        self._ascension = 0
        self._map_cache = None
        self._map_cache_key = None
        self._advance()
        return self.observe()

    def observe(self) -> Obs:
        gc = self.gc
        return Obs(
            vec=self._nn_vec(),
            hp=gc.cur_hp,
            max_hp=gc.max_hp,
            floor=gc.floor_num,
            gold=gc.gold,
            deck=[c.id.name for c in gc.deck],
            relics=[r.id.name for r in gc.relics],
            screen=gc.screen_state.name,
        )

    def candidates(self) -> list:
        """当前可选宏观动作；终局返回空列表。

        正常流程中非终局必然停在宏观决策点（pause_on_noncombat），
        因此非终局时返回值非空。
        """
        if self.gc is None or self.gc.outcome != sts.GameOutcome.UNDECIDED:
            self._actions = []
            return []
        acts = sts.get_all_actions(self.gc)
        self._actions = acts
        screen = self.gc.screen_state.name
        # 牌组流派强度每步算一次，供所有候选的交叉项复用（保持编码器纯函数）
        raw_obs = self._raw_nn_vec()
        deck_arch = self._deck_arch(raw_obs)
        # V10：SKIP 候选专用牌组质量上下文（16 维），仅 SKIP 候选填充；
        # 用与真机侧相同的 encode_skip_deck_context 构造，单一事实来源。
        deck_ctx = encode_skip_deck_context(
            raw_obs, card_index=CARD_INDEX,
            arch_table=CARD_ARCHETYPE, card_effects=CARD_EFFECTS)
        # KB 内容每屏只取一次（按屏分发，非对应屏为 None 安全跳过）
        # 实测映射规则见 candidate_encoder 段表注释与项目报告：
        #   REWARDS:  reward_type 区分类别，idx1 = 该类别子列表下标
        #             (POTION=3→potions[idx1]，RELIC=4→relics[idx1])，卡牌由 idx2 处理
        #   SHOP_ROOM: get_shop_contents 扁平元组列表，idx1 = 该类别子列表下标
        #   BOSS_RELIC_REWARDS: idx1=0..N-1→contents[idx1] 遗物，==len 为跳过项
        #   EVENT_SCREEN: event_name 作 event_key，bits.idx1 = option_indices 序号
        reward_contents = self.gc.get_reward_contents() if screen == "REWARDS" else None
        shop_contents = self.gc.get_shop_contents() if screen == "SHOP_ROOM" else None
        boss_relic_contents = self.gc.get_boss_relic_contents() if screen == "BOSS_RELIC_REWARDS" else None
        event_info = self.gc.get_event_info() if screen == "EVENT_SCREEN" else None
        # REST_ROOM 不在本次 KB 范围（get_rest_options 不产出 KB 特征）
        cands = []
        for i, a in enumerate(acts):
            label = a.describe(self.gc) or f"action_0x{a.bits:08x}"
            d = decode_action_bits(a.bits)
            kind = self._candidate_kind(screen, d)
            # KB 三段键：按屏 + bits 解析出候选对应的 KB 名/键（None 时对应段全零）
            potion_name = relic_name = None
            event_key = event_option_idx = None
            if screen == "REWARDS" and reward_contents is not None:
                if d["reward_type"] == 3:
                    pl = reward_contents.get("potions") or []
                    if d["idx1"] < len(pl):
                        potion_name = pl[d["idx1"]]
                elif d["reward_type"] == 4:
                    rl = reward_contents.get("relics") or []
                    if d["idx1"] < len(rl):
                        relic_name = rl[d["idx1"]]
            elif screen == "SHOP_ROOM" and shop_contents is not None:
                if d["reward_type"] == 3:
                    pl = [t[1] for t in shop_contents if t[0] == "potion"]
                    if d["idx1"] < len(pl):
                        potion_name = pl[d["idx1"]]
                elif d["reward_type"] == 4:
                    rl = [t[1] for t in shop_contents if t[0] == "relic"]
                    if d["idx1"] < len(rl):
                        relic_name = rl[d["idx1"]]
                # reward_type 0=卡牌(绑定未暴露→段全零)/5=删卡/6=跳过：无 KB
            elif screen == "BOSS_RELIC_REWARDS" and boss_relic_contents is not None:
                # 候选 reward_type 固定为 0（绑定怪异），idx1==len(contents) 为跳过项
                if d["idx1"] < len(boss_relic_contents):
                    relic_name = boss_relic_contents[d["idx1"]]
            elif screen == "EVENT_SCREEN" and event_info is not None:
                event_key = event_info.get("event_name")
                event_option_idx = d["idx1"]
            proto = Candidate(idx=i, kind=kind, desc_vec=[], label=label)
            cands.append(Candidate(
                idx=i, kind=kind,
                desc_vec=encode_candidate(
                    proto,
                    card=self._candidate_card(screen, d),
                    room=self._candidate_room(screen, d),
                    bits=a.bits,
                    lookahead_rooms=self._candidate_lookahead(screen, d),
                    deck_arch=deck_arch,
                    potion_name=potion_name,
                    relic_name=relic_name,
                    event_key=event_key,
                    event_option_idx=event_option_idx,
                    deck_ctx=deck_ctx,
                ),
                label=label,
            ))
        return cands

    def step(self, idx: int) -> StepResult:
        """执行候选动作并推进到下一个宏观决策点或终局。

        idx < 0 时不执行动作，仅继续推进（防御性保留；正常流程中
        非终局状态必有候选，外部只应传 candidates() 给出的 idx）。
        info.hp_delta = 执行前后 cur_hp 差（含中间所有战斗）。
        info.outcome = 当前 GameOutcome 名（UNDECIDED / PLAYER_VICTORY / PLAYER_LOSS）。
        """
        hp_before = self.gc.cur_hp
        if idx >= 0:
            acts = self._actions or sts.get_all_actions(self.gc)
            a = acts[idx]
            if a.is_valid(self.gc):
                a.execute(self.gc)
        self._advance()
        done = self.gc.outcome != sts.GameOutcome.UNDECIDED
        info = {"hp_delta": self.gc.cur_hp - hp_before,
                "outcome": self.gc.outcome.name}
        return StepResult(obs=self.observe(), done=done, info=info)

    # ---- 内部辅助 ----

    def _advance(self):
        """循环推进直到宏观决策点（有可选动作）或终局。

        combat_solver="beam" 时，playout 会在**战斗入口**暂停（pause_on_combat），
        此时改由 `_run_combat_beam` 用 Beam Search 驱动整场战斗。
        """
        gc = self.gc
        for _ in range(_MAX_ADVANCE_ROUNDS):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            if self._beam is not None and gc.screen_state == sts.ScreenState.BATTLE:
                self._run_combat_beam(gc)
                continue
            self._agent.playout(gc)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            if sts.get_all_actions(gc):
                return
        raise RuntimeError("_advance 超过迭代上限，状态可能异常")

    def _run_combat_beam(self, gc):
        """用 Beam Search 驱动一场战斗（Python 侧逐决策），结束后写回 gc。

        骨架取自 sts-agent-v2 `engine/runner_sim._run_combat`：
        建 BattleContext → 逐决策 execute → exit_battle 写回 GameContext。
        Beam 给出的是**跨回合完整路线**，但这里只执行第一步并重新求解
        （因为真实推进可能因模拟器与游戏差异而偏离预测）。
        """
        from combat_beam import ACTION_TYPE_BY_NAME
        bc = sts.BattleContext()
        bc.init_from_gc(gc)
        for _ in range(_MAX_COMBAT_STEPS):
            if bc.is_terminal():
                break
            plan = self._beam.solve(bc)
            if not plan:
                break
            t, i, j = plan[0]
            act = sts.BattleAction(ACTION_TYPE_BY_NAME[t], i, j)
            if not act.is_valid(bc):
                act = sts.BattleAction(sts.ActionType.END_TURN, 0, 0)
                if not act.is_valid(bc):
                    break
            act.execute(bc)
        bc.exit_battle(gc)

    def _raw_nn_vec(self) -> list:
        """取 C++ 原始 412 维 NN 观测。

        C++ 侧已修复（return_value_policy::reference）：单例可重复获取，
        且任意 GameContext 都可观测。Python 侧仍缓存单例减少调用开销。
        """
        global _nni_singleton
        if _nni_singleton is None:
            _nni_singleton = sts.getNNInterface()
        return list(_nni_singleton.getObservation(self.gc))

    def _nn_vec(self) -> list:
        """取 477 维观测：C++ 412 维 + Python 侧增强 65 维。

        增强段（药水 24 + 药水数 1 + 地图摘要 4 + Act 4 + 房间 10
        + 牌组流派强度 5 + 牌组派生统计 8）见 models/obs_encoder.py。
        """
        return enhance_observation(
            self._raw_nn_vec(), self.gc,
            card_index=CARD_INDEX,
            arch_table=CARD_ARCHETYPE,
            card_effects=CARD_EFFECTS,
        )

    def _deck_arch(self, raw: list) -> list:
        """牌组五流派强度（5 维），供候选编码器的流派交叉项使用。"""
        return encode_deck_stats(
            raw, card_index=CARD_INDEX,
            arch_table=CARD_ARCHETYPE, card_effects=CARD_EFFECTS,
        )[:5]

    # ---- 候选特征解析（供 candidate_encoder 编码） ----

    @staticmethod
    def _candidate_kind(screen: str, d: dict) -> str:
        """按屏幕 + bits 里的 reward_type 细化 kind。

        REWARDS 屏区分 CARD_REWARD/SKIP/REWARD；SHOP 屏区分 SKIP；
        TREASURE_ROOM 映射为 TREASURE；其余按屏幕粗映射，兜底 ACTION。
        """
        if screen == "REWARDS":
            rt = d["reward_type"]
            if rt == 0:
                return "CARD_REWARD"
            if rt == 6:
                return "SKIP"
            return "REWARD"
        if screen == "SHOP_ROOM":
            return "SKIP" if d["reward_type"] == 6 else "SHOP_ITEM"
        if screen == "TREASURE_ROOM":
            return "TREASURE"
        return _KIND_BY_SCREEN.get(screen, "ACTION")

    def _candidate_card(self, screen: str, d: dict):
        """卡牌奖励候选 → CardInfo；其余 None（编码器卡牌段置零）。

        idx2==5 是唱歌碗特殊动作（不拿牌）。get_card_reward() 返回最后一页
        卡牌奖励；多页奖励且 idx1 指向非末页时会张冠李戴（v1 简化，罕见）。
        商店/CARD_SELECT 的卡牌绑定层未暴露，编码为全零。
        """
        if screen != "REWARDS" or d["reward_type"] != 0 or d["idx2"] == 5:
            return None
        cards = self.gc.get_card_reward()
        if d["idx2"] >= len(cards):
            return None
        c = cards[d["idx2"]]
        return CardInfo(
            name=c.id.name, card_type=c.type.name, rarity=c.rarity.name,
            cost=0.0,  # 绑定层 get_card_reward 返回 Card 无 cost；编码时回退查 card_data.base_cost
            upgraded=c.upgraded,
        )

    def _candidate_room(self, screen: str, d: dict):
        """MAP_NODE 候选 → 目标节点 Room 类型名；其余 None。

        地图动作 idx1 = 目标节点 x 坐标；目标 y = 当前 y+1（起始 y=-1 → 0；
        y=14 时唯一动作通往 BOSS）。用 (seed, ascension, act) 重建 SpireMap
        查询房间类型（burning elite 只标记坐标不改房间类型，重建结果与
        GameContext 内部地图一致；act>=4 走 act4Map 特例，v1 不编码）。
        """
        if screen != "MAP_SCREEN":
            return None
        y = self.gc.cur_map_node_y
        if y == 14:
            return "BOSS"
        if self.gc.act >= 4:
            return None
        m = self._spire_map()
        target_y = 0 if y == -1 else y + 1
        name = m.get_room_type(d["idx1"], target_y).name
        return None if name in ("INVALID", "NONE") else name

    def _candidate_lookahead(self, screen: str, d: dict):
        """MAP_NODE 候选 → 向下 6 层贪婪路径的房间类型名序列（供前瞻段编码）。

        从候选节点 (idx1, target_y) 出发，每层取离当前 x 最近的子节点，
        返回每层房间类型名列表；越界/无效房间置 None（编码时该层全零）。
        非 MAP 候选 / boss 节点 / act>=4 返回 None。
        """
        if screen != "MAP_SCREEN":
            return None
        y = self.gc.cur_map_node_y
        if y == 14 or self.gc.act >= 4:
            return []
        m = self._spire_map()
        target_y = 0 if y == -1 else y + 1

        def room_fn(x, yy):
            name = m.get_room_type(x, yy).name
            return None if name in ("INVALID", "NONE") else name

        rooms = compute_map_lookahead(d["idx1"], target_y, room_fn,
                                      m.has_edge, layers=_LOOKAHEAD_LAYERS)
        return rooms

    def _spire_map(self):
        """按 (seed, act) 缓存重建的 SpireMap（一幕内地图不变）。"""
        key = (self.gc.seed, self.gc.act)
        if self._map_cache_key != key:
            self._map_cache = sts.SpireMap(
                self.gc.seed, self._ascension, self.gc.act, False)
            self._map_cache_key = key
        return self._map_cache
