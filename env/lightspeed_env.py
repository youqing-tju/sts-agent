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

from .base_env import Obs, Candidate, StepResult, StsEnv

# 候选动作 desc_vec 占位维度（M2 才接入真实特征编码）
_DESC_VEC_DIM = 368

# NN 观测维度
_NN_OBS_DIM = 412

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

# NNInterface 单例缓存（C++ 所有权 bug 已修复，缓存仅为减少调用开销）
_nni_singleton = None


class LightspeedEnv(StsEnv):
    def __init__(self, mcts_simulations: int = 200):
        self._agent = sts.Agent()
        self._agent.simulation_count_base = mcts_simulations
        self._agent.pause_on_noncombat = True  # 非战斗决策点交还控制权（核心开关）
        self.gc = None
        self._actions = []  # candidates() 缓存的动作列表，step 按 idx 取用

    def reset(self, seed: int) -> Obs:
        """建新局并推进到第一个宏观决策点。"""
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
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
        kind = _KIND_BY_SCREEN.get(self.gc.screen_state.name, "ACTION")
        cands = []
        for i, a in enumerate(acts):
            label = a.describe(self.gc) or f"action_0x{a.bits:08x}"
            cands.append(Candidate(
                idx=i, kind=kind,
                desc_vec=self._placeholder_desc_vec(),
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
        """循环 playout 直到宏观决策点（有可选动作）或终局。"""
        gc = self.gc
        for _ in range(_MAX_ADVANCE_ROUNDS):
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            self._agent.playout(gc)
            if gc.outcome != sts.GameOutcome.UNDECIDED:
                return
            if sts.get_all_actions(gc):
                return
        raise RuntimeError("_advance 超过迭代上限，状态可能异常")

    def _nn_vec(self) -> list:
        """取 412 维 NN 观测。

        C++ 侧已修复（return_value_policy::reference）：单例可重复获取，
        且任意 GameContext 都可观测。Python 侧仍缓存单例减少调用开销。
        """
        global _nni_singleton
        if _nni_singleton is None:
            _nni_singleton = sts.getNNInterface()
        return list(_nni_singleton.getObservation(self.gc))

    @staticmethod
    def _placeholder_desc_vec() -> list:
        """候选描述向量占位：M1 全零，M2 接真实特征编码。"""
        return [0.0] * _DESC_VEC_DIM
