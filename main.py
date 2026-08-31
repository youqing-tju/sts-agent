"""sts-agent 真实游戏入口（路线 B：模拟器当大脑外挂）。

协议（Communication Mod）：
- 启动后向 stdout 打印 "ready"
- 之后每个稳定游戏状态会通过 stdin 收到一行 JSON
- 向 stdout 写动作命令（PLAY / END / CHOOSE / START 等）

决策架构：
- 宏观决策（选路/卡牌奖励/篝火/事件/商店）：
  真实游戏 JSON → GameContext.init_from_json 灌入模拟器 → get_all_actions
  枚举动作 → 368 维候选编码 + 412 维观测 → Scorer 打分 argmax →
  把选中的 GameAction.bits 翻译成 Communication Mod 命令发回真实游戏。
- 战斗出牌：v1 保持简单启发式（出第一张能出的牌，出不了 END），
  MCTS 战斗接入是后续任务。
- 降级：init_from_json / 编码 / 打分 / 翻译任一步失败 → 回退 main_rule 规则决策。

与 env/lightspeed_env.py 的候选特征逻辑保持一致（_candidate_kind /
_candidate_card / _candidate_room 为三处的小拷贝，约束要求不动 env/ 文件）。
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(
    filename="sts_agent_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("sts-agent")

import slaythespire as sts
import torch

from models.candidate_encoder import (
    CardInfo,
    decode_action_bits,
    encode_candidate,
)
from models.scorer import Scorer
from env.base_env import Candidate

# 降级路径：规则 bot（战斗启发式 + 宏观规则）
from main_rule import combat_action, macro_action

_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CKPT = os.path.join(_ROOT, "checkpoints", "long.pt")

# screen_state 枚举名 → Candidate.kind（与 lightspeed_env._KIND_BY_SCREEN 一致）
_KIND_BY_SCREEN = {
    "MAP_SCREEN": "MAP_NODE",
    "REWARDS": "REWARD",
    "SHOP_ROOM": "SHOP_ITEM",
    "REST_ROOM": "REST_OPTION",
    "EVENT_SCREEN": "EVENT_OPTION",
    "BOSS_RELIC_REWARDS": "BOSS_RELIC",
}

# RewardsActionType 编码（见 models.candidate_encoder.decode_action_bits 注释）
_REWARD_CARD = 0
_REWARD_SKIP = 6

# 唱歌碗特殊动作（不拿牌），见 lightspeed_env._candidate_card
_SINGING_BOWL_IDX2 = 5


class SimBrain:
    """模拟器大脑：把真实游戏状态灌进 lightspeed 模拟器，用 RL Scorer 决策。"""

    def __init__(self, checkpoint_path: str = _DEFAULT_CKPT):
        self.gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 0, 0)
        self._nni = sts.getNNInterface()  # 缓存单例（C++ 所有权 bug 已修复）
        self.scorer = None
        if os.path.exists(checkpoint_path):
            try:
                self.scorer = Scorer(obs_dim=412, cand_dim=368, hidden=(128, 128))
                self.scorer.load_state_dict(
                    torch.load(checkpoint_path, weights_only=True))
                self.scorer.eval()
                log.info("scorer loaded: %s", checkpoint_path)
            except Exception:
                log.exception("scorer 加载失败，宏观决策将全程走规则降级")
                self.scorer = None
        else:
            log.warning("checkpoint 不存在: %s，宏观决策将全程走规则降级",
                        checkpoint_path)
        self._ascension = 0
        self._map_cache = None       # (seed, act) 对应的 SpireMap
        self._map_cache_key = None

    # ---- 决策入口 ----

    def decide(self, state: dict) -> str:
        if state.get("error"):
            log.warning("game error: %s", state["error"])
        if not state.get("ready_for_command"):
            return "STATE"

        # Communication Mod 的 available_commands 全是小写
        cmds = [c.lower() for c in state.get("available_commands", [])]

        # 战斗：Communication Mod 在战斗中 screen_type 是 NONE，靠 "play" 命令判断
        if "play" in cmds:
            return combat_action(state)

        # 游戏未开始（主菜单）
        if "start" in cmds:
            return "START IRONCLAD"

        # 宏观决策：模拟器 + RL；任何失败回退规则
        try:
            cmd = self._sim_decide(state, cmds)
        except Exception:
            log.exception("sim 决策异常，回退规则决策")
            cmd = None
        if cmd is None:
            cmd = macro_action(state)
            log.info("rule fallback -> %s", cmd)
        return cmd

    # ---- 模拟器决策路径 ----

    def _sim_decide(self, state: dict, cmds: list):
        """灌入模拟器 → RL 选动作 → 翻译成 comm 命令；不可翻译返回 None。"""
        if self.scorer is None:
            return None
        game = state.get("game_state", {})
        self.gc.init_from_json(json.dumps(state))
        gc = self.gc
        self._ascension = int(game.get("ascension_level", 0) or 0)
        screen = gc.screen_state.name

        actions = sts.get_all_actions(gc)
        if not actions:
            log.warning("sim 无可用动作（screen=%s），回退规则", screen)
            return None

        obs = list(self._nni.getObservation(gc))
        encoded = [self._encode(a, screen, i) for i, a in enumerate(actions)]
        with torch.no_grad():
            logits = self.scorer(
                torch.tensor(obs, dtype=torch.float32),
                torch.tensor(encoded, dtype=torch.float32),
            )
        best = int(torch.argmax(logits))
        chosen = actions[best]
        log.debug("screen=%s n_act=%d best=%d bits=0x%08x logits=%s",
                  screen, len(actions), best, chosen.bits,
                  [round(float(x), 3) for x in logits])
        return self._to_comm(chosen, actions, screen, cmds)

    # ---- bits → Communication Mod 命令翻译 ----

    def _to_comm(self, action, actions: list, screen: str, cmds: list):
        """把选中的 GameAction 翻译成 comm 命令；无法翻译返回 None（走规则）。"""
        d = decode_action_bits(action.bits)

        # v1 不翻译药水动作（bit31/30）
        if d["is_potion"] or d["is_potion_discard"]:
            return None

        if screen == "MAP_SCREEN":
            return self._map_to_comm(actions, d, cmds)

        if screen in ("REWARDS", "CARD_SELECT"):
            rt = d["reward_type"]
            if rt == _REWARD_CARD:
                idx = d["idx2"]
                if idx == _SINGING_BOWL_IDX2:
                    return None  # 唱歌碗：v1 不会翻译，回退规则
                cards = self.gc.get_card_reward()
                if idx >= len(cards):
                    return None
                # comm mod 卡牌奖励屏 choose 就是按下标选卡
                return f"CHOOSE {idx}" if "choose" in cmds else None
            if rt == _REWARD_SKIP:
                if "skip" in cmds:
                    return "SKIP"
                if "proceed" in cmds:
                    return "PROCEED"
                return None
            # GOLD/KEY/POTION/RELIC/CARD_REMOVE：comm mod 通常自动收取，v1 不翻译
            return None

        if screen == "REST_ROOM":
            return f"CHOOSE {d['idx1']}" if "choose" in cmds else None

        if screen == "EVENT_SCREEN":
            return f"CHOOSE {d['idx1']}" if "choose" in cmds else None

        if screen == "SHOP_ROOM":
            # v1 直接离开商店（不买）
            if "proceed" in cmds:
                return "PROCEED"
            if "leave" in cmds:
                return "LEAVE"
            return None

        return None

    @staticmethod
    def _map_to_comm(actions: list, d: dict, cmds: list):
        """MAP 屏：idx1 = 目标节点 x 坐标。

        comm mod 的 choice_list 是可选节点列表，顺序与下一层节点从左到右
        （x 升序）一致；模拟器 MAP 动作与可选节点一一对应，因此把所有地图
        动作的 x 排序后，选中 x 的位次即 CHOOSE 下标。
        """
        if "choose" not in cmds:
            return None
        xs = sorted(a.bits & 0xFF for a in actions)
        x = d["idx1"]
        if x not in xs:
            return None
        return f"CHOOSE {xs.index(x)}"

    # ---- 候选特征解析（与 lightspeed_env 保持一致的小拷贝） ----

    def _encode(self, action, screen: str, idx: int) -> list:
        d = decode_action_bits(action.bits)
        kind = self._candidate_kind(screen, d)
        label = action.describe(self.gc) or f"action_0x{action.bits:08x}"
        proto = Candidate(idx=idx, kind=kind, desc_vec=[], label=label)
        return encode_candidate(
            proto,
            card=self._candidate_card(screen, d),
            room=self._candidate_room(screen, d),
            bits=action.bits,
        )

    @staticmethod
    def _candidate_kind(screen: str, d: dict) -> str:
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
        if screen != "REWARDS" or d["reward_type"] != 0 or d["idx2"] == 5:
            return None
        cards = self.gc.get_card_reward()
        if d["idx2"] >= len(cards):
            return None
        c = cards[d["idx2"]]
        return CardInfo(
            name=c.id.name, card_type=c.type.name, rarity=c.rarity.name,
            cost=0.0,  # Card 绑定未暴露费用，置 0
            upgraded=c.upgraded,
        )

    def _candidate_room(self, screen: str, d: dict):
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

    def _spire_map(self):
        key = (self.gc.seed, self.gc.act)
        if self._map_cache_key != key:
            self._map_cache = sts.SpireMap(
                self.gc.seed, self._ascension, self.gc.act, False)
            self._map_cache_key = key
        return self._map_cache


def main():
    brain = SimBrain()
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    log.info("sts-agent bridge ready (sim brain)")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            log.error("bad json: %s", line[:200])
            continue

        # 把最近一次状态原样写盘，方便检查真实 JSON 结构
        try:
            with open("last_state.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

        cmd = brain.decide(state)
        log.debug("screen=%s -> %s",
                  state.get("game_state", {}).get("screen_type"), cmd)
        if cmd == "STATE":
            time.sleep(0.1)  # 避免在主菜单等无操作状态下疯狂自旋
        sys.stdout.write(cmd + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
