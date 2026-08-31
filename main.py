"""sts-agent 真实游戏入口：Communication Mod 桥接。

协议（Communication Mod）：
- 启动后向 stdout 打印 "ready"
- 之后每个稳定游戏状态会通过 stdin 收到一行 JSON
- 向 stdout 写动作命令（PLAY / END / CHOOSE / START 等）

M3 策略分配：
- 宏观决策（选路/选牌/商店/篝火）→ RulePolicy
- 战斗出牌 → 简单启发式（能出攻击就出，没费了就结束回合）

注意：训练好的 RL 策略暂不能直接用在真实游戏——它的输入是 lightspeed 的
412 维观测 + 368 维候选编码（依赖模拟器内部 bits），需要单独的 JSON→特征
翻译层才能迁移（见设计文档 M3 节）。
"""

import json
import sys
import logging

logging.basicConfig(
    filename="sts_agent_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("sts-agent")


# ---------- 战斗启发式 ----------

def combat_action(state: dict) -> str:
    """极简战斗策略：按顺序出能出的牌，不能出了就 END。"""
    game = state.get("game_state", {})
    hand = game.get("hand", [])
    energy = game.get("player", {}).get("energy", 0)

    for i, card in enumerate(hand):
        if card.get("cost", 99) <= energy and card.get("is_playable", False):
            # 攻击牌需要指定目标
            if card.get("type") == "ATTACK":
                monsters = game.get("monsters", [])
                target = 0
                for j, m in enumerate(monsters):
                    if m.get("current_hp", 0) > 0:
                        target = j
                        break
                return f"PLAY {i + 1} {target}"
            return f"PLAY {i + 1}"
    return "END"


# ---------- 宏观决策 ----------

def macro_action(state: dict) -> str:
    """宏观决策：用可用命令列表做简单规则。"""
    cmds = state.get("available_commands", [])
    game = state.get("game_state", {})
    screen_type = game.get("screen_type", "")

    # 选路
    if "CHOOSE" in cmds and screen_type == "MAP":
        # 优先精英（血够）或问号，否则第一个
        choices = game.get("choice_list", [])
        for i, ch in enumerate(choices):
            if "elite" in str(ch).lower():
                return f"CHOOSE {i}"
        return "CHOOSE 0"

    # 卡牌奖励：拿第一张或跳过
    if screen_type == "CARD_REWARD":
        cards = game.get("cards", [])
        if cards:
            return "CHOOSE 0"
        return "SKIP"

    # 篝火：血量低睡觉，否则升牌
    if screen_type == "REST":
        options = game.get("options", [])
        cur = game.get("current_hp", 80)
        maxhp = game.get("max_hp", 80)
        if cur / maxhp < 0.4:
            for i, o in enumerate(options):
                if "rest" in str(o).lower():
                    return f"CHOOSE {i}"
        for i, o in enumerate(options):
            if "smith" in str(o).lower():
                return f"CHOOSE {i}"
        return "CHOOSE 0"

    # 通用兜底
    for cmd in ("PROCEED", "CONFIRM", "CONTINUE", "LEAVE"):
        if cmd in cmds:
            return cmd
    if "CHOOSE" in cmds:
        return "CHOOSE 0"
    if "SKIP" in cmds:
        return "SKIP"
    if "END" in cmds:
        return "END"
    return "STATE"


def decide(state: dict) -> str:
    if state.get("error"):
        log.warning("game error: %s", state["error"])
    if not state.get("ready_for_command"):
        return "STATE"

    cmds = state.get("available_commands", [])
    game = state.get("game_state", {})
    screen = game.get("screen_type", "")

    # 战斗
    if screen == "COMBAT" or ("PLAY" in cmds and game.get("hand")):
        return combat_action(state)

    # 游戏未开始
    if "START" in cmds:
        return "START IRONCLAD"

    return macro_action(state)


def main():
    sys.stdout.write("ready\n")
    sys.stdout.flush()
    log.info("sts-agent bridge ready")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            log.error("bad json: %s", line[:200])
            continue

        cmd = decide(state)
        log.debug("screen=%s -> %s",
                  state.get("game_state", {}).get("screen_type"), cmd)
        sys.stdout.write(cmd + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
