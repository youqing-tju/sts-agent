"""sts-agent 真实游戏入口：Communication Mod 桥接。

协议（Communication Mod）：
- 启动后向 stdout 打印 "ready"
- 之后每个稳定游戏状态会通过 stdin 收到一行 JSON
- 向 stdout 写动作命令（PLAY / END / CHOOSE / START 等）

M3 策略分配：
- 宏观决策（选路/选牌/商店/篝火）→ RulePolicy
- 战斗出牌 → 简单启发式（能出攻击就出，没费了就结束回合）

注意：训练好的 RL 策略暂不能直接用在真实游戏——它的输入是 lightspeed 的
412 维观测 + 408 维候选编码（依赖模拟器内部 bits），需要单独的 JSON→特征
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
    """极简战斗策略：按顺序出能出的牌，出不了就先用药水，最后 END。

    Communication Mod 的战斗数据在 game_state.combat_state 里。
    """
    game = state.get("game_state", {})
    combat = game.get("combat_state", game)  # 兼容两种结构
    hand = combat.get("hand", [])
    energy = combat.get("player", {}).get("energy", 0)

    for i, card in enumerate(hand):
        cost = card.get("cost", 99)
        if cost < 0:
            cost = 0  # 诅咒/特殊牌 cost=-1
        if cost <= energy and card.get("is_playable", True):
            if card.get("type", "").upper() == "ATTACK":
                monsters = combat.get("monsters", [])
                target = 0
                for j, m in enumerate(monsters):
                    if m.get("current_hp", 0) > 0 and not m.get("is_gone", False):
                        target = j
                        break
                return f"PLAY {i + 1} {target}"
            return f"PLAY {i + 1}"

    # 没牌可出时用药水（Communication Mod：POTION USE <槽位> [目标]）
    for i, p in enumerate(game.get("potions", [])):
        if not p.get("can_use") or p.get("id") in (None, "Potion Slot"):
            continue
        if p.get("requires_target"):
            target = 0
            for j, m in enumerate(combat.get("monsters", [])):
                if m.get("current_hp", 0) > 0 and not m.get("is_gone", False):
                    target = j
                    break
            return f"POTION USE {i} {target}"
        return f"POTION USE {i}"
    return "END"


# ---------- 宏观决策 ----------

def macro_action(state: dict) -> str:
    """宏观决策：只用 available_commands 里实际存在的命令（小写）。"""
    cmds = [c.lower() for c in state.get("available_commands", [])]
    game = state.get("game_state", {})
    screen_type = (game.get("screen_type") or "").upper()

    # 选路
    if "choose" in cmds and screen_type == "MAP":
        choices = game.get("choice_list", [])
        for i, ch in enumerate(choices):
            if "elite" in str(ch).lower():
                return f"CHOOSE {i}"
        return "CHOOSE 0"

    # 卡牌奖励
    if screen_type == "CARD_REWARD":
        if "choose" in cmds:
            return "CHOOSE 0"      # 拿第一张牌
        if "skip" in cmds:
            return "SKIP"
        if "proceed" in cmds:
            return "PROCEED"

    # 篝火（注意：只有 choose 在可用命令里时才发 CHOOSE）
    if screen_type == "REST" and "choose" in cmds:
        options = game.get("choice_list", [])
        cur = game.get("current_hp", 80)
        maxhp = game.get("max_hp", 80)
        if maxhp and cur / maxhp < 0.4:
            for i, o in enumerate(options):
                if "rest" in str(o).lower():
                    return f"CHOOSE {i}"
        for i, o in enumerate(options):
            if "smith" in str(o).lower():
                return f"CHOOSE {i}"
        return "CHOOSE 0"

    # 战斗奖励：逐个领取金币/药水/遗物/卡牌，领完再 PROCEED
    if screen_type == "COMBAT_REWARD":
        if "choose" in cmds:
            return "CHOOSE 0"  # 逐个领取所有奖励（金币/药水/遗物先出现）
        if "proceed" in cmds:
            return "PROCEED"

    # 宝箱 / Boss 遗物：必须先拿（choose 优先于 proceed，否则兜底链会跳过奖励）
    if screen_type in ("CHEST", "BOSS_REWARD"):
        if "choose" in cmds:
            return "CHOOSE 0"
        if "proceed" in cmds:
            return "PROCEED"

    # 网格选牌屏（篝火升牌 / 商店删牌 / 事件变化牌等）
    if screen_type == "GRID":
        ss = game.get("screen_state", {})
        cards = ss.get("cards", [])
        if "choose" in cmds and cards:
            if ss.get("for_purge"):
                # 删牌：优先删打击/防御这类基础牌
                for i, c in enumerate(cards):
                    if str(c.get("id", "")).split("_")[0].lower() in ("strike", "defend"):
                        return f"CHOOSE {i}"
                return "CHOOSE 0"
            # 升牌：优先升级非打击的高费卡（打击升级收益最低）
            best, best_key = 0, None
            for i, c in enumerate(cards):
                cid = str(c.get("id", "")).lower()
                is_strike = cid.startswith("strike")
                key = (not is_strike, c.get("cost", 0) or 0)
                if best_key is None or key > best_key:
                    best, best_key = i, key
            return f"CHOOSE {best}"
        if "confirm" in cmds:
            return "CONFIRM"
        if "skip" in cmds:
            return "SKIP"

    # 通用兜底——只从实际可用的命令里选
    for cmd in ("proceed", "confirm", "continue", "leave"):
        if cmd in cmds:
            return cmd.upper()
    if "choose" in cmds:
        return "CHOOSE 0"
    if "skip" in cmds:
        return "SKIP"
    if "end" in cmds:
        return "END"
    return "STATE"


def decide(state: dict) -> str:
    if state.get("error"):
        log.warning("game error: %s", state["error"])
    if not state.get("ready_for_command"):
        return "STATE"

    # Communication Mod 的 available_commands 全是小写
    cmds = [c.lower() for c in state.get("available_commands", [])]
    game = state.get("game_state", {})
    screen = (game.get("screen_type") or "").upper()

    # 战斗：Communication Mod 在战斗中 screen_type 是 NONE，靠 "play" 命令判断
    if "play" in cmds:
        return combat_action(state)

    # 游戏未开始（主菜单）
    if "start" in cmds:
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

        # 把最近一次状态原样写盘，方便我们检查真实 JSON 结构
        try:
            with open("last_state.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=1)
        except Exception:
            pass

        cmd = decide(state)
        log.debug("screen=%s -> %s",
                  state.get("game_state", {}).get("screen_type"), cmd)
        if cmd == "STATE":
            import time
            time.sleep(0.1)  # 避免在主菜单等无操作状态下疯狂自旋
        sys.stdout.write(cmd + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
