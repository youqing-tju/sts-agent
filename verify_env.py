"""冒烟测试：用内置 MCTS Agent 打完一局，打印结果。

用法: python verify_env.py [seed]
"""

import sys

import slaythespire as sts


def main(seed: int = 12345) -> None:
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    agent = sts.Agent()
    agent.simulation_count_base = 1000  # 冒烟用低预算，快
    agent.pause_on_card_reward = False  # 让 Agent 全程自主决策
    agent.print_logs = False

    steps = 0
    while gc.outcome == sts.GameOutcome.UNDECIDED and steps < 500:
        agent.playout(gc)
        steps += 1

    print(f"seed={seed} floor={gc.floor_num} act={gc.act} outcome={gc.outcome} steps={steps}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12345)
