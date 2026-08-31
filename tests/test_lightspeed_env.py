from env.lightspeed_env import LightspeedEnv

def test_reset_and_candidates():
    env = LightspeedEnv(mcts_simulations=200)
    obs = env.reset(seed=42)
    assert obs.hp > 0 and obs.floor >= 0
    cands = env.candidates()
    assert len(cands) > 0            # 开局必然有宏观决策（选起始奖励或选路）

def test_full_random_game_terminates():
    import random
    env = LightspeedEnv(mcts_simulations=200)
    env.reset(seed=42)
    done, steps = False, 0
    while not done and steps < 500:
        cands = env.candidates()
        if not cands:                # 战斗内部，env 自动跑 MCTS
            res = env.step(-1)
        else:
            res = env.step(random.choice(cands).idx)
        done = res.done
        steps += 1
    assert done                      # 随机玩也必须在 500 步内结束
