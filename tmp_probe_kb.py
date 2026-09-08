"""临时探测 v4：确认 GameContext 是否暴露药水/遗物奖励的"名字"。"""
import slaythespire as sts
from env.lightspeed_env import LightspeedEnv

gc = sts.GameContext(sts.CharacterClass.IRONCLAD, 1, 0)
print("=== 完整 dir(gc) ===")
all_attrs = [x for x in dir(gc) if not x.startswith("__")]
for a in all_attrs:
    print("  ", a)
print()

# 探查 potion 相关属性
print("=== potion 相关属性类型/值 ===")
for a in ("potion_count", "potion_capacity", "potion_slots", "potion_chance"):
    try:
        v = getattr(gc, a)
        print(f"  {a} = {v!r} (type={type(v).__name__})")
    except Exception as e:
        print(f"  {a} -> ERR {e!r}")

# 跑一局，停在含 POTION/RELIC 奖励的屏幕，看 gc 是否能给出奖励项名字
print()
print("=== 停在 REWARDS(RELIC/POTION) 与 BOSS_RELIC 屏幕，探查奖励项 ===")
env = LightspeedEnv(mcts_simulations=50)
from models.candidate_encoder import decode_action_bits

def probe_screen(env):
    g = env.gc
    print(f"  screen={g.screen_state.name}")
    # 尝试各种可能的奖励访问器
    for meth in ("get_card_reward", "pick_reward_card", "skip_reward_cards"):
        pass
    # 列出所有带 reward/potion/relic 的方法并调用无参的
    for a in dir(g):
        if a.startswith("__"):
            continue
        if any(k in a.lower() for k in ("reward", "potion", "relic")) and callable(getattr(g, a)):
            try:
                r = getattr(g, a)()
                print(f"    {a}() -> {r!r}")
            except Exception as e:
                print(f"    {a}() -> ERR {type(e).__name__}: {e}")

for seed in (42,):
    env.reset(seed)
    for _ in range(400):
        cands = env.candidates()
        if not cands:
            break
        has_pr = any(decode_action_bits(env._actions[i].bits if i < len(env._actions) else 0)
                     for i in range(len(cands)))
        # 简单判断：本帧是否有 POTION/RELIC 候选
        frame_bits = [a.bits for a in env._actions]
        if any((b >> 27) & 0x7 in (3, 4) for b in frame_bits):
            print(f"\n[seed={seed}] 帧含 POTION/RELIC 奖励候选:")
            probe_screen(env)
        env.step(0)
        if env.gc.outcome != sts.GameOutcome.UNDECIDED:
            break

# 直接测试 potion_slots 内容
print()
print("=== 直接看 potion_slots 的元素结构 ===")
try:
    ps = gc.potion_slots
    print("potion_slots =", ps)
    if ps:
        for el in ps:
            print("  elem:", repr(el), "attrs:", [x for x in dir(el) if not x.startswith("__")][:40])
except Exception as e:
    print("potion_slots ERR:", e)
