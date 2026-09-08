"""Probe 2: 深挖事件身份字段 + 确认 describe() 是否对所有事件都空。

重点：
- cur_room / encounter 是不是携带事件名（Room 对象 / Encounter 对象）
- 跑 30 个 seed，统计：多少事件 describe 全空？候选数分布？idx1 是否从 0 连续？
- 是否任何事件能拿到名字
"""
import slaythespire as sts


def find_event(seed):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    agent = sts.Agent()
    agent.simulation_count_base = 200
    agent.pause_on_noncombat = True
    for _ in range(2000):
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            return None
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            return None
        if gc.screen_state.name == "EVENT_SCREEN":
            return gc
    return None


def deep_introspect(obj, name, depth=0, maxdepth=2):
    """递归打印对象字段，找事件名/标题。"""
    if depth > maxdepth:
        return
    if obj is None:
        return
    print("  " * depth + f"[{name}] type={type(obj).__name__}")
    for attr in dir(obj):
        if attr.startswith("_") or attr in ("name",):
            continue
        try:
            v = getattr(obj, attr)
        except Exception:
            continue
        if callable(v):
            continue
        if isinstance(v, str) and v:
            print("  " * (depth + 1) + f".{attr} = {v!r}")
        elif isinstance(v, (int, float, bool)):
            # 跳过纯数字，太多
            pass
        elif hasattr(v, "name"):
            print("  " * (depth + 1) + f".{attr} = <{type(v).__name__}:{getattr(v,'name',None)}>")
            if depth < maxdepth:
                deep_introspect(v, f"{attr}({getattr(v,'name',None)})", depth + 2, maxdepth)


def main():
    print("=== PROBE 2: 事件身份 & describe 普查 ===\n")
    total = 0
    all_empty = 0
    any_text = 0
    option_counts = {}
    samples_with_text = []
    seeds = list(range(100, 260))  # 连续 160 个 seed，覆盖更多事件
    for seed in seeds:
        gc = find_event(seed)
        if gc is None:
            continue
        total += 1
        acts = sts.get_all_actions(gc)
        labels = [a.describe(gc) for a in acts]
        nonempty = [l for l in labels if l]
        if nonempty:
            any_text += 1
            samples_with_text.append((seed, acts, labels))
        else:
            all_empty += 1
        option_counts[len(acts)] = option_counts.get(len(acts), 0) + 1
        # 检查 idx1 是否从 0 连续
        idxs = [a.bits & 0xFF for a in acts]
        if sorted(idxs) != list(range(len(acts))):
            print(f"  seed={seed}: idx1 不连续 {idxs}")

    print(f"命中事件屏总数: {total}")
    print(f"describe 全空: {all_empty}  有非空文本: {any_text}")
    print(f"候选数分布: {option_counts}")
    print(f"idx1 是否每个事件从 0 连续: 见上方提示（无提示=均连续）")

    if samples_with_text:
        print("\n-- 含非空 describe 的事件样本 --")
        for seed, acts, labels in samples_with_text[:5]:
            print(f"seed={seed}")
            for a, l in zip(acts, labels):
                print(f"   bits=0x{a.bits:08x} describe={l!r}")
    else:
        print("\n结论: 全部事件 describe() 均返回空串 → label 必然回退为 action_0x<bits>")

    # 深度探查 cur_room / encounter 是否携带事件信息
    print("\n-- 深度探查 cur_room / encounter --")
    for seed in [111, 130, 200]:
        gc = find_event(seed)
        if gc is None:
            continue
        print(f"\nseed={seed}:")
        deep_introspect(gc.cur_room, "cur_room", 0, 1)
        deep_introspect(gc.encounter, "encounter", 0, 1)


if __name__ == "__main__":
    main()
