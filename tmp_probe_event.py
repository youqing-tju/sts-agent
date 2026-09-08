"""Probe: 搞清楚事件候选在 C++ 模拟器里到底能拿到什么信息。

目标三个问题：
1. 事件候选的 label 到底是什么？（选项文本 / 事件名 / 空串回退 action_0x...?）
2. 能不能拿到当前是哪个事件？（gc 上有没有 event id / 名字字段？）
3. 一个事件内多个选项的 bits 有什么规律？（idx1 是不是选项序号？）

注意：不能用 agents/LightspeedEnv 这类业务封装（会引入 desc_vec 依赖），
这里直接拿裸 sts.GameContext，镜像 lightspeed_env._advance 的暂停逻辑。
"""
import sys
import slaythespire as sts

# Windows 上 OMP 重复加载保护（可选）
try:
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
except Exception:
    pass


def dump_gc_fields(gc):
    """列出 gc 上所有看起来像事件/id/name 的字段，帮助判断能否拿事件名。"""
    fields = {}
    for attr in dir(gc):
        if attr.startswith("_"):
            continue
        try:
            v = getattr(gc, attr)
        except Exception as e:
            continue
        if callable(v):
            continue
        # 只挑可能跟事件/当前屏幕相关的字符串/枚举型
        if isinstance(v, str):
            fields[attr] = v
        elif hasattr(v, "name") and isinstance(getattr(v, "name", None), str):
            fields[attr] = f"<{type(v).__name__}:{v.name}>"
    return fields


def decode_bits(bits):
    return {
        "is_potion": bool(bits & 0x80000000),
        "is_potion_discard": bool(bits & 0x40000000),
        "reward_type": (bits >> 27) & 0x7,
        "idx3": (bits >> 16) & 0xFF,
        "idx2": (bits >> 8) & 0xFF,
        "idx1": bits & 0xFF,
    }


def find_event(seed):
    gc = sts.GameContext(sts.CharacterClass.IRONCLAD, seed, 0)
    agent = sts.Agent()
    agent.simulation_count_base = 200
    agent.pause_on_noncombat = True  # 关键：非战斗决策点暂停，事件屏会停在这里

    for _ in range(2000):
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            return None
        agent.playout(gc)
        if gc.outcome != sts.GameOutcome.UNDECIDED:
            return None
        if gc.screen_state.name == "EVENT_SCREEN":
            return gc
        # 非事件屏一定有候选或被 playout 自动消化；EVENT_SCREEN 才是我们要找的
    return None


def main():
    print("=== PROBE: 事件候选真实形态 ===\n")
    found = []
    seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999, 1234,
             12345, 55555, 88888, 2024, 31415, 27182, 42424, 65535,
             1001, 2002, 3003, 4004, 5005, 6006, 7007, 8008]
    for seed in seeds:
        gc = find_event(seed)
        if gc is not None:
            found.append((seed, gc))
            if len(found) >= 3:  # 抓到 3 个事件够看了
                break

    if not found:
        print("未在任何测试 seed 上命中 EVENT_SCREEN（事件概率 ~70%，但可能种子不凑巧）")
        return

    for seed, gc in found:
        print(f"\n########## seed={seed} ##########")
        print("screen_state:", gc.screen_state.name)
        print("floor/act:", gc.floor_num, gc.act)

        # 问题2：gc 上能拿到的所有字符串/枚举字段（找事件名）
        fields = dump_gc_fields(gc)
        # 只打印可能相关的
        keys_of_interest = [k for k in fields if any(
            w in k.lower() for w in ("event", "screen", "state", "name", "id", "room", "current", "map", "node"))]
        print("\n-- gc 上可能相关的字段 --")
        for k in keys_of_interest:
            print(f"   {k} = {fields[k]}")

        # 列出全部字段名（便于人工发现漏网的事件字段）
        print("\n-- gc 全部非 callable 字段名 --")
        print("  ", [k for k in fields.keys()])

        # 问题1 & 3：候选的 label / kind / bits
        acts = sts.get_all_actions(gc)
        print(f"\n-- 候选数 = {len(acts)} --")
        for i, a in enumerate(acts):
            raw = a.describe(gc)
            label = raw or f"action_0x{a.bits:08x}"
            d = decode_bits(a.bits)
            print(f"  [{i}] kind_bits(idx1={d['idx1']},idx2={d['idx2']},idx3={d['idx3']},"
                  f"rt={d['reward_type']},potion={d['is_potion']})")
            print(f"      describe()  = {raw!r}")
            print(f"      label(回退) = {label!r}")
            print(f"      bits=0x{a.bits:08x}")

        # 尝试访问可能的事件名字段（多种猜测名）
        print("\n-- 尝试常见事件字段名 --")
        for guess in ("event_name", "current_event", "event", "event_id",
                      "get_event_name", "event_title", "room_event"):
            try:
                if hasattr(gc, guess):
                    val = getattr(gc, guess)
                    print(f"   gc.{guess} EXISTS -> {val!r}")
                else:
                    print(f"   gc.{guess} -> 不存在")
            except Exception as e:
                print(f"   gc.{guess} -> 访问异常 {e!r}")


if __name__ == "__main__":
    main()
