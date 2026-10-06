"""Probe 3: 决定性确认——gc 上到底有没有任何携带"当前事件身份"的属性/方法。

- 枚举 gc 全部属性名与方法名，过滤含 event/screen/room/title/name/option 的
- 打印实际的 cur_room.name / encounter.name（当前值，不是枚举成员列表）
- 尝试调用疑似方法：get_event / event / current_event / get_event_name 等
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


def main():
    gc = find_event(111)
    print("=== PROBE 3: gc 事件身份字段决定性探查 ===\n")
    print("gc.screen_state =", gc.screen_state.name)
    print("gc.cur_room (实际值) =", getattr(gc.cur_room, "name", gc.cur_room))
    print("gc.encounter (实际值) =", getattr(gc.encounter, "name", gc.encounter))
    print("gc.boss =", gc.boss)
    print("gc.outcome =", gc.outcome.name)

    print("\n-- gc 全部属性名（含下划线）中含 event/room/screen/title/option/name --")
    for attr in dir(gc):
        low = attr.lower()
        if any(w in low for w in ("event", "room", "screen", "title", "option", "name", "dialog", "page")):
            kind = "CALLABLE" if callable(getattr(gc, attr, None)) else "FIELD"
            print(f"   [{kind}] {attr}")

    print("\n-- 尝试调用疑似拿事件名的方法（带/不带 gc 参数） --")
    candidates = ["get_event", "event", "current_event", "get_event_name",
                  "event_name", "get_room", "get_current_room", "room",
                  "get_screen", "get_encounter", "encounter_name",
                  "get_event_screen", "event_screen", "get_dialog", "dialog"]
    for c in candidates:
        if hasattr(gc, c):
            try:
                v = getattr(gc, c)
                if callable(v):
                    try:
                        r = v()
                        print(f"   gc.{c}() -> {r!r}  (.name={getattr(r,'name',None)})")
                    except Exception as e:
                        try:
                            r = v(gc)
                            print(f"   gc.{c}(gc) -> {r!r}  (.name={getattr(r,'name',None)})")
                        except Exception as e2:
                            print(f"   gc.{c}() 调用失败: {e!r} / {e2!r}")
                else:
                    print(f"   gc.{c} = {v!r}")
            except Exception as e:
                print(f"   gc.{c} 访问失败: {e!r}")
        else:
            print(f"   gc.{c} -> 不存在")

    # screen_state 对象本身有没有携带事件
    print("\n-- gc.screen_state 对象探查 --")
    print("   type:", type(gc.screen_state).__name__)
    for attr in dir(gc.screen_state):
        if attr.startswith("_"):
            continue
        try:
            v = getattr(gc.screen_state, attr)
        except Exception:
            continue
        if not callable(v):
            print(f"   .{attr} = {v!r}")


if __name__ == "__main__":
    main()
