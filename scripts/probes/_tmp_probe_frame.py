"""定位含 cards 的选牌帧结构。"""
import json, sys
PATH = sys.argv[1]
def load_line(line):
    try:
        return json.loads(line)
    except Exception:
        raw = line.encode("utf-8", "surrogatepass")
        return json.loads(raw.decode("utf-8", "replace"))

with open(PATH, "r", encoding="utf-8") as f:
    for ln, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = load_line(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        # 找含 cards 列表的对象
        def find_cards(o, path=""):
            if isinstance(o, dict):
                cs = o.get("cards")
                if isinstance(cs, list) and cs and isinstance(cs[0], dict) and ("id" in cs[0] or "cost" in cs[0]):
                    return path, o
                for k, v in o.items():
                    r = find_cards(v, path + "/" + str(k))
                    if r:
                        return r
            elif isinstance(o, list):
                for i, v in enumerate(o[:3]):
                    r = find_cards(v, path + f"[{i}]")
                    if r:
                        return r
            return None
        res = find_cards(obj)
        if res:
            path, ss = res
            print(f"line {ln} cards-path={path}")
            print("  parent keys:", list(ss.keys()))
            print("  first card:", json.dumps(ss["cards"][0], ensure_ascii=False)[:200])
            print("  screen field nearby:", {k: ss.get(k) for k in ("screen", "screen_type", "room_type", "reward_type") if k in ss})
            break
