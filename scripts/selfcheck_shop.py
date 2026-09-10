# -*- coding: utf-8 -*-
"""真机商店血案的端到端回归：把 main.py 当子进程，按「历史死循环序列」对打。

血案回放（0911_001847 原始帧，两段卡死合计 29.5 分钟）：
  SHOP_SCREEN(gold 146) --买 purge(75)--> GRID 选牌 --> CONFIRM
  --> SHOP_SCREEN(gold 71) --买卡--> SHOP_SCREEN(无可买) --LEAVE-->
  SHOP_ROOM  ← ❌旧版在这里 CHOOSE 0 重开商店（逐帧死循环）
             ← ✅新版必须 PROCEED 走出商店 --> MAP

断言：
  1. 进店第一帧不是 CHOOSE（不盲点），是 STATE
  2. 商店界面：先买 purge、再买一张过阈值卡、然后 LEAVE
  3. SHOP_ROOM：必须 PROCEED；**全程任何 SHOP_ROOM 帧都不允许出现 CHOOSE**
  4. L2 决策日志包含 screen_raw=SHOP_ROOM 的记录（旧版这些帧完全不落盘）
  5. L3 事件含 shop_buy(purge/card) / shop_leave / shop_exit

用法：
    KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python.exe scripts/selfcheck_shop.py
"""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
_CKPT = os.path.join(_ROOT, "checkpoints", "ppo_v9_10000.pt")
_FIX = os.path.join(_ROOT, "tests", "fixtures")


def _load(name):
    with open(os.path.join(_FIX, name), encoding="utf-8") as f:
        return json.load(f)


class Bridge:
    def __init__(self, log_dir):
        env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE",
                   PYTHONIOENCODING="utf-8")
        self.proc = subprocess.Popen(
            [_PY, os.path.join(_ROOT, "main.py"),
             "--checkpoint", _CKPT, "--log-dir", log_dir],
            cwd=_ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            errors="replace", bufsize=1)
        self.q = queue.Queue()
        self.err = []
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.q),
                         daemon=True).start()
        threading.Thread(target=self._pump, args=(self.proc.stderr, self.err),
                         daemon=True).start()

    @staticmethod
    def _pump(stream, sink):
        try:
            for line in stream:
                if isinstance(sink, queue.Queue):
                    sink.put(line.rstrip("\n"))
                else:
                    sink.append(line.rstrip("\n"))
        except Exception:
            pass

    def send(self, obj):
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout=90.0):
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def _grid_confirm_frame(grid):
    g2 = json.loads(json.dumps(grid))
    g2["available_commands"] = ["potion", "confirm", "cancel", "key", "click",
                                "wait", "state"]
    g2["game_state"]["screen_state"]["selected_cards"] = \
        [g2["game_state"]["screen_state"]["cards"][0]]
    return g2


def _post_purchase_screen():
    """模拟买完 purge 之后的商店帧：金币 71、列表只剩买得起的。"""
    fr = _load("shop_screen.json")
    gs = fr["game_state"]
    ss = gs["screen_state"]
    gs["gold"] = 71
    affordable = [c["name"] for c in ss["cards"]
                  if c["price"] <= 71 and c["name"] in gs["choice_list"]]
    affordable += [p["name"] for p in ss["potions"]
                   if p["price"] <= 71 and p["name"] in gs["choice_list"]]
    gs["choice_list"] = affordable
    return fr


def _empty_screen():
    fr = _load("shop_screen.json")
    fr["game_state"]["gold"] = 17
    fr["game_state"]["choice_list"] = []
    return fr


def _map_state():
    return {"available_commands": ["choose", "return", "key", "click", "wait",
                                   "state"],
            "ready_for_command": True, "in_game": True,
            "game_state": {"screen_type": "MAP", "screen_name": "MAP",
                           "floor": 4, "act": 1, "current_hp": 70,
                           "max_hp": 80, "gold": 17, "choice_list": ["x=1"]}}


def main():
    log_dir = tempfile.mkdtemp(prefix="selfcheck_shop_")
    print(f"[selfcheck-shop] 日志目录: {log_dir}")
    b = Bridge(log_dir)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}"
              + (f"  ({detail})" if detail else ""))
        ok = ok and bool(cond)

    first = b.recv(timeout=90)
    check("启动打印 ready", first == "ready", f"got={first!r}")

    room = _load("shop_room.json")
    screen = _load("shop_screen.json")
    grid = _load("grid.json")

    # 1) 进店：不盲点
    b.send(room)
    r = b.recv()
    check("进店第一帧不是 CHOOSE（等待界面）", r == "STATE", f"got={r!r}")

    # 2) 商店界面：买 purge
    b.send(screen)
    r = b.recv()
    check("商店界面先买删卡服务 CHOOSE 0", r == "CHOOSE 0", f"got={r!r}")

    # 3) 删卡 GRID：选牌 -> 确认
    b.send(grid)
    r = b.recv()
    check("GRID 选牌 CHOOSE 0（删打击）", r == "CHOOSE 0", f"got={r!r}")
    b.send(_grid_confirm_frame(grid))
    r = b.recv()
    check("GRID 确认 CONFIRM", r == "CONFIRM", f"got={r!r}")

    # 4) 买完 purge 后的商店：应再买一张好卡（Shrug It Off）
    b.send(_post_purchase_screen())
    r = b.recv()
    check("再买一张过阈值的卡（CHOOSE 2 = Shrug It Off）", r == "CHOOSE 2",
          f"got={r!r}")

    # 5) 囊中羞涩：离开界面
    b.send(_empty_screen())
    r = b.recv()
    check("无可买 -> LEAVE 关闭商店界面", r == "LEAVE", f"got={r!r}")

    # 6) 房间：走出（核心修复）
    b.send(room)
    r = b.recv()
    check("SHOP_ROOM -> PROCEED 走出商店（旧版此处死循环）", r == "PROCEED",
          f"got={r!r}")

    # 7) 地图：不再与商店纠缠
    b.send(_map_state())
    r = b.recv()
    check("回到地图正常决策", r == "CHOOSE 0", f"got={r!r}")

    b.close()

    # ---- 日志层断言 ----
    dec = [json.loads(x) for x in
           open(os.path.join(log_dir, next(f for f in os.listdir(log_dir)
                                           if f.endswith("_decisions.jsonl"))),
                encoding="utf-8") if x.strip()]
    shop_room_recs = [d for d in dec if d.get("screen_raw") == "SHOP_ROOM"]
    check("L2 记录包含 SHOP_ROOM 帧（旧版完全丢失）",
          len(shop_room_recs) >= 1, f"n={len(shop_room_recs)}")
    check("L2 中不存在 SHOP_ROOM 上的 CHOOSE（核心不变量）",
          not any(str(d.get("cmd", "")).startswith("CHOOSE")
                  for d in shop_room_recs),
          str([d.get("cmd") for d in shop_room_recs]))

    evt_path = os.path.join(log_dir, next(f for f in os.listdir(log_dir)
                                          if f.endswith("_events.jsonl")))
    evts = [json.loads(x) for x in open(evt_path, encoding="utf-8") if x.strip()]
    kinds = [e.get("kind") for e in evts]
    check("L3 含 shop_buy / shop_purge_pick / shop_leave / shop_exit",
          all(k in kinds for k in ("shop_buy", "shop_purge_pick", "shop_leave",
                                   "shop_exit")), str(sorted(set(kinds))))

    print()
    print("[selfcheck-shop] 全部通过 ✅" if ok else "[selfcheck-shop] 有 FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
