"""桥接协议自检：把 main.py 当子进程拉起，按 Communication Mod 的协议对打。

**为什么需要**：真机桥接是「stdin 收 JSON / stdout 发命令」的进程协议，
一旦协议层写错，表现是游戏卡死或空转——而真机一局要 20–40 分钟，试错极贵。
本脚本在**没有游戏**的情况下把协议层跑通：

  1. 启动后必须立刻打印 "ready"（Communication Mod 靠它判定就绪）
  2. 主菜单状态（available_commands 含 start）→ 必须回 "START IRONCLAD"
  3. ready_for_command=false → 必须回 "STATE"（不空转、不猜）
  4. 畸形 JSON → 不能崩
  5. **防卡死守卫**：连续喂同一个「不会推进」的状态 25 次 → 必须触发 stall
     事件并最终主动退出（而不是像历史上那样空转 5853 次）

用法：
    KMP_DUPLICATE_LIB_OK=TRUE .venv/Scripts/python.exe scripts/selfcheck_bridge.py
"""
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
_CKPT = os.path.join(_ROOT, "checkpoints", "ppo_v9_10000.pt")


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

    def send(self, obj) -> None:
        self.proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def send_raw(self, text: str) -> None:
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def recv(self, timeout=10.0):
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


def _state_start():
    return {"available_commands": ["start"], "ready_for_command": True,
            "in_game": False}


def _state_not_ready():
    return {"available_commands": ["state"], "ready_for_command": False,
            "in_game": True,
            "game_state": {"screen_type": "COMBAT", "floor": 1, "act": 1,
                           "current_hp": 70, "max_hp": 80, "gold": 99}}


def _state_stuck():
    """一个不会让游戏推进的状态（模拟器无法处理 → 规则降级 → 命令无效）。

    关键点：连续喂它时游戏状态**完全不变**，正是历史上死循环的场景。
    """
    return {"available_commands": ["proceed"], "ready_for_command": True,
            "in_game": True,
            "game_state": {"screen_type": "SHOP_ROOM", "floor": 5, "act": 1,
                           "current_hp": 60, "max_hp": 80, "gold": 120,
                           "choice_list": []}}


def main():
    log_dir = tempfile.mkdtemp(prefix="selfcheck_")
    print(f"[selfcheck] 日志目录: {log_dir}")
    b = Bridge(log_dir)
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
        ok = ok and cond

    # 1) ready
    first = b.recv(timeout=60)
    check("启动打印 ready", first == "ready", f"got={first!r}")
    for line in b.err[:3]:
        print(f"        stderr: {line}")

    # 2) 主菜单 → START
    b.send(_state_start())
    r = b.recv()
    check("主菜单回 START IRONCLAD", r == "START IRONCLAD", f"got={r!r}")

    # 3) ready_for_command=false → STATE
    b.send(_state_not_ready())
    r = b.recv()
    check("未就绪回 STATE", r == "STATE", f"got={r!r}")

    # 4) 畸形 JSON 不崩
    b.send_raw("{not json at all")
    b.send(_state_start())
    r = b.recv(timeout=10)
    check("畸形 JSON 后仍能正常响应", r == "START IRONCLAD", f"got={r!r}")

    # 5) 防卡死守卫：连喂同一个不推进的状态
    #    真机 01:30 血案后调整语义：守卫 hard_limit 后**不再退出**，改为
    #    keep-alive（进程活着、只回 STATE）——退出意味着用户必须重启游戏才能
    #    恢复；keep-alive 下用户手动推进或状态变化，agent 立即自动恢复。
    stuck = _state_stuck()
    replies = []
    for _ in range(40):
        b.send(stuck)
        rr = b.recv(timeout=20)
        if rr is None:
            break
        replies.append(rr)
        if not b.proc.poll() is None:
            break
    time.sleep(1.0)
    guard_msg = [l for l in b.err if "[guard]" in l]
    check("触发防卡死守卫", len(guard_msg) >= 1,
          f"guard 消息 {len(guard_msg)} 条")
    if guard_msg:
        print(f"        {guard_msg[0]}")
    check("转入 keep-alive 而不退出（进程存活可人工接管）",
          b.proc.poll() is None, f"returncode={b.proc.poll()}")
    check("keep-alive 阶段只回 STATE（不再拿无效命令打游戏）",
          len(replies) >= 3 and all(r == "STATE" for r in replies[-3:]),
          f"尾部回复={replies[-3:]}")

    # 6) 事件日志落盘
    evt_files = [f for f in os.listdir(log_dir) if f.endswith("_events.jsonl")]
    stall_evt = False
    if evt_files:
        with open(os.path.join(log_dir, evt_files[0]), encoding="utf-8") as f:
            for line in f:
                if '"stall"' in line:
                    stall_evt = True
    check("stall 事件写入 L3 日志", stall_evt)
    giveup_evt = False
    if evt_files:
        with open(os.path.join(log_dir, evt_files[0]), encoding="utf-8") as f:
            for line in f:
                if '"stall_giveup"' in line:
                    giveup_evt = True
    check("stall_giveup 事件写入 L3 日志", giveup_evt)

    st_files = [f for f in os.listdir(log_dir) if f.endswith("_states.jsonl")]
    n_states = 0
    if st_files:
        with open(os.path.join(log_dir, st_files[0]), encoding="utf-8") as f:
            n_states = sum(1 for _ in f)
    check("L1 全量状态落盘", n_states > 0, f"{n_states} 条")

    b.close()
    print(f"\n[selfcheck] {'全部通过 ✅' if ok else '存在失败项 ❌'}")
    print(f"[selfcheck] 详细日志: {log_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
