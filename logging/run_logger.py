"""每局一个 JSONL 文件：decision / combat / outcome 三类记录。"""
import json, os, time


class RunLogger:
    def __init__(self, log_dir: str, seed: int):
        os.makedirs(log_dir, exist_ok=True)
        self.path = os.path.join(log_dir, f"run_{seed}_{int(time.time())}.jsonl")
        self._f = open(self.path, "a", encoding="utf-8")

    def _write(self, rec: dict):
        self._f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._f.flush()

    def log_decision(self, **kw): self._write({"type": "decision", **kw})
    def log_combat(self, **kw):   self._write({"type": "combat", **kw})
    def finish(self, final_floor: int, win: bool):
        self._write({"type": "outcome", "final_floor": final_floor, "win": win})
        self._f.close()
