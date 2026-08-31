import json, os
from logging.run_logger import RunLogger

def test_logger_writes_jsonl(tmp_path):
    lg = RunLogger(log_dir=str(tmp_path), seed=7)
    lg.log_decision(floor=3, screen="CARD_REWARD",
                    candidates=["Strike","Defend","SKIP"], chosen="Strike", hp=70)
    lg.log_combat(floor=4, hp_in=70, hp_out=62, kind="NORMAL")
    lg.finish(final_floor=12, win=False)
    lines = open(lg.path, encoding="utf-8").read().strip().split("\n")
    recs = [json.loads(l) for l in lines]
    assert recs[0]["type"] == "decision" and recs[1]["type"] == "combat"
    assert recs[-1]["type"] == "outcome" and recs[-1]["final_floor"] == 12
