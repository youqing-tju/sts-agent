from evaluation.evaluate import make_eval_seeds, summarize


def test_eval_seeds_deterministic_and_disjoint():
    a = make_eval_seeds(100)
    b = make_eval_seeds(100)
    assert a == b and len(set(a)) == 100


def test_summarize():
    s = summarize([{"final_floor": 10, "win": False},
                   {"final_floor": 50, "win": True}])
    assert s["avg_floor"] == 30.0 and s["win_rate"] == 0.5
