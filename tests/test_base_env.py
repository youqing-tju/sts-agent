from env.base_env import Obs, Candidate, StepResult

def test_obs_dataclass():
    o = Obs(vec=[0.0]*412, hp=70, max_hp=80, floor=5, gold=99,
            deck=["Strike"]*4, relics=["Burning Blood"], screen="MAP")
    assert o.hp == 70 and len(o.vec) == 412

def test_candidate_dataclass():
    c = Candidate(idx=0, kind="CARD_REWARD", desc_vec=[0.0]*408, label="Strike")
    assert c.kind == "CARD_REWARD"
