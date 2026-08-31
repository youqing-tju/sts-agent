from agents.rule_policy import RulePolicy
from env.base_env import Candidate


def c(idx, kind, label):
    return Candidate(idx=idx, kind=kind, desc_vec=[], label=label)


def test_skip_bad_card_when_deck_big():
    p = RulePolicy()
    class FakeObs: deck = ["Strike"]*20; hp=70; max_hp=80; floor=5; gold=0
    chosen = p.choose(FakeObs(), [c(0,"CARD_REWARD","Strike"), c(1,"CARD_REWARD","Defend"), c(2,"SKIP","SKIP")])
    assert chosen == 2                     # 卡组已臃肿 → 跳过


def test_rest_when_low_hp():
    p = RulePolicy()
    class FakeObs: deck=["Strike"]*4; hp=25; max_hp=80; floor=5; gold=0
    chosen = p.choose(FakeObs(), [c(0,"REST_OPTION","Rest"), c(1,"REST_OPTION","Smith")])
    assert chosen == 0                     # 血量 <40% → 睡觉


def test_map_prefers_elite_when_high_hp():
    p = RulePolicy()
    class FakeObs: deck=["Strike"]*4; hp=70; max_hp=80; floor=5; gold=0
    chosen = p.choose(FakeObs(), [c(0,"MAP_NODE","Monster"), c(1,"MAP_NODE","Unknown"), c(2,"MAP_NODE","Elite")])
    assert chosen == 2                     # hp>70% → 优先精英


def test_map_prefers_rest_when_low_hp():
    p = RulePolicy()
    class FakeObs: deck=["Strike"]*4; hp=25; max_hp=80; floor=5; gold=0
    chosen = p.choose(FakeObs(), [c(0,"MAP_NODE","Monster"), c(1,"MAP_NODE","Rest Site"), c(2,"MAP_NODE","Elite")])
    assert chosen == 1                     # hp<40% 且有篝火方向 → 优先生存向


def test_shop_always_skip():
    p = RulePolicy()
    class FakeObs: deck=["Strike"]*4; hp=70; max_hp=80; floor=5; gold=300
    chosen = p.choose(FakeObs(), [c(0,"SHOP_ITEM","Apotheosis"), c(1,"SKIP","SKIP")])
    assert chosen == 1                     # v1 保守：直接 SKIP


def test_event_pick_first():
    p = RulePolicy()
    class FakeObs: deck=["Strike"]*4; hp=70; max_hp=80; floor=5; gold=0
    chosen = p.choose(FakeObs(), [c(0,"EVENT_OPTION","[Leave]"), c(1,"EVENT_OPTION","action_0x00000001")])
    assert chosen == 0                     # v1 选第一个


def test_take_attack_card_when_deck_small():
    p = RulePolicy()
    class FakeObs: deck=["Strike"]*4; hp=70; max_hp=80; floor=5; gold=0
    chosen = p.choose(FakeObs(), [c(0,"CARD_REWARD","Strike"), c(1,"CARD_REWARD","Clash"), c(2,"SKIP","SKIP")])
    assert chosen != 2                     # 卡组 <15 → 不跳过（拿攻击牌）
