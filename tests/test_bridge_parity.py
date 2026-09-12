"""真机桥接路径 ↔ 训练路径的等价性测试（不依赖真实游戏）。

**为什么必须有**：`main.py`（真机桥接）与 `env/lightspeed_env.py`（训练 env）
是两套独立实现的"状态 → 模型输入"转换。一旦它们不一致，模型在真机上拿到的
输入与训练分布不同 → 行为整体错位，而且**极难排查**（表现为"通关率莫名下降"）。

本测试用模拟器生成真实决策点，断言两条路径产出**逐元素相同**的：
  1. 观测向量（477 维，含 enhance_observation 的增强段）
  2. 牌组流派强度（5 维，供候选交叉项）
  3. 候选描述向量（369 维，含 KB 三段 + V10 SKIP 牌组上下文段）

这样"真机与训练一致"就不再靠人工比对，而是可回归验证的。
"""
import pytest

from models.candidate_encoder import encode_candidate, DESC_VEC_DIM
from models.obs_encoder import enhance_observation, encode_deck_stats, encode_skip_deck_context
from models.card_index import CARD_INDEX
from models.card_data import CARD_ARCHETYPE, CARD_EFFECTS
from models.candidate_kb import candidate_kind, resolve_candidate_kb
from models.candidate_encoder import decode_action_bits
from env.base_env import Candidate

pytestmark = pytest.mark.filterwarnings("ignore")


def _collect_decisions(seeds=(20260831, 20260832, 20260833), max_depth=8):
    """跑模拟器，收集若干真实决策点（env 为参考实现）。

    注意：每个决策点用**独立的 env**（新建 → 推进到第 k 步 → 采集后即丢弃）。
    否则同一 env 被后续 step 推进，存下来的引用会指向错误的游戏状态
    （本测试第一版就栽在这里：kind 断言拿到了错位的屏幕）。
    """
    from env.lightspeed_env import LightspeedEnv

    out = []
    for seed in seeds:
        for k in range(max_depth):
            env = LightspeedEnv(mcts_simulations=200)
            try:
                env.reset(seed)
                for _ in range(k):
                    cands = env.candidates()
                    if not cands:
                        break
                    env.step(cands[0].idx)
                cands = env.candidates()
            except Exception:
                break
            if not cands:
                break
            out.append(env)          # 该 env 停在第 k 个决策点，之后不再推进
    return out


@pytest.fixture(scope="module")
def decisions():
    return _collect_decisions()


def test_obs_parity(decisions):
    """桥接的观测构造必须与 env._nn_vec() 逐元素相同（477 维）。"""
    assert decisions, "没收集到决策点"
    for env in decisions:
        raw = env._raw_nn_vec()
        bridge = enhance_observation(raw, env.gc, card_index=CARD_INDEX,
                                     arch_table=CARD_ARCHETYPE,
                                     card_effects=CARD_EFFECTS)
        ref = env._nn_vec()
        assert len(bridge) == len(ref) == 477
        assert bridge == ref, "桥接观测与 env._nn_vec() 不一致"


def test_deck_arch_parity(decisions):
    """牌组流派强度（5 维）必须一致——它喂给候选的流派交叉项。"""
    assert decisions, "没收集到决策点"
    for env in decisions:
        raw = env._raw_nn_vec()
        bridge = encode_deck_stats(raw, card_index=CARD_INDEX,
                                   arch_table=CARD_ARCHETYPE,
                                   card_effects=CARD_EFFECTS)[:5]
        assert bridge == env._deck_arch(raw)
        assert len(bridge) == 5


def test_candidate_parity(decisions):
    """候选向量（369 维）必须逐元素相同。

    桥接路径 = 共享的 candidate_kind/resolve_candidate_kb + env 的
    card/room/lookahead 解析器重建候选，再与 env.candidates() 的 desc_vec 比对。
    """
    assert decisions, "没收集到决策点"
    n_checked = 0
    for env in decisions:
        gc = env.gc
        screen = gc.screen_state.name
        actions = env._actions
        cands = env.candidates()
        assert len(actions) == len(cands)
        raw = env._raw_nn_vec()
        deck_arch = encode_deck_stats(raw, card_index=CARD_INDEX,
                                      arch_table=CARD_ARCHETYPE,
                                      card_effects=CARD_EFFECTS)[:5]
        # V10：与训练侧 env 共用同一函数构造 SKIP 牌组上下文
        deck_ctx = encode_skip_deck_context(raw, card_index=CARD_INDEX,
                                            arch_table=CARD_ARCHETYPE,
                                            card_effects=CARD_EFFECTS)
        for i, (a, ref_cand) in enumerate(zip(actions, cands)):
            d = decode_action_bits(a.bits)
            kb = resolve_candidate_kb(screen, d, gc)
            label = a.describe(gc) or f"action_0x{a.bits:08x}"
            proto = Candidate(idx=i, kind=candidate_kind(screen, d),
                              desc_vec=[], label=label)
            ctx = deck_ctx if candidate_kind(screen, d) == "SKIP" else None
            bridge = encode_candidate(
                proto,
                card=env._candidate_card(screen, d),
                room=env._candidate_room(screen, d),
                bits=a.bits,
                lookahead_rooms=env._candidate_lookahead(screen, d),
                deck_arch=deck_arch,
                deck_ctx=ctx,
                **kb,
            )
            assert bridge == list(ref_cand.desc_vec), (
                f"候选不一致 screen={screen} idx={i} kind={proto.kind}")
            assert len(bridge) == DESC_VEC_DIM
            n_checked += 1
    assert n_checked > 0


def test_kind_parity(decisions):
    """kind 判定必须一致。"""
    assert decisions, "没收集到决策点"
    for env in decisions:
        screen = env.gc.screen_state.name
        for a, ref_cand in zip(env._actions, env.candidates()):
            d = decode_action_bits(a.bits)
            assert candidate_kind(screen, d) == ref_cand.kind, (
                f"screen={screen} bits=0x{a.bits:08x}")


def test_kb_helper_no_crash_on_all_screens(decisions):
    """KB 解析在所有遇到的屏上都不抛异常，且返回四个键。"""
    assert decisions, "没收集到决策点"
    for env in decisions:
        screen = env.gc.screen_state.name
        for a in env._actions:
            d = decode_action_bits(a.bits)
            kb = resolve_candidate_kb(screen, d, env.gc)
            assert set(kb.keys()) == {"potion_name", "relic_name",
                                      "event_key", "event_option_idx"}
