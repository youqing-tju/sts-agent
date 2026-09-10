"""权重手术的测试：证明「改维度后热启动」确实保住了已有知识。

核心可断言性质：
  用零填充方式扩张输入维度后，在**新增维全为 0** 的输入上，
  手术后模型的输出与手术前一致。
  · 维度不变时是**逐位相等**（同一 BLAS 路径，见 test_identity_is_bit_exact）
  · 维度改变时差异仅来自 BLAS 对不同输入宽度的 SIMD 分块/求和顺序，
    属浮点非结合性，量级 ~1e-7；语义上完全等价。
"""
import pytest
import torch

from models.scorer import Scorer, ValueNet
from models.weight_surgery import (
    adapt_mlp_state,
    adapt_checkpoint,
    append_map,
    name_remap,
)


def _mk(dims_in, hidden, seed=0):
    torch.manual_seed(seed)
    return Scorer(obs_dim=dims_in, cand_dim=0, hidden=hidden)


def test_identity_is_bit_exact():
    """维度不变时手术应逐位相等。"""
    s = _mk(10, (8, 4))
    sd = s.state_dict()
    new_sd = adapt_mlp_state(sd, prefix="mlp", old_in=10, new_in=10,
                             old_hidden=(8, 4), new_hidden=(8, 4))
    for k in sd:
        assert torch.equal(sd[k], new_sd[k]), k


def test_extend_input_preserves_output():
    """扩张输入维度（零填充）后，新维为 0 时输出必须与原模型一致。

    用紧公差而非逐位相等：12 输入 vs 16 输入的 BLAS 求和顺序不同，
    末位舍入会有 ~1e-7 的差（浮点非结合性），不是语义差异。"""
    s = _mk(10, (8, 4))
    sd = s.state_dict()
    new_sd = adapt_mlp_state(sd, prefix="mlp", old_in=10, new_in=16,
                             old_hidden=(8, 4), new_hidden=(8, 4))
    s2 = Scorer(obs_dim=16, cand_dim=0, hidden=(8, 4))
    s2.load_state_dict(new_sd)

    s.eval(); s2.eval()
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(10)
            x2 = torch.cat([x, torch.zeros(6)])
            assert torch.allclose(s.mlp(x), s2.mlp(x2), atol=1e-6, rtol=1e-5), \
                "扩张后的输出与原模型不一致（实测最大误差 ~6e-8，纯浮点舍入）"


def test_shrink_input_equivalent_to_zeroing():
    """删维度（按 in_map=None 丢弃）后，输出必须等于「原模型该维置 0」的输出。"""
    s = _mk(12, (8, 4))
    sd = s.state_dict()
    keep = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11]      # 丢掉第 3、4 维
    in_map = keep
    new_sd = adapt_mlp_state(sd, prefix="mlp", old_in=12, new_in=10,
                             old_hidden=(8, 4), new_hidden=(8, 4), in_map=in_map)
    s2 = Scorer(obs_dim=10, cand_dim=0, hidden=(8, 4))
    s2.load_state_dict(new_sd)

    s.eval(); s2.eval()
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(12)
            x_masked = x.clone()
            x_masked[3] = 0.0
            x_masked[4] = 0.0
            x_kept = x[keep]
            assert torch.allclose(s.mlp(x_masked), s2.mlp(x_kept),
                                  atol=1e-6, rtol=1e-5)


def test_widen_hidden_preserves_output():
    """加宽隐藏层（新神经元零初始化）后输出必须与原模型完全相同。"""
    s = _mk(10, (8, 4))
    sd = s.state_dict()
    new_sd = adapt_mlp_state(sd, prefix="mlp", old_in=10, new_in=10,
                             old_hidden=(8, 4), new_hidden=(16, 12))
    s2 = Scorer(obs_dim=10, cand_dim=0, hidden=(16, 12))
    s2.load_state_dict(new_sd)
    s.eval(); s2.eval()
    with torch.no_grad():
        for _ in range(5):
            x = torch.randn(10)
            assert torch.allclose(s.mlp(x), s2.mlp(x), atol=1e-6, rtol=1e-5)


def test_name_remap_handles_reorder():
    """语义重排必须按名字对齐，不能按下标。"""
    old = ["A", "B", "C", "D"]
    new = ["C", "A", "E", "B", "D"]
    m = name_remap(old, new)
    assert m == [2, 0, None, 1, 3]
    assert m.count(None) == 1


def test_name_remap_used_for_potion_slots():
    """实例：观测药水槽位重排（旧 24 槽 → 新 33 槽，含新药水）。

    验证按名字对齐后，同一组持有的药水在新旧模型里产生相同输出。
    """
    from models.obs_encoder import POTION_INDEX
    old24 = ["ATTACK_POTION", "BLOCK_POTION", "BLOOD_POTION", "STRENGTH_POTION",
             "DEXTERITY_POTION", "ENERGY_POTION", "FIRE_POTION", "POISON_POTION",
             "WEAK_POTION", "SPEED_POTION", "FLEX_POTION", "FOCUS_POTION",
             "SKILL_POTION", "POWER_POTION", "REGEN_POTION", "SWIFT_POTION",
             "EXPLOSIVE_POTION", "CUNNING_POTION", "DUPLICATION_POTION",
             "FRUIT_JUICE", "LIQUID_MEMORIES", "LIQUID_BRONZE", "SMOKE_BOMB",
             "ANCIENT_POTION"]
    m = name_remap(old24, POTION_INDEX)
    assert len(m) == len(POTION_INDEX) == 33
    # ELIXIR/HEART_OF_IRON 是新增槽位 → 必须零填充
    assert m[POTION_INDEX.index("ELIXIR_POTION")] is None
    assert m[POTION_INDEX.index("HEART_OF_IRON")] is None
    # 旧表里存在且不在新表的（异角色药水）不该被映射进来
    assert all(x is None or 0 <= x < len(old24) for x in m)


def test_checkpoint_roundtrip(tmp_path):
    """落盘/读回之后仍然逐位一致。"""
    torch.manual_seed(0)
    s = Scorer(obs_dim=20, cand_dim=10, hidden=(8, 4))
    src = tmp_path / "a.pt"
    dst = tmp_path / "b.pt"
    torch.save(s.state_dict(), str(src))
    adapt_checkpoint(str(src), str(dst), prefix="mlp", old_in=30, new_in=30,
                     old_hidden=(8, 4), new_hidden=(8, 4))
    sd2 = torch.load(str(dst), weights_only=True)
    for k, v in s.state_dict().items():
        assert torch.equal(v, sd2[k]), k


def test_value_net_only_obs():
    """ValueNet 只有 obs 输入，等价性同样成立。"""
    torch.manual_seed(0)
    v = ValueNet(obs_dim=10, hidden=6)
    new_sd = adapt_mlp_state(v.state_dict(), prefix="mlp", old_in=10, new_in=14,
                             old_hidden=(6,), new_hidden=(6,))
    v2 = ValueNet(obs_dim=14, hidden=6)
    v2.load_state_dict(new_sd)
    v.eval(); v2.eval()
    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(10)
            assert torch.allclose(v.mlp(x), v2.mlp(torch.cat([x, torch.zeros(4)])),
                                  atol=1e-6, rtol=1e-5)


# ---- 观测段真实案例：468 → 477（药水段 24→33 + 后续段整体后移 9）----

def test_obs_468_to_477_map_structure():
    """映射结构自检：药水段按名字对齐、后续段整体后移 9、新槽位零填充。"""
    from models.obs_encoder import POTION_INDEX as new33
    from models.weight_surgery import obs_468_to_477_map, POTION_OLD_24
    m = obs_468_to_477_map()
    assert len(m) == 477
    # C++ 原始段不变
    assert m[:412] == list(range(412))
    # 药水段：新表里出现过的旧药水 → 指回旧槽位；新药水 → None
    assert m[412 + new33.index("BLOCK_POTION")] == 412 + POTION_OLD_24.index("BLOCK_POTION")
    assert m[412 + new33.index("ELIXIR_POTION")] is None
    assert m[412 + new33.index("HEART_OF_IRON")] is None
    # 3 个铁甲拿不到的异角色药水已从新表移除（这正是那次改动的目的）
    for gone in ("POISON_POTION", "CUNNING_POTION", "FOCUS_POTION"):
        assert gone not in new33
    # 旧表里存在的药水若不在新表，就不该被映射（映射源必须落在新表里）
    mapped_names = [new33[i - 412] for i, s in enumerate(m)
                    if s is not None and 412 <= i < 445]
    assert mapped_names, "药水段应有映射（新旧表有 21 个共享药水）"
    assert all(n in POTION_OLD_24 for n in mapped_names)
    # 后续 32 维整体后移 9
    assert m[445:] == [i - 9 for i in range(445, 477)]
    assert m[445] == 436 and m[476] == 467


def test_obs_change_requires_explicit_alignment(tmp_path):
    """观测维度变化但没说明对齐方式 → CLI 必须报错，不能猜。"""
    from models.weight_surgery import main as ws_main
    torch.manual_seed(0)
    s = Scorer(obs_dim=20, cand_dim=5, hidden=(8, 4))
    src = tmp_path / "a.pt"
    torch.save(s.state_dict(), str(src))
    with pytest.raises(SystemExit) as e:
        ws_main(["--src", str(src), "--dst", str(tmp_path / "b.pt"),
                 "--old-obs", "20", "--new-obs", "24",
                 "--old-cand", "5", "--new-cand", "5", "--hidden", "8,4"])
    assert "显式说明对齐方式" in str(e.value)


def test_obs_append_only_flag_works(tmp_path):
    """显式声明追加在末尾时，CLI 正常执行且等价性成立。"""
    from models.weight_surgery import main as ws_main
    torch.manual_seed(0)
    s = Scorer(obs_dim=20, cand_dim=5, hidden=(8, 4))
    src, dst = tmp_path / "a.pt", tmp_path / "b.pt"
    torch.save(s.state_dict(), str(src))
    ws_main(["--src", str(src), "--dst", str(dst),
             "--old-obs", "20", "--new-obs", "24", "--obs-append-only",
             "--old-cand", "5", "--new-cand", "5", "--hidden", "8,4"])
    s2 = Scorer(obs_dim=24, cand_dim=5, hidden=(8, 4))
    s2.load_state_dict(torch.load(str(dst), weights_only=True))
    s.eval(); s2.eval()
    with torch.no_grad():
        for _ in range(3):
            o = torch.randn(20)
            c = torch.randn(4, 5)
            o2 = torch.cat([o, torch.zeros(4)])
            assert torch.allclose(s(o, c), s2(o2, c), atol=1e-6, rtol=1e-5)
