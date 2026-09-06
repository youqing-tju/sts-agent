import torch

from models.scorer import Scorer


def test_output_shape_single_obs():
    torch.manual_seed(0)
    s = Scorer()
    out = s(torch.randn(468), torch.randn(5, 367))
    assert out.shape == (5,)


def test_output_shape_batched_obs():
    torch.manual_seed(0)
    s = Scorer()
    out = s(torch.randn(3, 468), torch.randn(5, 367))
    assert out.shape == (3, 5)


def test_scores_differ_across_candidates():
    torch.manual_seed(0)
    s = Scorer()
    out = s(torch.randn(468), torch.randn(6, 367))
    # 随机权重下不同候选几乎不可能同分
    assert not torch.allclose(out, out[0].expand_as(out))


def test_gradient_flows_to_all_params():
    torch.manual_seed(0)
    s = Scorer()
    out = s(torch.randn(468), torch.randn(4, 367))
    out.sum().backward()
    for p in s.parameters():
        assert p.grad is not None
        assert p.grad.abs().sum() > 0


def test_custom_hidden_dims():
    s = Scorer(obs_dim=10, cand_dim=4, hidden=(8,))
    out = s(torch.randn(10), torch.randn(3, 4))
    assert out.shape == (3,)
