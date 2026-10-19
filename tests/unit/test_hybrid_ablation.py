import pytest
import torch
from oeis_learn.encoder.ablation import InputMask
from oeis_learn.encoder.hybrid import ChunkHybridEncoder
from oeis_learn.encoder.film_fusion import HierarchicalFilmFusion


@pytest.mark.parametrize("group", ["magnitude", "modular", "difference"])
def test_fusion_masks_remove_bias_and_input_dependence(group):
    torch.manual_seed(4)
    fusion = HierarchicalFilmFusion(d_model=8, dropout=0).eval()
    mask = InputMask(**{group: False})
    values = [torch.randn(1, 3, 8) for _ in range(3)]
    expected = fusion(*values, group_mask=mask)
    values[["magnitude", "modular", "difference"].index(group)] = torch.randn(1, 3, 8) * 100
    torch.testing.assert_close(fusion(*values, group_mask=mask), expected, atol=0, rtol=0)
    # No scaling/renormalization of the surviving paths.
    s1, s2, s3 = values
    p1 = fusion.film1(s2) if mask.modular else torch.zeros(1, 3, 16)
    p2 = fusion.film2(s3) if mask.difference else torch.zeros(1, 3, 16)
    g1, b1 = p1.chunk(2, -1)
    g2, b2 = p2.chunk(2, -1)
    manual = fusion.layer_norm(
        (1 + g2) * ((1 + g1) * (s1 if mask.magnitude else torch.zeros_like(s1)) + b1) + b2
    )
    torch.testing.assert_close(expected, manual)


def test_retraining_freezes_matching_boundary_and_checkpoint_mask():
    model = ChunkHybridEncoder(group_mask=InputMask(exact_bytes=False), ablation_mode="retrained")
    assert not any(p.requires_grad for p in model.chunk_encoder.parameters())
    report = model.ablation_report()
    assert report["active_parameters"] < report["total_parameters"]
    model.forward_from_sequences([[1, 2, 3]]).sum().backward()
    assert all(p.grad is None for p in model.chunk_encoder.parameters())
    other = ChunkHybridEncoder()
    with pytest.raises(RuntimeError, match="mask"):
        other.load_state_dict(model.state_dict())
    same = ChunkHybridEncoder(group_mask=InputMask(exact_bytes=False), ablation_mode="retrained")
    same.load_state_dict(model.state_dict())
