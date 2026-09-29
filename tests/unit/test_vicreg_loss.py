"""Unit tests for non-contrastive VICReg loss function."""

import torch
from oeis_learn.discovery.vicreg_loss import VicRegLoss


def test_vicreg_loss_computation():
    loss_fn = VicRegLoss(inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0)
    batch_size, d = 32, 64

    z_a = torch.randn(batch_size, d, dtype=torch.float32, requires_grad=True)
    z_b = torch.randn(batch_size, d, dtype=torch.float32, requires_grad=True)

    loss = loss_fn(z_a, z_b)

    assert loss.dim() == 0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert z_a.grad is not None and not torch.isnan(z_a.grad).any()
    assert z_b.grad is not None and not torch.isnan(z_b.grad).any()


def test_vicreg_identical_embeddings_have_zero_invariance():
    loss_fn = VicRegLoss(inv_coeff=25.0, var_coeff=0.0, cov_coeff=0.0)
    z = torch.randn(16, 32, dtype=torch.float32)

    loss = loss_fn(z, z)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_vicreg_additive_homomorphism_and_shift_loss():
    from oeis_learn.discovery.vicreg_loss import compute_rank_dispersion_ratio

    d_model = 32
    loss_fn = VicRegLoss(
        inv_coeff=1.0,
        var_coeff=1.0,
        cov_coeff=1.0,
        homomorphism_coeff=10.0,
        shift_coeff=5.0,
        d_model=d_model,
    )
    batch_size = 16

    z_a = torch.randn(batch_size, d_model, dtype=torch.float32, requires_grad=True)
    z_b = torch.randn(batch_size, d_model, dtype=torch.float32, requires_grad=True)
    z_sum = z_a + z_b  # Exact homomorphism
    z_shift = torch.randn(batch_size, d_model, dtype=torch.float32)

    loss = loss_fn(z_a, z_b, z_a_plus_b=z_sum, z_shift_a=z_shift)
    assert loss.item() > 0.0
    loss.backward()
    assert z_a.grad is not None

    # Test Rank Dispersion Ratio
    rdr = compute_rank_dispersion_ratio(z_a.detach())
    assert 0.0 <= rdr <= 1.0
    assert rdr > 0.40  # Full rank random Gaussian should have high dispersion
