"""Non-contrastive VICReg Loss (Variance-Invariance-Covariance Regularization)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VicRegLoss(nn.Module):
    """Computes non-contrastive VICReg loss with additive homomorphism and shift equivariance

    to learn continuous latent representations without dimension collapse or negative-pair collisions.
    """

    def __init__(
        self,
        inv_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        homomorphism_coeff: float = 10.0,
        shift_coeff: float = 5.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
        d_model: Optional[int] = None,
    ):
        super().__init__()
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.homomorphism_coeff = homomorphism_coeff
        self.shift_coeff = shift_coeff
        self.gamma = gamma
        self.eps = eps

        # Learnable linear shift operator M_shift in R^(d x d)
        if d_model is not None:
            self.shift_operator = nn.Linear(d_model, d_model, bias=False)
            nn.init.orthogonal_(self.shift_operator.weight)
        else:
            self.shift_operator = None

    def forward(
        self,
        z_a: torch.Tensor,
        z_b: torch.Tensor,
        z_a_plus_b: Optional[torch.Tensor] = None,
        z_shift_a: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes VICReg loss with optional homomorphism and shift equivariance terms.

        Args:
            z_a: Embeddings for view A (batch_size, d)
            z_b: Embeddings for view B (batch_size, d)
            z_a_plus_b: Optional embeddings for sum sequence A + B (batch_size, d)
            z_shift_a: Optional embeddings for shifted sequence T_1(A) (batch_size, d)
        """
        assert z_a.shape == z_b.shape, f"Shape mismatch: {z_a.shape} vs {z_b.shape}"
        batch_size, d = z_a.shape

        # 1. Invariance Loss (Mean Squared Error)
        sim_loss = F.mse_loss(z_a, z_b)

        # 2. Variance Loss (Hinge loss on standard deviation along each dimension)
        std_z_a = torch.sqrt(z_a.var(dim=0) + self.eps)
        std_z_b = torch.sqrt(z_b.var(dim=0) + self.eps)
        var_loss = torch.mean(F.relu(self.gamma - std_z_a)) + torch.mean(F.relu(self.gamma - std_z_b))

        # 3. Covariance Loss (Off-diagonal penalty on feature covariance matrix)
        z_a_centered = z_a - z_a.mean(dim=0)
        z_b_centered = z_b - z_b.mean(dim=0)

        cov_z_a = (z_a_centered.T @ z_a_centered) / max(1, batch_size - 1)
        cov_z_b = (z_b_centered.T @ z_b_centered) / max(1, batch_size - 1)

        off_diag_a = cov_z_a.pow(2).sum() - cov_z_a.diag().pow(2).sum()
        off_diag_b = cov_z_b.pow(2).sum() - cov_z_b.diag().pow(2).sum()
        cov_loss = (off_diag_a + off_diag_b) / d

        total_loss = (
            self.inv_coeff * sim_loss
            + self.var_coeff * var_loss
            + self.cov_coeff * cov_loss
        )

        # 4. Optional Additive Homomorphism Loss: ||f(A+B) - (f(A) + f(B))||_2^2
        if z_a_plus_b is not None:
            homo_loss = F.mse_loss(z_a_plus_b, z_a + z_b)
            total_loss = total_loss + (self.homomorphism_coeff * homo_loss)

        # 5. Optional Shift Equivariance Loss: ||f(T_1 A) - M_shift f(A)||_2^2
        if z_shift_a is not None and self.shift_operator is not None:
            predicted_shift = self.shift_operator(z_a)
            shift_loss = F.mse_loss(z_shift_a, predicted_shift)
            total_loss = total_loss + (self.shift_coeff * shift_loss)

        return total_loss


def compute_rank_dispersion_ratio(z: torch.Tensor, threshold: float = 1e-4) -> float:
    """Computes the Rank Dispersion Ratio (RDR = Rank(Cov(Z)) / d) to detect dimensional collapse."""
    batch_size, d = z.shape
    if batch_size < 2:
        return 1.0

    z_centered = z - z.mean(dim=0)
    cov = (z_centered.T @ z_centered) / (batch_size - 1)

    # Compute singular values / eigenvalues
    eigenvalues = torch.linalg.eigvalsh(cov)
    # Count eigenvalues above threshold relative to max eigenvalue
    max_ev = eigenvalues.max().item()
    if max_ev <= 0.0:
        return 0.0

    effective_rank = int((eigenvalues > (threshold * max_ev)).sum().item())
    return effective_rank / d
