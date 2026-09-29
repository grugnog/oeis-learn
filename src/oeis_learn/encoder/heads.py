"""Auxiliary pre-training and supervision prediction heads."""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from oeis_learn.encoder.modulo_stream import BASE_MODULI


class TriStreamPredictionHeads(nn.Module):
    """Auxiliary prediction heads for pre-training and representation grounding."""

    def __init__(self, d_model: int = 256, selected_moduli: Tuple[int, ...] = (2, 3, 5, 7, 10, 11, 13)):
        super().__init__()
        self.d_model = d_model
        self.selected_moduli = selected_moduli

        # Next term continuous magnitude MSE head
        self.magnitude_head = nn.Sequential(
            nn.Linear(d_model, d_model, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(d_model, 1, dtype=torch.float32),
        )

        # Sign classification head (0: negative, 1: zero, 2: positive)
        self.sign_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(d_model // 2, 3, dtype=torch.float32),
        )

        # Moduli residue classification heads for selected key primes/bases
        self.moduli_heads = nn.ModuleDict({
            f"mod_{m}": nn.Linear(d_model, m, dtype=torch.float32) for m in selected_moduli
        })

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute predictions from fused sequence representations z (batch, seq_len, d_model)."""
        z = z.to(dtype=torch.float32)
        outputs: Dict[str, torch.Tensor] = {
            "pred_magnitude": self.magnitude_head(z).squeeze(-1),
            "pred_sign_logits": self.sign_head(z),
        }
        for m in self.selected_moduli:
            outputs[f"pred_mod_{m}_logits"] = self.moduli_heads[f"mod_{m}"](z)
        return outputs


class SummaryRegressionHeads(nn.Module):
    """Auxiliary regression heads predicting sequence-wide linear slope m_hat and geometric ratio r_hat

    from learnable summary tokens z_affine and z_geom.
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # Slope regression head from z_affine
        self.slope_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(d_model // 2, 1, dtype=torch.float32),
        )

        # Geometric ratio regression head from z_geom
        self.ratio_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2, dtype=torch.float32),
            nn.GELU(),
            nn.Linear(d_model // 2, 1, dtype=torch.float32),
        )

    def forward(self, z_affine: torch.Tensor, z_geom: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes predicted slope and geometric ratio.

        z_affine: (batch, d_model)
        z_geom: (batch, d_model)
        Returns: (pred_slope, pred_ratio) of shapes (batch, 1)
        """
        pred_slope = self.slope_head(z_affine.to(dtype=torch.float32))
        pred_ratio = self.ratio_head(z_geom.to(dtype=torch.float32))
        return pred_slope, pred_ratio

    def compute_auxiliary_loss(
        self,
        pred_slope: torch.Tensor,
        pred_ratio: torch.Tensor,
        target_slope: torch.Tensor,
        target_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Computes auxiliary Mean Squared Error loss:

        L_aux = MSE(pred_slope, target_slope) + MSE(pred_ratio, target_ratio).
        """
        loss_slope = nn.functional.mse_loss(pred_slope, target_slope.to(dtype=torch.float32))
        loss_ratio = nn.functional.mse_loss(pred_ratio, target_ratio.to(dtype=torch.float32))
        return loss_slope + loss_ratio

