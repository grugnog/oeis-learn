"""Modulo-Spectrum Stream (S2): 100-moduli trigonometric Fourier phase embeddings."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union, cast
import torch
import torch.nn as nn

MODULI_COUNT = 100
BASE_MODULI = list(range(2, 2 + MODULI_COUNT))  # 2 to 101


def compute_fourier_phase_vector(
    val: Union[int, float], base_moduli: Optional[Sequence[int]] = None
) -> List[float]:
    """Computes the Fourier phase vector for given moduli m.

    For each m: [sin(2*pi*(x mod m)/m), cos(2*pi*(x mod m)/m)]
    """
    if isinstance(val, float):
        int_val = int(val)
    else:
        int_val = int(val)

    active_moduli = base_moduli if base_moduli is not None else BASE_MODULI
    features = []
    two_pi = 2.0 * math.pi
    for m in active_moduli:
        rem = int_val % m
        theta = (two_pi * rem) / m
        features.append(math.sin(theta))
        features.append(math.cos(theta))
    return features


class ModuloSpectrumStream(nn.Module):
    """Encodes cyclic modular congruences into continuous Fourier phase embeddings."""

    def __init__(
        self,
        d_model: int = 256,
        moduli_count: int = MODULI_COUNT,
        base_moduli: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        if base_moduli is not None:
            self.base_moduli = list(base_moduli)
            self.moduli_count = len(self.base_moduli)
        else:
            self.moduli_count = moduli_count
            self.base_moduli = list(range(2, 2 + moduli_count))

        self.input_dim = self.moduli_count * 2
        self.projection = nn.Sequential(
            nn.Linear(self.input_dim, d_model, dtype=torch.float32),
            nn.LayerNorm(d_model, dtype=torch.float32),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward_vectors(self, phase_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass taking precomputed phase tensor of shape (batch, seq_len, input_dim)."""
        phase_tensor = phase_tensor.to(dtype=torch.float32)
        return cast(torch.Tensor, self.projection(phase_tensor))

    def forward(self, sequences: Sequence[Sequence[Union[int, float]]], device: Optional[torch.device] = None) -> torch.Tensor:
        """Processes raw batch of sequences of integers/floats."""
        batch_size = len(sequences)
        max_len = max((len(seq) for seq in sequences), default=0)

        phases = torch.zeros((batch_size, max_len, self.input_dim), dtype=torch.float32, device=device)
        for b_idx, seq in enumerate(sequences):
            for t_idx, term in enumerate(seq):
                vec = compute_fourier_phase_vector(term, base_moduli=self.base_moduli)
                phases[b_idx, t_idx] = torch.tensor(vec, dtype=torch.float32, device=device)

        return cast(torch.Tensor, self.projection(phases))


PRIME_FIELDS_16 = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]


class ModuloStreamEncoder(ModuloSpectrumStream):
    """Tri-Stream Encoder v2 Modulo Stream using 16 orthogonal Prime Fourier Embeddings (PFE)."""

    def __init__(
        self,
        d_model: int = 256,
        primes: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
    ):
        active_primes = list(primes) if primes is not None else PRIME_FIELDS_16
        super().__init__(
            d_model=d_model,
            moduli_count=len(active_primes),
            base_moduli=active_primes,
            dropout=dropout,
        )

    def compute_raw_pfe(self, seq: Sequence[Union[int, float]]) -> torch.Tensor:
        """Computes raw Prime Fourier Embeddings across 16 odd prime fields:

        PFE(y) = [cos(2*pi*y / p), sin(2*pi*y / p)] for p in P_16.
        """
        n = len(seq)
        num_primes = len(self.base_moduli)
        pfe_matrix = torch.zeros((n, num_primes * 2), dtype=torch.float32)
        two_pi = 2.0 * math.pi

        for i, term in enumerate(seq):
            int_val = int(term)
            for p_idx, p in enumerate(self.base_moduli):
                theta = (two_pi * (int_val % p)) / float(p)
                # cos in even pos, sin in odd pos
                pfe_matrix[i, 2 * p_idx] = math.cos(theta)
                pfe_matrix[i, 2 * p_idx + 1] = math.sin(theta)

        return pfe_matrix

    def forward_from_terms(
        self, sequences: Sequence[Sequence[Union[int, float]]], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Processes raw sequences using Prime Fourier Embeddings."""
        return self.forward(sequences, device=device)

