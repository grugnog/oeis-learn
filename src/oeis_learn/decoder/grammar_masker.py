"""Dynamic Grammar Masker enforcing S-expression syntax and lexical scoping."""

from __future__ import annotations

import time
from typing import List, Optional, Set, Union
import torch
from oeis_learn.decoder.environment_tracker import EnvironmentTracker
from oeis_learn.decoder.wat_grammar import ID_TO_TOKEN, TOKEN_TO_ID, VOCAB_SIZE


class GrammarMasker:
    """Computes dynamic token logit masks during autoregressive decoding to guarantee

    100% syntactically and semantically valid WAT programs.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size

    def compute_mask(self, tracker: EnvironmentTracker, device: Optional[torch.device] = None) -> torch.Tensor:
        """Computes boolean / float logit mask for next token.

        Allowed tokens get 0.0, forbidden tokens get -inf.
        """
        valid_ids = tracker.get_valid_next_tokens()
        mask = torch.full((self.vocab_size,), float("-inf"), dtype=torch.float32, device=device)
        for t_id in valid_ids:
            if 0 <= t_id < self.vocab_size:
                mask[t_id] = 0.0
        return mask

    def compute_batch_mask(
        self, trackers: List[EnvironmentTracker], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Computes batch logit mask of shape (batch_size, vocab_size)."""
        batch_size = len(trackers)
        masks = torch.zeros((batch_size, self.vocab_size), dtype=torch.float32, device=device)
        for b_idx, tracker in enumerate(trackers):
            masks[b_idx] = self.compute_mask(tracker, device=device)
        return masks
