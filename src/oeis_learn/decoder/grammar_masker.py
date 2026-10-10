"""Dynamic Grammar Masker enforcing S-expression syntax and lexical scoping."""

from __future__ import annotations

from typing import List, Optional
import torch
from oeis_learn.decoder.environment_tracker import EnvironmentTracker
from oeis_learn.decoder.wat_grammar import VOCAB_SIZE


class GrammarMasker:
    """Combines vocabulary bounds with dynamic grammar support. Execution
    correctness, numeric limits and completion still require independent checks.
    """

    def __init__(self, vocab_size: int = VOCAB_SIZE, static_legal_ids=None):
        self.vocab_size = vocab_size
        self._support_buffer = None
        self.static_legal_ids = frozenset(
            range(vocab_size) if static_legal_ids is None else static_legal_ids
        )
        if any(type(i) is not int or not 0 <= i < vocab_size for i in self.static_legal_ids):
            raise ValueError("static grammar vocabulary mismatch")
        self._static_masks = {}

    def compute_mask(
        self, tracker: EnvironmentTracker, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Computes boolean / float logit mask for next token.

        Allowed tokens get 0.0, forbidden tokens get -inf.
        """
        if hasattr(tracker, "vocab_size") and tracker.vocab_size != self.vocab_size:
            raise ValueError("tracker and mask vocabularies differ")
        support = self.fill_support([tracker.get_valid_next_tokens()], device=device)[0]
        return torch.zeros(self.vocab_size, device=device).masked_fill(~support, float("-inf"))

    def compute_batch_mask(
        self, trackers: List[EnvironmentTracker], device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Computes batch logit mask of shape (batch_size, vocab_size)."""
        lists = [tracker.get_valid_next_tokens() for tracker in trackers]
        support = self.fill_support(lists, device=device)
        return torch.zeros(support.shape, device=device).masked_fill(~support, float("-inf"))

    def fill_support(self, legal_lists, device=None):
        """Reuse vocabulary-sized storage and transfer flattened dynamic IDs once."""
        target = torch.device(device or "cpu")
        shape = (len(legal_lists), self.vocab_size)
        if (
            self._support_buffer is None
            or self._support_buffer.shape != shape
            or self._support_buffer.device != target
        ):
            self._support_buffer = torch.empty(shape, dtype=torch.bool, device=target)
        self._support_buffer.zero_()
        flattened = []
        for row, ids in enumerate(legal_lists):
            if any(type(i) is not int or not 0 <= i < self.vocab_size for i in ids):
                raise ValueError("mask vocabulary would truncate allowed tokens")
            if not set(ids) <= self.static_legal_ids:
                raise ValueError("dynamic support exceeds the frozen static grammar")
            flattened.extend(row * self.vocab_size + i for i in ids)
        if flattened:
            self._support_buffer.view(-1)[
                torch.tensor(flattened, device=target, dtype=torch.long)
            ] = True
        if target not in self._static_masks:
            static = torch.zeros(self.vocab_size, device=target, dtype=torch.bool)
            static[list(sorted(self.static_legal_ids))] = True
            self._static_masks[target] = static
        self._support_buffer &= self._static_masks[target]
        return self._support_buffer
