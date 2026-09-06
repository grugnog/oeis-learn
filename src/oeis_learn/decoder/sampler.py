"""Autoregressive WAT Program Sampler with dynamic grammar constraints."""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from oeis_learn.decoder.environment_tracker import EnvironmentTracker, StructuralPhase
from oeis_learn.decoder.grammar_masker import GrammarMasker
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import (
    BOS_ID,
    EOS_ID,
    ID_TO_TOKEN,
    PAD_ID,
    TOKEN_TO_ID,
    decode_wat_tokens,
)


def top_p_filtering(
    logits: torch.Tensor,
    top_p: float = 0.95,
    filter_value: float = -1e9,
) -> torch.Tensor:
    """Filter a distribution of logits using nucleus (top-p) filtering."""
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices to the right to keep at least the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, filter_value)


def finalize_wat_tokens(token_ids: List[int], tracker: EnvironmentTracker) -> Tuple[str, List[int]]:
    """Cleanly balances and soundly closes open paren depth and control frames if generation truncated."""
    if tracker.paren_depth > 0:
        if tracker.phase == StructuralPhase.LOCAL_NAME:
            token_ids.append(TOKEN_TO_ID["$temp"])
            tracker.update("$temp")
        if tracker.phase == StructuralPhase.LOCAL_TYPE:
            token_ids.append(TOKEN_TO_ID["i64"])
            tracker.update("i64")
        if tracker.phase == StructuralPhase.LOCAL_CLOSE:
            token_ids.append(TOKEN_TO_ID[")"])
            tracker.update(")")
        if tracker.pending_const_type is not None:
            token_ids.append(TOKEN_TO_ID["0"])
            tracker.update("0")
        if tracker.pending_var_op is not None:
            token_ids.append(TOKEN_TO_ID["$n"])
            tracker.update("$n")
        if tracker.pending_branch_op is not None:
            token_ids.append(TOKEN_TO_ID["0"])
            tracker.update("0")

        # Clean up control frames by popping or dropping excess items to match baseline
        while tracker.control_stack:
            frame = tracker.control_stack[-1]
            while len(tracker.operand_stack) > frame.baseline_stack_depth:
                token_ids.append(TOKEN_TO_ID["drop"])
                tracker.update("drop")
            while len(tracker.operand_stack) < frame.baseline_stack_depth:
                token_ids.extend([TOKEN_TO_ID["i64.const"], TOKEN_TO_ID["0"]])
                tracker.update("i64.const")
                tracker.update("0")
            token_ids.append(TOKEN_TO_ID[")"])
            tracker.update(")")

        # Now in func body (paren_depth == 2)
        # Drop excess items until 0 or 1
        while len(tracker.operand_stack) > 1:
            token_ids.append(TOKEN_TO_ID["drop"])
            tracker.update("drop")

        if tracker.operand_stack != ["i64"]:
            if not tracker.operand_stack:
                token_ids.extend([TOKEN_TO_ID["i64.const"], TOKEN_TO_ID["0"]])
                tracker.update("i64.const")
                tracker.update("0")
            elif tracker.operand_stack[-1] == "i32":
                token_ids.append(TOKEN_TO_ID["i64.extend_i32_u"])
                tracker.update("i64.extend_i32_u")

        while tracker.paren_depth > 0:
            token_ids.append(TOKEN_TO_ID[")"])
            tracker.update(")")

    if EOS_ID in token_ids:
        eos_idx = token_ids.index(EOS_ID)
        token_ids = token_ids[:eos_idx]
    code = decode_wat_tokens(token_ids)
    return code, token_ids


class WatProgramSampler:
    """Generates WAT programs using temperature / nucleus sampling under strict grammar constraints."""

    def __init__(
        self,
        decoder: WatTransformerDecoder,
        grammar_masker: Optional[GrammarMasker] = None,
        max_length: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ):
        self.decoder = decoder
        self.grammar_masker = grammar_masker or GrammarMasker()
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p

    def sample_candidate(
        self,
        memory: torch.Tensor,
        seed: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        use_grammar_mask: bool = True,
        max_length: Optional[int] = None,
        prefix_wat: Optional[str] = None,
    ) -> Tuple[str, torch.Tensor]:
        """Samples a single candidate program using a local deterministic generator."""
        self.decoder.eval()
        device = memory.device
        temp = temperature if temperature is not None else self.temperature
        p_val = top_p if top_p is not None else self.top_p
        limit_len = max_length if max_length is not None else self.max_length

        # Isolated local generator (seed in [0, 2^31 - 1])
        gen = torch.Generator(device="cpu")
        local_seed = abs(int(seed)) % (2**31 - 1)
        gen.manual_seed(local_seed)

        # Single prompt batch
        tracker = EnvironmentTracker()
        tracker.reset()
        tracker.update("<bos>")

        if prefix_wat:
            from oeis_learn.decoder.wat_grammar import tokenize_wat, UNK_ID
            p_tokens = tokenize_wat(prefix_wat)
            p_ids = [TOKEN_TO_ID.get(t, UNK_ID) for t in p_tokens]
            for t in p_tokens:
                tracker.update(t)
            generated = torch.tensor([[BOS_ID] + p_ids], dtype=torch.long, device=device)
        else:
            generated = torch.full((1, 1), BOS_ID, dtype=torch.long, device=device)

        finished = False
        start_step = generated.size(1) - 1

        with torch.no_grad():
            for step in range(start_step, limit_len):
                logits = self.decoder(generated, memory[:1])
                next_token_logits = logits[:, -1, :].clone()

                if use_grammar_mask:
                    mask = self.grammar_masker.compute_batch_mask([tracker], device=device)
                    next_token_logits = next_token_logits + mask

                if temp > 0.0:
                    scaled = next_token_logits / temp
                    if p_val < 1.0:
                        scaled = top_p_filtering(scaled, top_p=p_val)
                    probs = F.softmax(scaled, dim=-1)
                    # Sample on CPU with isolated generator
                    next_token_cpu = torch.multinomial(probs.cpu(), num_samples=1, generator=gen)
                    next_token = next_token_cpu.to(device).squeeze(1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)

                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
                tok_id = int(next_token.item())
                tok_str = ID_TO_TOKEN.get(tok_id, "<unk>")
                tracker.update(tok_str)

                if tok_id == EOS_ID or (tracker.paren_depth == 0 and step > 10):
                    finished = True
                    break

        code, final_tokens = finalize_wat_tokens(generated[0].tolist(), tracker)
        final_tensor = torch.tensor(final_tokens + [EOS_ID], dtype=torch.long, device=device)
        return code, final_tensor

    def sample(
        self,
        memory: torch.Tensor,
        temperature: Optional[float] = None,
        use_grammar_mask: bool = True,
        prefix_wat: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """Autoregressively sample candidate programs for a batch of encoded sequence representations.

        Args:
            memory: (batch, src_len, d_model) encoded latent embeddings
            temperature: Sampling temperature
            use_grammar_mask: Whether to apply dynamic grammar logit masking
            prefix_wat: Optional WAT prefix to initialize the generation
            max_length: Optional maximum token length override

        Returns:
            Tuple of (generated WAT code strings, generated token IDs tensor)
        """
        self.decoder.eval()
        device = memory.device
        batch_size = memory.size(0)
        temp = temperature if temperature is not None else self.temperature
        limit_len = max_length if max_length is not None else self.max_length

        # Initialize trackers and generated tokens
        trackers = [EnvironmentTracker() for _ in range(batch_size)]

        if prefix_wat:
            from oeis_learn.decoder.wat_grammar import tokenize_wat, UNK_ID
            p_tokens = tokenize_wat(prefix_wat)
            p_ids = [TOKEN_TO_ID.get(t, UNK_ID) for t in p_tokens]
            for tracker in trackers:
                tracker.reset()
                tracker.update("<bos>")
                for t in p_tokens:
                    tracker.update(t)
            generated = torch.tensor([[BOS_ID] + p_ids] * batch_size, dtype=torch.long, device=device)
        else:
            generated = torch.full((batch_size, 1), BOS_ID, dtype=torch.long, device=device)
            for tracker in trackers:
                tracker.reset()
                tracker.update("<bos>")

        finished = [False] * batch_size
        start_step = generated.size(1) - 1

        with torch.no_grad():
            for step in range(start_step, limit_len):
                logits = self.decoder(generated, memory)  # (batch, cur_len, vocab_size)
                next_token_logits = logits[:, -1, :].clone()  # (batch, vocab_size)

                if use_grammar_mask:
                    mask = self.grammar_masker.compute_batch_mask(trackers, device=device)
                    next_token_logits = next_token_logits + mask

                # Temperature scaling & top-p filtering
                if temp > 0.0:
                    scaled_logits = next_token_logits / temp
                    if self.top_p < 1.0:
                        scaled_logits = top_p_filtering(scaled_logits, top_p=self.top_p)
                    probs = F.softmax(scaled_logits, dim=-1)
                    # Sample from valid distribution
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                # Append newly sampled tokens
                generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)

                for b_idx in range(batch_size):
                    if not finished[b_idx]:
                        tok_id = int(next_tokens[b_idx].item())
                        tok_str = ID_TO_TOKEN.get(tok_id, "<unk>")
                        trackers[b_idx].update(tok_str)
                        if tok_id == EOS_ID or (trackers[b_idx].paren_depth == 0 and step > 10):
                            finished[b_idx] = True

                if all(finished):
                    break

        # Decode token sequences to WAT code strings using shared sound closing
        wat_codes = []
        final_token_lists = []
        for b_idx in range(batch_size):
            token_ids = generated[b_idx].tolist()
            tracker = trackers[b_idx]
            code, final_tokens = finalize_wat_tokens(token_ids, tracker)
            wat_codes.append(code)
            final_token_lists.append(final_tokens + [EOS_ID])

        max_final_len = max((len(t) for t in final_token_lists), default=1)
        final_generated = torch.full((batch_size, max_final_len), PAD_ID, dtype=torch.long, device=device)
        for b_idx, t_seq in enumerate(final_token_lists):
            final_generated[b_idx, : len(t_seq)] = torch.tensor(t_seq, dtype=torch.long, device=device)

        return wat_codes, final_generated
