"""Autoregressive WAT Program Sampler with dynamic grammar constraints."""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn.functional as F
from oeis_learn.decoder.environment_tracker import EnvironmentTracker, StructuralPhase
from oeis_learn.decoder.grammar_masker import GrammarMasker
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


def finalize_wat_tokens(
    token_ids: List[int],
    tracker: EnvironmentTracker,
    max_length: int = 512,
) -> Tuple[str, List[int]]:
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
        while tracker.control_stack and len(token_ids) < max_length - 8:
            frame = tracker.control_stack[-1]
            while (
                len(tracker.operand_stack) > frame.baseline_stack_depth
                and len(token_ids) < max_length - 8
            ):
                token_ids.append(TOKEN_TO_ID["drop"])
                tracker.update("drop")
            while (
                len(tracker.operand_stack) < frame.baseline_stack_depth
                and len(token_ids) < max_length - 8
            ):
                token_ids.extend([TOKEN_TO_ID["i64.const"], TOKEN_TO_ID["0"]])
                tracker.update("i64.const")
                tracker.update("0")
            token_ids.append(TOKEN_TO_ID[")"])
            tracker.update(")")

        # Now in func body (paren_depth == 2)
        target_types = tracker.result_types if tracker.result_types else ["i64"]
        target_len = len(target_types)

        # Drop excess items until matching target_len
        while len(tracker.operand_stack) > target_len and len(token_ids) < max_length - 8:
            token_ids.append(TOKEN_TO_ID["drop"])
            tracker.update("drop")

        while len(tracker.operand_stack) < target_len and len(token_ids) < max_length - 8:
            token_ids.extend([TOKEN_TO_ID["i64.const"], TOKEN_TO_ID["0"]])
            tracker.update("i64.const")
            tracker.update("0")

        while tracker.paren_depth > 0 and len(token_ids) < max_length:
            token_ids.append(TOKEN_TO_ID[")"])
            tracker.update(")")

    if EOS_ID in token_ids:
        eos_idx = token_ids.index(EOS_ID)
        token_ids = token_ids[:eos_idx]
    code = decode_wat_tokens(token_ids)
    return code, token_ids


from dataclasses import dataclass
from oeis_learn.rl.policy import scoring_mode, masked_distribution


@dataclass(frozen=True)
class SampledTrajectory:
    source: str
    tokens: tuple[int, ...]
    supports: tuple[tuple[int, ...], ...]
    log_probabilities: tuple[float | None, ...]
    forced: tuple[bool, ...]
    temperature: float
    seed: int
    complete: bool
    failure: str | None

    def record(
        self,
        *,
        encoder_sha256,
        decoder_sha256,
        prompt_sha256,
        execution_sha256,
        reward_components,
        eos_id,
    ):
        from oeis_learn.experiments.qualified_models import seal_record
        from oeis_learn.experiments.artifacts import compute_canonical_digest

        return seal_record(
            "rollout_record",
            behavior_encoder_sha256=encoder_sha256,
            behavior_decoder_sha256=decoder_sha256,
            visible_prompt_sha256=prompt_sha256,
            actions=list(self.tokens),
            supports=[list(v) for v in self.supports],
            temperatures=[self.temperature] * len(self.tokens),
            log_probabilities=list(self.log_probabilities),
            forced_mask=list(self.forced),
            pad_mask=[False] * len(self.tokens),
            eos_mask=[v == eos_id for v in self.tokens],
            rng_sha256=compute_canonical_digest(["per-candidate-torch/v1", self.seed]),
            reward_components=reward_components,
            execution_sha256=execution_sha256,
            top_p=1,
            top_k=None,
            rollout_dropout=False,
            scoring_dropout=False,
        )


class WatProgramSampler:
    """Shared seeded rollout; diagnostic repair tokens never become policy actions."""

    def __init__(
        self,
        decoder,
        grammar_masker=None,
        max_length=256,
        temperature=0.8,
        top_p=0.95,
        codec_profile=None,
        use_cache=False,
    ):
        self.decoder, self.codec = decoder, None
        if codec_profile is not None:
            from oeis_learn.decoder.wat_grammar import get_program_codec

            self.codec = get_program_codec(codec_profile)
            if getattr(decoder, "codec_sha256", None) != self.codec.codec_digest():
                raise ValueError("decoder has no matching foundation codec identity")
        self.grammar_masker = grammar_masker or GrammarMasker(
            self.codec.FOUNDATION_VOCAB_SIZE if self.codec else decoder.vocab_size
        )
        self.max_length, self.temperature, self.top_p = max_length, temperature, top_p
        self.use_cache, self.last_trajectories = use_cache, ()

    def sample_trajectories(
        self,
        memory,
        seeds,
        *,
        policy=False,
        temperature=None,
        top_p=None,
        prefix_wat=None,
        max_length=None,
        use_grammar_mask=True,
        budget_hook=None,
    ):
        import math
        from oeis_learn.decoder.environment_tracker import FoundationEnvironmentTracker
        from oeis_learn.decoder.wat_grammar import encode_wat

        temp = self.temperature if temperature is None else temperature
        nucleus = self.top_p if top_p is None else top_p
        limit = self.max_length if max_length is None else max_length
        if not math.isfinite(temp) or temp < 0 or not 0 < nucleus <= 1:
            raise ValueError("invalid temperature/top_p")
        if policy and (temp == 0 or nucleus != 1 or not use_grammar_mask):
            raise ValueError(
                "policy gradients require positive temperature, top_p=1 and exact grammar"
            )
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ValueError("bounded context in 1..1024 required")
        if (
            not seeds
            or any(type(s) is not int or not -(2**63) <= s < 2**64 for s in seeds)
            or memory.ndim != 3
            or len(seeds) != memory.size(0)
        ):
            raise ValueError("one independent seed required for each visible prompt")
        if self.codec and (prefix_wat is not None or not use_grammar_mask):
            raise self.codec.CodecError(
                "foundation forbids source scaffolds and unmasked generation"
            )
        codec = self.codec
        bos, eos, pad = (
            (codec.BOS_ID, codec.EOS_ID, codec.PAD_ID) if codec else (BOS_ID, EOS_ID, PAD_ID)
        )
        words = codec.ID_TO_TOKEN if codec else ID_TO_TOKEN
        trackers = [
            FoundationEnvironmentTracker() if codec else EnvironmentTracker() for _ in seeds
        ]
        prefix = [bos] + ([] if prefix_wat is None else encode_wat(prefix_wat))
        # Foundation BOS is outside the body cap; legacy scaffolding is inside it.
        prefix_body_length = len(prefix) - int(bool(codec))
        if prefix_body_length >= limit:
            raise ValueError("forced prefix exhausts context")
        for tracker in trackers:
            for token in prefix[1:] if codec else prefix:
                tracker.update(words[token])
        batch = len(seeds)
        generators = [
            torch.Generator(device=memory.device).manual_seed(int(seed)) for seed in seeds
        ]
        buffer = torch.full((batch, limit + 1), pad, dtype=torch.long, device=memory.device)
        buffer[:, : len(prefix)] = torch.tensor(prefix, device=memory.device)
        token_rows = [list(prefix) for _ in seeds]
        # Retain the emitted prefix if an external time/token budget interrupts.
        self.active_token_rows = token_rows
        self.eos_traces = [[] for _ in seeds]
        supports = [[(v,) for v in prefix] for _ in seeds]
        logs = [[None] * len(prefix) for _ in seeds]
        forced = [[True] * len(prefix) for _ in seeds]
        finished, failures = [False] * batch, [None] * batch
        with scoring_mode(self.decoder), torch.no_grad():
            cached = self.use_cache and hasattr(self.decoder, "prefill")
            if cached:
                logits, cache = self.decoder.prefill(
                    buffer[:, : len(prefix)], memory, capacity=limit
                )
                logits = logits[:, -1]
            for step in range(len(prefix), limit + int(codec is not None)):
                if budget_hook is not None:
                    budget_hook(sum(not done for done in finished))
                if not cached:
                    logits = self.decoder(buffer[:, :step], memory)[:, -1]
                if (
                    logits.shape != (batch, self.grammar_masker.vocab_size)
                    or not torch.isfinite(logits).all()
                ):
                    raise ValueError("incompatible or nonfinite decoder logits")
                legal_lists = [
                    tracker.get_valid_next_tokens()
                    if use_grammar_mask
                    else list(range(logits.size(-1)))
                    for tracker in trackers
                ]
                for row in range(batch):
                    if finished[row]:
                        legal_lists[row] = [pad]
                    elif not legal_lists[row]:
                        finished[row], failures[row] = True, "grammar_dead_end"
                        legal_lists[row] = [pad]
                support = self.grammar_masker.fill_support(legal_lists, device=memory.device)
                log_distribution, probabilities, _ = masked_distribution(
                    logits, support, temp if temp else 1.0
                )
                if nucleus < 1:
                    filtered = top_p_filtering(
                        logits.masked_fill(~support, float("-inf")) / (temp or 1.0),
                        nucleus,
                        float("-inf"),
                    )
                    probabilities = F.softmax(filtered, -1)
                    log_distribution = F.log_softmax(filtered, -1)
                if getattr(self, "capture_diagnostics", False):
                    raw_eos = logits.softmax(-1)[:, eos].cpu().tolist()
                    supported_eos = probabilities[:, eos].cpu().tolist()
                    for row in range(batch):
                        if not finished[row]:
                            self.eos_traces[row].append(dict(
                                position=step, raw_probability=raw_eos[row],
                                grammar_probability=supported_eos[row],
                                allowed=eos in legal_lists[row],
                            ))
                chosen = torch.stack(
                    [
                        probabilities[row].argmax()
                        if temp == 0
                        else torch.multinomial(probabilities[row], 1, generator=generators[row])[0]
                        for row in range(batch)
                    ]
                )
                chosen_logs = (
                    log_distribution.gather(-1, chosen[:, None]).squeeze(-1).cpu().tolist()
                )
                ids = chosen.cpu().tolist()
                buffer[:, step] = chosen
                for row, token in enumerate(ids):
                    if finished[row]:
                        continue
                    token_rows[row].append(token)
                    supports[row].append(tuple(legal_lists[row]))
                    logs[row].append(chosen_logs[row] if temp else None)
                    forced[row].append(temp == 0)
                    trackers[row].update(words[token])
                    if token == eos:
                        finished[row] = True
                if all(finished):
                    break
                if cached and step + 1 < limit + int(codec is not None):
                    cache.finished = list(finished)
                    logits = self.decoder.decode_step(chosen, cache)[:, 0]
        results = []
        for row in range(batch):
            tokens = token_rows[row]
            complete = tokens[-1] == eos and failures[row] is None
            if not complete:
                failures[row] = failures[row] or "token_cap_without_eos"
            if codec:
                source = codec.decode_body(tokens[1:]) if complete else ""
            else:
                source, repaired = finalize_wat_tokens(
                    list(tokens), trackers[row], max_length=limit
                )
                if not complete:
                    for token in repaired[len(tokens) :] + [eos]:
                        supports[row].append((token,))
                        logs[row].append(None)
                        forced[row].append(True)
                    tokens = repaired + [eos]
            results.append(
                SampledTrajectory(
                    source,
                    tuple(tokens),
                    tuple(supports[row]),
                    tuple(logs[row]),
                    tuple(forced[row]),
                    temp or 1.0,
                    int(seeds[row]),
                    complete,
                    failures[row],
                )
            )
        self.last_trajectories = tuple(results)
        return results

    def sample_candidate(self, memory, seed, **kwargs):
        if memory.size(0) != 1:
            raise ValueError("one prompt required for sample_candidate")
        trajectory = self.sample_trajectories(memory, [seed], **kwargs)[0]
        if self.codec and not trajectory.complete:
            raise self.codec.CodecError(trajectory.failure)
        tokens = trajectory.tokens[1:] if self.codec else trajectory.tokens
        return trajectory.source, torch.tensor(tokens, device=memory.device, dtype=torch.long)

    def sample(
        self, memory, temperature=None, use_grammar_mask=True, prefix_wat=None, max_length=None
    ):
        if self.codec:
            raise ValueError("foundation requires explicit per-attempt seeds")
        seeds = torch.randint(0, 2**31 - 1, (memory.size(0),)).tolist()
        trajectories = self.sample_trajectories(
            memory,
            seeds,
            temperature=temperature,
            use_grammar_mask=use_grammar_mask,
            prefix_wat=prefix_wat,
            max_length=max_length,
        )
        size = max(len(t.tokens) for t in trajectories)
        output = torch.full(
            (len(trajectories), size), PAD_ID, device=memory.device, dtype=torch.long
        )
        for row, trajectory in enumerate(trajectories):
            output[row, : len(trajectory.tokens)] = torch.tensor(
                trajectory.tokens, device=memory.device
            )
        return [t.source for t in trajectories], output


def prompt_length_buckets(sequences, *, context_limit=1024):
    """Stable exact-length buckets avoid cross-attending to fabricated padding.

    The code capacity is part of the bucket identity; neural inputs contain only
    visible values. Bucket ordering never changes per-candidate random streams.
    """
    if type(context_limit) is not int or not 2 <= context_limit <= 1024:
        raise ValueError("bounded model context required")
    groups = {}
    for row, values in enumerate(sequences):
        if not 1 <= len(values) <= 20:
            raise ValueError("visible lengths must be in 1..20")
        groups.setdefault((len(values), context_limit), []).append(row)
    return tuple((key, tuple(rows)) for key, rows in sorted(groups.items()))


def sample_bucketed(encoder, sampler, sequences, seeds, *, policy=True):
    if len(sequences) != len(seeds) or not sequences:
        raise ValueError("one named seed per visible prompt required")
    results = [None] * len(sequences)
    device = next(encoder.parameters()).device
    with scoring_mode(encoder, sampler.decoder), torch.no_grad():
        for _, rows in prompt_length_buckets(sequences, context_limit=sampler.max_length + 1):
            memory = encoder.forward_from_sequences([sequences[i] for i in rows], device=device)
            trajectories = sampler.sample_trajectories(
                memory, [seeds[i] for i in rows], policy=policy
            )
            for row, trajectory in zip(rows, trajectories):
                results[row] = trajectory
    return tuple(results)
