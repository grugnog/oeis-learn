"""Bounded production sampling shared by visible portfolios and cost measurement."""

from dataclasses import dataclass
import math
import time

GENERATION_BATCH_SIZE = 8
GENERATION_POLICY = "bigint-generation-batches/v1"


@dataclass(frozen=True)
class GenerationRequest:
    memory: object
    seed: int
    temperature: float
    deadline: float
    max_length: int = 1024


def sample_batch(decoder, requests, *, batch_size=8, deadline_policy="production60/v1", bookkeeping=None, capacity_qualified=False):
    """Decode at most eight compatible requests; no deadline or seed is renewed."""
    import torch
    from .sampler import WatProgramSampler

    if batch_size not in (1,8,16,32) or not 1 <= len(requests) <= batch_size:
        raise ValueError("registered generation batch1/8/16 required")
    if batch_size==32 and capacity_qualified is not True:
        raise ValueError('batch32 requires measured capacity admission')
    if deadline_policy not in ('production60/v1','fixed-work180/v1'):
        raise ValueError('unknown deadline policy')
    if bookkeeping is not None:
        if bookkeeping not in ('baseline','hoisted/v1'):
            raise ValueError('unknown decoder choice')
        decoder.cache_bookkeeping = bookkeeping
    first = requests[0]
    for request in requests:
        if (
            request.memory.ndim != 3
            or request.memory.size(0) != 1
            or request.memory.shape[1:] != first.memory.shape[1:]
            or request.memory.device != first.memory.device
            or request.memory.dtype != first.memory.dtype
            or request.temperature != first.temperature
            or not math.isfinite(request.temperature)
            or request.temperature < 0
            or type(request.seed) is not int
            or not -(2**63) <= request.seed < 2**64
            or not math.isfinite(request.deadline)
            or type(request.max_length) is not int
            or not 1 <= request.max_length <= 1024
            or request.max_length != first.max_length
        ):
            raise ValueError("compatible visible memories, seeds, temperatures and caps required")
    start = time.monotonic()
    deadline = min(start + (60 if deadline_policy == 'production60/v1' else 180), *(request.deadline for request in requests))
    sampler = WatProgramSampler(
        decoder,
        max_length=first.max_length,
        temperature=first.temperature,
        top_p=1,
        codec_profile="bigint-codec/v1",
        use_cache=True,
    )

    def hook(_):
        if time.monotonic() >= deadline:
            raise TimeoutError("visible portfolio deadline")

    try:
        hook(0)
        with torch.no_grad():
            memory = torch.cat([r.memory for r in requests], dim=0)
            trajectories = sampler.sample_trajectories(
                memory, [r.seed for r in requests], budget_hook=hook
            )
        if len(trajectories) != len(requests):
            raise RuntimeError("generation batch returned incomplete row mapping")
        # A completed tensor operation may return after the last per-token hook.
        hook(0)
    except TimeoutError:
        partial = getattr(sampler, "active_token_rows", [[] for _ in requests])
        return [
            dict(
                source="",
                complete=False,
                tokens=list(partial[i][1 : first.max_length + 1]),
                generation_seconds=time.monotonic() - start,
                generation_batch_size=len(requests),
                failure="deadline",
            )
            for i in range(len(requests))
        ]
    elapsed = time.monotonic() - start
    return [
        dict(
            source=r.source if r.complete else "",
            complete=r.complete,
            tokens=list(r.tokens[1:]),
            generation_seconds=elapsed,
            generation_batch_size=len(requests),
            failure=getattr(r, "failure", None),
        )
        for r in trajectories
    ]
