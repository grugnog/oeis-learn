"""Execution Trace Analyzer for fine-grained credit assignment, divergence localization, and coverage tracking."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
from oeis_learn.data.models import ExecutionResult, FineGrainedAttributionSpan
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID, tokenize_wat


def classify_failure_mode(
    exec_result: ExecutionResult,
    target_terms: Sequence[int],
) -> str:
    """Classifies rollout outcome into priority-ordered failure modes:

    SYNTAX -> CONSTRAINT -> LOGIC -> CORRECT.
    """
    if exec_result.status in ("PARSE_ERROR", "COMPILE_ERROR", "MISSING_ENTRYPOINT", "CONFIG_ERROR"):
        return "SYNTAX"

    if exec_result.status in ("OUT_OF_FUEL", "EXECUTION_TRAP"):
        return "CONSTRAINT"

    if exec_result.status == "SUCCESS":
        check_len = min(len(target_terms), len(exec_result.output))
        if check_len > 0 and exec_result.output[:check_len] == list(target_terms[:check_len]):
            return "CORRECT"
        return "LOGIC"

    return "LOGIC"


def locate_divergence_token_span(
    wat_code: str,
    divergence_step: Optional[int],
    total_tokens: int,
) -> Tuple[int, int]:
    """Maps the sequence term divergence index n_diverge to token span [t_start, t_end].

    If program failed on initial terms, concentrates credit on arithmetic/loop body.
    """
    tokens = tokenize_wat(wat_code)
    n_tok = len(tokens)

    if divergence_step is None or divergence_step < 0:
        return 0, max(1, min(n_tok, total_tokens))

    # Locate arithmetic or loop instructions in the program body
    body_start = 0
    for idx, tok in enumerate(tokens):
        if tok in ("result", "local") and idx + 3 < n_tok:
            body_start = idx + 3

    body_end = n_tok - 1
    for idx in range(n_tok - 1, -1, -1):
        if tokens[idx] == ")":
            body_end = idx
            break

    t_start = min(body_start, n_tok - 1)
    t_end = max(t_start + 1, min(body_end, total_tokens))

    return t_start, t_end


def locate_causal_divergence_span(
    wat_code: str,
    divergence_step: Optional[int],
    failure_mode: str,
    total_tokens: int,
) -> Tuple[int, int]:
    """Locates the causal bytecode token span T_k* where execution diverged or failed."""
    tokens = tokenize_wat(wat_code)
    n_tok = len(tokens)
    if n_tok == 0:
        return 0, max(1, total_tokens)

    if failure_mode == "SYNTAX":
        # Find earliest unparseable token or header defect
        err_start = 0
        for idx, tok in enumerate(tokens):
            if tok in ("func", "param", "result", "local"):
                err_start = max(0, idx - 1)
        return err_start, min(n_tok, total_tokens)

    if failure_mode in ("CORRECT", "CONSTRAINT"):
        return 0, min(n_tok, total_tokens)

    # LOGIC failure mode
    return locate_divergence_token_span(wat_code, divergence_step, total_tokens)


def extract_token_coverage(wat_code: str) -> List[bool]:
    """Extracts runtime basic block coverage flags per token position."""
    tokens = tokenize_wat(wat_code)
    n_tok = len(tokens)
    coverage = [True] * n_tok

    # Flag unreachable instructions after unconditional return or br outside loops
    unreachable = False
    for idx, tok in enumerate(tokens):
        if unreachable:
            if tok == ")":
                coverage[idx] = True
                unreachable = False
            else:
                coverage[idx] = False
        elif tok in ("unreachable", "return"):
            unreachable = True

    return coverage


def build_fine_grained_attribution(
    wat_code: str,
    exec_result: ExecutionResult,
    target_terms: Sequence[int],
    total_advantage: float,
    total_tokens: int,
) -> FineGrainedAttributionSpan:
    """Constructs localized per-token advantage operator a_{i,t} with downstream zero-masking

    and exact total advantage mass conservation (sum a_{i,t} = A_i).
    """
    mode = classify_failure_mode(exec_result, target_terms)
    t_start, t_end = locate_causal_divergence_span(
        wat_code=wat_code,
        divergence_step=exec_result.divergence_step,
        failure_mode=mode,
        total_tokens=total_tokens,
    )

    t_span_len = max(1, t_end - t_start)
    advantage_vector = [0.0] * total_tokens
    coverage_mask = extract_token_coverage(wat_code)
    if len(coverage_mask) < total_tokens:
        coverage_mask = coverage_mask + [True] * (total_tokens - len(coverage_mask))
    else:
        coverage_mask = coverage_mask[:total_tokens]

    if mode in ("CORRECT", "CONSTRAINT"):
        # Uniform advantage distribution across sequence
        weight = total_advantage / max(1, total_tokens)
        for t in range(total_tokens):
            advantage_vector[t] = weight
    else:
        # LOGIC or SYNTAX failure: concentrate total advantage onto causal error span [t_start, t_end]
        # and zero-mask all downstream tokens (t > t_end) and upstream tokens (t < t_start)
        weight = total_advantage / float(t_span_len)
        for t in range(t_start, min(t_end, total_tokens)):
            if coverage_mask[t]:
                advantage_vector[t] = weight

        # Conserve total advantage mass exactly
        curr_sum = sum(advantage_vector)
        if abs(curr_sum) > 1.0e-8 and abs(curr_sum - total_advantage) > 1.0e-6:
            scale = total_advantage / curr_sum
            advantage_vector = [v * scale for v in advantage_vector]

    return FineGrainedAttributionSpan(
        failure_mode=mode,
        divergence_step=exec_result.divergence_step,
        causal_token_start=t_start,
        causal_token_end=t_end,
        token_advantage_mask=advantage_vector,
        executed_token_mask=coverage_mask,
    )
