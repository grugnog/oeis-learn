"""Bounded production-decoder checks on admitted synthetic examples only."""

import time
from collections import Counter

from .common import digest


def execute_values(runner, source, indices, request_id, deadline):
    result = runner.run_single(
        source, indices, request_id=request_id, deadline_ns=deadline
    ).to_dict()
    if result["outcome"] is not None:
        return None, result["outcome"]
    if result["outputs"] != result["reference_outputs"]:
        raise RuntimeError("native/reference disagreement in diagnostic execution")
    return result["outputs"], None


def choose_rows(rows, count, stage=None):
    selected = [r for r in rows if stage is None or r.get("stage", 0) == stage]
    return sorted(selected, key=lambda r: r["program_id"])[:count]


def strata(row):
    from oeis_learn.decoder.program_codec import decode_body

    length = len(row["tokens"])
    source = decode_body(row["tokens"])
    complexity = (
        "loop"
        if "loop" in source
        else "branch"
        if "if" in source
        else "assignment"
        if "local.set" in source
        else "expression"
    )
    return dict(
        target_tokens=length,
        length_band=next(
            label
            for upper, label in (
                (64, "1-64"),
                (256, "65-256"),
                (512, "257-512"),
                (1024, "513-1024"),
            )
            if length <= upper
        ),
        complexity=complexity,
        stage=row.get("stage"),
    )


def summarize(attempts):
    outputs = [r["output_sha256"] for r in attempts if r.get("output_sha256")]
    comparisons = [r["generation_parity"] for r in attempts if "generation_parity" in r]
    return dict(
        denominator=len(attempts),
        evaluated=sum(r.get("evaluated", False) for r in attempts),
        exact=sum(r["matched"] for r in attempts),
        shuffled_exact=sum(r["shuffled_matched"] for r in attempts),
        shuffled_denominator=sum(r["shuffled_evaluated"] for r in attempts),
        completed=sum(r["complete"] for r in attempts),
        legal=sum(r.get("legal", False) for r in attempts),
        emitted_eos=sum(r.get("emitted_eos", False) for r in attempts),
        prefix_exact=sum(r.get("prefix_matched", False) for r in attempts),
        continuation_exact=sum(r.get("continuation_matched", False) for r in attempts),
        output_diversity=dict(executed=len(outputs), distinct=len(set(outputs))),
        failures=dict(Counter(r.get("failure") or "exact_match" for r in attempts)),
        outcomes=dict(Counter(r.get("outcome", "not_evaluated") for r in attempts)),
        shuffled_outcomes=dict(
            Counter(r.get("shuffled_outcome", "not_evaluated") for r in attempts)
        ),
        generation_parity=dict(
            requested=len(comparisons),
            evaluated=sum(c["evaluated"] for c in comparisons),
            passed=bool(comparisons) and all(c["evaluated"] and c["passed"] for c in comparisons),
        ),
    )


def evaluate(
    encoder,
    decoder,
    examples,
    learning,
    runner,
    *,
    stage=None,
    full_development=False,
    compare_generation=False,
    detail=False,
):
    import torch
    from oeis_learn.decoder.program_codec import BOS_ID, EOS_ID, decode_body
    from oeis_learn.evaluation.search import ModelProposer, SearchLedger, SearchExhausted
    from oeis_learn.rl.policy import scoring_mode
    from oeis_learn.sandbox.wat_ast import parse

    deadline = time.monotonic_ns() + learning["diagnostic_seconds"] * 10**9
    proposer = ModelProposer(
        encoder, decoder, context_limit=learning["diagnostic_tokens"] + 1, diagnostics=True
    )
    uncached = ModelProposer(
        encoder,
        decoder,
        context_limit=learning["diagnostic_tokens"] + 1,
        use_cache=False,
        diagnostics=True,
    )
    report = {"version": "bootstrap-diagnostic/v3", "stage": stage, "splits": {}}
    selected = {
        split: choose_rows(
            rows,
            len(rows) if full_development and split != "training" else learning["diagnostic_rows"],
            stage,
        )
        for split, rows in examples.items()
        if split in {"training", "development", "parameter_development"}
    }
    remaining_rows = sum(map(len, selected.values()))
    for split, rows in selected.items():
        attempts = []
        representatives = sorted(rows, key=lambda r: (len(r["tokens"]), r["program_id"]))
        compare_ids = {r["program_id"] for r in representatives[:: max(1, len(rows) // 3)][:3]}
        for i, row in enumerate(rows):
            # Divide remaining time fairly: one long decode cannot consume the
            # entire development view. Unfinished rows remain in the denominator.
            now = time.monotonic_ns()
            row_deadline = min(deadline, now + max(0, (deadline - now) // max(1, remaining_rows)))
            generation_deadline = (
                now + (row_deadline - now) * (4 if detail and split == "training" else 5) // 5
            )
            remaining_rows -= 1
            record = dict(
                program_id=row["program_id"],
                matched=False,
                shuffled_matched=False,
                shuffled_evaluated=False,
                evaluated=False,
                tokens=[],
                complete=False,
                eos_trace=[],
                **strata(row),
            )
            attempts.append(record)
            compare = compare_generation and (
                split == "training" or row["program_id"] in compare_ids
            )
            if compare:
                record["generation_parity"] = dict(evaluated=False, passed=False)
            if now >= deadline:
                record["failure"] = "diagnostic_deadline"
                record["outcome"] = "timeout"
                continue
            visible = list(map(int, row["values"]))
            expected, failure = execute_values(
                runner,
                decode_body(row["tokens"]),
                range(100),
                digest([split, row["program_id"], "oracle"]),
                row_deadline,
            )
            if expected is None:
                if time.monotonic_ns() >= row_deadline:
                    record["failure"] = "diagnostic_deadline"
                    record["outcome"] = "timeout"
                    continue
                raise RuntimeError(f"admitted diagnostic oracle failed: {failure}")
            variants = ["correct", "shuffled"]
            if compare:
                variants.append("uncached")
            for variant in variants:
                conditioning = visible
                if variant == "shuffled":
                    alternatives = (
                        list(map(int, rows[(i + offset) % len(rows)]["values"]))
                        for offset in range(1, len(rows))
                    )
                    conditioning = next(
                        (values for values in alternatives if values != visible), None
                    )
                    if conditioning is None:
                        record["shuffle_unavailable"] = True
                        continue
                # Independent shares prevent a slow correct decode from consuming
                # the substituted-prompt and cache-control evaluations as well.
                variant_deadline = time.monotonic_ns() + max(
                    0,
                    (generation_deadline - time.monotonic_ns())
                    // (len(variants) - variants.index(variant)),
                )
                budget = min(300_000_000_000, variant_deadline - time.monotonic_ns())
                failure_key = "failure" if variant == "correct" else variant + "_failure"
                if budget <= 0:
                    record[failure_key] = "diagnostic_deadline"
                    record["outcome" if variant == "correct" else variant + "_outcome"] = "timeout"
                    continue
                ledger = SearchLedger(
                    max_attempts=1,
                    max_tokens=learning["diagnostic_tokens"],
                    max_expansions=learning["diagnostic_tokens"],
                    budget_ns=budget,
                )
                try:
                    ledger.charge(attempts=1)
                    source, evidence = (uncached if variant == "uncached" else proposer).stochastic(
                        conditioning, i, ledger, greedy=True
                    )
                    values, outcome = (
                        execute_values(
                            runner,
                            source,
                            range(100),
                            digest([split, row["program_id"], variant]),
                            variant_deadline,
                        )
                        if evidence["complete"] and variant != "uncached"
                        else (None, evidence["failure"])
                    )
                    matched = values is not None and values == expected
                    timed_out = time.monotonic_ns() >= variant_deadline
                    matched = matched and not timed_out
                    outcome_class = (
                        "timeout"
                        if timed_out
                        else "incomplete"
                        if not evidence["complete"]
                        else "execution_failure"
                        if outcome
                        else "correct"
                        if matched
                        else "incorrect"
                    )
                    if variant == "shuffled":
                        record.update(
                            shuffled_evaluated=not timed_out
                            and outcome_class != "execution_failure",
                            shuffled_matched=matched,
                            shuffled_failure=outcome,
                            shuffled_outcome=outcome_class,
                        )
                    elif variant == "uncached":
                        equal = evidence["tokens"] == record["tokens"]
                        record["generation_parity"] = dict(
                            evaluated=not timed_out and record.get("evaluated", False),
                            passed=not timed_out
                            and record.get("evaluated", False)
                            and equal
                            and evidence["complete"] == record["complete"],
                            tokens=evidence["tokens"],
                            complete=evidence["complete"],
                            matched=record["matched"] if equal else None,
                            failure=outcome,
                            token_trace=evidence.get("token_trace", []),
                        )
                    else:
                        legal = False
                        if evidence["complete"]:
                            parse(source)  # Complete grammar output must also parse independently.
                            legal = True
                        prefix = values[:20] if values is not None else None
                        if legal and values is None and time.monotonic_ns() < variant_deadline:
                            prefix, _ = execute_values(
                                runner,
                                source,
                                range(20),
                                digest([split, row["program_id"], "prefix"]),
                                variant_deadline,
                            )
                        if time.monotonic_ns() >= variant_deadline:
                            timed_out, matched, outcome_class = True, False, "timeout"
                        record.update(
                            source=source,
                            evaluated=not timed_out and outcome_class != "execution_failure",
                            legal=legal,
                            model_provenance={
                                k: evidence[k]
                                for k in ("method", "encoder_sha256", "decoder_sha256")
                            },
                            tokens=evidence["tokens"],
                            eos_trace=evidence.get("eos_trace", []),
                            token_trace=evidence.get("token_trace", []),
                            complete=evidence["complete"],
                            emitted_eos=EOS_ID in evidence["tokens"],
                            matched=matched,
                            prefix_matched=not timed_out and prefix == expected[:20],
                            continuation_matched=not timed_out
                            and values is not None
                            and values[20:] == expected[20:],
                            output_sha256=digest(values) if values is not None else None,
                            failure=outcome if outcome else None if matched else "wrong_value",
                            outcome=outcome_class,
                        )
                except SearchExhausted as exc:
                    record[failure_key] = str(exc)
                    timeout = exc.reason == "deadline" or time.monotonic_ns() >= variant_deadline
                    record["outcome" if variant == "correct" else variant + "_outcome"] = (
                        "timeout" if timeout else "incomplete"
                    )
                    if variant == "correct":
                        record["evaluated"] = not timeout
                    elif variant == "shuffled":
                        record["shuffled_evaluated"] = not timeout
                    if variant == "correct":
                        record.update(
                            tokens=list(getattr(exc, "tokens", [])),
                            eos_trace=list(getattr(exc, "eos_trace", [])),
                            token_trace=list(getattr(exc, "token_trace", [])),
                        )
            if detail and split == "training":
                record["visible_values"] = row["values"]
                record["target_token_ids"] = row["tokens"]
                if time.monotonic_ns() < row_deadline:
                    from .token_diagnostic import teacher_trace

                    record["teacher_forced"] = teacher_trace(encoder, decoder, row)
                else:
                    record["teacher_forced_failure"] = "diagnostic_deadline"
        summary = summarize(attempts)
        summary.update(
            attempts=attempts, available=len(examples[split]), by_length={}, by_complexity={}
        )
        for field, target in (("length_band", "by_length"), ("complexity", "by_complexity")):
            for value in sorted({r[field] for r in attempts}):
                summary[target][value] = summarize([r for r in attempts if r[field] == value])
        summary["coverage_complete"] = summary["evaluated"] == summary["denominator"]
        summary["conditioning_coverage_complete"] = (
            summary["shuffled_denominator"] == summary["denominator"]
        )
        report["splits"][split] = summary
    # Teacher-forced logit equivalence is separate from free-running token parity.
    row = min(examples["training"], key=lambda r: len(r["tokens"]))
    device = next(encoder.parameters()).device
    with scoring_mode(encoder, decoder), torch.no_grad():
        memory = encoder.forward_from_sequences([list(map(int, row["values"]))], device=device)
        tokens = torch.tensor([[BOS_ID] + row["tokens"][:-1][:31]], device=device)
        full = decoder(tokens, memory)
        cached, _ = decoder.prefill(tokens, memory, capacity=max(32, tokens.shape[1]))
        error = (full - cached).abs().max().item()
        report["cache_parity"] = dict(
            max_absolute_error=error,
            passed=bool(torch.allclose(full, cached, atol=2e-5, rtol=2e-4)),
        )
    comparisons = [
        a["generation_parity"]
        for s in report["splits"].values()
        for a in s["attempts"]
        if "generation_parity" in a
    ]
    report["generation_parity"] = dict(
        required=compare_generation,
        requested=len(comparisons),
        evaluated=sum(c["evaluated"] for c in comparisons),
        passed=bool(comparisons) and all(c["evaluated"] and c["passed"] for c in comparisons),
    )
    teacher_checks = [
        a["teacher_forced"]["cache_parity"]
        for s in report["splits"].values()
        for a in s["attempts"]
        if "teacher_forced" in a
    ]
    if teacher_checks:
        report["cache_parity"]["passed"] &= all(c["passed"] for c in teacher_checks)
        report["cache_parity"]["max_absolute_error"] = max(
            report["cache_parity"]["max_absolute_error"],
            *(c["max_absolute_error"] for c in teacher_checks),
        )
    report["identity"] = digest(report)
    return report


def competence(report, split="development"):
    row = report["splits"][split]
    return row["exact"] / row["denominator"] if row["denominator"] else None
