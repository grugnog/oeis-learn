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
    report = {"version": "bootstrap-diagnostic/v2", "stage": stage, "splits": {}}
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
            if now >= deadline:
                record["failure"] = "diagnostic_deadline"
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
                    continue
                raise RuntimeError(f"admitted diagnostic oracle failed: {failure}")
            variants = ["correct", "shuffled"]
            if compare_generation and row["program_id"] in compare_ids:
                variants.append("uncached")
                record["generation_parity"] = dict(evaluated=False, passed=False)
            for variant in variants:
                if variant == "shuffled" and len(rows) < 2:
                    record["shuffle_unavailable"] = True
                    continue
                conditioning = (
                    list(map(int, rows[(i + 1) % len(rows)]["values"]))
                    if variant == "shuffled"
                    else visible
                )
                budget = row_deadline - time.monotonic_ns()
                failure_key = "failure" if variant == "correct" else variant + "_failure"
                if budget <= 0:
                    record[failure_key] = "diagnostic_deadline"
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
                            row_deadline,
                        )
                        if evidence["complete"]
                        else (None, evidence["failure"])
                    )
                    matched = values is not None and values == expected
                    if variant == "shuffled":
                        record.update(
                            shuffled_evaluated=time.monotonic_ns() < row_deadline,
                            shuffled_matched=matched,
                            shuffled_failure=outcome,
                        )
                    elif variant == "uncached":
                        equal = evidence["tokens"] == record["tokens"]
                        record["generation_parity"] = dict(
                            evaluated=True,
                            passed=equal and evidence["complete"] == record["complete"],
                            tokens=evidence["tokens"],
                            complete=evidence["complete"],
                            matched=matched,
                            failure=outcome,
                        )
                    else:
                        legal = False
                        if evidence["complete"]:
                            parse(source)  # Complete grammar output must also parse independently.
                            legal = True
                        prefix = values[:20] if values is not None else None
                        if legal and values is None and time.monotonic_ns() < row_deadline:
                            prefix, _ = execute_values(
                                runner,
                                source,
                                range(20),
                                digest([split, row["program_id"], "prefix"]),
                                row_deadline,
                            )
                        record.update(
                            source=source,
                            evaluated=True,
                            legal=legal,
                            model_provenance={
                                k: evidence[k]
                                for k in ("method", "encoder_sha256", "decoder_sha256")
                            },
                            tokens=evidence["tokens"],
                            eos_trace=evidence.get("eos_trace", []),
                            complete=evidence["complete"],
                            emitted_eos=EOS_ID in evidence["tokens"],
                            matched=matched,
                            prefix_matched=prefix == expected[:20],
                            continuation_matched=values is not None
                            and values[20:] == expected[20:],
                            output_sha256=digest(values) if values is not None else None,
                            failure=outcome if outcome else None if matched else "wrong_value",
                        )
                except SearchExhausted as exc:
                    record[failure_key] = str(exc)
                    if variant == "correct":
                        record.update(
                            tokens=list(getattr(exc, "tokens", [])),
                            eos_trace=list(getattr(exc, "eos_trace", [])),
                        )
        summary = summarize(attempts)
        summary.update(
            attempts=attempts, available=len(examples[split]), by_length={}, by_complexity={}
        )
        for field, target in (("length_band", "by_length"), ("complexity", "by_complexity")):
            for value in sorted({r[field] for r in attempts}):
                summary[target][value] = summarize([r for r in attempts if r[field] == value])
        summary["coverage_complete"] = summary["evaluated"] == summary["denominator"]
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
    report["identity"] = digest(report)
    return report


def competence(report, split="development"):
    row = report["splits"][split]
    return row["exact"] / row["denominator"] if row["denominator"] else None
