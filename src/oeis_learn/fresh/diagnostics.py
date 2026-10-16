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


def evaluate(encoder, decoder, examples, learning, runner, *, stage=None):
    import torch
    from oeis_learn.decoder.program_codec import BOS_ID, EOS_ID, decode_body
    from oeis_learn.evaluation.search import ModelProposer, SearchLedger, SearchExhausted
    from oeis_learn.rl.policy import scoring_mode

    deadline = time.monotonic_ns() + learning["diagnostic_seconds"] * 10**9
    proposer = ModelProposer(encoder, decoder, context_limit=learning["diagnostic_tokens"] + 1)
    report = {"version": "bootstrap-diagnostic/v1", "stage": stage, "splits": {}}
    for split in ("training", "development"):
        rows = choose_rows(examples[split], learning["diagnostic_rows"], stage)
        attempts, failures = [], Counter()
        for i, row in enumerate(rows):
            record = {
                "program_id": row["program_id"],
                "matched": False,
                "shuffled_matched": False,
                "shuffled_evaluated": False,
                "tokens": [],
                "complete": False,
            }
            if time.monotonic_ns() >= deadline:
                record["failure"] = "diagnostic_deadline"
                failures[record["failure"]] += 1
                attempts.append(record)
                continue
            visible = list(map(int, row["values"]))
            expected, failure = execute_values(
                runner,
                decode_body(row["tokens"]),
                range(100),
                digest([split, row["program_id"], "oracle"]),
                deadline,
            )
            if expected is None and time.monotonic_ns() >= deadline:
                record["failure"] = "diagnostic_deadline"
                failures[record["failure"]] += 1
                attempts.append(record)
                continue
            if expected is None:
                raise RuntimeError(f"admitted diagnostic oracle failed: {failure}")
            for shuffled in (False, True):
                if shuffled and len(rows) < 2:
                    record["shuffle_unavailable"] = True
                    continue
                conditioning = (
                    list(map(int, rows[(i + 1) % len(rows)]["values"])) if shuffled else visible
                )
                budget = deadline - time.monotonic_ns()
                if budget <= 0:
                    record["shuffle_failure" if shuffled else "failure"] = "diagnostic_deadline"
                    continue
                ledger = SearchLedger(
                    max_attempts=1,
                    max_tokens=learning["diagnostic_tokens"],
                    max_expansions=learning["diagnostic_tokens"],
                    budget_ns=budget,
                )
                try:
                    ledger.charge(attempts=1)
                    source, evidence = proposer.stochastic(conditioning, i, ledger, greedy=True)
                    values, outcome = (
                        execute_values(
                            runner,
                            source,
                            range(100),
                            digest([split, row["program_id"], shuffled]),
                            deadline,
                        )
                        if evidence["complete"]
                        else (None, evidence["failure"])
                    )
                    matched = values == expected
                    if shuffled:
                        record["shuffled_evaluated"] = time.monotonic_ns() < deadline
                        record["shuffled_matched"] = matched
                        record["shuffle_failure"] = outcome
                    else:
                        record.update(
                            source=source,
                            model_provenance={
                                k: evidence[k]
                                for k in ("method", "encoder_sha256", "decoder_sha256")
                            },
                            tokens=evidence["tokens"],
                            complete=evidence["complete"],
                            emitted_eos=EOS_ID in evidence["tokens"],
                            matched=matched,
                            failure=outcome if outcome else None if matched else "wrong_value",
                        )
                except SearchExhausted as exc:
                    record["shuffle_failure" if shuffled else "failure"] = str(exc)
                    if not shuffled:
                        record["tokens"] = list(getattr(exc, "tokens", []))
            failures[record.get("failure") or "exact_match"] += 1
            attempts.append(record)
        denominator = len(rows)
        report["splits"][split] = dict(
            denominator=denominator,
            attempts=attempts,
            exact=sum(r["matched"] for r in attempts),
            shuffled_exact=sum(r["shuffled_matched"] for r in attempts),
            shuffled_denominator=sum(r["shuffled_evaluated"] for r in attempts),
            completed=sum(r["complete"] for r in attempts),
            failures=dict(failures),
        )
    # Compare logits on the same teacher-forced prefix, independently of sampling.
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
