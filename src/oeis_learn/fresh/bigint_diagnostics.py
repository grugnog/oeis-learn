"""Visible-only generation/selection and separately mounted finite scoring."""

import time
from collections import Counter, defaultdict

from oeis_learn.synthesis.admission import VisiblePrompt, admit, reference_identity, source_identity
from oeis_learn.synthesis.contracts import canonical_integer, identity
from oeis_learn.synthesis.discovery_status import validate_discovery
from oeis_learn.synthesis.reference import evaluate as reference_evaluate

from .candidate_archive import retain_alternatives
from .conditioning_controls import condition_memory, derangement
from .synthesis_policy import Portfolio, rescue_identity, select_visible, transfer_success
from .template_baseline import template_bank


def _status(complete, outcome, reason=None):
    return dict(complete=bool(complete), outcome=outcome, reason=reason)


def _raw(client, source, prompt, deadline):
    production = client.execute(source, prompt.indices, deadline=deadline)
    if production.get("outcome") != "ok":
        return None, _status(False, "production_failure", production.get("reason"))
    reference = reference_evaluate(
        source, prompt.indices, client.limits, deadline=min(deadline, time.monotonic() + 2)
    )
    if reference.outcome != "ok":
        return None, _status(False, "reference_failure", reference.reason)
    if len(production.get("values", [])) != len(prompt.indices) or reference.coverage != len(
        prompt.indices
    ):
        return None, _status(False, "partial_visible_evidence")
    if list(reference.values) != production["values"]:
        raise RuntimeError("raw production/reference disagreement; reopen Phase2 qualification")
    if list(reference.values) != list(prompt.values):
        return None, _status(True, "visible_mismatch")
    record = dict(
        outcome="assignment_found",
        source=source,
        source_sha256=source_identity(source),
        original_source_sha256=source_identity(source),
        transformed_source=source,
        occurrences=[],
        assignment={},
        indices=list(prompt.indices),
        values=production["values"],
        prompt_sha256=prompt.sha256,
        resource_sha256=client.limits.identity,
        runtime_sha256=client.runtime_identity("bigint-body/v1"),
        reference_sha256=reference_identity(),
        language_profile="bigint-body/v1",
        coverage=len(production["values"]),
        coefficient_bits=4096,
        uniqueness="unchecked",
    )
    result = admit(
        record,
        prompt,
        runtime_sha256=record["runtime_sha256"],
        limits=client.limits,
        deadline=deadline,
    )
    if result.outcome == "verification_failure":
        raise RuntimeError("raw independent admission disagreement")
    found = (
        dict(record, verification=result.verification)
        if result.outcome == "verified_solution"
        else None
    )
    return found, _status(found is not None, result.outcome, result.reason)


def _arm_evidence(details, expected, *, raw=False):
    """An absent selection is resolved only by this arm's own completed work."""
    generation = [item["generation"] for item in details]
    fitting = [item["raw" if raw else "fitting"] for item in details]
    complete = len(details) == expected and all(item["complete"] for item in generation + fitting)
    failure = next((item for item in generation + fitting if not item["complete"]), None)
    reason = (
        (failure["reason"] or failure["outcome"])
        if failure
        else ("attempts_unstarted" if len(details) != expected else fitting[-1]["outcome"])
    )
    return dict(
        complete=complete,
        reason=reason,
        generation=dict(
            expected=expected,
            attempted=len(details),
            completed=sum(item["complete"] for item in generation),
        ),
        fitting=dict(
            required=not raw,
            completed=sum(item["complete"] for item in fitting),
            outcomes=[item["outcome"] for item in fitting],
        ),
    )


def compact_attempts(details):
    """No tokens, hidden values, native assignment payloads or unbounded traces."""
    return [
        {
            key: item[key]
            for key in ("position", "seed", "source_sha256", "generation", "raw", "fitting")
        }
        for item in details
    ]


def _seal(items, prompt, *, cost):
    chosen = select_visible(
        [
            dict(
                source=r["source"],
                source_sha256=r["source_sha256"],
                verified=True,
                visible_count=20,
                cost=r.get("_cost", cost),
            )
            for r in items
        ]
    )
    if chosen is None:
        return None
    record = next(r for r in items if r["source_sha256"] == chosen["source_sha256"])
    return dict(
        selected=chosen,
        verification=record["verification"],
        assignment=record["assignment"],
        original_source=record.get("_original_source", record["transformed_source"]),
        transformed_source=record["transformed_source"],
        original_source_sha256=record["original_source_sha256"],
        prompt_sha256=prompt.sha256,
    )


def _template_candidate(position):
    """Serialize the frozen template through the same codec as generated sources."""
    from oeis_learn.decoder.bigint_codec import decode_body, encode_body

    tokens = encode_body(template_bank()["sources"][position])
    return decode_body(tokens), tokens


def _portfolio(
    decoder,
    memory,
    row,
    recipe,
    client,
    *,
    deadline,
    attempt_count=8,
    template=False,
    compare_cache=False,
):
    from oeis_learn.decoder.sampler import WatProgramSampler

    from .bigint_program_identity import structural_identity

    prompt = VisiblePrompt(tuple(row["indices"]), tuple(row["values"]))
    start = time.monotonic()
    policy = Portfolio(
        start=start,
        seed=recipe["model"]["seed"] + int(identity(row["id"])[7:14], 16),
        beam=recipe["search"]["beam"],
    )
    deadline = min(deadline, policy.deadline)
    fitted = []
    raw = []
    rescued = []
    details = []
    incomplete = False
    parity = None
    for position in range(attempt_count):
        if time.monotonic() >= deadline:
            incomplete = True
            break
        attempt = policy.begin(now=time.monotonic())
        source = ""
        complete = False
        tokens = []
        generated = 0.0
        if template:
            source, tokens = _template_candidate(position)
            complete = True
        else:
            from oeis_learn.decoder.bigint_generation import GenerationRequest

            def budget_hook(_):
                if time.monotonic() >= deadline:
                    raise TimeoutError("visible portfolio deadline")

            result = yield GenerationRequest(
                memory, attempt["seed"], 0 if position == 0 else 0.8, deadline
            )
            source, complete, tokens = result["source"], result["complete"], result["tokens"]
            generated = result["generation_seconds"]
            if time.monotonic() >= deadline:
                source, complete = "", False
            if compare_cache and position == 0 and complete:
                uncached = WatProgramSampler(
                    decoder,
                    max_length=1024,
                    temperature=0,
                    top_p=1,
                    codec_profile="bigint-codec/v1",
                    use_cache=False,
                )
                try:
                    other = uncached.sample_trajectories(
                        memory, [attempt["seed"]], budget_hook=budget_hook
                    )[0]
                    parity = bool(tuple(tokens) == other.tokens[1:] and complete == other.complete)
                    if not parity:
                        raise RuntimeError("cached/full generation disagreement")
                except TimeoutError:
                    parity = None
                    incomplete = True
        policy.finish_generation(
            attempt["id"], tokens=tokens, complete=complete, source=source, now=time.monotonic()
        )
        detail = dict(
            tokens=tokens,
            complete=complete,
            generation_seconds=generated,
            source=source if complete else "",
            outcome="incomplete",
            generation_batch_size=1 if template else result["generation_batch_size"],
            position=position,
            seed=attempt["seed"] % 2**32,
            source_sha256=source_identity(source) if complete else None,
            generation=_status(
                complete,
                "decoded" if complete else "generation_incomplete",
                None if template or complete else result.get("failure"),
            ),
            raw=_status(False, "not_attempted"),
            fitting=_status(False, "not_attempted"),
        )
        details.append(detail)
        if not complete:
            incomplete = True
            continue
        if position == 0 and not template:
            found, detail["raw"] = _raw(client, source, prompt, min(deadline, time.monotonic() + 4))
            if found:
                raw.append(found)
        fitting_start = time.monotonic()
        if recipe["fitting"]["disabled"]:
            found, raw_status = _raw(client, source, prompt, min(deadline, time.monotonic() + 4))
            fit = dict(
                outcome="raw_only",
                admitted=[found] if found else [],
                complete=raw_status["complete"],
                reason=raw_status["reason"],
            )
        else:
            fit = client.fit(
                source,
                prompt,
                policy="none" if "(param.get " in source else "eligible-literals/v1",
                deadline=min(deadline, time.monotonic() + 4),
                seed=attempt["seed"] % 2**32,
            )
        for admitted in fit["admitted"]:
            admitted["_cost"] = generated + time.monotonic() - fitting_start
            admitted["_original_source"] = source
        detail.update(
            outcome=fit["outcome"], native=fit.get("native"), fit_cache=fit.get("metrics")
        )
        native = fit.get("native") or {}
        detail["fitting"] = _status(
            fit.get(
                "complete",
                fit["outcome"]
                in {
                    "verified_solution",
                    "raw_only",
                    "unsupported",
                    "unsat_in_scope",
                    "invalid_source",
                },
            ),
            fit["outcome"],
            fit.get("reason") or native.get("reason"),
        )
        detail["fitting"].update(
            {
                key: native.get(key)
                for key in ("phase", "completed_bounds", "coefficient_bits", "uniqueness")
            }
        )
        detail["fitting"]["admission"] = fit.get("admission", [])
        if fit["outcome"] == "verification_failure":
            raise RuntimeError("independent admission disagreement")
        if fit["outcome"] in {"unknown", "resource_exhausted"}:
            incomplete = True
        fitted.extend(fit["admitted"])
        if position == 0 and not template:
            for record in fit["admitted"]:
                rescue_identity(source_identity(source), record["original_source_sha256"])
            rescued.extend(fit["admitted"])
    cost = time.monotonic() - start
    distinct = {}
    for r in fitted:
        if r["source_sha256"] not in distinct or r["_cost"] < distinct[r["source_sha256"]]["_cost"]:
            distinct[r["source_sha256"]] = r
    archive = (
        retain_alternatives(
            [
                dict(
                    id=r["source_sha256"],
                    structure=structural_identity(r["transformed_source"]),
                    source=r["source"],
                    cost=r["_cost"],
                    verified=True,
                    verification=identity(r["verification"]),
                    track=recipe["track"]["kind"],
                    profile="bigint-body/v1",
                    assignment=r["assignment"],
                )
                for r in distinct.values()
            ]
        )
        if fitted
        else dict(retained=[], omitted=[])
    )
    return dict(
        raw=_seal(raw, prompt, cost=cost),
        rescue=_seal(rescued, prompt, cost=cost),
        equal_budget=_seal(fitted, prompt, cost=cost),
        complete=not incomplete and len(details) == attempt_count,
        elapsed_seconds=cost,
        details=details,
        archive=archive,
        cache_parity=parity,
        arm_evidence=dict(
            raw=_arm_evidence(details[:1], 1, raw=True),
            rescue=_arm_evidence(details[:1], 1),
            equal_budget=_arm_evidence(details, attempt_count),
        ),
    )


def _drive_portfolios(decoder, generators):
    """Interleave decoding only. Fitting/admission remain serial and prompt-local."""
    from oeis_learn.decoder.bigint_generation import GENERATION_BATCH_SIZE, sample_batch

    results, pending = [None] * len(generators), []

    def advance(index, generator, value=None, *, first=False):
        try:
            request = next(generator) if first else generator.send(value)
            pending.append((index, generator, request))
        except StopIteration as done:
            results[index] = done.value

    for index, generator in enumerate(generators):
        advance(index, generator, first=True)
    while pending:
        # Greedy and stochastic distributions never share a sampling call.
        first = pending[0][2]
        group = [
            p
            for p in pending
            if p[2].temperature == first.temperature
            and p[2].memory.shape[1:] == first.memory.shape[1:]
        ][:GENERATION_BATCH_SIZE]
        selected = {p[0] for p in group}
        pending[:] = [p for p in pending if p[0] not in selected]
        sampled = sample_batch(decoder, [p[2] for p in group])
        for (index, generator, _), value in zip(group, sampled):
            advance(index, generator, value)
    return results


def portfolio(decoder, memory, row, recipe, client, **kwargs):
    return _drive_portfolios(decoder, [_portfolio(decoder, memory, row, recipe, client, **kwargs)])[
        0
    ]


def generate(
    encoder, decoder, rows, recipe, client, *, deadline, periodic=False, compare_cache=False
):
    import torch

    if any(len(r["values"]) != 20 or len(r["indices"]) != 20 for r in rows):
        raise ValueError("only20 visible terms enter generation")
    if not rows:
        return dict(records={}, controls={})
    encoder.eval()
    decoder.eval()
    device = next(encoder.parameters()).device
    with torch.no_grad():
        memory = encoder.forward_from_sequences(
            [[canonical_integer(v) for v in r["values"]] for r in rows], device=device
        )
    padding = torch.zeros(memory.shape[:2], dtype=torch.bool, device=device)
    ids = [r["id"] for r in rows]
    mapping = derangement(ids, recipe["model"]["seed"]) if not periodic else {}
    order = [ids.index(mapping[k]) for k in ids] if mapping else None
    modes = ["correct"] if periodic else ["correct", "shuffled", "absent", "template"]
    records = {r["id"]: dict(selections={}, arm_evidence={}, complete=True) for r in rows}
    for mode in modes:
        controlled = (
            None
            if mode == "template"
            else condition_memory(
                memory, padding, mode, order=order if mode == "shuffled" else None
            )[0]
        )
        from oeis_learn.decoder.bigint_generation import GENERATION_BATCH_SIZE

        # A group spends one shared slice of the remaining global budget. Every
        # prompt also retains Portfolio's original sixty-second ceiling.
        size = 1 if mode == "template" else GENERATION_BATCH_SIZE
        for offset in range(0, len(rows), size):
            group = rows[offset : offset + size]
            remaining = (len(modes) - modes.index(mode) - 1) * len(rows) + len(rows) - offset
            now = time.monotonic()
            allowance = max(0, (deadline - now) * len(group) / max(1, remaining))
            until = min(deadline, now + allowance, now + 60)
            generators = [
                _portfolio(
                    decoder,
                    None if controlled is None else controlled[i : i + 1],
                    row,
                    recipe,
                    client,
                    deadline=until,
                    attempt_count=1 if periodic else 8,
                    template=mode == "template",
                    compare_cache=compare_cache and mode == "correct" and i < 3,
                )
                for i, row in enumerate(group, offset)
            ]
            for row, result in zip(group, _drive_portfolios(decoder, generators)):
                record = records[row["id"]]
                record["complete"] &= result["complete"]
                if mode == "correct":
                    record["selections"].update(
                        {key: result[key] for key in ("raw", "rescue", "equal_budget")}
                    )
                    record["arm_evidence"].update(result["arm_evidence"])
                else:
                    record["selections"][mode] = result["equal_budget"]
                    record["arm_evidence"][mode] = result["arm_evidence"]["equal_budget"]
                record.setdefault("details", {})[mode] = result
                record["family"] = row.get("family", row["id"])
                record["strata"] = row.get("strata", strata(row))
    return dict(
        records=records,
        controls=dict(
            derangement=mapping,
            sha256=identity(mapping),
            absent="one-valid-token:first-coordinate-one-rest-zero",
        ),
    )


def score(selections, truth, client, *, deadline):
    # Each arm carries precisely one previously sealed visible-only selection.
    counts = defaultdict(lambda: Counter(denominator=len(truth), complete=0, success=0))
    outcomes = {}
    for key, target in truth.items():
        record = selections.get(key)
        outcomes[key] = {}
        for mode in ("raw", "rescue", "equal_budget", "shuffled", "absent", "template"):
            chosen = record.get("selections", {}).get(mode) if record else None
            evidence = record.get("arm_evidence", {}).get(mode) if record else None
            success = False
            complete = False
            scoring = _status(False, "not_attempted")
            reason = "selection_record_missing" if record is None else "legacy_arm_evidence_missing"
            if chosen is not None and time.monotonic() < deadline:
                expected_prompt = identity(
                    dict(indices=target["indices"][:20], values=target["values"][:20])
                )
                if (
                    chosen["prompt_sha256"] != expected_prompt
                    or chosen["verification"]["prompt_sha256"] != expected_prompt
                ):
                    raise ValueError("sealed selection belongs to another prompt")
                source = chosen["selected"]["source"]
                source_hash = source_identity(source)
                production = client.execute(
                    source, target["indices"], deadline=min(deadline, time.monotonic() + 30)
                )
                reference = reference_evaluate(
                    source,
                    target["indices"],
                    client.limits,
                    deadline=min(deadline, time.monotonic() + 30),
                )
                p = dict(
                    source_sha256=source_hash,
                    indices=target["indices"],
                    values=production.get("values", []),
                    outcome=production.get("outcome"),
                )
                r = dict(
                    source_sha256=source_hash,
                    indices=target["indices"],
                    values=list(reference.values),
                    outcome=reference.outcome,
                )
                success = transfer_success(
                    chosen["selected"], target["indices"], target["values"], p, r
                )
                complete = (
                    p["outcome"] == r["outcome"] == "ok"
                    and len(p["values"]) == len(r["values"]) == len(target["indices"]) == 100
                )
                if complete and p["values"] != r["values"]:
                    raise RuntimeError("production/reference full-horizon disagreement")
                reason = (
                    "success"
                    if success
                    else "continuation_mismatch"
                    if complete
                    else "scoring_incomplete"
                )
                scoring = dict(
                    complete=complete,
                    outcome=reason,
                    reason=None,
                    production=dict(
                        outcome=p["outcome"],
                        reason=production.get("reason"),
                        coverage=len(p["values"]),
                    ),
                    reference=dict(
                        outcome=r["outcome"], reason=reference.reason, coverage=len(r["values"])
                    ),
                )
            elif chosen is not None:
                reason = "scoring_deadline"
                scoring = _status(False, "not_attempted", "deadline")
            elif evidence is not None:
                complete = evidence["complete"] is True
                reason = evidence["reason"]
                scoring = _status(complete, "no_selected_candidate", reason)
            counts[mode]["success"] += success
            counts[mode]["complete"] += complete
            outcomes[key][mode] = dict(
                success=success,
                complete=complete,
                reason=reason,
                candidate_sha256=chosen["selected"]["source_sha256"] if chosen else None,
                upstream=evidence,
                scoring=scoring,
            )
    strata_counts = {}
    for key, record in selections.items():
        for dimension, band in record.get("strata", {}).items():
            for mode, outcome in outcomes.get(key, {}).items():
                name = f"{dimension}:{band}:{mode}"
                group = strata_counts.setdefault(name, dict(denominator=0, success=0, complete=0))
                group["denominator"] += 1
                group["success"] += outcome["success"]
                group["complete"] += outcome["complete"]
    return dict(
        strata=strata_counts,
        visible_fit20={
            mode: sum(bool(row.get("selections", {}).get(mode)) for row in selections.values())
            for mode in counts
        },
        metric="continuation_accuracy_100",
        portfolio="exact_equal_budget",
        splits={
            k: dict(
                v,
                accuracy=v["success"] / v["denominator"] if v["denominator"] else 0.0,
                verified_admissions=True,
                disagreement=False,
                reasons=dict(Counter(row[k]["reason"] for row in outcomes.values())),
            )
            for k, v in counts.items()
        },
        outcomes=outcomes,
        discovery=validate_discovery(
            dict(
                finite_match=identity(outcomes),
                conjecture=None,
                bounded_verification=None,
                proved_statement=None,
                novelty_review=None,
                historical_status=None,
            )
        ),
        sensors=dict(
            utilization=None, reason="no portable utilization sensor in algorithm integration"
        ),
    )


def strata(row):
    from oeis_learn.synthesis.contracts import canonical_integer

    sizes = [abs(canonical_integer(v)).bit_length() for v in row["values"]]
    maximum = max(sizes, default=0)
    length = len(row.get("tokens", []))
    source = row.get("source", "")
    return dict(
        size=next(
            label
            for upper, label in (
                (64, "0-64"),
                (256, "65-256"),
                (512, "257-512"),
                (1024, "513-1024"),
                (4096, "1025-4096"),
            )
            if maximum <= upper
        ),
        length=next(
            label
            for upper, label in (
                (64, "0-64"),
                (256, "65-256"),
                (512, "257-512"),
                (1024, "513-1024"),
            )
            if length <= upper
        ),
        complexity="loop" if "(loop " in source else "branch" if "(if " in source else "expression",
    )
