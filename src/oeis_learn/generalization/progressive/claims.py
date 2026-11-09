"""Sealed, training-ineligible discovery dossiers with separate evidence axes.

A review policy is a frozen trust input, never inferred from a model label or a
hash. Finite execution does not establish novelty or an all-domain theorem.
"""

import copy
import json
import time
from pathlib import Path

from .contracts import atomic_write, digest, load, record, reference, seal, validate, verify_refs

STATEMENTS = {"finite_agreement", "conjecture", "bounded_verified", "proved"}
NOVELTY = {"not_reviewed", "known", "uncertain", "potentially_novel", "independently_confirmed"}


def validate_claim(claim, *, reviewers, source=None, candidate=None, proof=None):
    validate_reviewers(reviewers)
    if (
        not isinstance(claim, dict)
        or set(claim) != {"statement", "novelty", "proof", "review"}
        or claim["statement"] not in STATEMENTS
        or claim["novelty"] not in NOVELTY
    ):
        raise ValueError("closed independent statement/novelty axes required")
    if claim["statement"] == "proved":
        if source is None or claim["proof"] is None:
            raise ValueError("source-bound registered proof required")
        try:
            from .polynomial_certificate import check
        except ImportError:
            raise ValueError("registered bigint polynomial checker unavailable") from None
        checked = check(source, proof if proof is not None else claim["proof"])
        if checked["statement"] != "proved":
            raise ValueError("unsupported proof promotion")
    elif claim["proof"] is not None:
        raise ValueError("unreplayed proof cannot accompany an unsupported claim")
    review = claim["review"]
    if claim["novelty"] == "independently_confirmed":
        if (
            not isinstance(review, dict)
            or set(review) != {"reviewer", "evidence", "candidate", "scope"}
            or review["reviewer"] not in reviewers
            or candidate is None
            or review["candidate"] != candidate
        ):
            raise ValueError(
                "named authorized independent reviewer and candidate-bound evidence required"
            )
        trusted = reviewers[review["reviewer"]]
        if (
            set(trusted) != {"name", "independent", "evidence", "candidate", "scope"}
            or trusted["independent"] is not True
            or not isinstance(trusted["name"], str)
            or not trusted["name"].strip()
            or review["reviewer"].lower() in {"model", "llm", "self"}
            or any(trusted[k] != review[k] for k in ("evidence", "candidate", "scope"))
            or not isinstance(review["scope"], str)
            or not review["scope"].strip()
        ):
            raise ValueError("original independent reviewer policy differs")
    elif review is not None:
        raise ValueError("review authority is reserved for independently confirmed scoped novelty")
    return copy.deepcopy(claim)


def _selected(root, selection):
    value = json.loads(verify_refs(root, [selection["artifact"]], max_bytes=256 << 20)[0])
    if (
        set(value)
        != {
            "version",
            "identity",
            "model",
            "checkpoint",
            "model_control",
            "recipe",
            "methods",
            "prompts",
            "records",
            "clock",
            "counts",
            "elapsed_us",
        }
        or value.get("version") != "discovery-candidates/v1"
        or seal(value) != value
        or selection["method"] != "equal_budget"
    ):
        raise ValueError("original sealed generator selection required")
    chosen = value["records"][selection["task"]]["selections"][selection["method"]]
    if chosen is None or chosen["selected"]["verified"] is not True:
        raise ValueError("concrete selected candidate required")
    return value, chosen


def seal_candidate(
    root,
    candidates,
    task,
    *,
    source_snapshot,
    offset,
    budget,
    output,
    library=None,
    purpose="research",
    track="synthetic_strict",
):
    from .budgets import vector
    from oeis_learn.synthesis.contracts import canonical_integer
    from oeis_learn.decoder.bigint_codec import encode_body

    selection = dict(artifact=candidates, task=task, method="equal_budget")
    generated, chosen = _selected(root, selection)
    vector(budget)
    attempts = sum(
        len(v.get("details", []))
        for v in generated["records"][task]["costs"].values()
        if isinstance(v, dict)
    )
    if (
        type(generated["elapsed_us"]) is not int
        or not 0 <= generated["elapsed_us"] <= budget["elapsed_us"]
        or attempts > budget["attempts"]
    ):
        raise ValueError("original generation exceeds recorded budget")
    source = chosen["selected"]["source"]
    if bool(generated["records"][task]["costs"].get("components")) != (library is not None):
        raise ValueError("selected component library provenance must be retained")
    encode_body(source)
    canonical_integer(offset)
    if library is not None:
        from .component_search import validate_input

        lib = validate_input(json.loads(verify_refs(root, [library], max_bytes=256 << 20)[0]))
        if (
            lib["library"]["track"] != track
            or lib["library"]["purpose"] != purpose
            or generated["records"][task]["costs"]["components"]["library"]
            != lib["library"]["identity"]
        ):
            raise ValueError("original selected library/source track differs")
    verify_refs(root, [source_snapshot])
    value = seal(
        dict(
            version="progressive-claim-candidate/v1",
            source=source,
            source_hash=digest(source),
            selection=selection,
            source_snapshot=source_snapshot,
            offset=offset,
            generation_budget=vector(budget),
            library=library,
            purpose=purpose,
            track=track,
        )
    )
    path = Path(root) / output
    from oeis_learn.pipeline_validation.identity import confined

    path = confined(root, output, exists=False)
    atomic_write(path, value)
    return reference(root, path)


def candidate(root, ref):
    value = json.loads(verify_refs(root, [ref], max_bytes=256 << 20)[0])
    fields = {
        "version",
        "identity",
        "source",
        "source_hash",
        "selection",
        "source_snapshot",
        "offset",
        "generation_budget",
        "library",
        "purpose",
        "track",
    }
    if (
        set(value) != fields
        or value["version"] != "progressive-claim-candidate/v1"
        or seal(value) != value
        or value["source_hash"] != digest(value["source"])
    ):
        raise ValueError("original candidate sealed before analysis required")
    _, selected = _selected(root, value["selection"])
    if selected["selected"]["source"] != value["source"]:
        raise ValueError("candidate source differs from original selection")
    verify_refs(root, [value["source_snapshot"]])
    if value["library"] is not None:
        from .component_search import validate_input

        validate_input(json.loads(verify_refs(root, [value["library"]], max_bytes=256 << 20)[0]))
    return value


def create_dossier(
    root,
    candidate_ref,
    *,
    indices,
    expected,
    output,
    client=None,
    assumptions=(),
    known_baselines=(),
    untested_scope=(
        "All indices outside the tested range; unbounded checked-machine safety unverified",
    ),
    budget=None,
):
    from oeis_learn.synthesis.admission import reference_identity
    from oeis_learn.synthesis.contracts import canonical_integer
    from oeis_learn.synthesis.reference import evaluate

    c = candidate(root, candidate_ref)
    if c["purpose"] == "research" and client is None:
        raise ValueError("actual native independent dossier execution required")
    if len(indices) != 100 or len(expected) != 100:
        raise ValueError("exact bounded hundred-term dossier evidence required")
    xs = [canonical_integer(x) for x in indices]
    for v in expected:
        canonical_integer(v)
    if xs != list(range(xs[0], xs[0] + 100)) or str(xs[0]) != c["offset"]:
        raise ValueError("source offset and tested range differ")
    if budget:
        budget.consume(attempts=1, rejections=1)
    deadline = min(time.monotonic() + 4, budget.deadline) if budget else time.monotonic() + 4
    result = evaluate(c["source"], indices, deadline=deadline)
    if result.outcome != "ok":
        raise ValueError("independent finite execution unresolved")
    native = client.execute(c["source"], indices, deadline=deadline) if client else None
    if native is not None and (
        native.get("outcome") != "ok" or native.get("values") != list(result.values)
    ):
        raise ValueError("native/reference dossier execution disagreement")
    if time.monotonic() >= deadline:
        raise TimeoutError("late dossier verification")
    verify_refs(root, list(known_baselines))
    from oeis_learn.pipeline_validation.identity import confined

    execution = seal(
        dict(
            version="progressive-claim-execution/v1",
            source=c["source_hash"],
            indices=indices,
            expected=digest(expected),
            observed=list(result.values),
            reference=reference_identity(),
            native=native,
            runtime=client.runtime_identity("bigint-body/v1") if client else None,
        )
    )
    execution_path = confined(root, str(output) + ".execution.json", exists=False)
    atomic_write(execution_path, execution)
    counterexamples = [
        dict(index=i, expected=e, observed=v)
        for i, e, v in zip(indices, expected, result.values)
        if e != v
    ]
    payload = dict(
        candidate=candidate_ref,
        source=c["source"],
        source_hash=c["source_hash"],
        selection=c["selection"],
        source_snapshot=c["source_snapshot"],
        offset=c["offset"],
        library=c["library"],
        generation_budget=c["generation_budget"],
        tested=dict(
            indices=indices,
            expected=expected,
            observed=list(result.values),
            agreement=not counterexamples,
            native=client.runtime_identity("bigint-body/v1") if client else None,
            reference=reference_identity(),
            scope="finite_100_terms_only",
            execution=reference(root, execution_path),
        ),
        assumptions=list(assumptions),
        known_baselines=list(known_baselines),
        counterexamples=counterexamples,
        untested_scope=list(untested_scope),
        training_eligible=False,
        machine_safety="finite_execution_only_no_unbounded_safety",
        claim=dict(
            statement="conjecture" if counterexamples else "finite_agreement",
            novelty="not_reviewed",
            proof=None,
            review=None,
        ),
    )
    value = record(
        "dossier",
        payload,
        created_at="2026-11-09T00:00:00Z",
        purpose=c["purpose"],
        track=c["track"],
    )
    validate_dossier(root, value)
    from oeis_learn.pipeline_validation.identity import confined

    atomic_write(confined(root, output, exists=False), value)
    return value


def references(root, dossier):
    p = dossier["payload"]
    c = candidate(root, p["candidate"])
    refs = [
        p["candidate"],
        c["selection"]["artifact"],
        c["source_snapshot"],
        p["tested"]["execution"],
        *p["known_baselines"],
    ]
    if c["library"] is not None:
        refs.append(c["library"])
    if p["claim"]["proof"] is not None:
        refs.append(p["claim"]["proof"])
    if p["claim"]["review"] is not None:
        refs.append(p["claim"]["review"]["evidence"])
    paths = {}
    for ref in refs:
        if ref["path"] in paths and paths[ref["path"]] != ref:
            raise ValueError("conflicting original dossier reference")
        paths[ref["path"]] = ref
    return list(paths.values())


def validate_dossier(root, value, *, reviewers=None):
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.admission import reference_identity

    v = validate("dossier", value)
    p = v["payload"]
    c = candidate(root, p["candidate"])
    for key in (
        "source",
        "source_hash",
        "offset",
        "library",
        "selection",
        "source_snapshot",
        "generation_budget",
    ):
        if p[key] != c[key]:
            raise ValueError("original dossier candidate binding differs")
    if v["purpose"] != c["purpose"] or v["track"] != c["track"]:
        raise ValueError("dossier/candidate scope differs")
    t = p["tested"]
    if (
        t["indices"] != list(map(str, range(int(p["offset"]), int(p["offset"]) + 100)))
        or t["reference"] != reference_identity()
    ):
        raise ValueError("original source offset/reference differs")
    actual = evaluate(p["source"], t["indices"])
    if actual.outcome != "ok" or list(actual.values) != t["observed"]:
        raise ValueError("independent dossier execution differs")
    execution = json.loads(verify_refs(root, [t["execution"]], max_bytes=32 << 20)[0])
    if (
        set(execution)
        != {
            "version",
            "identity",
            "source",
            "indices",
            "expected",
            "observed",
            "reference",
            "native",
            "runtime",
        }
        or seal(execution) != execution
        or execution["version"] != "progressive-claim-execution/v1"
        or execution["source"] != p["source_hash"]
        or execution["indices"] != t["indices"]
        or execution["expected"] != digest(t["expected"])
        or execution["observed"] != t["observed"]
        or execution["reference"] != t["reference"]
        or execution["runtime"] != t["native"]
    ):
        raise ValueError("original independent execution receipt differs")
    if t["native"] is not None and (
        not isinstance(execution["native"], dict)
        or execution["native"].get("outcome") != "ok"
        or execution["native"].get("values") != t["observed"]
    ):
        raise ValueError("original native result differs from dossier")
    expected = [
        dict(index=i, expected=e, observed=o)
        for i, e, o in zip(t["indices"], t["expected"], t["observed"])
        if e != o
    ]
    if expected != p["counterexamples"] or t["agreement"] != (not expected):
        raise ValueError("complete counterexample inventory differs")
    if p["claim"]["statement"] in {"finite_agreement", "bounded_verified"} and not t["agreement"]:
        raise ValueError("counterexample cannot establish finite/bounded agreement")
    if p["claim"]["statement"] == "bounded_verified" and t["native"] is None:
        raise ValueError("bounded verification requires independent native evidence")
    if v["purpose"] == "research" and t["native"] is None:
        raise ValueError("research dossier lacks native execution")
    if p["claim"]["novelty"] == "known" and not p["known_baselines"]:
        raise ValueError("known novelty status requires original comparison evidence")
    refs = references(root, v)
    verify_refs(root, refs, max_bytes=512 << 20)
    proof = (
        json.loads(verify_refs(root, [p["claim"]["proof"]])[0])
        if p["claim"]["proof"] is not None
        else None
    )
    validate_claim(
        p["claim"],
        reviewers=reviewers or {},
        source=p["source"],
        candidate=p["candidate"]["sha256"],
        proof=proof,
    )
    if p["claim"]["review"] is not None:
        evidence = json.loads(verify_refs(root, [p["claim"]["review"]["evidence"]])[0])
        if evidence.get("candidate") != p["candidate"]["sha256"] or not evidence.get("prior_work"):
            raise ValueError("candidate-bound scoped prior-work comparison required")
    return v


def review_dossier(root, dossier, claim, *, reviewers, output):
    # Revalidate the original with the same frozen trust policy; never rewrite it.
    original = validate_dossier(root, dossier, reviewers=reviewers)
    p = copy.deepcopy(original["payload"])
    p["claim"] = copy.deepcopy(claim)
    value = record(
        "dossier",
        p,
        created_at="2026-11-09T00:00:00Z",
        parents=[original["identity"]],
        purpose=original["purpose"],
        track=original["track"],
    )
    validate_dossier(root, value, reviewers=reviewers)
    from oeis_learn.pipeline_validation.identity import confined

    if output is not None:
        atomic_write(confined(root, output, exists=False), value)
    return value


def validate_reviewers(reviewers):
    from .contracts import _check, schema
    import re

    if not isinstance(reviewers, dict) or len(reviewers) > 64:
        raise ValueError("bounded frozen reviewer registry required")
    for key, value in reviewers.items():
        if (
            not isinstance(key, str)
            or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", key)
            or set(value) != {"name", "independent", "evidence", "candidate", "scope"}
            or value["independent"] is not True
            or any(
                not isinstance(value[k], str) or not 1 <= len(value[k]) <= 4096
                for k in ("name", "scope")
            )
        ):
            raise ValueError("closed named independent reviewer policy required")
        _check(schema("common")["$defs"]["artifact"], value["evidence"], schema("common"))
        _check(schema("common")["$defs"]["hash"], value["candidate"], schema("common"))
    return reviewers


def review_references(root, kind, data):
    if kind == "sealed_dossier":
        validate("dossier", data)
        return references(root, data)
    if kind == "reviewer_registry":
        if set(data) != {"reviewers"}:
            raise ValueError("closed reviewer registry envelope required")
        return [v["evidence"] for v in validate_reviewers(data["reviewers"]).values()]
    if kind == "review_evidence":
        from .contracts import _check, schema

        if set(data) != {"claim"}:
            raise ValueError("closed requested claim required")
        doc = schema("dossier")
        _check(doc["properties"]["payload"]["properties"]["claim"], data["claim"], doc)
        refs = []
        if data["claim"]["proof"] is not None:
            refs.append(data["claim"]["proof"])
        if data["claim"]["review"] is not None:
            refs.append(data["claim"]["review"]["evidence"])
        return refs
    raise ValueError("unregistered review input")
