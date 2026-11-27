"""Read-only, layered comparison of frozen measurements with original installed 028 receipts."""

from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    _check,
    digest,
    load,
    read_bound_record,
    schema,
    seal,
)
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_records import SESSION, physical_root_id
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_research_inputs import checked_file
from .progressive_round028_pilot import AREA, PilotRole, score_projection
from .progressive_velocity_donor import PARENT, PILOT_REPORT, reuse_layers

ARMS = {"00": "ordinary", "first_divergence": "first_divergence", "D": "incumbent"}
FIELDS = {"weights", "lineage", "inference", "selection", "scorer", "truth"}


def _registration(root, rows):
    if not isinstance(rows, list):
        raise ValueError("ordered proposed reuse measurements required")
    for r in rows:
        if (
            set(r) != {"method", "look", "segment", "score_attempt", "request"}
            or r["method"] not in ARMS
            or type(r["look"]) is not int
            or r["look"] not in (0, 1_000_000, 4_000_000)
            or r["method"] == "D"
            and r["look"] != 4_000_000
            or set(r["request"]) != FIELDS
        ):
            raise ValueError("closed proposed original look and measurement coordinates required")
        integer(r["segment"])
        integer(r["score_attempt"])
        for val in r["request"].values():
            _check(schema("common")["$defs"]["hash"], val, {})
    return seal(
        dict(
            version="round029-reuse-request/v1",
            state_root=physical_root_id(root),
            parent=PILOT_REPORT,
            rows=rows,
        )
    )


def register_requests(root, rows):
    """Custodian freezes requested identities. Original scores are always reopened separately."""
    reg = _registration(root, rows)
    folder = confined(root, f"run-state/discovery/{SESSION}/preparation/reuse", exists=False)
    folder.mkdir(parents=True, exist_ok=True)
    with RunStore.owner(folder):
        retain(folder / (reg["identity"][7:] + ".json"), reg)
        current = folder / "registration.json"
        if not current.exists() or load(confined(folder, current)) != reg:
            revise(current, reg)
    return reg


def _role(root, area, cut, block, role, *, expected_file=None):
    from .progressive_pilot_recovery import assigned_request

    claim = confined(area, f"assignments/{block}-{role}.json")
    key = assigned_request(area, claim)
    _check(schema("common")["$defs"]["hash"], key, {})
    folder = confined(area, "roles/" + key[7:])
    req = load(confined(folder, "request.json"))
    data = load(confined(folder, "data.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
    phase = "hip" if role == "search" else "cpu"
    if (
        req != seal(req)
        or req["identity"] != key
        or req["version"] != PilotRole.request_version
        or req["source"] != cut["source"]
        or req["registration"] != cut["registration"]
        or req["image"] != cut["images"][phase]
        or req["role"] != role
        or data != seal(data)
        or data["identity"] != req["data"]
        or data["block"] != block
    ):
        raise ValueError("original assigned pilot worker, frozen inputs or scope differs")
    if role == "search":
        # Authenticate the actual mounted weight bytes against the cut pinned by the public pilot.
        boundary = load(confined(folder, "boundary.json"))
        inspection = load(
            checked_file(folder, boundary["inspection"]), maximum=CAMPAIGN_DOCUMENT_BYTES
        )
        dest = "/work/inputs/weights.pt"
        filename = next(
            (m["Source"] for m in inspection[0]["Mounts"] if m["Destination"] == dest), None
        )
        if not filename or data["weights"]["sha256"] != cut["weights"]:
            raise ValueError("inference checkpoint differs")
        original = confined(root, Path(filename))
        input_path = checked_file(root, dict(data["weights"], path=str(original.relative_to(root))))
    else:
        dest = "/work/inputs/candidates.json"
        input_path = expected_file
        checked_file(expected_file.parent, dict(data["candidates"], path=expected_file.name))
    reader = PilotRole(None, None, root, None, area, None, None)
    reader.phase = phase
    fields = reader.read_committed(
        folder,
        req,
        [
            (folder / "request.json", "/work/inputs/request.json"),
            (folder / "data.json", "/work/inputs/data.json"),
            (input_path, dest),
        ],
    )
    name, version = (
        ("candidates", "discovery-candidates/v1")
        if role == "search"
        else ("scores", "discovery-scoring/v1")
    )
    ref = fields[name]
    record = read_bound_record(reader.last_output, ref, version=version)
    return data, req, reader.native, record, checked_file(reader.last_output, ref)


def historical_probe(root, cut, row):
    from oeis_learn.generalization.progressive.program_search_bank import selected_source
    from .pilot_clocks import CORRECT_FULL, DESCRIPTIVE

    if not cut.get("weights") or not cut.get("registration"):
        return None
    area = confined(root, f"{PARENT}/{AREA}")
    name = ARMS[row["method"]] + "-" + str(row["look"])
    block = name + (f"-resume-{row['segment']}" if row["segment"] else "")
    search, worker, native, candidates, path = _role(root, area, cut, block, "search")
    if (
        search["arm"] != ARMS[row["method"]]
        or search["target"] != row["look"]
        or search["weights"]["sha256"] != cut["weights"]
        or candidates["methods"] != ["correct"]
    ):
        raise ValueError("original method, look, or proposal mode differs")
    policy = CORRECT_FULL if row["look"] == 4_000_000 else DESCRIPTIVE
    prompts = search["rows"]
    common = dict(
        weights=cut["weights"],
        lineage=cut["lineage"],
        inference=digest(
            dict(
                weights=cut["weights"],
                runtime=worker["source"],
                image=worker["image"],
                native=native,
                model=candidates["model"],
                recipe=candidates["recipe"],
                mode="correct",
                prompts=prompts,
                opportunity=policy,
                inference_seed=search["inference_seed"],
            )
        ),
        selection=digest([worker["source"], "equal_budget", "correct"]),
    )
    clock = candidates["clock"]
    if (
        candidates["checkpoint"] != cut["weights"]
        or candidates["recipe"] != digest(search["recipe"])
        or candidates["prompts"] != digest(sorted(prompts, key=lambda p: p["id"]))
        or clock != seal(clock)
        or clock["policy"] != policy
        or clock.get("inference_seed") != search["inference_seed"]
        or set(candidates["records"]) != {p["id"] for p in prompts}
    ):
        raise ValueError("original model, ordered recipients or opportunity differs")
    choices = []
    for prompt in prompts:
        envelope = candidates["records"][prompt["id"]]["selections"]["equal_budget"]
        if envelope is not None:
            from oeis_learn.fresh.synthesis_policy import validate_selection_envelope

            selected = validate_selection_envelope(envelope)
            selected_source(
                envelope, dict(candidate_sha256=selected["source_sha256"], success=False), prompt
            )
        choices.append([prompt["id"], envelope])
    common.update(choice=digest(choices), truth=None, scorer=None, score=None)
    scored_block = block + (f"-scoreretry-{row['score_attempt']}" if row["score_attempt"] else "")
    try:
        score_in, scorer, score_native, scores, _ = _role(
            root, area, cut, scored_block, "score", expected_file=path
        )
    except (OSError, ValueError, KeyError):
        # Complete original inference remains usable when the score-side closure is unavailable.
        return seal(common)
    truth = score_in["rows"]
    projected = [
        {k: (r[k] if k == "id" else r[k][:20]) for k in ("id", "indices", "values")} for r in truth
    ]
    if projected != prompts:
        raise ValueError("original ordered private truth differs from inference recipients")
    outcomes, comparable = score_projection(
        candidates,
        scores,
        score_in["candidates"]["sha256"],
        prompts,
        search["weights"],
        search["recipe"],
        policy,
        inference_seed=search["inference_seed"],
    )
    for prompt in prompts:
        key = prompt["id"]
        selected_source(
            candidates["records"][key]["selections"]["equal_budget"],
            scores["outcomes"][key]["equal_budget"],
            prompt,
        )
    return seal(
        dict(
            common,
            truth=digest(truth),
            scorer=digest(
                [scorer["source"], scorer["image"], score_native, "independent_exact_100"]
            ),
            score=digest([scores["identity"], outcomes]) if comparable else None,
        )
    )


def add_look_costs(cuts, costs):
    """A missing earlier clock leaves the actual later model eligible at its own useful-token look."""
    for cut in cuts:
        if cut["method"] == "D":
            continue
        base = ARMS[cut["method"]] + "-" + str(cut["look"])
        rows = [
            r
            for r in costs
            if r["role"] == "train"
            and (r["block"] == base or r["block"].startswith(base + "-resume-"))
        ]
        cut["look_work_us"] = sum(integer(r["charged_us"]) for r in rows) if rows else None
        cut["clock_reason"] = None if rows else "original training clock unavailable for this look"
    for method in ("00", "first_divergence"):
        cumulative = 0
        for cut in sorted((c for c in cuts if c["method"] == method), key=lambda c: c["look"]):
            delta = cut["look_work_us"]
            cumulative = None if delta is None or cumulative is None else cumulative + delta
            cut["training_to_look_us"] = cumulative


def reuse_inventory(root, cuts):
    folder = confined(root, f"run-state/discovery/{SESSION}/preparation/reuse", exists=False)
    result = dict(
        measurements=[],
        distinct_score_credits=0,
        research_dispatches=0,
        registration=None,
        reason="no prospective probe identities registered",
    )
    if not (folder / "registration.json").is_file():
        return result
    reg = load(confined(folder, "registration.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
    if reg != _registration(root, reg["rows"]) or reg != load(
        confined(folder, reg["identity"][7:] + ".json"), maximum=CAMPAIGN_DOCUMENT_BYTES
    ):
        raise ValueError("frozen request registration differs")
    result.update(registration=reg["identity"], reason=None)
    prior, credited = {}, set()
    for r in reg["rows"]:
        method, look = r["method"], r["look"]
        cut = next(
            c for c in cuts if c["method"] == method and (c["look"] == look or method == "D")
        )
        slot = method, look, r["segment"], r["score_attempt"]
        if slot not in prior:
            try:
                prior[slot] = historical_probe(root, cut, r), None
            except (ValueError, OSError, KeyError) as exc:
                prior[slot] = None, str(exc)
        old, gap = prior[slot]
        layers = reuse_layers(cut, old, r["request"])
        score_key = old["score"] if layers["action"] == "reuse_score" else None
        new_credit = bool(score_key and score_key not in credited)
        if new_credit:
            credited.add(score_key)
        result["measurements"].append(
            dict(
                method=method,
                look=look,
                **layers,
                look_work_us=cut.get("look_work_us"),
                historical_gap=gap,
                credited=new_credit,
                score=score_key,
            )
        )
    result["distinct_score_credits"] = len(credited)
    return result
