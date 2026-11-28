"""Frozen operation order and the single cumulative authority for comparative exploration."""

from collections import Counter
import secrets
from pathlib import Path

from oeis_learn.generalization.progressive.budgets import KEYS, role_work_requirements
from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.graph import velocity_study_looks
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.trials import integer
from oeis_learn.generalization.progressive.velocity_records import SESSION, physical_root_id
from . import progressive_velocity_checkpoint as checkpoint
from . import progressive_velocity_trial_preparation as preparation
from . import progressive_velocity_conditioning as conditioning
from .identity import confined, file_hash
from .preparation_recovery import retain
from .progressive_research_inputs import checked_file
from .progressive_velocity_broad_host import MAX_INPUT

VERSION = "round029-comparative-wave/v1"
MARKER = "comparative/authorization.json"
DRIVER = "src/oeis_learn/pipeline_validation/progressive_velocity_wave_role.py"
EVALUATOR = "src/oeis_learn/pipeline_validation/progressive_evaluation.py"


CONTROLS = "src/oeis_learn/pipeline_validation/progressive_velocity_controls.py"


def driver_identity(source):
    return digest([file_hash(confined(source, path)) for path in (DRIVER, EVALUATOR, CONTROLS)])


def targets(learner, requested):
    """Compute the exact whole-update crossings once, using the frozen circular order."""
    batch = integer(learner["policy"]["effective_batch"], 1)
    counts = [integer(len(learner["rows"][key]["tokens"]), 1) for key in learner["order"]]
    if not counts or requested != sorted(set(requested)):
        raise ValueError("ordered targets and nonempty positive stream required")
    cursor = useful = updates = last = 0
    result = {}
    for target in requested:
        integer(target)
        while useful < target:
            last = sum(counts[(cursor + i) % len(counts)] for i in range(batch))
            cursor += batch
            useful += last
            updates += 1
        result[target] = dict(
            useful_tokens=useful,
            updates=updates,
            last_update_tokens=last,
            cursor=cursor,
            overshoot=useful - target,
        )
    return result


def probe_work(look, phase):
    cases = len(look["members"])
    if phase == "search":
        work = role_work_requirements(
            "generate", [dict(kind="visible_prompt")] * cases + [dict(kind="oeis_generation")]
        )
        if look["profile"] == "common":
            work.update(draws=cases, attempts=cases, rejections=cases)
    else:
        work = role_work_requirements("score", [dict(kind="development_truth")] * cases)
    return {k: v for k, v in work.items() if k != "elapsed_us" and v}


def plan(accepted, registration, packages, sample=None):
    graph = velocity_study_looks(registration)
    if (
        accepted["registration"] != registration["identity"]
        or set(packages) != set(accepted["supported"])
        or set(accepted["trials"]) != set(packages) | set(accepted["omitted"])
    ):
        raise ValueError("registered recipes or exclusions differ")
    cuts = {}
    for method, learner in packages.items():
        rows = [r for r in accepted["training"] if r["method"] == method]
        if any(r["learner"] != learner["identity"] for r in rows):
            raise ValueError("training package differs from the accepted recipe")
        cuts[method] = targets(learner, [r["target"] for r in rows])
    ops = []
    previous = {}
    look_index = {
        look["identity"]: look
        for profiles in graph.values()
        for series in profiles.values()
        for look in series
    }
    eligible = {
        look["identity"]
        for method, profiles in graph.items()
        if method not in accepted["omitted"]
        for series in profiles.values()
        for look in series
    }
    if len(eligible) != len(accepted["schedule"]) or eligible != {
        r["roster"] for r in accepted["schedule"]
    }:
        raise ValueError("missing, repeated or unregistered comparative look")
    # Finish the matched common boundary across recipes before advancing to the next.
    # A 1M diagnostic cannot remove an arm from its registered 4M opportunity.
    for row in sorted(
        accepted["schedule"], key=lambda r: (r["look"], r["profile"] != "common", r["method"])
    ):
        method, target = row["method"], row["look"]
        look = look_index[row["roster"]]
        if (look["method"], look["look"], look["profile"], len(look["members"])) != (
            method,
            target,
            row["profile"],
            row["members"],
        ):
            raise ValueError("accepted evaluation coordinate differs")
        if method in cuts and row["profile"] == "common":
            cut = cuts[method][target]
            prior = previous.get(method, dict(useful_tokens=0, updates=0))
            ops.append(
                dict(
                    role="train",
                    roster=row["roster"],
                    cost=dict(
                        tokens=cut["useful_tokens"] - prior["useful_tokens"],
                        attempts=cut["updates"] - prior["updates"],
                    ),
                )
            )
            previous[method] = cut
        shared_zero = target == 0 and method != min(packages)
        for role in ("search", "score"):
            work = {} if shared_zero else probe_work(look, role)
            ops.append(dict(role=role, roster=row["roster"], cost=work))
    if sample is not None:
        # Even an empty sample commits its price and measured supply disposition.
        ops[0]["conditioning_plan"] = sample["identity"]
        paired = list(conditioning.opportunities(sample, registration, accepted["supported"]))
        for role in ("search", "score"):
            for ref, look, mode in paired:
                look_index[ref] = look
                ops.append(dict(stage=conditioning.STAGE, role=role, mode=mode, roster=ref,
                                cost=probe_work(look, role)))
    return graph, look_index, cuts, ops


def reopen_treatments(folder, native, reg, view, matrix):
    """Rebuild the priced slot/donor projection from the original native witnesses."""
    from . import progressive_velocity_broad as broad
    from . import progressive_velocity_checkpoint_host as checkpoints

    pointer = checkpoints.origin(folder)
    fresh = load(confined(folder, "qualification/" + pointer["grant"][7:] + "/receipt.json"))
    if (fresh != seal(fresh) or fresh["identity"] != pointer["receipt"]
            or fresh["identity"] != matrix["origin"] or fresh["supply"]["matching"] != view["support"]
            or not matrix["methods"].keys() <= fresh["device"]["methods"].keys()):
        raise ValueError("eligible learner packages require their paid support and optimizer prices")
    saved = load(confined(native, "fresh-ordinary.json"), maximum=MAX_INPUT)
    ordinary = saved["stream"]
    start = saved["start"]
    ordinary_rows = {row["draw_identity"]: row for row in saved["ordinary_rows"]}
    if (saved != seal(saved) or ordinary != seal(ordinary) or ordinary["identity"] != reg["ordinary"]
            or start != reg["training_start"]["origin"] or start != fresh["start"]
            or len(ordinary_rows) != len(saved["ordinary_rows"])
            or set(ordinary["order"]) != ordinary_rows.keys()
            or digest([ordinary_rows[k]["identity"] for k in ordinary["order"]]) != ordinary["row_identities"]
            or ordinary["seeds"]["training_families"] != reg["stream_seed"]):
        raise ValueError("ordinary witnesses differ from the registered training start")
    parent = confined(native, "run-state/discovery/" + SESSION + "/preparation/composition")
    receipts = list(parent.glob("*/receipt.json"))
    if len(receipts) != 1:
        raise ValueError("one original composition receipt required")
    receipt = load(confined(parent, receipts[0]), maximum=MAX_INPUT)

    def reopen(ref):
        return load(checked_file(receipts[0].parent, ref), maximum=ref["bytes"])

    stream, audit = reopen(receipt["stream"]), reopen(receipt["audit"])
    rows = {k: reopen(ref) for k, ref in receipt["rows"].items()}
    if (receipt != seal(receipt) or stream != seal(stream) or audit != seal(audit)
            or stream["audit"] != audit["identity"] or stream["ordinary_ref"] != reg["ordinary"]
            or (stream["identity"] if stream["available"] else None) != reg["broad"]
            or any(row["identity"] != audit["witnesses"][k] for k, row in rows.items())):
        raise ValueError("broader witnesses differ from the registered stream or composition audit")
    manifests = load(confined(native, "pairs.json"), maximum=MAX_INPUT)
    origin = dict(stream=ordinary, ordinary_rows=saved["ordinary_rows"], start=start, frame=True,
                  pair_seed=reg["stream_seed"], initial_weights=start["initial_weights"],
                  profile=view["streams"]["ordinary"]["profile"])
    reopened, pairs = broad.matching_views(origin, stream, rows, recorded=manifests)
    if reopened != view or pairs != manifests:
        raise ValueError("learner donor map, pair proofs or stream projection differs")
    if {"02", "03"}.intersection(matrix["methods"]) and (
        matrix.get("calibration") != fresh["device"]["calibration"]
    ):
        raise ValueError("matching weight differs from the paid ordinary calibration")


def inputs(root, folder, accepted):
    from oeis_learn.generalization.progressive.campaign import validate_velocity_registration

    area = confined(folder, "qualification/" + accepted["preparation_grant"][7:])
    private = load(confined(area, "trial-inputs.json"))
    if (
        private != seal(private)
        or private["identity"] != accepted["inputs"]
        or private["targets"] != accepted["training"]
    ):
        raise ValueError("funded trial input commitment differs")
    reg = validate_velocity_registration(load(checked_file(root, private["registration"])))
    path = checked_file(root, private["projection"])
    view = load(path, maximum=MAX_INPUT)
    matrix = private["matrix"]
    if (
        view != seal(view)
        or matrix != seal(matrix)
        or reg["identity"] != accepted["registration"]
        or view["fresh_start"] != reg["training_start"]["origin"]["identity"]
        or view["streams"]["ordinary"]["plan"] != reg["ordinary"]
        or view["identity"] != matrix["projection"]
        or matrix["methods"] != accepted["learners"]
        or set(matrix["methods"]) != set(accepted["supported"])
        or matrix["origin"] != accepted["sources"]["fresh"]
    ):
        raise ValueError("funded training projection differs")
    if set(matrix["methods"]) - set(checkpoint.METHODS):
        reopen_treatments(folder, path.parent, reg, view, matrix)
    packages = {m: checkpoint.treatment_package(view, m, matrix["microbatch"], matrix["origin"],
                                                matrix.get("calibration")) for m in matrix["methods"]}
    if any(p["identity"] != matrix["methods"][m]["package"] for m, p in packages.items()):
        raise ValueError("reopened learner package differs from its paid matrix")
    return reg, packages


def family_key(folder, *, create=False):
    path = confined(folder, "comparative/family-key.json", exists=not create)
    if not path.exists():
        retain(path, seal(dict(key=secrets.token_hex(32))))
    value = load(path)
    if (
        value != seal(value)
        or set(value) != {"key", "identity"}
        or len(bytes.fromhex(value["key"])) != 32
    ):
        raise ValueError("original private family-label key differs")
    return value


def authority(root, revision, minutes, accepted, operations, driver, key_ref, *, qualification=None):
    total = integer(minutes, 1) * 60_000_000
    if total < accepted["proposed_total_us"]:
        raise ValueError("grant is below the frozen full-wave tail and protected follow-up")
    work = Counter(dict.fromkeys(KEYS, 0))
    for op in operations:
        work.update(op["cost"])
    # Bound recovery to one additional largest dispatch in each resource. All
    # attempts share the original wall clock and the protected follow-up reserve.
    for key in KEYS:
        work[key] += max((op["cost"].get(key, 0) for op in operations), default=0)
    work["elapsed_us"] = total
    terms = dict(
        version="round029-grant/v1",
        study_id=SESSION,
        scope="exploration",
        registration_or_parent_ref=accepted["identity"],
        controller_ref=digest(
            [VERSION, accepted["identity"], revision, driver, key_ref, digest(operations)]
            + ([qualification] if qualification else [])
        ),
        state_root_id=physical_root_id(root),
        total_us=total,
        holdback_us=accepted["followup_us"],
        allowed_operations=["run"],
        resources=dict(work),
    )
    approval = seal(dict(version="round029-authorization/v1", terms=digest(terms)))
    grant = seal(dict(terms, authorizing_record_ref=approval["identity"]))
    marker = seal(
        dict(
            version=VERSION,
            controller=revision,
            family_labels=key_ref,
            driver=driver,
            grant=grant["identity"],
            accepted=accepted["identity"],
            operations=digest(operations),
            **({"qualification": qualification} if qualification else {}),
        )
    )
    return grant, approval, marker


def authorize(root, source, minutes, expected):
    from .progressive_round028 import candidate_revision

    folder = confined(root, "run-state/discovery/" + SESSION)
    revision = candidate_revision(Path(source), root)
    with RunStore.owner(folder):
        accepted = preparation.accepted_preparation(folder)
        if accepted is None or accepted["identity"] != expected:
            raise ValueError("operator's accepted pre-score commitment differs")
        preparation.published(root, folder)
        reg, packages = inputs(root, folder, accepted)
        from . import progressive_velocity_wave_commitment as commitment

        sample = conditioning.original(root, folder, source, accepted, reg,
                                       total_us=integer(minutes, 1) * 60_000_000)
        design = plan(accepted, reg, packages, sample)
        ops = design[-1]
        from .progressive_velocity_wave_qualification import required
        qualified = required(root, folder, source, accepted)
        key = family_key(folder, create=True)
        grant, approval, marker = authority(
            root,
            revision,
            minutes,
            accepted,
            ops,
            driver_identity(source),
            key["identity"], qualification=qualified,
        )
        # One owner per accepted wave, including concurrent or repeated start requests.
        retain(folder / MARKER, marker)
        retain(folder / conditioning.PATH, sample)
        directory = confined(root, "run-state/discovery-authorizations")
        retain(directory / (approval["identity"][7:] + ".json"), approval)
        path = directory / ("grant-" + grant["identity"][7:] + ".json")
        retain(path, grant)
        commitment.install(root, grant, commitment.prescore(accepted, design, grant, approval, marker,
                                                           bytes.fromhex(key["key"]), sample))
        return path
