"""Immutable expanded training targets, separate from OEIS admission authority."""

import copy
from .contracts import digest, seal
from .component_search import validate_input as validate_library_input
from .replay import Mixer

RULE = dict(
    version="progressive-component-distillation/v1",
    fraction=[1, 4],
    sampling="task-first-then-alternative",
    weighting="charged-target-tokens",
    freshness="one-frozen-task-cut",
    overshoot="whole-example",
    library_update="complete-task-boundary",
)


def validate_target(value, library):
    from oeis_learn.generalization.progressive.admission import validate_member
    from oeis_learn.generalization.progressive.components import expand
    from oeis_learn.decoder.bigint_codec import encode_body, decode_body
    from oeis_learn.synthesis.reference import evaluate

    if (
        set(value)
        != {
            "identity",
            "version",
            "library",
            "proposal",
            "member",
            "row",
            "purpose",
            "native",
            "reference",
        }
        or value["version"] != "progressive-component-target/v1"
        or seal(value) != value
    ):
        raise ValueError("closed independently verified component target required")
    member = validate_member(value["member"])
    if (
        value["purpose"] not in {"acceptance", "research"}
        or value["library"] != library["library"]["identity"]
    ):
        raise ValueError("component target scope/library differs")
    proposal = value["proposal"]
    row = value["row"]
    from .contracts import _check, schema
    from oeis_learn.synthesis.admission import reference_identity

    _check(schema("common")["$defs"]["hash"], value["reference"], schema("common"))
    if value["native"] is not None:
        _check(schema("common")["$defs"]["hash"], value["native"], schema("common"))
    if type(row["native_verified"]) is not bool or value["reference"] != reference_identity():
        raise ValueError("original component verification producer required")
    expanded = expand(library["library"], proposal["expression"])
    if expanded["calls"] < 1:
        raise ValueError("distillation requires an actual learned component call")
    source = decode_body(encode_body(expanded["source"]))
    if (
        proposal["source"] != source
        or row["source"] != source
        or row["membership"] != "training"
        or seal(row) != row
        or row["tokens"] != encode_body(row["target_source"])
    ):
        raise ValueError("original expanded target/source/codec differs")
    result = evaluate(source, member["indices"])
    if (
        result.outcome != "ok"
        or list(result.values) != member["values"]
        or row["values"] != member["values"]
        or row["indices"] != member["indices"]
    ):
        raise ValueError("independent hundred-term component verification differs")
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from .generator import structure_id

    expected, reason = verify_draw(
        dict(
            source=source,
            ordinal=0,
            draw_identity=digest([value["library"], proposal, member]),
            source_policy=digest(RULE),
            structure=structure_id(source),
            detector_limit="finite training-only component distillation; no novelty claim",
        )
    )
    if expected is None or seal(dict(expected, native_verified=row["native_verified"])) != row:
        raise ValueError("independent grounded target binding differs: " + str(reason))
    if row["draw_identity"] != digest([value["library"], proposal, member]):
        raise ValueError("component draw provenance differs")
    if value["purpose"] == "research" and (
        not row["native_verified"]
        or value["native"] is None
        or library["library"]["purpose"] != "research"
    ):
        raise ValueError("native research component target required")
    return value


def freeze_distillation(library, targets, *, arm):
    value = seal(
        dict(
            version="progressive-component-input/v1",
            arm=arm,
            policy=RULE,
            library=library,
            targets=targets,
        )
    )
    ComponentMixer(value)
    return value


class ComponentMixer(Mixer):
    rule = RULE

    def __init__(self, package):
        if (
            set(package) != {"identity", "version", "arm", "policy", "library", "targets"}
            or seal(package) != package
            or package["version"] != "progressive-component-input/v1"
            or package["policy"] != RULE
            or package["arm"] not in {"library", "no_library"}
        ):
            raise ValueError("closed frozen component distillation input required")
        self.original = copy.deepcopy(package)
        self.package = dict(arm="replay" if package["arm"] == "library" else "continued_synthetic")
        self.groups = {}
        self.archive = None
        if package["arm"] == "no_library":
            if package["library"] is not None or package["targets"] != []:
                raise ValueError("no-library control cannot receive discoveries")
            return
        lib = validate_library_input(package["library"])
        if not isinstance(package["targets"], list) or not 1 <= len(package["targets"]) <= 64:
            raise ValueError("bounded nonempty expanded target cut required")
        seen = set()
        for item in package["targets"]:
            validate_target(item, lib)
            if item["identity"] in seen:
                raise ValueError("duplicate component target")
            seen.add(item["identity"])
            self.groups.setdefault(item["member"]["id"], []).append(item)
        if any(len(v) > 2 for v in self.groups.values()):
            raise ValueError("at most two alternatives per training task")
        self.archive = digest([lib["library"]["identity"], sorted(seen)])

    def target_row(self, a, task):
        validate_target(a, self.original["library"])
        return copy.deepcopy(a["row"])


def compare_utility(rows, roster, seeds, design, *, costs, panel):
    """Diagnostic matched harder-panel comparison, never a promotion authority."""
    from .decisions import recompute

    if panel != "components-harder" or set(costs) != {"baseline", "candidate"}:
        raise ValueError("frozen harder-panel and matched arm costs required")
    totals = {}
    for arm, value in costs.items():
        if set(value) != {"discovery", "validation", "search", "execution", "training"} or any(
            type(v) is not int or v < 0 for v in value.values()
        ):
            raise ValueError("complete nonoverlapping charged cost breakdown required")
        totals[arm] = sum(value.values())
        if totals[arm] <= 0:
            raise ValueError("positive total cost required")
    metrics, verdict = recompute(
        rows, roster, seeds, design, costs=totals if design["claim"] == "efficiency" else None
    )
    return seal(
        dict(
            version="progressive-component-utility/v1",
            panel=panel,
            metrics=metrics,
            verdict=verdict,
            costs=costs,
            total_cost_us=totals,
            cost_reduction=1 - totals["candidate"] / totals["baseline"],
            compression_is_utility=False,
            executable=False,
            next_action="RETAIN_BASELINE"
            if verdict not in {"CONFIRMED_GAIN", "CONFIRMED_EFFICIENCY"}
            else "REQUIRES_FROZEN_PRODUCER_GATE",
        )
    )
