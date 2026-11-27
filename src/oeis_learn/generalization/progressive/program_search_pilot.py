"""Prospective continuation of the measured round028 generator study."""

import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction

from .contracts import seal
from .program_search_outcome import archive_bytes, read_canonical, verify_workstation
from .program_search_rehearsal_report import encode
from .training_rows import GENERATOR_LOOKS as LOOKS, pair_seeds

ARMS = ("ordinary", "first_divergence")
SHIFT_US = 11 * 3600 * 1_000_000
STRATA = ("composition", "unseen_structure", "parameter", "oeis", "retention")
PILOT_MEMBERS = ["pilot.json", "round028-parent.zip"]


# Operational envelopes are independent of the registered 60 seconds per case.
RESIDENT_SUPERVISOR_GRACE_SECONDS = 120


def resident_seconds(cases):
    if type(cases) is not int or not 1 <= cases <= 160:
        raise ValueError("resident pilot roster must contain 1 to 160 cases")
    return 1800 + 120 * cases


def allocation(parent):
    if (
        parent["status"] != "VERIFIED"
        or parent["next_action"] != "RETAIN_BASELINE"
        or parent["workstation"]["status"] != "VERIFIED"
        or parent["comparison"]["fresh_scored_cases"] != 0
        or parent["comparison"]["assessment"] != "SEALED"
    ):
        raise ValueError("verified unspent parent workstation decision required")
    cost = parent["funding"]
    return dict(
        shift_us=SHIFT_US,
        forecast_us=cost["pilot_us"],
        headroom_us=SHIFT_US - cost["pilot_us"],
        repeat_funded=False,
        eight_case_batch_headroom_us=SHIFT_US - cost["eight_case_batch_pilot_us"],
        repeat_us=cost["repeat_if_earned_us"],
        schedule="one_process_per_arm_and_look",
    )


def pilot_registration(parent, candidate, source):
    funding = allocation(parent)
    if funding["headroom_us"] < 0:
        raise ValueError("measured pilot exceeds the revised allocation")
    if not re.fullmatch("[0-9a-f]{40}", candidate) or not re.fullmatch(
        "sha256:[0-9a-f]{64}", source
    ):
        raise ValueError("pinned candidate and source required")
    return seal(
        dict(
            version="round028-pilot-registration/v1",
            predecessor=parent["report"],
            workstation=parent["workstation"]["identity"],
            audit=parent["audit"],
            timing=parent["timing"],
            candidate=candidate,
            source=source,
            funding=funding,
            arms=list(ARMS),
            reference="retained_D",
            looks=list(LOOKS),
            pilot_pair=0,
            training_supply="original_native_draw_count_then_same_counter_stream_if_signal_missing",
            multiplicity=3,
            attempts=[1, 1, 8],
            portfolio_seconds=60,
            fit_seconds=4,
            tokens_per_draw=1024,
            repeat="UNFUNDED",
            confirmation="UNFUNDED",
            assessment="SEALED",
        )
    )


def family_counts(roster, outcomes, arms):
    """Release only within-family joint counts. Group numbers are local to this publication."""
    groups = {g: i for i, g in enumerate(sorted({r["group"] for r in roster}))}
    cells = defaultdict(Counter)
    for row in roster:
        status = []
        for arm in arms:
            v = outcomes[arm][row["task"]]
            code = {"success": "S", "failure": "F", "unresolved": "U"}[v["outcome"]]
            if v["complete"] != (code != "U"):
                raise ValueError("contradictory paired measurement")
            status.append(code)
        cells[(row["stratum"], groups[row["group"]])]["".join(status)] += 1
    return [
        dict(stratum=s, group=g, counts=dict(sorted(c.items())))
        for (s, g), c in sorted(cells.items())
    ]


def frozen_family_sizes(graph, registration, private_freeze, prices):
    """Project only archive-local group numbers and sizes before opening scores."""
    sizes = {}
    for target, looks in graph.items():
        rows = looks[ARMS[0]]["members"]
        numbers = {key: i for i, key in enumerate(sorted({r["group"] for r in rows}))}
        counts = Counter((r["stratum"], numbers[r["group"]]) for r in rows)
        sizes[target] = [dict(stratum=s, group=g, size=n) for (s, g), n in sorted(counts.items())]
    return seal(dict(version="round028-pilot-family-freeze/v1", registration=registration,
                     private_freeze=private_freeze, sizes=sizes,
                     prices=sorted([block, role, price] for (block, role),price in prices.items())))


def summary(cells, number):
    """Compute exact equal-family contrasts with both completions of every unresolved case."""
    by_stratum = defaultdict(list)
    seen = set()
    for cell in cells:
        if set(cell) != {"stratum", "group", "counts"}:
            raise ValueError("invalid released family cell")
        s, g, counts = cell["stratum"], cell["group"], cell["counts"]
        if (
            s not in STRATA
            or type(g) is not int
            or not 0 <= g < 160
            or (s, g) in seen
            or not counts
            or any(
                not re.fullmatch("[SFU]{" + str(number) + "}", key)
                or type(n) is not int
                or not 1 <= n <= 32
                for key, n in counts.items()
            )
        ):
            raise ValueError("invalid joint family counts")
        seen.add((s, g))
        by_stratum[s].append(counts)
    contrasts = {}
    for baseline, candidate in ((0, 1), (2, 0), (2, 1)) if number == 3 else ((0, 1),):
        name = (
            ("ordinary", "first_divergence", "incumbent")[candidate]
            + "_minus_"
            + ("ordinary", "first_divergence", "incumbent")[baseline]
        )
        strata = {}
        for s, families in by_stratum.items():
            gains, lower, upper = [], [], []
            for counts in families:
                n = sum(counts.values())
                delta = sum(
                    v * ((k[candidate] == "S") - (k[baseline] == "S")) for k, v in counts.items()
                )
                gains.append(Fraction(delta, n))
                lower.append(
                    Fraction(delta - sum(v for k, v in counts.items() if k[baseline] == "U"), n)
                )
                upper.append(
                    Fraction(delta + sum(v for k, v in counts.items() if k[candidate] == "U"), n)
                )
            strata[s] = dict(
                cases=sum(sum(c.values()) for c in families),
                families=len(families),
                gain=sum(gains) / len(gains),
                lower=sum(lower) / len(lower),
                upper=sum(upper) / len(upper),
            )
        primary = ["composition", "unseen_structure"]
        if not all(s in strata for s in primary):
            raise ValueError("missing primary stratum")
        contrasts[name] = {
            k: float(sum(strata[s][k] for s in primary) / 2) for k in ("gain", "lower", "upper")
        }
        contrasts[name]["strata"] = {
            s: {k: float(v) if isinstance(v, Fraction) else v for k, v in row.items()}
            for s, row in strata.items()
        }
    return contrasts


def spending(cells, paired, comparable):
    matched = summary(cells, 3)["first_divergence_minus_ordinary"]
    retention = [c["counts"] for c in cells if c["stratum"] == "retention"]
    n = sum(sum(c.values()) for c in retention)
    if not n:
        raise ValueError("missing retention observations")
    delta = sum(v * ((k[0] == "S") - (k[1] == "S")) for c in retention for k, v in c.items())
    best = Fraction(delta - sum(v for c in retention for k, v in c.items() if k[1] == "U"), n)
    worst = Fraction(delta + sum(v for c in retention for k, v in c.items() if k[0] == "U"), n)
    if not paired or not comparable:
        return "INCOMPARABLE"
    if matched["upper"] <= 0 or best > Fraction(2, 100):
        return "RETAIN_REFERENCE"
    if matched["lower"] <= 0 or worst > Fraction(2, 100):
        return "UNRESOLVED_SPENDING_CONTRAST"
    return "REPLICATION_UNFUNDED"


LINEAGE_FIELDS = ("initial", "stream", "codec", "common_policy")


def verify_prescore(receipt, registration, acceptance):
    if (not isinstance(receipt, dict)
            or set(receipt) != {"version", "identity", "registration", "acceptance", "design", "lineage"}
            or receipt != seal(receipt) or receipt["version"] != "round028-pre-score/v1"
            or receipt["registration"] != registration or receipt["acceptance"] != acceptance
            or not isinstance(receipt["lineage"], dict) or set(receipt["lineage"]) != set(LINEAGE_FIELDS)
            or not all(isinstance(v, str) and re.fullmatch("sha256:[0-9a-f]{64}", v)
                       for v in [receipt["design"], *receipt["lineage"].values()])):
        raise ValueError("invalid pre-score qualification commitment")


def verify_pilot(path, *, qualification=None):
    from .trials import paired_generator_exposure

    records = read_canonical(path, tuple(PILOT_MEMBERS))
    raw = records["pilot.json"]
    doc = json.loads(raw)
    original = records["round028-parent.zip"]
    parent = verify_workstation(io.BytesIO(original))
    with (
        zipfile.ZipFile(io.BytesIO(original)) as wrapper,
        zipfile.ZipFile(io.BytesIO(wrapper.read("round028.zip"))) as evidence,
    ):
        panel = json.loads(evidence.read("panels.json"))
        rehearsal = json.loads(evidence.read("rehearsal.json"))
        timing = json.loads(evidence.read("timing.json"))
    fields = {
        "version",
        "identity",
        "registration",
        "freeze",
        "acceptance",
        "looks",
        "action",
        "research_gpu_us",
        "research_role_us",
        "costs",
        "verification",
    }
    anchored = doc.get("version") == "round028-generator-pilot/v3"
    current = anchored or doc.get("version") == "round028-generator-pilot/v2"
    if anchored:
        fields.add("prescore")
        committed = doc["prescore"]
        verify_prescore(committed, doc["registration"]["identity"], doc["acceptance"])
        if committed["design"] != doc["freeze"]:
            raise ValueError("family allocation differs from pre-score qualification")
    if qualification is not None:
        independent = verify_qualification(qualification)
        if (not anchored or independent.get("prescore") != committed
                or independent["scientific_status"] != "PILOT_UNMEASURED"):
            raise ValueError("pilot differs from independent pre-score qualification")
    if current:
        fields.add("design")
    targets = [row["target"] for row in doc["looks"]]
    if (
        set(doc) != fields
        or seal(doc) != doc
        or raw != encode(doc)
        or doc["version"] not in ("round028-generator-pilot/v1", "round028-generator-pilot/v2", "round028-generator-pilot/v3")
        or doc["registration"]
        != pilot_registration(
            parent, doc["registration"]["candidate"], doc["registration"]["source"]
        )
        or doc["verification"] != "private_choices_and_checkpoints"
        or any(not re.fullmatch("sha256:[0-9a-f]{64}", doc[k]) for k in ("freeze", "acceptance"))
        or type(doc["research_gpu_us"]) is not int
        or doc["research_gpu_us"] < 0
        or not targets or targets != list(LOOKS[:len(targets)])
        or not current and len(targets) != len(LOOKS)
    ):
        raise ValueError("invalid pilot registration or endpoint")
    if current:
        design = doc["design"]
        from .program_search_timing import public_pilot_prices
        prices = public_pilot_prices(panel, rehearsal, timing)
        if (set(design) != {"version", "registration", "identity", "private_freeze", "sizes", "prices"}
                or design["prices"] != sorted([b, r, p] for (b,r),p in prices.items())
                or design != seal(design) or design["identity"] != doc["freeze"]
                or design["version"] != "round028-pilot-family-freeze/v1"
                or design["registration"] != doc["registration"]["identity"]
                or not re.fullmatch("sha256:[0-9a-f]{64}", design["private_freeze"])
                or set(design["sizes"]) != set(map(str, LOOKS))):
            raise ValueError("invalid prospectively frozen pilot families")
    known = set()
    total = 0
    for item in doc["costs"]:
        if (
            set(item)
            != {
                "identity",
                "block",
                "role",
                "lease_us",
                "wall_us",
                "charged_us",
                "covered_us",
                "busy_us",
            }
            or item != seal(item)
            or not re.fullmatch(
                r"(ordinary|first_divergence|incumbent)-(0|1000000|4000000)(-resume-[1-9][0-9]*)?(-scoreretry-[1-9][0-9]*)?",
                item["block"],
            )
            or item["role"] not in ("train", "search", "score")
            or "-scoreretry-" in item["block"]
            and item["role"] != "score"
            or (item["block"], item["role"]) in known
            or type(item["lease_us"]) is not int
            or item["lease_us"] < 0
            or item["role"] == "score"
            and item["lease_us"] != 0
            or type(item["charged_us"]) is not int
            or item["charged_us"] < item["lease_us"]
            or item["wall_us"] is not None
            and (type(item["wall_us"]) is not int or item["wall_us"] != item["charged_us"])
            or type(item["covered_us"]) is not int
            or not 0 <= item["covered_us"] <= item["charged_us"]
            or item["busy_us"] is not None
            and (type(item["busy_us"]) is not int or not 0 <= item["busy_us"] <= item["covered_us"])
        ):
            raise ValueError("invalid reported research cost")
        known.add((item["block"], item["role"]))
        total += item["lease_us"]
    role_us = sum(item["charged_us"] for item in doc["costs"])
    if (
        not doc["costs"]
        or total != doc["research_gpu_us"]
        or role_us != doc["research_role_us"]
        or not 0 <= total <= role_us
    ):
        raise ValueError("pilot lease total differs from its registered spending")
    required_costs, permitted_costs = set(), set()
    previous = None
    for look in doc["looks"]:
        if previous and previous["action"] != "CONTINUE_TO_RESERVED_ENDPOINT":
            raise ValueError("observations after a terminal pilot decision")
        target = look["target"]
        count = 3 if target == LOOKS[-1] else 2
        if set(look) != {
            "target",
            "cuts",
            "counts",
            "contrasts",
            "comparable",
            "zero_reused",
            "action",
        } or set(look["cuts"]) != set(ARMS):
            raise ValueError("invalid pilot look")
        if type(look["comparable"]) is not bool:
            raise ValueError("invalid comparison flag")
        cuts = look["cuts"]
        for arm, cut in cuts.items():
            if anchored and any(cut.get(k) != v for k, v in committed["lineage"].items()):
                raise ValueError("checkpoint differs from pre-score training lineage")
            if (
                cut["target"] != target
                or cut["registration"] != doc["registration"]["identity"]
                or cut["pair"] != 0
                or cut["seeds"] != pair_seeds(cut["registration"], 0)
                or cut["arm"] != arm
            ):
                raise ValueError("foreign matched training cut")
            if previous:
                prior = previous["cuts"][arm]
                if any(cut[k] != prior[k] for k in ("initial", "stream", "codec", "common_policy")) or any(
                    cut[k] < prior[k] for k in ("updates", "cursor", "charged", "useful_tokens", "discarded")
                ):
                    raise ValueError("pilot training lineage changed between looks")
        paired = paired_generator_exposure(*(cuts[k] for k in ARMS)) == "MATCHED"
        counts = look["counts"]
        contrasted = summary(counts, count)
        if any(sum(c["counts"].values()) > (2 if count == 3 else 1) for c in counts):
            raise ValueError("family size exceeds the registered allocation")
        if current:
            observed = [dict(stratum=c["stratum"], group=c["group"], size=sum(c["counts"].values())) for c in counts]
            if observed != design["sizes"][str(target)]:
                raise ValueError("observed family sizes differ from pilot freeze")
        shared = look["zero_reused"]
        if (
            type(shared) is not bool
            or shared
            and (
                target != 0
                or not paired
                or any(code[0] != code[1] for cell in counts for code in cell["counts"])
            )
        ):
            raise ValueError("invalid shared zero observation")
        for i, arm in enumerate((*ARMS, "incumbent") if count == 3 else ARMS):
            block = f"{arm}-{target}"
            if arm in ARMS:
                required_costs.add((block, "train"))
                permitted_costs.add((block, "train"))
            if shared and arm == ARMS[1]:
                continue
            required_costs.add((block, "search"))
            permitted_costs.update(((block, "search"), (block, "score")))
            if any(code[i] != "U" for cell in counts for code in cell["counts"]):
                required_costs.add((block, "score"))
        strata = contrasted["first_divergence_minus_ordinary"]["strata"]
        expected = set(STRATA if count == 3 else STRATA[:-1])
        if set(strata) != expected or any(
            (v["cases"] != 32 if count == 3 else not 1 <= v["cases"] <= 32)
            or s in STRATA[:2]
            and v["families"] < 16
            for s, v in strata.items()
        ):
            raise ValueError("registered measurement denominator differs")
        for stratum in expected:
            split = "development_retention" if stratum == "retention" else "development"
            original_cell = panel["summary"]["panels"][split][stratum]
            expected_cases = original_cell["cases"] if count == 3 else original_cell["families"]
            if (strata[stratum]["cases"] != expected_cases
                    or strata[stratum]["families"] != original_cell["families"]):
                raise ValueError("pilot denominator differs from the authenticated reservation")
        action = (
            spending(counts, paired, look["comparable"])
            if count == 3
            else "CONTINUE_TO_RESERVED_ENDPOINT"
            if paired and look["comparable"]
            else "ENDPOINT_NOT_ADMITTED"
        )
        if contrasted != look["contrasts"] or action != look["action"]:
            raise ValueError("reported pilot evidence differs from family counts")
        previous = look
    if targets[-1] != LOOKS[-1] and previous["action"] != "ENDPOINT_NOT_ADMITTED":
        raise ValueError("unfinished pilot lacks terminal decision")
    recorded = {(re.split("-(?:resume|scoreretry)-", block)[0], role) for block, role in known}
    if not required_costs <= recorded <= permitted_costs:
        raise ValueError("research cost coverage differs from the observed arms and roles")
    if doc["action"] != doc["looks"][-1]["action"]:
        raise ValueError("unverified pilot spending action")
    return dict(
        status="VERIFIED",
        scope="archive_only_pilot",
        **({"prescore": committed["identity"]} if anchored else {}),
        report=doc["identity"],
        next_action=doc["action"],
        registered_role_us=SHIFT_US,
        overrun_us=max(0, role_us - SHIFT_US),
        research_gpu_us=doc["research_gpu_us"],
        research_role_us=doc["research_role_us"],
        endpoint=doc["looks"][-1]["contrasts"] if targets[-1] == LOOKS[-1] else None,
        **({"unmeasured_looks": list(LOOKS[len(targets):]),
            "forecast_without_sharing_us": sum(prices.values()),
            "forecast_with_observed_sharing_us": sum(prices.values()) - (sum(prices[(ARMS[1]+"-0",role)] for role in ("search","score")) if doc["looks"][0]["zero_reused"] else 0)} if current else {}),
        assessment="SEALED",
        repeat_funded=False,
    )


def pilot_archive(report, parent):
    return archive_bytes({"pilot.json": encode(report), "round028-parent.zip": parent})


def verify_score_control(control, scale):
    fields = {"identity", "historical_candidates", "candidates", "scores", "checked_choices",
              "scored_bytes", "worker_us", "supervisor_us", "supervisor_margin_us", "measurements"}
    if not isinstance(control, dict) or set(control) != fields or control != seal(control):
        raise ValueError("invalid exposed scorer control record")
    parts, wall = control["measurements"], control["supervisor_us"]
    digests = (control[k] for k in ("historical_candidates", "candidates", "scores"))
    if (
        scale["selected_choices"] != 0
        or type(control["checked_choices"]) is not int or control["checked_choices"] != scale["cases"]
        or any(type(control[k]) is not int or control[k] <= 0 for k in ("scored_bytes", "worker_us"))
        or any(not isinstance(d, str) or not re.fullmatch("sha256:[0-9a-f]{64}", d) for d in digests)
        or control["candidates"] == scale["candidate"] or control["scores"] == scale["scores"]
        or (wall is not None and (type(wall) is not int or wall < control["worker_us"]))
        or control["supervisor_margin_us"] != (None if wall is None else scale["score_supervisor_allowance_us"] - wall)
        or (parts is not None and (
            not isinstance(parts, dict) or set(parts) != {"score_choices_us", "publish_scores_us"}
            or any(type(v) is not int or v < 0 for v in parts.values()) or sum(parts.values()) > control["worker_us"]
        ))
    ):
        raise ValueError("invalid exposed scorer control measurements")


def resident_measurement_gaps(scale, portfolio_seconds):
    """Count missing physical evidence before opening a fresh pilot look."""
    minimum = scale.get("minimum_case_allowance_us")
    control = scale.get("score_control")
    gaps = {
        "missing_full_case_allowance": int(type(minimum) is not int or minimum != portfolio_seconds * 1_000_000),
        "missing_selected_program": int(not scale.get("selected_choices") and control is None),
        "missing_phase_timings": sum(scale.get(k + "_measurements") is None for k in ("search", "score")),
        "missing_supervisor_timings": sum(scale.get(k + "_supervisor_us") is None for k in ("search", "score")),
    }
    if "score_control" in scale:
        verify_score_control(control, scale)
        gaps["missing_phase_timings"] += int(control["measurements"] is None)
        gaps["missing_supervisor_timings"] += int(control["supervisor_us"] is None)
    return {key: count for key, count in gaps.items() if count}


def verify_qualification(path):
    """Recompute the release identity, forecast and timing margins from the copied receipt."""
    from .program_search_timing import public_pilot_prices

    files = read_canonical(path, ("qualification.json", "round028-parent.zip"))
    raw = files["qualification.json"]
    doc = json.loads(raw)
    parent_bytes = files["round028-parent.zip"]
    parent = verify_workstation(io.BytesIO(parent_bytes))
    with (
        zipfile.ZipFile(io.BytesIO(parent_bytes)) as outer,
        zipfile.ZipFile(io.BytesIO(outer.read("round028.zip"))) as evidence,
    ):
        prices = public_pilot_prices(
            *(
                json.loads(evidence.read(name + ".json"))
                for name in ("panels", "rehearsal", "timing")
            )
        )
    reg = doc["registration"]
    reconciled = doc["version"] == "round028-resident-qualification/v4"
    anchored = doc["version"] == "round028-resident-qualification/v3" or reconciled and "prescore" in doc
    current = reconciled or anchored or doc["version"] == "round028-resident-qualification/v2"
    if reconciled:
        from .program_search_timing import projected_funding
        expected_price = projected_funding(prices, doc["balance"], reg["identity"])
        if expected_price > SHIFT_US:
            raise ValueError("pilot funding deficit")
        stop = doc["balance"]["stopped_after"]
        if stop is not None:
            terminal = doc["terminal"]
            verified = verify_pilot(io.BytesIO(pilot_archive(terminal, parent_bytes)))
            if (terminal["registration"] != reg or terminal["acceptance"] != doc["acceptance"]
                    or terminal["looks"][-1]["target"] != stop or verified["report"] != terminal["identity"]
                    or sorted(doc["balance"]["charges"]) != sorted([c["block"], c["role"], c["charged_us"]] for c in terminal["costs"])):
                raise ValueError("qualification differs from terminal pilot")
            expected_status = "PILOT_MEASURED" if stop == LOOKS[-1] else "PILOT_INCOMPLETE"
        else:
            expected_status = "PILOT_INCOMPLETE" if doc["balance"]["charges"] or doc["balance"]["opened"] else "PILOT_UNMEASURED"
        if doc["scientific_status"] != expected_status:
            raise ValueError("scientific status differs from pilot funding evidence")
    else:
        expected_price = sum(prices.values())
    if anchored:
        verify_prescore(doc["prescore"], reg["identity"], doc["acceptance"])
    if (
        set(doc)
        != {
            "version",
            "identity",
            "controller",
            "registration",
            "acceptance",
            "scale",
            "projected_us",
            "registered_us",
            "scientific_status",
            *({"workspace_space"} if current else ()),
            *({"prescore"} if anchored else ()),
            *({"balance"} if reconciled else ()),
            *({"terminal"} if reconciled and stop is not None else ()),
        }
        or doc["version"] not in ("round028-resident-qualification/v1", "round028-resident-qualification/v2", "round028-resident-qualification/v3", "round028-resident-qualification/v4")
        or doc != seal(doc)
        or raw != encode(doc)
        or not re.fullmatch("[0-9a-f]{40}", doc["controller"])
        or reg != pilot_registration(parent, reg["candidate"], reg["source"])
        or not re.fullmatch("sha256:[0-9a-f]{64}", doc["acceptance"])
        or doc["registered_us"] != SHIFT_US
        or doc["projected_us"] != expected_price
        or doc["scientific_status"]
        not in {"PILOT_UNMEASURED", "PILOT_INCOMPLETE", "PILOT_MEASURED"}
    ):
        raise ValueError("invalid pilot qualification release or forecast")
    scale = doc["scale"]
    keys = {
        "identity",
        "cases",
        "exposed_problems",
        "selected_choices",
        "no_selection",
        "cold_attempt",
        "candidates_bytes",
        "scored_bytes",
        "candidate",
        "scores",
        "search_us",
        "search_allowance_us",
        "search_margin_us",
        "score_allowance_us",
        "score_margin_us",
        "slowest_case_us",
        *({"minimum_case_allowance_us", "slowest_case_margin_us"} if current else ()),
        "search_measurements",
        "score_measurements",
        *(
            f"{phase}_{field}_us"
            for phase in ("search", "score")
            for field in ("worker", "supervisor", "supervisor_allowance", "supervisor_margin")
        ),
    }
    if current and "score_control" in scale:
        keys.add("score_control")
    if (
        set(scale) != keys
        or scale != seal(scale)
        or scale["cases"] != 160
        or scale["exposed_problems"] != 8
        or not all(re.fullmatch("sha256:[0-9a-f]{64}", scale[k]) for k in ("candidate", "scores"))
        or any(
            type(scale[k]) is not int or scale[k] < 0
            for k in (
                "selected_choices",
                "no_selection",
                "cold_attempt",
                "slowest_case_us",
                "search_us",
            )
        )
        or scale["selected_choices"] + scale["no_selection"] != scale["cases"]
        or any(
            type(scale[k]) is not int or scale[k] <= 0
            for k in (
                "candidates_bytes",
                "scored_bytes",
                "search_allowance_us",
                "search_worker_us",
                "score_worker_us",
            )
        )
        or scale["search_allowance_us"] != resident_seconds(scale["cases"]) * 1_000_000
        or scale["score_allowance_us"] != scale["search_allowance_us"]
        or scale["search_us"] > scale["search_worker_us"]
        or scale["search_margin_us"] != scale["search_allowance_us"] - scale["search_us"]
    ):
        raise ValueError("invalid resident roster or search measurements")
    for phase, names in (
        ("search", {"load_and_identify_checkpoint_us", "publish_candidates_us"}),
        ("score", {"score_choices_us", "publish_scores_us"}),
    ):
        wall = scale[phase + "_supervisor_us"]
        limit = scale[phase + "_supervisor_allowance_us"]
        margin = scale[phase + "_supervisor_margin_us"]
        parts = scale[phase + "_measurements"]
        if (
            type(limit) is not int
            or limit != scale["search_allowance_us"] + RESIDENT_SUPERVISOR_GRACE_SECONDS * 1_000_000
            or (wall is not None and (type(wall) is not int or wall < scale[phase + "_worker_us"]))
            or margin != (None if wall is None else limit - wall)
            or (
                parts is not None
                and (
                    set(parts) != names
                    or any(type(v) is not int or v < 0 for v in parts.values())
                    or sum(parts.values()) > scale[phase + "_worker_us"]
                )
            )
        ):
            raise ValueError("inconsistent resident role measurements")
    measured_score = scale["score_measurements"]
    if scale["score_margin_us"] != (
        None
        if measured_score is None
        else scale["score_allowance_us"] - measured_score["score_choices_us"]
    ):
        raise ValueError("inconsistent independent scoring margin")
    if current:
        space = doc["workspace_space"]
        if (
            resident_measurement_gaps(scale, reg["portfolio_seconds"])
            or type(scale["slowest_case_margin_us"]) is not int
            or scale["slowest_case_margin_us"] != scale["minimum_case_allowance_us"] - scale["slowest_case_us"]
            or not isinstance(space, dict)
            or set(space) != {"free_bytes", "free_inodes"}
            or any(type(v) is not int or v < 0 for v in space.values())
        ):
            raise ValueError("incomplete resident case, scorer or filesystem measurements")
    return dict(
        status="VERIFIED",
        scope="archive_only_pilot_qualification",
        **({"prescore": doc["prescore"]} if anchored else {}),
        controller=doc["controller"],
        worker=reg["candidate"],
        registration=reg["identity"],
        acceptance=doc["acceptance"],
        scientific_status=doc["scientific_status"],
        scale=scale,
        **({"workspace_space": space} if current else {}),
        **({"balance": doc["balance"]} if reconciled else {}),
        projected_us=doc["projected_us"],
        registered_us=SHIFT_US,
        research_headroom_us=SHIFT_US - doc["projected_us"],
    )
