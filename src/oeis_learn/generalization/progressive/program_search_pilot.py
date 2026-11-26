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
from .training_rows import pair_seeds

ARMS = ("ordinary", "first_divergence")
LOOKS = (0, 1_000_000, 4_000_000)
SHIFT_US = 11 * 3600 * 1_000_000
STRATA = ("composition", "unseen_structure", "parameter", "oeis", "retention")
PILOT_MEMBERS = ["pilot.json", "round028-parent.zip"]


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


def verify_pilot(path):
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
    if (
        set(doc) != fields
        or seal(doc) != doc
        or raw != encode(doc)
        or doc["version"] != "round028-generator-pilot/v1"
        or doc["registration"]
        != pilot_registration(
            parent, doc["registration"]["candidate"], doc["registration"]["source"]
        )
        or doc["verification"] != "private_choices_and_checkpoints"
        or any(not re.fullmatch("sha256:[0-9a-f]{64}", doc[k]) for k in ("freeze", "acceptance"))
        or type(doc["research_gpu_us"]) is not int
        or doc["research_gpu_us"] < 0
        or [r["target"] for r in doc["looks"]] != list(LOOKS)
    ):
        raise ValueError("invalid pilot registration or endpoint")
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
    for look in doc["looks"]:
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
            if (
                cut["target"] != target
                or cut["registration"] != doc["registration"]["identity"]
                or cut["pair"] != 0
                or cut["seeds"] != pair_seeds(cut["registration"], 0)
                or cut["arm"] != arm
            ):
                raise ValueError("foreign matched training cut")
        paired = paired_generator_exposure(*(cuts[k] for k in ARMS)) == "MATCHED"
        counts = look["counts"]
        contrasted = summary(counts, count)
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
            if strata[stratum]["cases"] != expected_cases:
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
    recorded = {(re.split("-(?:resume|scoreretry)-", block)[0], role) for block, role in known}
    if not required_costs <= recorded <= permitted_costs:
        raise ValueError("research cost coverage differs from the observed arms and roles")
    if doc["action"] != doc["looks"][-1]["action"]:
        raise ValueError("unverified pilot spending action")
    return dict(
        status="VERIFIED",
        scope="archive_only_pilot",
        report=doc["identity"],
        next_action=doc["action"],
        registered_role_us=SHIFT_US,
        overrun_us=max(0, role_us - SHIFT_US),
        research_gpu_us=doc["research_gpu_us"],
        research_role_us=doc["research_role_us"],
        endpoint=doc["looks"][-1]["contrasts"],
        assessment="SEALED",
        repeat_funded=False,
    )


def pilot_archive(report, parent):
    return archive_bytes({"pilot.json": encode(report), "round028-parent.zip": parent})
