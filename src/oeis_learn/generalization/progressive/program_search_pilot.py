"""Prospective continuation of the measured round028 generator study."""

import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from fractions import Fraction

from .contracts import seal
from .program_search_outcome import archive_bytes, verify_workstation
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

    with zipfile.ZipFile(path) as z:
        if (
            z.namelist() != PILOT_MEMBERS
            or z.comment
            or any(i.file_size > 2 << 20 or i.extra or i.comment for i in z.infolist())
        ):
            raise ValueError("invalid pilot archive")
        raw = z.read("pilot.json")
        doc = json.loads(raw)
        parent = verify_workstation(io.BytesIO(z.read("round028-parent.zip")))
    fields = {
        "version",
        "identity",
        "registration",
        "freeze",
        "acceptance",
        "looks",
        "action",
        "research_gpu_us",
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
    for look in doc["looks"]:
        target = look["target"]
        count = 3 if target == LOOKS[-1] else 2
        if set(look) != {"target", "cuts", "counts", "contrasts", "comparable", "action"} or set(
            look["cuts"]
        ) != set(ARMS):
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
        strata = contrasted["first_divergence_minus_ordinary"]["strata"]
        expected = set(STRATA if count == 3 else STRATA[:-1])
        if set(strata) != expected or any(
            (v["cases"] != 32 if count == 3 else not 1 <= v["cases"] <= 32)
            or s in STRATA[:2]
            and v["families"] < 16
            for s, v in strata.items()
        ):
            raise ValueError("registered measurement denominator differs")
        action = (
            spending(counts, paired, look["comparable"])
            if count == 3
            else "CONTINUE_TO_RESERVED_ENDPOINT"
            if paired and look["comparable"]
            else "ENDPOINT_NOT_ADMITTED"
        )
        if contrasted != look["contrasts"] or action != look["action"]:
            raise ValueError("reported pilot evidence differs from family counts")
    if doc["action"] != doc["looks"][-1]["action"]:
        raise ValueError("unverified pilot spending action")
    return dict(
        status="VERIFIED",
        scope="archive_only_pilot",
        report=doc["identity"],
        next_action=doc["action"],
        registered_gpu_us=SHIFT_US,
        research_gpu_us=doc["research_gpu_us"],
        endpoint=doc["looks"][-1]["contrasts"],
        assessment="SEALED",
        repeat_funded=False,
    )


def pilot_archive(report, parent):
    return archive_bytes({"pilot.json": encode(report), "round028-parent.zip": parent})
