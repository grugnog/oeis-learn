"""Prospective selector replication and an anonymous, jointly paired formal comparison."""

from collections import Counter, defaultdict
import io
import json
import re

from .contracts import digest, seal
from .decisions import recompute
from .metrics import PRIMARY
from .report import MAX_ROWS
from .program_search_outcome import read_canonical
from .program_search_rehearsal_report import encode
from .training_rows import pair_seeds

MEMBERS = ("replication.json", "development.zip")
STRATA = {"composition", "unseen_structure", "parameter", "oeis", "retention"}
HEX = re.compile(r"sha256:[0-9a-f]{64}")
OUTCOMES = {"S": "success", "F": "failure", "U": "unresolved"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def freeze_panel(registration, roster, seconds):
    require(
        type(seconds) is int and seconds >= 0, "nonnegative prospective CPU reservation required"
    )
    require(len({m["task"] for m in roster}) == len(roster), "duplicate replication member")
    groups = {name: i for i, name in enumerate(sorted({m["group"] for m in roster}))}
    sizes = Counter((m["stratum"], groups[m["group"]]) for m in roster)
    return seal(
        dict(
            version="round028-selector-replication-design/v1",
            registration=registration,
            panel=digest(roster),
            sizes=[[s, g, n] for (s, g), n in sorted(sizes.items())],
            links=group_links(sizes),
            seeds=[pair_seeds(registration, i)["training_families"] for i in (0, 1)],
            comparisons=1,
            resamples=10_000,
            seconds=seconds,
        )
    )


def group_links(sizes):
    paired = sorted({g for s, g in sizes if s == PRIMARY[0] and (PRIMARY[1], g) in sizes})
    return [[f"{s}/{g}" for s in PRIMARY] for g in paired]


def joint_cells(roster, scores):
    require(
        len(scores) == 2 and all(set(p) == {m["task"] for m in roster} for p in scores),
        "both seeds require the complete common replication panel",
    )
    groups = {name: i for i, name in enumerate(sorted({m["group"] for m in roster}))}
    cells = defaultdict(Counter)
    for member in roster:
        code = "".join(p[member["task"]][a] for p in scores for a in ("reference", "live_learned"))
        cells[member["stratum"], groups[member["group"]]][code] += 1
    return [[s, g, dict(c)] for (s, g), c in sorted(cells.items())]


def formal_result(design, models, cells):
    fields = {
        "version",
        "registration",
        "panel",
        "sizes",
        "links",
        "seeds",
        "comparisons",
        "resamples",
        "seconds",
        "identity",
    }
    require(
        set(design) == fields
        and design == seal(design)
        and design["version"] == "round028-selector-replication-design/v1"
        and HEX.fullmatch(design["panel"])
        and design["comparisons"] == 1
        and design["resamples"] == 10_000
        and type(design["seconds"]) is int
        and design["seconds"] > 0,
        "invalid registered selector replication",
    )
    seeds = [pair_seeds(design["registration"], i)["training_families"] for i in (0, 1)]
    require(
        design["seeds"] == seeds and len(models) == 2 and len(set(seeds)) == 2,
        "two registered training seeds required",
    )
    for seed, model in zip(seeds, models, strict=True):
        require(
            set(model) == {"seed", "model", "stream", "training"}
            and model["seed"] == seed
            and all(HEX.fullmatch(model[k]) for k in ("model", "stream", "training")),
            "selector model provenance differs",
        )
    require(
        all(models[0][k] != models[1][k] for k in ("model", "stream", "training")),
        "independent selector streams required",
    )
    roster, rows, sizes = [], [], {}
    require(
        all(type(n) is int and n > 0 for _, _, n in design["sizes"])
        and sum(n for _, _, n in design["sizes"]) <= MAX_ROWS // 4,
        "bounded registered replication roster required",
    )
    registered = {(s, g): n for s, g, n in design["sizes"]}
    for stratum, group, counts in cells:
        require(
            stratum in STRATA
            and type(group) is int
            and group >= 0
            and (stratum, group) not in sizes
            and isinstance(counts, dict)
            and counts,
            "invalid replication family",
        )
        n = 0
        for code, count in sorted(counts.items()):
            require(
                len(code) == 4
                and set(code) <= OUTCOMES.keys()
                and type(count) is int
                and 0 < count <= registered.get((stratum, group), 0) - n,
                "invalid joint selector counts",
            )
            for ordinal in range(n, n + count):
                member = dict(
                    task=f"{stratum}/{group}/{ordinal}", group=str(group), stratum=stratum
                )
                roster.append(member)
                for i, symbol in enumerate(code):
                    rows.append(
                        dict(
                            member,
                            seed=seeds[i // 2],
                            arm=("baseline", "candidate")[i % 2],
                            outcome=OUTCOMES[symbol],
                        )
                    )
            n += count
        sizes[stratum, group] = n
    require(
        design["sizes"] == [[s, g, n] for (s, g), n in sorted(sizes.items())]
        and design["links"] == group_links(sizes),
        "replication sizes or independent-group links differ",
    )
    metrics, verdict = recompute(
        rows,
        roster,
        seeds,
        dict(
            claim="capability",
            costs=None,
            comparisons=design["comparisons"],
            bootstrap_seed=int(
                digest([design["registration"], "selector_family_resampling"])[7:15], 16
            ),
            links=design["links"],
        ),
    )
    # The common metrics include per-arm, per-seed parameter and OEIS strata independently.
    return dict(
        metrics=json.loads(encode(metrics)),
        verdict=verdict,
        action=(
            "RECOMMEND_SEPARATELY_BUDGETED_CONFIRMATION"
            if verdict == "CONFIRMED_GAIN"
            else "RETAIN_REFERENCE"
        ),
    )


def verify_replication(path):
    from .program_search_selector import verify_selector

    raw = read_canonical(path, MEMBERS)
    dev = verify_selector(io.BytesIO(raw["development.zip"]))
    development = json.loads(
        read_canonical(io.BytesIO(raw["development.zip"]), ("selector.json",))["selector.json"]
    )
    data = json.loads(raw["replication.json"])
    require(
        set(data)
        == {
            "version",
            "design",
            "development",
            "models",
            "cells",
            "result",
            "charged_us",
            "identity",
        }
        and data["version"] == "round028-selector-replication/v1"
        and data == seal(data)
        and encode(data) == raw["replication.json"]
        and dev["next_action"] == "RUN_RESERVED_REPLICATION"
        and dev["report"] == data["development"]
        and dev["replication"] == data["design"]["identity"]
        and development["design"]["registration"] == data["design"]["registration"]
        and development["design"]["replication"]["seconds"] == data["design"]["seconds"]
        and development["design"]["model"] == data["models"][0]["model"],
        "replication is not bound to a funded development decision",
    )
    require(
        type(data["charged_us"]) is int
        and 0 <= data["charged_us"] <= data["design"]["seconds"] * 1_000_000,
        "replication exceeds its frozen authorization",
    )
    result = formal_result(data["design"], data["models"], data["cells"])
    require(encode(result) == encode(data["result"]), "formal selector verdict differs")
    return dict(
        status="VERIFIED",
        scope="archive_only_selector_replication",
        report=data["identity"],
        next_action=result["action"],
        verdict=result["verdict"],
        final_assessment="SEALED",
    )
