"""Standalone supply and cost evidence for a prospective generator, with no private member keys."""

import io
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

from .cohorts import STRATA
from .contracts import seal
from .program_search_costing import PROXIES, forecast
from .program_search_panels import BANDS, RESERVATIONS
from .program_search_rehearsal_report import count, encode, fields, require, verify_transport

VERSION = "round028-generator-panel-supply/v1"
MEMBERS = ["panels.json", "round028-rehearsal.zip"]


def reduce(summary, cells, training):
    fields(summary, {"history", "supply", "extension", "panels", "calibration", "calibrated"})
    h, supply, extra = summary["history"], summary["supply"], summary["extension"]
    fields(
        h,
        {
            "original_reservations",
            "corrected_reservations",
            "native_training",
            "older_exposure_rows",
        },
    )
    fields(supply, {"candidates", "historical_rows", "eligible", "excluded", "components"})
    fields(extra, {"native_draws", "native_accepted", "source_selected", "source_remaining"})
    for v in (
        *h.values(),
        *supply.values(),
        extra["native_draws"],
        extra["native_accepted"],
        extra["source_selected"],
    ):
        count(v)
    if extra["source_remaining"] is not None:
        count(extra["source_remaining"])
    require(
        sum(h.values()) == supply["historical_rows"]
        and supply["eligible"] + supply["excluded"] == supply["candidates"]
        and supply["components"] <= supply["eligible"]
        and extra["native_accepted"] <= extra["native_draws"] <= 2048
        and extra["source_selected"] <= 4096
    )
    panels = summary["panels"]
    fields(panels, RESERVATIONS)
    misses, assigned = {}, 0
    for name, panel in panels.items():
        fields(panel, {"retention"} if name.endswith("_retention") else STRATA)
        for s, cell in panel.items():
            fields(cell, {"cases", "families", "bands"})
            count(cell["cases"])
            count(cell["families"])
            require(
                cell["cases"] <= 32
                and (cell["cases"] + 1) // 2 <= cell["families"] <= cell["cases"]
                and isinstance(cell["bands"], dict)
                and not set(cell["bands"]) - set(BANDS)
            )
            for n in cell["bands"].values():
                count(n, 1)
            require(sum(cell["bands"].values()) == cell["cases"])
            assigned += cell["cases"]
            if cell["cases"] < 32:
                misses[name + "/" + s] = {
                    "cases": 32 - cell["cases"],
                    "families": max(0, 16 - cell["families"]),
                }
    cal = summary["calibration"]
    fields(cal, {"reserved", "observed", "exact", "measured"})
    for n in cal.values():
        count(n)
    require(
        cal["observed"] in (0, cal["reserved"])
        and cal["reserved"] <= min(48, supply["components"])
        and cal["exact"] <= cal["measured"] <= 3 * cal["observed"]
        and assigned + cal["reserved"] <= supply["eligible"]
        and type(summary["calibrated"]) is bool
        and summary["calibrated"] == (cal["measured"] > 0)
    )
    if not cal["observed"]:
        require(
            all(not (set(c["bands"]) - {"unknown"}) for p in panels.values() for c in p.values())
        )
    fields(cells, PROXIES)
    for proxy in cells:
        fields(cells[proxy], STRATA)
    probes = {
        s: (p["families"] if p["cases"] == 32 and summary["calibrated"] else 32)
        for s, p in panels["development"].items()
    }
    costs = forecast(cells, training, probes)
    action = (
        "EXTEND_FRESH_SUPPLY"
        if misses
        else "MEASURE_DIFFICULTY_SUPPORT"
        if not summary["calibrated"]
        else "MEASURE_CANDIDATE_EVALUATION_COST"
    )
    return {
        "deficits": misses,
        "descriptive_cases_per_arm_look": probes,
        "historical_forecast": costs,
        "action": action,
    }


def make_report(registration, trainer, summary, cells, *, timings):
    return seal(
        dict(
            version=VERSION,
            registration=registration["identity"],
            candidate=registration["candidate"],
            rehearsal=trainer["evidence"],
            audit=trainer["audit"],
            scope="supply_on_registered_donor_training_support",
            gpu_dispatches=0,
            summary=summary,
            historical_clocks=cells,
            timings=timings,
            **reduce(summary, cells, trainer["forecast"]),
        )
    )


def verify_value(value, trainer):
    fields(
        value,
        {
            "version",
            "identity",
            "registration",
            "candidate",
            "rehearsal",
            "audit",
            "scope",
            "gpu_dispatches",
            "summary",
            "historical_clocks",
            "timings",
            "deficits",
            "descriptive_cases_per_arm_look",
            "historical_forecast",
            "action",
        },
    )
    require(
        seal(value) == value
        and value["version"] == VERSION
        and value["rehearsal"] == trainer["evidence"]
        and value["audit"] == trainer["audit"]
        and value["scope"] == "supply_on_registered_donor_training_support"
        and value["gpu_dispatches"] == 0
        and type(value["gpu_dispatches"]) is int
        and re.fullmatch("[0-9a-f]{40}", value["candidate"])
        and re.fullmatch("sha256:[0-9a-f]{64}", value["registration"])
    )
    fields(value["timings"], {"elapsed_us", "role_us", "unknown_role_intervals", "image_build_us"})
    for n in value["timings"].values():
        count(n)
    reduced = reduce(value["summary"], value["historical_clocks"], trainer["forecast"])
    require(all(value[k] == v for k, v in reduced.items()))
    return {
        "status": "VERIFIED",
        "scope": "archive_only_panel_supply",
        "candidate": value["candidate"],
        "evidence": value["identity"],
        "registration": value["registration"],
        "action": value["action"],
        "deficits": value["deficits"],
        "historical_forecast": value["historical_forecast"],
        "gpu_dispatches": 0,
    }


def verify_panels(path):
    with zipfile.ZipFile(path) as archive:
        require(
            archive.namelist() == MEMBERS
            and not archive.comment
            and all(
                not i.comment and not i.extra and i.file_size < 1 << 20 for i in archive.infolist()
            )
        )
        raw = archive.read(MEMBERS[0])
        value = json.loads(raw)
        require(encode(value) == raw)
        parent = archive.read(MEMBERS[1])
        with zipfile.ZipFile(io.BytesIO(parent)) as original:
            require(original.namelist() == ["rehearsal.json", "round028.zip"])
        trainer = verify_transport(io.BytesIO(parent))
        return verify_value(value, trainer)


def publish(path, value, rehearsal):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent) as staged:
        with zipfile.ZipFile(staged, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(MEMBERS[0], encode(value))
            archive.write(rehearsal, MEMBERS[1])
        staged.flush()
        os.fsync(staged.fileno())
        receipt = verify_panels(staged.name)
        os.link(staged.name, path)
    return receipt
