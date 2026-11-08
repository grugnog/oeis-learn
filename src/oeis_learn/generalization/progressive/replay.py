"""Deterministic whole-example token mixing over one immutable training cut."""

import copy
import hashlib
import json
import random

from .archive import Archive
from .contracts import digest, seal, validate

RULE = dict(
    version="progressive-oeis-mixture/v1",
    fraction=[1, 4],
    sampling="task-first-then-alternative",
    weighting="charged-target-tokens",
    freshness="one-frozen-cycle",
    overshoot="whole-example",
)


def freeze_replay(archive, cut, *, arm="replay"):
    if arm not in {"replay", "continued_synthetic"}:
        raise ValueError("paired replay/control arm required")
    entries = archive.entries(cut)
    return seal(
        dict(
            version="progressive-replay-input/v1",
            arm=arm,
            policy=RULE,
            archive=cut if arm == "replay" else None,
            admissions=entries if arm == "replay" else [],
        )
    )


def validate_charge(value):
    if (
        not isinstance(value, dict)
        or set(value)
        != {"archive", "oeis_tokens", "synthetic_tokens", "oeis_examples", "synthetic_examples"}
        or any(type(value[k]) is not int or value[k] < 0 for k in value if k != "archive")
    ):
        raise ValueError("closed irreversible replay debit required")
    if (
        value["oeis_examples"] > value["oeis_tokens"]
        or value["synthetic_examples"] > value["synthetic_tokens"]
    ):
        raise ValueError("whole nonempty target debit required")
    return value


class Mixer:
    def __init__(self, package):
        if (
            not isinstance(package, dict)
            or set(package) != {"version", "identity", "arm", "policy", "archive", "admissions"}
            or package["version"] != "progressive-replay-input/v1"
            or seal(package) != package
            or package["policy"] != RULE
            or package["arm"] not in {"replay", "continued_synthetic"}
        ):
            raise ValueError("closed frozen replay input required")
        self.package = copy.deepcopy(package)
        self.archive = package["archive"]["identity"] if package["archive"] else None
        self.groups = {}
        if package["arm"] == "continued_synthetic":
            if self.archive is not None or package["admissions"]:
                raise ValueError("synthetic control cannot receive discoveries")
            return
        p = validate("archive", package["archive"])["payload"]
        if not 1 <= len(package["admissions"]) <= p["capacity"] or len(p["entries"]) != len(
            package["admissions"]
        ):
            raise ValueError("bounded original archive inventory required")
        identities = set()
        for ref, item in zip(p["entries"], package["admissions"]):
            a = Archive.validate_entry(item)
            raw = (json.dumps(a, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            if (
                ref["bytes"] != len(raw)
                or ref["sha256"] != "sha256:" + hashlib.sha256(raw).hexdigest()
                or a["identity"] in identities
                or a["payload"]["native_verified"] is not True
            ):
                raise ValueError("original committed archive bytes required")
            identities.add(a["identity"])
            self.groups.setdefault(a["payload"]["task"], []).append(a)
        if any(
            len(v) > 2 or len({a["payload"]["structure"] for a in v}) != len(v)
            for v in self.groups.values()
        ):
            raise ValueError("task-balanced distinct archive alternatives required")

    def initial(self):
        return seal(
            dict(
                version="progressive-replay-state/v1",
                policy=digest(RULE),
                archive=self.archive,
                oeis_tokens=0,
                synthetic_tokens=0,
                oeis_examples=0,
                synthetic_examples=0,
                journal_index=0,
            )
        )

    def check(self, state):
        if (
            set(state) != set(self.initial())
            or seal(state) != state
            or any(state[k] != self.initial()[k] for k in ("version", "policy", "archive"))
            or any(
                type(state[k]) is not int or state[k] < 0
                for k in (
                    "oeis_tokens",
                    "synthetic_tokens",
                    "oeis_examples",
                    "synthetic_examples",
                    "journal_index",
                )
            )
        ):
            raise ValueError("exact checkpoint replay cut required")
        return state

    def plan(self, synthetic, state, *, count):
        state = self.check(state)
        if type(count) is not int or not 1 <= count <= 64 or not synthetic:
            raise ValueError("bounded whole-example mixture required")
        totals = {
            k: state[k]
            for k in ("oeis_tokens", "synthetic_tokens", "oeis_examples", "synthetic_examples")
        }
        rows = []
        for i in range(count):
            use_oeis = (
                self.package["arm"] == "replay"
                and 4 * totals["oeis_tokens"] <= totals["oeis_tokens"] + totals["synthetic_tokens"]
            )
            if use_oeis:
                rng = random.Random(int(digest([self.archive, totals["oeis_examples"]])[7:], 16))
                task = rng.choice(sorted(self.groups))
                a = rng.choice(self.groups[task])
                p = Archive.validate_entry(a)["payload"]
                row = dict(
                    draw_identity=a["identity"],
                    task=task,
                    group=p["group"],
                    values=p["visible_values"],
                    tokens=p["tokens"],
                    membership=p["membership"],
                )
            else:
                row = copy.deepcopy(synthetic[i % len(synthetic)])
            if row["membership"] != "training" or not row["tokens"]:
                raise ValueError("training membership must be rechecked before consumption")
            kind = "oeis" if use_oeis else "synthetic"
            totals[kind + "_tokens"] += len(row["tokens"])
            totals[kind + "_examples"] += 1
            rows.append(row)
        charge = dict(archive=self.archive, **{k: totals[k] - state[k] for k in totals})
        return rows, validate_charge(charge)

    def reconcile(self, state, events, *, task):
        value = dict(self.check(state))
        for event in events:
            if event["index"] < state["journal_index"]:
                continue
            if event["event"] == "RESERVE" and event["operation"].startswith(task + "/update/"):
                charge = validate_charge(event.get("details", {}).get("replay"))
                if (
                    charge["archive"] != self.archive
                    or charge["oeis_tokens"] + charge["synthetic_tokens"] != event["cost"]["tokens"]
                ):
                    raise ValueError("replay debit differs from original task/archive charge")
                for k in charge:
                    if k != "archive":
                        value[k] += charge[k]
            value["journal_index"] = max(value["journal_index"], event["index"] + 1)
        return seal(value)
