"""Immutable OEIS admission archive cuts; sample task before alternative."""

import random
from pathlib import Path

from .contracts import atomic_write, digest, load, record, reference, validate, verify_refs


class Archive:
    def __init__(self, root, *, capacity=4096, seed=13):
        if type(capacity) is not int or not 1 <= capacity <= 4096 or type(seed) is not int:
            raise ValueError("frozen bounded archive policy required")
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError("archive symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.seed = seed

    def publish(self, admissions, *, previous=None):
        values = {}
        if previous is not None:
            for r in self.entries(previous):
                values[r["identity"]] = r
        for r in admissions:
            r = validate("admission", r)
            if r["track"] != "oeis_derived" or r["payload"]["status"] != "COMMITTED":
                raise ValueError("committed OEIS targets only")
            values[r["identity"]] = r
        grouped = {}
        for r in values.values():
            grouped.setdefault(r["payload"]["task"], []).append(r)
        selected = []
        alternatives = []
        # Stable task order is the declared eviction rule, independent of score.
        for task in sorted(grouped)[: self.capacity]:
            candidates = sorted(
                grouped[task], key=lambda r: (len(r["payload"]["tokens"]), r["identity"])
            )
            keep = [candidates[0]]
            other = next(
                (
                    r
                    for r in candidates[1:]
                    if r["payload"]["structure"] != keep[0]["payload"]["structure"]
                ),
                None,
            )
            if other is not None:
                keep.append(other)
            selected.append(keep[0])
            alternatives.extend(keep[1:])
        selected += alternatives[: self.capacity - len(selected)]
        refs = []
        for value in selected:
            path = self.root / ("entry-" + value["identity"].split(":")[1] + ".json")
            if path.exists():
                if load(path) != value:
                    raise ValueError("conflicting archive target")
            else:
                atomic_write(path, value)
            refs.append(reference(self.root, path))
        return record(
            "archive",
            {
                "capacity": self.capacity,
                "seed": self.seed,
                "entries": refs,
                "previous": previous["identity"] if previous else None,
                "sampling": "task-first-then-alternative",
                "eviction": "stable-task-id",
                "maximum_alternatives": 2,
            },
            created_at="2026-11-08T00:00:00Z",
            parents=[previous["identity"]] if previous else [],
            purpose=selected[0]["purpose"] if selected else "acceptance",
            track="oeis_derived",
        )

    def entries(self, cut):
        p = validate("archive", cut)["payload"]
        if p["capacity"] != self.capacity or p["seed"] != self.seed:
            raise ValueError("archive policy changed")
        import json

        return [validate("admission", json.loads(b)) for b in verify_refs(self.root, p["entries"])]

    def sample(self, cut, *, ordinal, count=1):
        if (
            type(ordinal) is not int
            or ordinal < 0
            or type(count) is not int
            or not 1 <= count <= 64
        ):
            raise ValueError("bounded archive cursor required")
        tasks = {}
        for r in self.entries(cut):
            tasks.setdefault(r["payload"]["task"], []).append(r)
        if not tasks:
            raise ValueError("empty archive")
        result = []
        for i in range(count):
            rng = random.Random(
                int(digest([self.seed, cut["identity"], ordinal + i]).split(":")[1], 16)
            )
            key = rng.choice(sorted(tasks))
            value = rng.choice(tasks[key])
            p = value["payload"]
            result.append(
                {
                    "task": key,
                    "admission": value["identity"],
                    "values": p["visible_values"],
                    "tokens": p["tokens"],
                    "charged_tokens": len(p["tokens"]),
                    "membership": "training",
                }
            )
        return result
