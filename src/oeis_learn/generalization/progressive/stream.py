"""Durable reserve-before-draw stream and immutable verified shard cuts."""

import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path

from oeis_learn.pipeline_validation.identity import confined, seal

from .contracts import atomic_write, digest, load, record, reference, validate, verify_refs
from .generator import draw, validate_policy


class StreamStore:
    def __init__(self, root, policy, *, purpose="acceptance"):
        if purpose not in {"acceptance", "research"}:
            raise ValueError("declared stream purpose required")
        self.purpose = purpose
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = copy.deepcopy(validate_policy(policy))
        self.binding = seal(
            {"version": "progressive-stream-binding/v1", "policy": self.policy, "purpose": purpose}
        )
        if (self.root / "binding.json").exists():
            if load(self.root / "binding.json") != self.binding:
                raise ValueError("immutable stream policy differs")
        else:
            atomic_write(self.root / "binding.json", self.binding)
        with self._lock():
            self._recover()

    @contextmanager
    def _lock(self):
        import fcntl

        with (self.root / "stream.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _read(self, *, cold=False):
        path = self.root / "events.jsonl"
        cached = getattr(self, "_cache", None)
        offset = getattr(self, "_offset", 0)
        info = path.stat() if path.exists() else None
        if cached is not None and info is not None and hasattr(self, "_inode"):
            if info.st_ino != self._inode or info.st_size < offset:
                raise ValueError("journal replaced or truncated")
            if info.st_size == offset and info.st_mtime_ns != self._mtime:
                raise ValueError("journal changed without append")
            if info.st_size == offset and not cold:
                return cached
        if not cold and cached is not None:
            state = dict(cached, rows=list(cached["rows"]), shards=list(cached["shards"]))
        else:
            offset = 0
            state = {
                "tip": self.binding["identity"],
                "pending": None,
                "draws": 0,
                "work_units": 0,
                "sealed_shards": 0,
                "shards": [],
                "rows": [],
                "rejected": 0,
                "preparation_elapsed_us": 0,
                "preparation_timing_complete": True,
            }
        if info is not None:
            if info.st_size > 64 << 20:
                raise ValueError("bounded stream journal required")
            with path.open("rb") as stream:
                stream.seek(offset)
                for line in stream:
                    e = json.loads(line)
                    if seal(e) != e or e["previous"] != state["tip"]:
                        raise ValueError("stream journal chain differs")
                    state["tip"] = e["identity"]
                    if e["kind"] == "reserve":
                        if state["pending"] is not None or e["ordinal"] != state["draws"]:
                            raise ValueError("ordinal skipped or duplicated")
                        state["pending"] = state["draws"]
                        state["draws"] += 1
                        state["work_units"] += 1
                    elif e["kind"] == "replay":
                        state["preparation_timing_complete"] = False
                        if state["pending"] != e["ordinal"]:
                            raise ValueError("replay has no reservation")
                        state["work_units"] += 1
                    elif e["kind"] == "result":
                        if state["pending"] != e["ordinal"]:
                            raise ValueError("unreserved draw result")
                        if "elapsed_us" not in e:
                            state["preparation_timing_complete"] = False
                        elapsed = e.get("elapsed_us", 0)
                        if type(elapsed) is not int or elapsed < 0:
                            raise ValueError("invalid measured preparation time")
                        state["preparation_elapsed_us"] += elapsed
                        state["pending"] = None
                        if e["row"] is not None:
                            state["rows"].append(e["row"])
                        else:
                            state["rejected"] += 1
                    elif e["kind"] == "seal":
                        if state["pending"] is not None:
                            raise ValueError("cannot seal an active draw")
                        validate("stream", load(confined(self.root, e["manifest"]["path"])))
                        verify_refs(self.root, [e["manifest"]])
                        state["shards"].append(e["manifest"])
                        state["sealed_shards"] += 1
                        state["rows"] = []
                        state["rejected"] = 0
                    else:
                        raise ValueError("unknown stream event")
                self._offset = stream.tell()
            current = path.stat()
            self._inode = current.st_ino
            self._mtime = current.st_mtime_ns
            if (
                cold
                and cached is not None
                and info.st_size == getattr(self, "_prior_size", -1)
                and state["tip"] != cached["tip"]
            ):
                raise ValueError("journal prefix differs from observed cut")
            self._prior_size = current.st_size
        else:
            self._offset = 0
        if (
            state["draws"] > self.policy["max_draws"]
            or state["work_units"] > self.policy["max_work_units"]
        ):
            raise ValueError("journal exceeds frozen resources")
        self._cache = state
        return state

    def _append(self, kind, ordinal, **data):
        s = self._read()
        event = seal(dict(previous=s["tip"], kind=kind, ordinal=ordinal, **data))
        from .storage import append_json

        append_json(self.root / "events.jsonl", event)
        fd = os.open(self.root, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _recover(self):
        s = self._read()
        path = self.root / "shards" / f"{s['sealed_shards']:06d}.json"
        if path.exists():
            manifest = validate("stream", load(path))
            if manifest["payload"]["ledger_tip"] != s["tip"]:
                raise ValueError("uncommitted shard cut differs")
            self._append("seal", s["draws"], manifest=reference(self.root, path))

    def state(self):
        with self._lock():
            return copy.deepcopy(self._read())

    def shards(self):
        return [load(confined(self.root, r["path"])) for r in self.state()["shards"]]

    def step(self, *, client=None, fault=None, budget=None):
        with self._lock():
            self._recover()
            s = self._read()
            p = self.policy
            if fault not in (None, "after_reserve"):
                raise ValueError("registered diagnostic fault required")
            if (
                s["work_units"] >= p["max_work_units"]
                or len(s["rows"]) >= p["shard_rows"]
                or s["pending"] is None
                and s["draws"] >= p["max_draws"]
            ):
                raise TimeoutError("frozen draw/work/shard allowance exhausted")
            import time

            began = time.monotonic_ns()
            ordinal = s["pending"] if s["pending"] is not None else s["draws"]
            if budget:
                budget.consume(draws=1, attempts=1, rejections=1)
            self._append("replay" if s["pending"] is not None else "reserve", ordinal)
            if fault == "after_reserve":
                raise InterruptedError("diagnostic interruption after reservation")
            value = draw(p, ordinal)
            from oeis_learn.pipeline_validation.progressive_preparation import verify_draw

            row, reason = verify_draw(value, client=client, budget=budget)
            if row is not None:
                if row["structure"] in p["forbidden_structures"]:
                    row, reason = None, "held_out_structure"
                elif row["behavior"] in p["forbidden_behaviors"]:
                    row, reason = None, "finite_behavior_exclusion"
                elif any(
                    digest(row["values"][i : i + 20]) in p["forbidden_prefixes"] for i in range(81)
                ):
                    row, reason = None, "finite_prefix_or_shift_exclusion"
            if row is not None:
                raw = (json.dumps(row, sort_keys=True, indent=2) + "\n").encode()
                if sum(r["bytes"] for r in s["rows"]) + len(raw) > p["shard_bytes"]:
                    row, reason = None, "shard_byte_limit"
            ref = None
            if row is not None:
                path = self.root / "rows" / f"{ordinal:08d}.json"
                if path.exists():
                    if load(path) != row:
                        raise ValueError("replayed row bytes differ")
                else:
                    atomic_write(path, row)
                ref = reference(self.root, path)
            self._append(
                "result",
                ordinal,
                row=ref,
                reason=reason,
                elapsed_us=(time.monotonic_ns() - began) // 1000,
            )
            return {
                "ordinal": ordinal,
                "status": "accepted" if ref else "rejected",
                "row": ref,
                "reason": reason,
            }

    def seal_shard(self, *, fault=None):
        with self._lock():
            self._recover()
            s = self._read(cold=True)
            verify_refs(self.root, s["rows"], max_bytes=256 << 20)
            if s["pending"] is not None or not s["rows"] and not s["rejected"]:
                raise ValueError("terminal nonempty draw cut required")
            if fault not in (None, "after_shard_write"):
                raise ValueError("registered diagnostic fault required")
            rows = [load(confined(self.root, r["path"])) for r in s["rows"]]
            payload = {
                "preparation_elapsed_us": s["preparation_elapsed_us"]
                if s["preparation_timing_complete"]
                else None,
                "source_policy": digest(self.policy),
                "rows": s["rows"],
                "draw_high": s["draws"],
                "accepted": len(rows),
                "rejected": s["rejected"],
                "work_units": s["work_units"],
                "bytes": sum(r["bytes"] for r in s["rows"]),
                "membership": "training",
                "native_verified": bool(rows) and all(r["native_verified"] for r in rows),
                "ledger_tip": s["tip"],
                "behavior_scope": "finite_100_terms_only",
            }
            manifest = record(
                "stream",
                payload,
                created_at="2026-11-08T00:00:00Z",
                purpose=self.purpose,
                track="synthetic_strict",
            )
            path = self.root / "shards" / f"{s['sealed_shards']:06d}.json"
            atomic_write(path, manifest)
            if fault == "after_shard_write":
                raise InterruptedError("diagnostic interruption after shard publication")
            self._append("seal", s["draws"], manifest=reference(self.root, path))
            return manifest


def load_shard(root, manifest):
    p = validate("stream", manifest)["payload"]
    if not p["native_verified"]:
        raise ValueError("native verified sealed shard required for learning")
    blobs = verify_refs(root, p["rows"], max_bytes=256 << 20)
    rows = [json.loads(b) for b in blobs]
    if len(rows) != p["accepted"] or sum(r["bytes"] for r in p["rows"]) != p["bytes"]:
        raise ValueError("shard inventory differs")
    if any(seal(r) != r or r["membership"] != "training" or not r["native_verified"] for r in rows):
        raise ValueError("ineligible learner row")
    return rows
