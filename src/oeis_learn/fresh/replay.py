"""Opt-in, independently admitted shorter targets for training prompts only."""

import time
from .common import digest


def restore(records, rows, archive):
    by_id = {row["program_id"]: row for row in rows}
    targets = {}
    if len(records) > 32:
        raise ValueError("replay cap exceeded")
    for record in records:
        packet = archive.validate_packet(record["packet"], record["key"])
        original = by_id.get(record["program_id"])
        if (
            original is None
            or not packet["entry"]["training_eligible"]
            or packet["entry"]["parent_sha256"] != original["program_id"]
            or packet["entry"]["conditioning_values"] != original["values"]
            or len(packet["tokens"]) >= len(original["tokens"])
        ):
            raise ValueError("replay is not an eligible shorter training target")
        targets[original["program_id"]] = packet["tokens"]
    return targets


def admit(report, rows, archive, runner, existing, *, seconds=20):
    from oeis_learn.decoder.program_codec import decode_body, encode_body
    from oeis_learn.sandbox.pipeline import digest_bytes
    from oeis_learn.experiments.artifacts import compute_canonical_digest
    from .diagnostics import execute_values

    deadline = time.monotonic_ns() + seconds * 10**9
    by_id = {row["program_id"]: row for row in rows}
    retained = {record["program_id"]: record for record in existing}

    class BoundedRunner:
        def run_single(self, source, indices, *, request_id):
            return runner.run_single(source, indices, request_id=request_id, deadline_ns=deadline)

    for candidate in report["splits"]["training"]["attempts"][:8]:
        if time.monotonic_ns() >= deadline or len(retained) >= 32:
            break
        row = by_id[candidate["program_id"]]
        old_tokens = (
            retained.get(row["program_id"], {}).get("packet", {}).get("tokens", row["tokens"])
        )
        if (
            not candidate["matched"]
            or not candidate["complete"]
            or len(encode_body(candidate["source"])) >= len(old_tokens)
        ):
            continue
        source = candidate["source"]
        if not archive.holdouts.decision(source, list(map(int, row["values"])))[0]:
            continue
        expected, failure = execute_values(
            runner,
            decode_body(row["tokens"]),
            range(100),
            digest([row["program_id"], "replay-oracle"]),
            deadline,
        )
        if failure:
            continue
        evidence = runner.run_single(
            source,
            range(20),
            request_id=digest([row["program_id"], source, "replay-prefix"]),
            deadline_ns=deadline,
        ).to_dict()
        provenance = dict(
            version="search-hypothesis/v1",
            source_sha256=digest_bytes(source.encode()),
            body_tokens=encode_body(source),
            visible_prompt_sha256=compute_canonical_digest(row["values"]),
            execution=evidence,
            parent=None,
            **candidate["model_provenance"],
        )
        provenance["identity_sha256"] = compute_canonical_digest(provenance)
        try:
            key = archive.admit(
                source,
                list(map(int, row["values"])),
                list(map(int, expected)),
                origin="training_prefix_hypothesis",
                provenance={"search": provenance},
                runner=BoundedRunner(),
                request_id=digest([row["program_id"], source, "replay-admit"]),
                model_sha256=compute_canonical_digest(
                    [provenance["encoder_sha256"], provenance["decoder_sha256"]]
                ),
                parent_sha256=row["program_id"],
            )
        except ValueError:
            if time.monotonic_ns() >= deadline:
                break  # Preserve earlier admitted targets; no partial admission.
            raise
        packet = archive.get(key)
        retained[row["program_id"]] = dict(program_id=row["program_id"], packet=packet, key=key)
    result = [retained[key] for key in sorted(retained)]
    restore(result, rows, archive)
    return result


class SynthesisReplay:
    """Phase 2 pure replay policy; retained profile routing above is unchanged."""

    FIELDS = {
        "target",
        "split",
        "track",
        "source",
        "original_source",
        "assignment",
        "profile",
        "verification",
        "independent",
        "tokens",
    }

    def __init__(
        self,
        capacity=4096,
        track="generic_only",
        representation_comparison=False,
        training_targets=(),
    ):
        if (
            type(capacity) is not int
            or not 1 <= capacity <= 4096
            or track not in {"generic_only", "assisted"}
            or type(representation_comparison) is not bool
        ):
            raise ValueError("invalid replay contract")
        self.capacity = capacity
        self.track = track
        self.representation_comparison = representation_comparison
        if len(set(training_targets)) != len(training_targets) or any(
            not isinstance(k, str) or not k for k in training_targets
        ):
            raise ValueError("frozen unique training-target roster required")
        self.training_targets = tuple(sorted(training_targets))
        self.records = {}
        self.counter = 0
        self.cursor = 0

    def admit(self, row):
        import copy
        from oeis_learn.synthesis.contracts import canonical_integer
        from oeis_learn.synthesis.state import require_hash

        if (
            self.representation_comparison
            or set(row) != self.FIELDS
            or row["split"] != "training"
            or row["independent"] is not True
            or row["track"] not in {"generic_only", "assisted"}
            or self.track == "generic_only"
            and row["track"] != "generic_only"
            or row["profile"] != "bigint-body/v1"
            or not row["tokens"]
            or len(row["tokens"]) > 1024
        ):
            raise ValueError("replay requires verified compatible training ancestry")
        if row["target"] not in self.training_targets:
            raise ValueError("target absent from frozen training roster")
        from oeis_learn.decoder.bigint_codec import encode_body
        from oeis_learn.synthesis.admission import source_identity
        from oeis_learn.synthesis.reference_parser import parse

        parse(row["source"])
        parse(row["original_source"], allow_parameters=True)
        if row["tokens"] != encode_body(row["source"]):
            raise ValueError("replay source/token mismatch")
        verification = row["verification"]
        if (
            not isinstance(verification, dict)
            or set(verification)
            != {
                "target",
                "source_sha256",
                "runtime_sha256",
                "reference_sha256",
                "indices",
                "values",
                "admitted",
            }
            or verification["admitted"] is not True
            or verification["target"] != row["target"]
            or verification["source_sha256"] != source_identity(row["source"])
            or len(verification["indices"]) != 100
            or len(verification["values"]) != 100
        ):
            raise ValueError("full training-target independent verification required")
        for h in (
            verification["source_sha256"],
            verification["runtime_sha256"],
            verification["reference_sha256"],
        ):
            require_hash(h)
        ns = [canonical_integer(n) for n in verification["indices"]]
        if any(a >= b for a, b in zip(ns, ns[1:])):
            raise ValueError("verification indices must increase")
        for n in verification["values"]:
            canonical_integer(n, bits=16384)
        if not isinstance(row["assignment"], dict) or len(row["assignment"]) > 8:
            raise ValueError("invalid assignment")
        for n in row["assignment"].values():
            canonical_integer(n)
        key = digest(
            [row["target"], row["profile"], row["source"], row["assignment"], row["track"]]
        )
        if key in self.records:
            return key
        if self.counter >= 65536:
            raise ValueError("replay lifetime admission budget exhausted")
        self.counter += 1
        self.records[key] = dict(row=copy.deepcopy(row), order=self.counter)
        if len(self.records) > self.capacity:
            victim = min(self.records, key=lambda k: (self.records[k]["order"], k))
            del self.records[victim]
        return key

    def sample(self, batch):
        if type(batch) is not int or batch < 1:
            raise ValueError("invalid batch")
        if self.representation_comparison:
            return []
        keys = sorted(self.records)
        n = min(batch // 4, len(keys))
        if not n:
            return []
        import copy

        chosen = [
            copy.deepcopy(self.records[keys[(self.cursor + i) % len(keys)]]["row"])
            for i in range(n)
        ]
        self.cursor = (self.cursor + n) % len(keys)
        return chosen

    def snapshot(self):
        import copy

        return dict(
            version="synthesis-replay/v1",
            capacity=self.capacity,
            track=self.track,
            representation_comparison=self.representation_comparison,
            training_targets=self.training_targets,
            records=copy.deepcopy(self.records),
            counter=self.counter,
            cursor=self.cursor,
        )

    @classmethod
    def restore(cls, state):
        if (
            set(state)
            != {
                "version",
                "capacity",
                "track",
                "representation_comparison",
                "records",
                "counter",
                "cursor",
                "training_targets",
            }
            or state["version"] != "synthesis-replay/v1"
        ):
            raise ValueError("invalid replay state")
        result = cls(
            state["capacity"],
            state["track"],
            state["representation_comparison"],
            state["training_targets"],
        )
        if (
            len(state["records"]) > result.capacity
            or type(state["counter"]) is not int
            or type(state["cursor"]) is not int
            or not 0 <= state["counter"] <= 65536
            or state["cursor"] < 0
        ):
            raise ValueError("invalid replay capacity/cursor")
        seen = set()
        for key, value in sorted(state["records"].items(), key=lambda kv: kv[1]["order"]):
            if (
                set(value) != {"row", "order"}
                or type(value["order"]) is not int
                or not 0 < value["order"] <= state["counter"]
                or value["order"] in seen
            ):
                raise ValueError("invalid replacement history")
            seen.add(value["order"])
            if result.admit(value["row"]) != key:
                raise ValueError("replay identity mismatch")
            result.records[key]["order"] = value["order"]
        if result.records and state["cursor"] >= len(result.records):
            raise ValueError("invalid replay cursor")
        result.counter = state["counter"]
        result.cursor = state["cursor"]
        return result
