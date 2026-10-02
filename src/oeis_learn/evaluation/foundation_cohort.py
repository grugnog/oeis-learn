"""Indexed first-100 cohorts with exact witnesses and separate evaluator views."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import uuid
import yaml

from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    atomic_write,
    canonical_bytes,
    compute_canonical_digest as digest,
    compute_file_hash,
    load_json,
)
from oeis_learn.experiments.config import _UniqueLoader
from oeis_learn.experiments.models import INTEGER_TEXT_RE, validate_artifact
from oeis_learn.experiments.profiles import I256_MIN, I256_MAX, profile_digests

GROUPING = {
    "version": "equal-prefix-shift/v1",
    "horizon": 100,
    "prefix": 20,
    "window": 80,
    "max_shift": 20,
}


def exact_keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name}: missing or unknown fields")


def write_json(root, path, value):
    target = ArtifactPath(Path(root), path)
    atomic_write(target, canonical_bytes(value))
    return {"path": path, "sha256": compute_file_hash(target.as_path())}


def read_ref(root, ref):
    exact_keys(ref, ("path", "sha256"), "artifact reference")
    path = ArtifactPath(Path(root), ref["path"]).as_path()
    if compute_file_hash(path) != ref["sha256"]:
        raise ValueError("artifact digest mismatch")
    return load_json(path.read_bytes())


def build_cohort(rows, *, seed, dev_count, final_count):
    for name, value in [("seed", seed), ("dev_count", dev_count), ("final_count", final_count)]:
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative exact integer")
    if dev_count + final_count == 0:
        raise ValueError("at least one requested representative required")
    if not isinstance(rows, list):
        raise ValueError("source records must be an array")
    records, exclusions, seen = {}, [], set()
    available = Counter()
    census = dict(
        total_records=len(rows),
        complete20=0,
        in_range_complete20=0,
        complete100=0,
        in_range_complete100=0,
    )
    for row in rows:
        exact_keys(
            row, ("record_id", "first_index", "indices", "values", "metadata"), "source record"
        )
        identity = row["record_id"]
        if not isinstance(identity, str) or ":" not in identity or identity in seen:
            raise ValueError("unique source-qualified record IDs required")
        seen.add(identity)
        reason, indexed = None, {}
        if type(row["first_index"]) is not int or not isinstance(row["metadata"], dict):
            reason = "invalid_index_or_metadata"
        elif (
            not isinstance(row["indices"], list)
            or not isinstance(row["values"], list)
            or len(row["indices"]) != len(row["values"])
        ):
            reason = "index_value_length_mismatch"
        else:
            for index, value in zip(row["indices"], row["values"]):
                if (
                    type(index) is not int
                    or not isinstance(value, str)
                    or not INTEGER_TEXT_RE.fullmatch(value)
                    or len(value) > 4096
                ):
                    reason = "invalid_index_or_integer_text"
                    break
                if index in indexed and indexed[index] != value:
                    reason = "conflicting_index"
                    break
                indexed[index] = value
        if reason is None:
            first = row["first_index"]
            count = 0
            while first + count in indexed:
                count += 1
            available[str(count)] += 1
            for horizon in (20, 100):
                complete = all(i in indexed for i in range(first, first + horizon))
                if complete:
                    census[f"complete{horizon}"] += 1
                    if all(
                        I256_MIN <= int(indexed[i]) <= I256_MAX
                        for i in range(first, first + horizon)
                    ):
                        census[f"in_range_complete{horizon}"] += 1
            if not all(i in indexed for i in range(first, first + 100)):
                reason = "incomplete_or_gapped_first100"
            elif any(
                not I256_MIN <= int(indexed[i]) <= I256_MAX for i in range(first, first + 100)
            ):
                reason = "numeric_range"
            else:
                records[identity] = {
                    **row,
                    "indices": list(range(first, first + 100)),
                    "values": [indexed[i] for i in range(first, first + 100)],
                }
        if reason:
            exclusions.append({"record_id": identity, "reason": reason})
    records = dict(sorted(records.items()))
    parent = {name: name for name in records}
    witnesses = []

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b, kind, shift, overlap):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        parent[max(ra, rb)] = min(ra, rb)
        witnesses.append({"a": a, "b": b, "kind": kind, "shift": shift, "matched_overlap": overlap})

    for horizon, kind in [(100, "equal100"), (20, "equal_prefix")]:
        buckets = {}
        for name, row in records.items():
            values = tuple(row["values"][:horizon])
            if values in buckets:
                union(buckets[values], name, kind, 0, horizon)
            else:
                buckets[values] = name
    # Full-overlap hashes refine the 80-term buckets. Identical prefixes retain
    # one witness representative, so constant/duplicate buckets are not O(N^2).
    for shift in range(1, 21):
        buckets = defaultdict(list)
        for name, row in records.items():
            values = row["values"]
            key = (digest(values[:80]), digest(values[: 100 - shift]))
            bucket = buckets[key]
            if not any(
                records[other]["values"][: 100 - shift] == values[: 100 - shift] for other in bucket
            ):
                bucket.append(name)
        for name, row in records.items():
            values = row["values"]
            key = (digest(values[shift : shift + 80]), digest(values[shift:100]))
            for other in buckets.get(key, ()):
                if (
                    find(name) != find(other)
                    and values[shift:] == records[other]["values"][: 100 - shift]
                ):
                    union(name, other, "shift", shift, 100 - shift)
    components = defaultdict(list)
    for name in records:
        components[find(name)].append(name)
    component_edges = defaultdict(list)
    for edge in witnesses:
        component_edges[find(edge["a"])].append(edge)
    groups = []
    for members in components.values():
        group_id = digest({"members": members, "grouping_profile": digest(GROUPING)})
        groups.append(
            {
                "group_id": group_id,
                "members": members,
                "representative": min(members),
                "witnesses": component_edges[find(members[0])],
            }
        )
    groups.sort(key=lambda g: digest([seed, g["group_id"]]))
    if len(groups) < dev_count + final_count:
        raise ValueError(
            f"cohort shortfall: requested {dev_count + final_count}, available groups {len(groups)}"
        )
    for index, group in enumerate(groups):
        group["partition"] = (
            "development"
            if index < dev_count
            else "final"
            if index < dev_count + final_count
            else "reserved"
        )
    census["available_contiguous_terms_distribution"] = dict(sorted(available.items()))
    census["groups"] = len(groups)
    census["group_size_distribution"] = dict(
        sorted(Counter(str(len(g["members"])) for g in groups).items())
    )
    return {
        "records": records,
        "groups": groups,
        "exclusions": sorted(exclusions, key=lambda x: x["record_id"]),
        "census": census,
    }


def freeze_cohort(source: Path, config: Path, output: Path):
    raw = yaml.load(Path(config).read_text(), Loader=_UniqueLoader)
    exact_keys(
        raw, ("schema_version", "profile", "seed", "dev_count", "final_count"), "cohort config"
    )
    if raw["schema_version"] != "foundation/v1" or raw["profile"] != "prefix20_total100_v1":
        raise ValueError("legacy20+100 or unknown cohort profile")
    source_path = ArtifactPath(Path(source), "records.json").as_path()
    rows = load_json(source_path.read_bytes())
    if not isinstance(rows, list):
        raise ValueError("source records must be an array")
    built = build_cohort(rows, **{key: raw[key] for key in ("seed", "dev_count", "final_count")})
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("cohort output must be empty")
    groups = []
    membership = set()
    for group in built["groups"]:
        name = group["group_id"][7:]
        representative = built["records"][group["representative"]]
        prompt = {
            "kind": "visible_prompt",
            "schema_version": "foundation/v1",
            "request_nonce": str(uuid.uuid4()),
            "language_profile": profile_digests()["language"],
            "observed_terms": representative["values"][:20],
        }
        validate_artifact(prompt)
        groups.append(
            {
                **group,
                "prompt": write_json(output, f"visible/{name}.json", prompt),
                "truth": write_json(output, f"private/{name}.json", representative),
            }
        )
        if group["partition"] != "reserved":
            membership.update(
                digest(built["records"][member]["values"][:20]) for member in group["members"]
            )
    private = write_json(
        output,
        "private/source-records.json",
        {"records": built["records"], "exclusions": built["exclusions"]},
    )
    membership_ref = write_json(output, "admission/prefix-membership.json", sorted(membership))
    manifest = {
        "schema_version": "foundation/v1",
        "kind": "cohort_manifest",
        "profile": "prefix20_total100_v1",
        "source_sha256": compute_file_hash(source_path),
        "config": raw,
        "grouping_profile": GROUPING,
        "groups": groups,
        "census": built["census"],
        "source_records": private,
        "prefix_membership": membership_ref,
    }
    manifest["cohort_id"] = digest(manifest)
    write_json(output, "manifest.json", manifest)
    return manifest


def load_cohort(root: Path):
    root = Path(root)
    manifest = load_json((root / "manifest.json").read_bytes())
    exact_keys(
        manifest,
        (
            "schema_version",
            "kind",
            "profile",
            "source_sha256",
            "config",
            "grouping_profile",
            "groups",
            "census",
            "source_records",
            "prefix_membership",
            "cohort_id",
        ),
        "cohort manifest",
    )
    if (
        manifest["schema_version"] != "foundation/v1"
        or manifest["profile"] != "prefix20_total100_v1"
        or manifest["grouping_profile"] != GROUPING
        or digest(manifest, "cohort_id") != manifest["cohort_id"]
    ):
        raise ValueError("invalid or legacy cohort identity/profile")
    if manifest["groups"] != sorted(
        manifest["groups"], key=lambda g: digest([manifest["config"]["seed"], g["group_id"]])
    ):
        raise ValueError("frozen group ordering changed")
    members, ids, reps = set(), set(), set()
    # Verify evaluator/admission artifacts without decoding private records.
    for name in ("source_records", "prefix_membership"):
        ref = manifest[name]
        exact_keys(ref, ("path", "sha256"), "artifact reference")
        if compute_file_hash(ArtifactPath(root, ref["path"]).as_path()) != ref["sha256"]:
            raise ValueError("cohort source/membership digest mismatch")
    counts = Counter()
    for position, group in enumerate(manifest["groups"]):
        expected_partition = (
            "development"
            if position < manifest["config"]["dev_count"]
            else "final"
            if position < manifest["config"]["dev_count"] + manifest["config"]["final_count"]
            else "reserved"
        )
        if group["partition"] != expected_partition:
            raise ValueError("frozen split assignment changed")
        exact_keys(
            group,
            ("group_id", "members", "representative", "witnesses", "partition", "prompt", "truth"),
            "split group",
        )
        names = group["members"]
        if (
            not names
            or names != sorted(set(names))
            or members.intersection(names)
            or group["representative"] != names[0]
            or group["group_id"] != digest({"members": names, "grouping_profile": digest(GROUPING)})
        ):
            raise ValueError("invalid/disjoint group membership")
        members.update(names)
        ids.add(group["group_id"])
        reps.add(group["representative"])
        if group["partition"] not in ("development", "final", "reserved"):
            raise ValueError("unknown partition")
        counts[group["partition"]] += 1
        prompt = read_ref(root, group["prompt"])
        validate_artifact(prompt)
        # Check existence/integrity before scoring; do not decode hidden values.
        truth = ArtifactPath(root, group["truth"]["path"]).as_path()
        if compute_file_hash(truth) != group["truth"]["sha256"]:
            raise ValueError("missing/corrupt truth")
    if (
        len(ids) != len(manifest["groups"])
        or len(members) != manifest["census"]["in_range_complete100"]
        or len(ids) != manifest["census"]["groups"]
    ):
        raise ValueError("cohort census mismatch")
    for part, key in [("development", "dev_count"), ("final", "final_count")]:
        if counts[part] != manifest["config"][key]:
            raise ValueError("frozen count mismatch")
    return manifest
