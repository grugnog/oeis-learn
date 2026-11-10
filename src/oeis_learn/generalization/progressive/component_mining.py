"""Deterministic connected-subtree mining and reserve/verify/commit publication."""

import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path

from oeis_learn.pipeline_validation.identity import confined
from .contracts import atomic_write, digest, load, reference, seal, verify_refs
from .components import (
    BOUNDS,
    _Expansion,
    _count,
    _render,
    _source,
    make_library,
    validate_library,
    expand,
)
from .generator import structure_id
from .storage import StorageGuard, append_json


def policy(*, maximum_work=131072):
    return seal(
        dict(
            version="progressive-component-mining/v1",
            maximum_sources=4096,
            maximum_source_bytes=64 << 20,
            maximum_candidates=4096,
            maximum_candidate_bytes=32 << 20,
            maximum_components=64,
            pattern_depth=3,
            minimum_operators=2,
            maximum_origins=256,
            maximum_work=maximum_work,
        )
    )


def validate_policy(p):
    default = policy()
    if set(p) != set(default) or seal(p) != p or p["version"] != default["version"]:
        raise ValueError("closed frozen component mining policy required")
    if any(
        type(p[k]) is not int or not 1 <= p[k] <= default[k]
        for k in default
        if k not in {"version", "identity"}
    ):
        raise ValueError("finite positive component work bounds required")
    if p["minimum_operators"] != 2:
        raise ValueError("connected nontrivial fragment law required")
    return p


def _walk(ast, path=()):
    yield ast, list(path)
    if isinstance(ast, (tuple, list)):
        for i, child in enumerate(ast[1:], 1):
            if isinstance(child, (tuple, list)) or child == "$index":
                yield from _walk(child, (*path, i))


def _at(ast, path):
    for i in path:
        if not isinstance(ast, (tuple, list)) or type(i) is not int or not 1 <= i < len(ast):
            raise ValueError("original connected subtree path differs")
        ast = ast[i]
    return ast


def _list(ast):
    return [_list(x) for x in ast] if isinstance(ast, (tuple, list)) else ast


def _pattern(ast, depth, limits, *, charge=None, types=None):
    holes = {}
    bindings = []
    arguments = []
    operators = 0
    types = {} if types is None else types

    def kind_of(node):
        if charge:
            charge(1)
        if node not in types:
            worker = _Expansion({}, BOUNDS, limits)
            original = worker.tick

            def tick(n=1):
                original(n)
                if charge:
                    charge(n)

            worker.tick = tick
            types[node] = worker.expand(node)[1]
        return types[node]

    def visit(node, left):
        nonlocal operators
        kind = kind_of(node)
        leaf = not isinstance(node, (tuple, list)) or node[0] in {"int", "control", "true", "false"}
        if not left or leaf:
            key = digest([kind, _list(node)])
            if key not in holes:
                if len(holes) == 8:
                    raise ValueError("typed hole bound exceeded")
                name = f"h{len(holes)}"
                holes[key] = name
                arguments.append(dict(name=name, type=kind))
                bindings.append(_list(node))
            return ["arg", holes[key]]
        operators += 1
        return [node[0], *[visit(child, left - 1) for child in node[1:]]]

    body = visit(ast, depth)
    kind = kind_of(ast)
    return body, arguments, bindings, operators, kind


class LibraryStore:
    def __init__(
        self,
        root,
        output,
        *,
        purpose,
        track,
        selected=None,
        max_bytes=512 << 20,
        max_inodes=131072,
        store_root=None,
        generation_policies=None,
    ):
        self.root = Path(root)
        self.path = (
            Path(store_root) if store_root is not None else confined(root, output, exists=False)
        )
        if self.path.is_symlink() or any(p.is_symlink() for p in self.path.parents):
            raise ValueError("owned library state alias")
        self.path.mkdir(parents=True, exist_ok=True)
        if purpose not in {"acceptance", "research"} or track not in {
            "synthetic_strict",
            "oeis_derived",
            "external_assisted",
        }:
            raise ValueError("explicit library provenance required")
        self.purpose, self.track = purpose, track
        self.policy = copy.deepcopy(validate_policy(selected or policy()))
        from .generator import validate_policy as validate_generation

        self.generation_policies = copy.deepcopy(generation_policies or {})
        if not isinstance(self.generation_policies, dict) or len(self.generation_policies) > 64:
            raise ValueError("bounded original mechanical source laws required")
        for identity, law in self.generation_policies.items():
            if identity != digest(validate_generation(law)):
                raise ValueError("original mechanical source policy identity differs")

        self.storage = StorageGuard([self.path], max_bytes=max_bytes, max_inodes=max_inodes)
        self.work = self.path / "spent-work.jsonl"
        self._put(
            "binding.json",
            seal(
                dict(
                    implementation=digest(
                        [
                            Path(__file__).read_text(),
                            Path(__file__).with_name("components.py").read_text(),
                            Path(__file__).with_name("generator.py").read_text(),
                        ]
                    ),
                    policy=self.policy,
                    generation_policies=self.generation_policies,
                    purpose=purpose,
                    track=track,
                    max_bytes=max_bytes,
                    max_inodes=max_inodes,
                )
            ),
        )

    def _put(self, name, value):
        path = self.path / name
        if path.exists():
            if load(path, maximum=256 << 20) != value:
                raise ValueError("immutable library publication changed")
        else:
            self.storage.check(
                len((json.dumps(value, sort_keys=True, indent=2) + "\n").encode()), 2
            )
            atomic_write(path, value)
        return reference(self.path, path)

    @contextmanager
    def lease(self):
        import fcntl

        with (self.path / "publication.lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("another owned library publication is active") from None
            yield

    def spent(self):
        if not self.work.exists():
            return 0
        if self.work.is_symlink() or self.work.stat().st_size > 64 << 20:
            raise ValueError("bounded original component work journal required")
        total = 0
        for i, line in enumerate(self.work.read_text().splitlines()):
            item = json.loads(line)
            if (
                set(item) != {"identity", "ordinal", "phase", "key", "units"}
                or seal(item) != item
                or item["ordinal"] != i
                or type(item["units"]) is not int
                or item["units"] < 1
            ):
                raise ValueError("original irreversible component work required")
            total += item["units"]
        return total

    def debit(self, phase, key, units=1):
        if time.monotonic() >= self.deadline or self.spent() + units > self.policy["maximum_work"]:
            raise TimeoutError("component verification/mining work reservation exhausted")
        ordinal = len(self.work.read_text().splitlines()) if self.work.exists() else 0
        value = seal(dict(ordinal=ordinal, phase=phase, key=key, units=units))
        self.storage.check(len(json.dumps(value).encode()) + 64, 1)
        append_json(self.work, value)
        if self.budget:
            self.budget.consume(attempts=1, rejections=1)

    def committed(self):
        values = []
        for path in sorted(self.path.glob("commit-*.json")):
            receipt = load(path)
            if (
                seal(receipt) != receipt
                or receipt["policy"] != self.policy["identity"]
                or receipt["status"] != "COMMITTED"
            ):
                raise ValueError("original library commit authority required")
            value = json.loads(verify_refs(self.path, [receipt["library"]], max_bytes=256 << 20)[0])
            validate_library(value)
            verify_refs(self.path, value["payload"]["verification"]["proofs"], max_bytes=256 << 20)
            origins = {
                o["artifact"]["path"]: o["artifact"]
                for c in value["payload"]["components"]
                for o in c["origins"]
            }
            verify_refs(self.root, list(origins.values()), max_bytes=256 << 20)
            values.append(value)
        return values

    def _origin(self, ref, client):
        from oeis_learn.generalization.progressive.admission import validate_member
        from oeis_learn.generalization.progressive.archive import Archive
        from oeis_learn.synthesis.reference import evaluate
        from oeis_learn.synthesis.reference_parser import parse

        self.debit("source-verification", ref["sha256"])
        value = json.loads(verify_refs(self.root, [ref], max_bytes=4 << 20)[0])
        if value.get("version") == "discovery-admission/v1":
            value = Archive.validate_entry(value)
            p = value["payload"]
            if value["track"] != self.track or value["purpose"] != self.purpose:
                raise ValueError("training origin track/purpose differs")
            source, indices, expected, group = (
                p["source"],
                p["indices"],
                p["values_hash"],
                p["group"],
            )
        elif value.get("version") == "progressive-training-row/v1":
            required = {
                "version",
                "identity",
                "ordinal",
                "draw_identity",
                "source_policy",
                "source",
                "structure",
                "behavior",
                "indices",
                "values",
                "target_source",
                "tokens",
                "target_disposition",
                "membership",
                "native_verified",
                "control_events",
                "branch_events",
                "operations",
                "detector_limit",
            }
            if set(value) != required or seal(value) != value or value["membership"] != "training":
                raise ValueError(
                    "closed verified training origin required; evaluation witnesses excluded"
                )
            if self.purpose == "research" and value["native_verified"] is not True:
                raise ValueError("native verified training source required")
            p = value
            law = self.generation_policies.get(p["source_policy"])
            if law is None and (self.purpose == "research" or self.generation_policies):
                raise ValueError("original mechanical generation provenance required")
            if law is not None:
                from .generator import draw

                original = draw(law, p["ordinal"])
                if (
                    p["structure"] in law["forbidden_structures"]
                    or p["behavior"] in law["forbidden_behaviors"]
                    or digest(p["values"][:20]) in law["forbidden_prefixes"]
                ):
                    raise ValueError(
                        "held-out/exposed mechanical source is not a permitted training origin"
                    )
                if any(
                    original[k] != p[k]
                    for k in ("source", "structure", "source_policy", "draw_identity", "ordinal")
                ):
                    raise ValueError("mechanical draw provenance differs from frozen law")
            if p["behavior"] != digest(p["values"]):
                raise ValueError("original finite behavior identity differs")
            source, indices, expected, group = (
                p["source"],
                p["indices"],
                digest(p["values"]),
                p["behavior"],
            )
            validate_member(
                dict(
                    id=p["draw_identity"],
                    group=group,
                    role="training",
                    source_snapshot=p["source_policy"],
                    indices=indices,
                    values=p["values"],
                )
            )
        else:
            raise ValueError(
                "only committed admissions or verified training rows are component origins"
            )
        if p.get("membership") != "training" or p["structure"] != structure_id(source):
            raise ValueError("original training membership/structure differs")
        deadline = min(self.deadline, time.monotonic() + 4)
        result = evaluate(source, indices, self.limits, deadline=deadline)
        if result.outcome != "ok" or digest(list(result.values)) != expected:
            raise ValueError("original independent training source verification differs")
        if client is not None:
            native = client.execute(source, indices, deadline=deadline)
            if (
                native.get("outcome") != "ok"
                or native.get("values") != list(result.values)
                or time.monotonic() >= deadline
            ):
                raise ValueError("native/reference source disagreement or late result")
        program = parse(source, self.limits)
        return dict(
            ref=ref,
            target=value.get("draw_identity", value["identity"]),
            source=source,
            ast=program.result,
            indices=indices,
            structure=p["structure"],
            group=group,
        )

    def _mine(self, verified):
        candidates = {}
        candidate_bytes = 0
        for item in verified:
            credits = 0
            types = {}

            def charge(units):
                nonlocal credits
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("component mining deadline exhausted")
                while credits < units:
                    reserve = min(512, self.policy["maximum_work"] - self.spent())
                    if reserve <= 0:
                        raise TimeoutError("component mining work reservation exhausted")
                    self.debit("connected-subtree-mining", item["ref"]["sha256"], reserve)
                    credits += reserve
                credits -= units

            for node, path in _walk(item["ast"]):
                for depth in range(1, self.policy["pattern_depth"] + 1):
                    try:
                        body, args, bindings, operators, kind = _pattern(
                            node, depth, self.limits, charge=charge, types=types
                        )
                    except ValueError:
                        continue  # captures/unsupported typed fragments are ineligible
                    if operators < self.policy["minimum_operators"]:
                        continue
                    key = digest(dict(body=body, arguments=args, returns=kind))
                    proposed = dict(
                        id="c_" + key[7:31],
                        arguments=args,
                        returns=kind,
                        body=body,
                        dependencies=[],
                        origins=[],
                        assumptions=[
                            "pure bigint-body/v1 expressions",
                            "checked numerical/resource profile",
                            "finite verification is not a discovery proof",
                        ],
                    )
                    if key not in candidates:
                        added = len(json.dumps(proposed).encode())
                        if (
                            len(candidates) >= self.policy["maximum_candidates"]
                            or candidate_bytes + added > self.policy["maximum_candidate_bytes"]
                        ):
                            raise TimeoutError(
                                "component candidate inventory reservation exhausted"
                            )
                        candidates[key] = proposed
                        candidate_bytes += added
                    c = candidates[key]
                    origin = dict(
                        artifact=item["ref"],
                        target=item["target"],
                        indices=item["indices"],
                        role="training",
                        group=item["group"],
                        structure=item["structure"],
                        source=digest(item["source"]),
                        path=path,
                        bindings=bindings,
                    )
                    added = len(json.dumps(origin).encode())
                    if candidate_bytes + added > self.policy["maximum_candidate_bytes"]:
                        raise TimeoutError("component candidate byte reservation exhausted")
                    candidate_bytes += added
                    c["origins"].append(origin)
        selected = []
        for key, c in sorted(candidates.items()):
            if (
                len({o["structure"] for o in c["origins"]}) < 3
                or len({o["group"] for o in c["origins"]}) < 2
            ):
                continue
            origins = sorted(
                c["origins"],
                key=lambda o: (o["structure"], o["group"], o["artifact"]["sha256"], o["path"]),
            )
            keep = []
            seen_structures = set()
            seen_groups = set()
            for o in origins:
                if o["structure"] not in seen_structures or o["group"] not in seen_groups:
                    keep.append(o)
                    seen_structures.add(o["structure"])
                    seen_groups.add(o["group"])
            keep = (keep + [o for o in origins if o not in keep])[: self.policy["maximum_origins"]]
            if len({o["structure"] for o in keep}) < 3 or len({o["group"] for o in keep}) < 2:
                continue
            c["origins"] = keep
            selected.append(c)
            if len(selected) == self.policy["maximum_components"]:
                break
        return selected

    def publish(self, sources, *, client=None, previous=None, seconds=60, fault=None, budget=None):
        from oeis_learn.synthesis.contracts import ResourceProfile

        if (
            type(seconds) not in (int, float)
            or not 0 < seconds <= 21600
            or fault not in {None, "after-reserve", "after-verify"}
            or fault
            and self.purpose != "acceptance"
        ):
            raise ValueError("bounded component transaction required")
        if self.purpose == "research" and client is None:
            raise ValueError("actual native component verification required")
        if (
            not isinstance(sources, list)
            or not 1 <= len(sources) <= self.policy["maximum_sources"]
            or len({r["path"] for r in sources}) != len(sources)
        ):
            raise ValueError("finite distinct original source inventory required")
        self.deadline = (
            min(time.monotonic() + seconds, budget.deadline)
            if budget
            else time.monotonic() + seconds
        )
        self.budget = budget
        self.limits = ResourceProfile()
        if client is not None and getattr(client, "limits", self.limits) != self.limits:
            raise ValueError("matching native/reference component resource profile required")
        ordered = sorted(sources, key=lambda r: (r["sha256"], r["path"]))
        request = seal(
            dict(
                sources=ordered,
                previous=previous["identity"] if previous else None,
                policy=self.policy["identity"],
                purpose=self.purpose,
                track=self.track,
                native=client.runtime_identity("bigint-body/v1") if client else None,
            )
        )
        key = request["identity"][7:]
        with self.lease():
            done = self.path / ("commit-" + key + ".json")
            if done.exists():
                receipt = load(done)
                if receipt["request"] != request["identity"]:
                    raise ValueError("changed library request")
                return next(
                    v
                    for v in self.committed()
                    if v["identity"]
                    == json.loads(verify_refs(self.path, [receipt["library"]])[0])["identity"]
                )
            if previous is not None and previous not in self.committed():
                raise ValueError("previous library is not an owned committed cut")
            if previous is not None:
                supplied = {r["sha256"] for r in ordered}
                if any(
                    o["artifact"]["sha256"] not in supplied
                    for c in previous["payload"]["components"]
                    for o in c["origins"]
                ):
                    raise ValueError(
                        "previous library origins must remain explicit permitted inputs"
                    )
            self._put("reserve-" + key + ".json", request)
            if fault == "after-reserve":
                raise InterruptedError("diagnostic component interruption after reserve")
            verify_refs(self.root, ordered, max_bytes=self.policy["maximum_source_bytes"])
            verified = [self._origin(ref, client) for ref in ordered]
            definitions = self._mine(verified)
            if previous:
                old = {
                    c["id"]: {k: v for k, v in c.items() if k != "expanded_ast"}
                    for c in previous["payload"]["components"]
                }
                for c in definitions:
                    if c["id"] in old and any(
                        c[k] != old[c["id"]][k]
                        for k in ("body", "arguments", "returns", "dependencies")
                    ):
                        raise ValueError("immutable prior component identity changed")
                    old.setdefault(c["id"], c)
                prior_ids = [c["id"] for c in previous["payload"]["components"]]
                order = prior_ids + [k for k in sorted(old) if k not in prior_ids]
                definitions = [old[k] for k in order][: self.policy["maximum_components"]]
            provisional = make_library(definitions, purpose="acceptance", track=self.track)
            lookup = {v["ref"]["sha256"]: v for v in verified}
            proofs = []
            from oeis_learn.synthesis.reference import evaluate
            from oeis_learn.synthesis.reference_parser import parse
            from oeis_learn.synthesis.admission import reference_identity
            import oeis_learn.synthesis.reference_parser as parser

            for c in provisional["payload"]["components"]:
                for o in c["origins"]:
                    self.debit("full-expansion-verification", digest([c["id"], o]))
                    item = lookup.get(o["artifact"]["sha256"])
                    if item is None:
                        item = self._origin(o["artifact"], client)
                    node = _at(item["ast"], o["path"])
                    value = expand(
                        provisional, ["call", c["id"], *o["bindings"]], limits=self.limits
                    )
                    if value["expanded_ast"] != _list(node):
                        raise ValueError(
                            "expanded subtree changed operand order or captured a variable"
                        )
                    _, kind = _Expansion({}, BOUNDS, self.limits).expand(node)
                    original = _source(_list(node), kind)
                    parse(value["source"], self.limits)
                    deadline = min(self.deadline, time.monotonic() + 4)
                    a = evaluate(original, item["indices"], self.limits, deadline=deadline)
                    b = evaluate(value["source"], item["indices"], self.limits, deadline=deadline)
                    if a.outcome != b.outcome or a.outcome != "ok" or a.values != b.values:
                        raise ValueError("independent full-expansion reference disagreement")
                    if client is not None:
                        native = client.execute(value["source"], item["indices"], deadline=deadline)
                        if (
                            native.get("outcome") != "ok"
                            or native.get("values") != list(b.values)
                            or time.monotonic() >= deadline
                        ):
                            raise ValueError(
                                "native/reference expanded candidate disagreement or late result"
                            )
                    proof = seal(
                        dict(
                            version="progressive-component-verification/v1",
                            component=c["id"],
                            origin=o,
                            expanded_source=digest(value["source"]),
                            indices=item["indices"],
                            values=digest(list(b.values)),
                            native=request["native"],
                            reference=reference_identity(),
                            scope="finite_checked_expansion",
                            implementation=load(self.path / "binding.json")["implementation"],
                            reference_steps=a.steps + b.steps,
                            expanded_nodes=value["expanded_nodes"],
                            expanded_bytes=value["expanded_bytes"],
                        )
                    )
                    proofs.append(self._put("proof-" + proof["identity"][7:] + ".json", proof))
            verification = dict(
                status="NATIVE_VERIFIED" if client else "DIAGNOSTIC_ONLY",
                native=request["native"],
                reference=reference_identity(),
                parser=digest(Path(parser.__file__).read_text()),
                proofs=proofs,
            )
            self.debit("library-publication", request["identity"])
            library = make_library(
                definitions,
                purpose=self.purpose,
                track=self.track,
                previous=previous["identity"] if previous else None,
                verification=verification,
            )
            if fault == "after-verify":
                raise InterruptedError(
                    "diagnostic component interruption after verify; no library cut committed"
                )
            ref = self._put("library-" + library["identity"][7:] + ".json", library)
            if time.monotonic() >= self.deadline:
                raise TimeoutError("library publication deadline exhausted; no committed cut")
            self._put(
                "commit-" + key + ".json",
                seal(
                    dict(
                        version="progressive-library-commit/v1",
                        request=request["identity"],
                        policy=self.policy["identity"],
                        library=ref,
                        status="COMMITTED",
                    )
                ),
            )
            return library
