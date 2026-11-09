"""Append-only refutation overlays over a closed, typed dependency DAG.

Edges point from a consumer to its prerequisites. Later consumers inherit earlier
refutations. Original artifacts and the cost journal are never rewritten.
"""

from contextlib import contextmanager
import copy
import fcntl
from pathlib import Path
from .contracts import atomic_write, digest, load, seal, _check, schema

KINDS = {"target", "archive", "library", "consumed_update", "checkpoint", "dossier"}
MAX_NODES = 65536


def _graph(graph):
    if type(graph) is not list or len(graph) > MAX_NODES:
        raise ValueError("bounded lineage required")
    nodes = {}
    for node in graph:
        if (
            type(node) is not dict
            or set(node) != {"id", "kind", "parents"}
            or type(node["id"]) is not str
            or not 1 <= len(node["id"]) <= 256
            or node["kind"] not in KINDS
            or type(node["parents"]) is not list
            or len(node["parents"]) > MAX_NODES
            or any(type(p) is not str for p in node["parents"])
            or len(set(node["parents"])) != len(node["parents"])
            or node["id"] in nodes
        ):
            raise ValueError("invalid typed lineage")
        nodes[node["id"]] = node
    if sum(len(n["parents"]) for n in graph) > 262144:
        raise ValueError("lineage edge budget exhausted")
    children = {i: [] for i in nodes}
    degree = {}
    for i, n in nodes.items():
        degree[i] = len(n["parents"])
        for p in n["parents"]:
            if p not in nodes:
                raise ValueError("missing lineage parent")
            children[p].append(i)
    ready = [i for i in nodes if not degree[i]]
    visited = 0
    while ready:
        visited += 1
        for c in children[ready.pop()]:
            degree[c] -= 1
            if not degree[c]:
                ready.append(c)
    if visited != len(nodes):
        raise ValueError("cyclic lineage")
    return nodes, children


def refute(graph, *, roots):
    nodes, children = _graph(graph)
    if (
        type(roots) is not list
        or not roots
        or len(roots) > 256
        or any(type(r) is not str or r not in nodes for r in roots)
        or len(set(roots)) != len(roots)
    ):
        raise ValueError("known unique refutation roots required")
    affected, todo = set(), list(roots)
    while todo:
        i = todo.pop()
        if i not in affected:
            affected.add(i)
            todo.extend(children[i])
    return seal(
        dict(
            version="progressive-quarantine/v1",
            graph=digest(graph),
            roots=sorted(roots),
            affected=sorted(affected),
        )
    )


class Journal:
    def __init__(self, root):
        self.root = Path(root) / "quarantine"
        if self.root.is_symlink():
            raise ValueError("symlink quarantine")

    @contextmanager
    def lock(self):
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "lock"
        if path.is_symlink():
            raise ValueError("symlink quarantine lock")
        with path.open("a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield

    def read(self):
        graph, roots, tip = [], set(), digest("progressive-quarantine/v1")
        paths = sorted(self.root.glob("event-*.json"))
        if len(paths) > MAX_NODES:
            raise ValueError("quarantine journal exhausted")
        for i, path in enumerate(paths):
            e = load(path, maximum=32 << 20)
            _check(schema("quarantine"), e, schema("quarantine"))
            if path.name != f"event-{i:05d}.json" or seal(e) != e or e["previous"] != tip:
                raise ValueError("invalid quarantine chain")
            if e["operation"] == "register":
                graph.extend(e["nodes"])
                _graph(graph)
            else:
                checked = refute(graph, roots=e["roots"])
                if checked != e["overlay"]:
                    raise ValueError("refutation closure differs")
                roots.update(e["roots"])
            tip = e["identity"]
        affected = set(refute(graph, roots=sorted(roots))["affected"]) if roots else set()
        return graph, affected, tip, len(paths)

    def append(self, operation, *, nodes=None, roots=None, reason=""):
        with self.lock():
            graph, _, tip, ordinal = self.read()
            if ordinal >= MAX_NODES:
                raise ValueError("quarantine journal exhausted")
            if operation == "register":
                _graph(graph + nodes)
                overlay = None
            else:
                if type(reason) is not str or not 1 <= len(reason) <= 4096:
                    raise ValueError("counterexample reason required")
                overlay = refute(graph, roots=roots)
            event = seal(
                dict(
                    version="progressive-quarantine-event/v1",
                    previous=tip,
                    operation=operation,
                    nodes=nodes or [],
                    roots=roots or [],
                    reason=reason,
                    overlay=overlay,
                )
            )
            atomic_write(self.root / f"event-{ordinal:05d}.json", event)
            return overlay

    def register(self, nodes):
        return self.append("register", nodes=copy.deepcopy(nodes))

    def refute(self, roots, *, reason):
        return self.append("refute", roots=roots, reason=reason)

    def require_clean(self, identities):
        _, affected, _, _ = self.read()
        if affected.intersection(identities):
            raise ValueError("quarantined lineage cannot resume/adopt")

    def recovery(self, ancestors, *, spent):
        _, affected, _, _ = self.read()
        graph, _, _, _ = self.read()
        known = {n["id"] for n in graph if n["kind"] == "checkpoint"}
        clean = next((a for a in ancestors if a in known and a not in affected), None)
        return dict(
            checkpoint=clean,
            status="CLEAN_ANCESTOR" if clean else "CONTAMINATED",
            spent=copy.deepcopy(spent),
        )

    def observe(self, nodes):
        """Idempotently bind real artifact dependencies; changed history is invalid."""
        with self.lock():
            graph, _, _, _ = self.read()
            known = {n["id"]: n for n in graph}
            pending = []
            for n in nodes:
                if n["id"] in known:
                    if known[n["id"]] != n:
                        raise ValueError("changed quarantine lineage")
                else:
                    pending.append(n)
                    known[n["id"]] = n
            if not pending:
                return
        # register rechecks closure while holding the write lock.
        self.register(pending)

    def checkpoint(self, record, *, previous=None):
        cuts = record["payload"]["cuts"]
        graph, _, _, _ = self.read()
        known = {n["id"] for n in graph}
        nodes = []
        parents = []
        for key, kind in [
            ("archive", "archive"),
            ("library", "library"),
        ]:
            identity = cuts.get(key)
            if identity is not None and identity != digest(None):
                if identity not in known:
                    nodes.append(dict(id=identity, kind=kind, parents=[]))
                    known.add(identity)
                parents.append(identity)
        if cuts.get("consumed") is not None and cuts["consumed"] != digest(None):
            update = digest(["consumed_update", cuts["consumed"], sorted(parents)])
            nodes.append(dict(id=update, kind="consumed_update", parents=sorted(set(parents))))
            parents = [update]
        if previous is not None:
            parents.append(previous["identity"])
        self.require_clean(parents)
        nodes.append(dict(id=record["identity"], kind="checkpoint", parents=sorted(set(parents))))
        self.observe(nodes)

    def training(self, package):
        nodes = []

        def add(identity, kind, parents=()):
            nodes.append(dict(id=identity, kind=kind, parents=sorted(set(parents))))

        replay = package.get("oeis_replay")
        if replay and replay["archive"]:
            for a in replay["admissions"]:
                add(a["identity"], "target")
            add(
                replay["archive"]["identity"],
                "archive",
                [a["identity"] for a in replay["admissions"]],
            )
        component = package.get("component_distillation")
        if component and component["library"]:
            lib = component["library"]["library"]
            origins = []
            for c in lib["payload"]["components"]:
                for o in c["origins"]:
                    identity = o["artifact"]["sha256"]
                    if identity not in origins:
                        origins.append(identity)
                        add(identity, "target")
            add(lib["identity"], "library", origins)
            for t in component["targets"]:
                add(t["identity"], "target", [lib["identity"]])
            from .component_distillation import ComponentMixer

            add(
                ComponentMixer(component).archive,
                "archive",
                [t["identity"] for t in component["targets"]],
            )
        self.observe(nodes)
        self.require_clean([n["id"] for n in nodes])

    def dossier(self, value, *, checkpoints):
        """Bind review artifacts to their declared checkpoint/library producers."""
        parents = list(checkpoints)
        library = value["payload"].get("library")
        if library is not None:
            parents.append(library["sha256"])
        self.observe([dict(id=value["identity"], kind="dossier", parents=sorted(set(parents)))])
        return dict(dossier=value["identity"], invalidated=value["identity"] in self.read()[1])
