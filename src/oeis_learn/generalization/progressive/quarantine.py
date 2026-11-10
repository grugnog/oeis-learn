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
        # Event root limits bound each publication, not cumulative history.
        # Traverse the validated graph once for the union, including late consumers.
        _, children = _graph(graph)
        affected, todo = set(), list(roots)
        while todo:
            node = todo.pop()
            if node not in affected:
                affected.add(node)
                todo.extend(children[node])
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

    def checkpoint(self, record, *, previous=None, entries=(), weights=None):
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
        known = {n['id'] for n in graph}
        consumed = sorted(set(entries) & known)
        if consumed:
            update = digest(['exact-consumed-update',record['identity'],consumed])
            nodes.append(dict(id=update,kind='consumed_update',parents=consumed))
            parents.append(update)
        if previous is not None:
            parents.append(previous["identity"])
        self.require_clean(parents)
        nodes.append(dict(id=record["identity"], kind="checkpoint", parents=sorted(set(parents))))
        self.observe(nodes)
        if weights is not None:
            path=self.root/'weights'/weights[7:]/(record['identity'][7:]+'.json')
            value=seal(dict(checkpoint=record['identity'],weights=weights))
            if path.exists():
                if load(path)!=value:raise ValueError('changed checkpoint weight binding')
            else:
                from .storage import StorageGuard
                StorageGuard([self.root],max_bytes=64<<20,max_inodes=MAX_NODES*4).check(512,3)
                atomic_write(path,value)

    def weight_checkpoints(self, weights):
        result=[]
        for path in (self.root/'weights'/weights[7:]).glob('*.json'):
            value=load(path)
            if (set(value)!={'identity','checkpoint','weights'} or seal(value)!=value
                    or value['weights']!=weights or path.stem!=value['checkpoint'][7:]):
                raise ValueError('checkpoint weight binding differs')
            result.append(value['checkpoint'])
        return sorted(result)

    def training(self, package):
        nodes = []

        def add(identity, kind, parents=()):
            nodes.append(dict(id=identity, kind=kind, parents=sorted(set(parents))))

        known=None
        for shard in package.get('shards',[]):
            for row in shard:
                add(row['draw_identity'],'target')
                if 'source' in row and 'indices' in row:
                    self.bind_source(row['draw_identity'],row['source'],row['indices'])
                else:
                    if known is None:known={n['id'] for n in self.read()[0] if n['kind']=='target'}
                    if row['draw_identity'] not in known:
                        raise ValueError('consumed target lacks its original registered source binding')
        replay = package.get("oeis_replay")
        if replay and replay["archive"]:
            for a in replay["admissions"]:
                add(a["identity"], "target")
                self.bind_source(a['identity'],a['payload']['source'],a['payload']['indices'])
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
                self.bind_source(t['identity'],t['row']['source'],t['row']['indices'])
            from .component_distillation import ComponentMixer

            add(
                ComponentMixer(component).archive,
                "archive",
                [t["identity"] for t in component["targets"]],
            )
        self.observe(nodes)
        self.require_clean([n["id"] for n in nodes])

    def dossier(self, value, *, checkpoints, library_identity=None):
        """Bind review artifacts to their declared checkpoint/library producers."""
        parents = list(checkpoints)
        library = value["payload"].get("library")
        if library is not None:
            parents.append(library_identity or library['sha256'])
        self.observe([dict(id=value["identity"], kind="dossier", parents=sorted(set(parents)))])
        return dict(dossier=value["identity"], invalidated=value["identity"] in self.read()[1])

    def bind_source(self, identity, source, indices):
        value=seal(dict(identity_key=identity,source=source,indices=indices))
        path=self.root/'sources'/(digest(identity)[7:]+'.json')
        if path.exists():
            if load(path)!=value:raise ValueError('changed target source binding')
        else:
            from .storage import StorageGuard
            import json
            StorageGuard([self.root],max_bytes=64<<20,max_inodes=MAX_NODES*4).check(len(json.dumps(value).encode())+64,3)
            atomic_write(path,value)

    def refute_dossier(self, value, *, evidence):
        counterexamples=value['payload']['counterexamples']
        if not counterexamples:return None
        indices={c['index'] for c in counterexamples}
        roots=[value['identity']]
        for path in sorted((self.root/'sources').glob('*.json')):
            entry=load(path)
            if seal(entry)!=entry:raise ValueError('target source binding differs')
            if entry['source']==value['payload']['source'] and indices.intersection(entry['indices']):
                roots.append(entry['identity_key'])
        affected=set()
        import json
        for i in range(0,len(roots),256):
            affected.update(self.refute(roots[i:i+256],reason=json.dumps(dict(evidence=evidence,dossier=value['identity'])))['affected'])
        return dict(roots=sorted(roots),affected=sorted(affected))
