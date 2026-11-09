"""Bounded public-table export. No recursive workspace collection is permitted."""

import hashlib
import json
from pathlib import Path
import zipfile
from .contracts import digest, seal, verify_refs, _check, schema

KINDS = ("roster", "outcomes", "accounting", "ancestry", "qualifications", "observations", "comparisons", "authorities")
MAX_BYTES = 512 << 20
MAX_PART = 4 << 20
MAX_ROWS = 262144


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value):
    import re

    if type(value) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
        raise ValueError("public content-bound identity required")


def validate_rows(kind, rows):
    if type(rows) is not list or len(rows) > MAX_ROWS:
        raise ValueError("bounded public table required")
    fields = {
        "roster": {"task", "group", "stratum"},
        "outcomes": {"task", "group", "stratum", "seed", "arm", "outcome"},
        "accounting": {
            "task",
            "arm",
            "tokens",
            "useful_tokens",
            "discarded_tokens",
            "elapsed_us",
            "token_cap",
            "elapsed_cap_us",
        },
        "ancestry": {"child", "parent", "relation"},
        "qualifications": {
            "identity",
            "source",
            "scope",
            "device",
            "status",
            "requested",
            "executed",
            "receipt",
        },
    }
    if kind in {'observations','comparisons','authorities'}:
        for row in rows:
            expected={'observations':{'producer','data'},'comparisons':{'id','rows','roster','seeds','design','verdict','action'},'authorities':{'artifact','raw'}}[kind]
            if not isinstance(row,dict) or set(row)!=expected:raise ValueError('closed public auxiliary row required')
            if len(encoded(row))>MAX_PART:raise ValueError('public auxiliary row byte cap')
            if kind=='observations':
                _hash(row['producer'])
                if not isinstance(row['data'],dict):raise ValueError('closed numeric observation required')
        return rows
    for r in rows:
        if type(r) is not dict or set(r)-({'resources'} if kind=='accounting' else set()) != fields[kind]:
            raise ValueError("closed public table row required")
        for k in ("task", "group", "child", "parent", "identity", "source"):
            if k in r:
                _hash(r[k])
        for k in (
            "seed",
            "tokens",
            "useful_tokens",
            "discarded_tokens",
            "elapsed_us",
            "token_cap",
            "elapsed_cap_us",
        ):
            if k in r and (type(r[k]) is not int or not 0 <= r[k] <= 2**63 - 1):
                raise ValueError("exact bounded public count required")
        if kind in {"roster", "outcomes"} and r["stratum"] not in {
            "parameter",
            "unseen_structure",
            "composition",
            "oeis",
            "retention",
        }:
            raise ValueError("unknown stratum")
        if kind == "outcomes" and (
            r["arm"] not in {"baseline", "candidate"}
            or r["outcome"] not in {"success", "failure", "unresolved"}
        ):
            raise ValueError("unknown outcome")
        if kind=='accounting' and 'resources' in r:
            from .budgets import vector
            if set(r['resources'])!={'spent','caps'}:raise ValueError('closed resource accounting required')
            vector(r['resources']['spent']);vector(r['resources']['caps'])
        if kind == "accounting" and r["arm"] not in {"baseline", "candidate"}:
            raise ValueError("closed accounting arm required")
        if kind == "ancestry" and r["relation"] not in {
            "continuation",
            "target",
            "archive",
            "library",
            "consumed_update",
            "dossier",
        }:
            raise ValueError("unknown ancestry relation")
        if kind == "qualifications":
            if (
                r["status"] not in {"PASS", "FAIL"}
                or r["device"] not in {"actual_HIP", "cpu"}
                or type(r["scope"]) is not str
                or not 1 <= len(r["scope"]) <= 256
            ):
                raise ValueError("closed qualification scope required")
            for k in ("requested", "executed"):
                if (
                    type(r[k]) is not list
                    or len(r[k]) > 4096
                    or any(type(x) is not str or len(x) > 256 for x in r[k])
                ):
                    raise ValueError("bounded producer case inventory")
            _check(schema("common")["$defs"]["artifact"], r["receipt"], schema("common"))
    return rows


def _error(error):
    if error is None:
        return
    if type(error) is not dict or set(error) != {"type", "code", "producer", "local_traceback"}:
        raise ValueError("public error projection required")
    for k in ("type", "code", "producer"):
        if type(error[k]) is not str or len(error[k]) > 256:
            raise ValueError("bounded public error required")
    ref = error["local_traceback"]
    if ref is not None:
        _check(schema("common")["$defs"]["artifact"], ref, schema("common"))


def validate_manifest(m):
    doc = schema("report")
    _check(doc, m, doc)
    if seal(m) != m:
        raise ValueError("report seal differs")
    _error(m["producer_error"])
    for e in m["collector_errors"]:
        _error(e)
    return m


def build(root, destination, request, *, max_bytes=MAX_BYTES):
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("report metadata cap")
    if (
        set(request)
        != {
            "version",
            "identity",
            "campaign",
            "source",
            "seeds",
            "design",
            "scope",
            "tables",
            "decisions",
            "producer_error",
            "collector_errors",
            "campaign_record",
            "scope_record",
            "terminal",
        }
        or request["version"] != "progressive-public-report-input/v1"
        or seal(request) != request
    ):
        raise ValueError("closed public report request required")
    if set(request["tables"]) - set(KINDS):
        raise ValueError("private/unknown report table")
    if sum(r["bytes"] for refs in request["tables"].values() for r in refs) > max_bytes:
        raise ValueError("REPORT_INCOMPLETE: input metadata byte budget")
    docs = {}
    if request["scope_record"] is not None:
        docs["authority/scope.json"] = verify_refs(
            root, [request["scope_record"]], max_bytes=MAX_PART
        )[0]
    totals = {}
    row_count = 0
    for kind in KINDS:
        rows = []
        refs = request["tables"].get(kind, [])
        if len(refs) > 4096:
            raise ValueError("report reference budget")
        for blob in verify_refs(root, refs, max_bytes=max_bytes):
            rows.extend(validate_rows(kind, json.loads(blob)))
            if len(rows) > MAX_ROWS:
                raise ValueError("public table row budget")
        if kind == "qualifications":
            for row in rows:
                blob = verify_refs(root, [row["receipt"]], max_bytes=MAX_PART)[0]
                validate_producer(json.loads(blob), row)
                docs["producers/" + row["identity"].split(":")[1] + ".json"] = blob
        totals[kind] = len(rows)
        row_count += len(rows)
        # Canonical deterministic partitions by original frozen row order.
        partitions = []
        part = []
        size = 2
        for row in rows:
            n = len(encoded(row)) + 1
            if n > MAX_PART:
                raise ValueError("REPORT_INCOMPLETE: required row cannot fit")
            if size + n > MAX_PART:
                partitions.append(part)
                part = []
                size = 2
            part.append(row)
            size += n
        partitions.append(part)
        for i, part in enumerate(partitions):
            docs[f"tables/{kind}-{i:04d}.json"] = encoded(part)
    if row_count > MAX_ROWS:
        raise ValueError("report total row budget")
    trace_missing=[]
    for e in [request['producer_error'],*request['collector_errors']]:
        if not e or e['local_traceback'] is None:continue
        ref=e['local_traceback']
        try:
            raw=verify_refs(root,[ref],max_bytes=64<<10)[0]
            docs['errors/'+ref['sha256'][7:]+'.txt']=raw
        except (ValueError,OSError):trace_missing.append('traceback:'+ref['sha256'])
    inventory = [
        dict(path=n, bytes=len(b), sha256="sha256:" + hashlib.sha256(b).hexdigest())
        for n, b in sorted(docs.items())
    ]
    missing = sorted((set(request['scope']['required'])-set(request['tables'])) | set(trace_missing))
    manifest = seal(
        dict(
            version="progressive-report/v1",
            campaign_record=request["campaign_record"],
            scope_record=request["scope_record"],
            terminal=request["terminal"],
            campaign=request["campaign"],
            source=request["source"],
            seeds=request["seeds"],
            design=request["design"],
            scope=request["scope"],
            inventory=inventory,
            summary=totals,
            decisions=request["decisions"],
            producer_error=request["producer_error"],
            collector_errors=request["collector_errors"],
            missing=missing,
            status="REPORT_INCOMPLETE" if missing else "COMPLETE",
        )
    )
    validate_manifest(manifest)
    docs["manifest.json"] = encoded(manifest)
    if len(docs["manifest.json"]) > MAX_PART:
        raise ValueError("REPORT_INCOMPLETE: manifest cannot fit")
    if sum(map(len, docs.values())) > max_bytes:
        raise ValueError("REPORT_INCOMPLETE: metadata byte budget")
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("symlink report output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("xb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as z,
    ):
        for name, blob in sorted(docs.items()):
            z.writestr(name, blob)
    return manifest


def validate_producer(value, row):
    if (
        type(value) is not dict
        or set(value)
        != {
            "version",
            "identity",
            "source",
            "scope",
            "device",
            "status",
            "requested",
            "executed",
            "measurements",
        }
        or value["version"] != "progressive-hip-case/v1"
        or seal(value) != value
        or any(
            value[k] != row[k]
            for k in ("identity", "source", "scope", "device", "status", "requested", "executed")
        )
    ):
        raise ValueError("original qualification producer differs")
    metrics = value["measurements"]
    if (
        type(metrics) is not dict
        or len(metrics) > 64
        or any(
            type(k) is not str or len(k) > 128 or (type(v) not in (int, float, bool, type(None)))
            for k, v in metrics.items()
        )
    ):
        raise ValueError("closed numeric qualification measurements required")
    required = [
        "finite_update",
        "parameter_change",
        "full_restore",
        "deterministic_continuation",
        "generation",
        "native_reference_score",
        "cleanup",
    ]
    if value["status"] == "PASS" and (
        value["requested"] != required
        or value["executed"] != required
        or any(metrics.get(k) is not True for k in required)
    ):
        raise ValueError("mandatory qualification case evidence missing")
    if value["status"] == "PASS" and (
        value["device"] != "actual_HIP"
        or not metrics.get("finite_update")
        or not metrics.get("parameter_change")
        or not metrics.get("full_restore")
        or not metrics.get("generation")
        or not metrics.get("cleanup")
    ):
        raise ValueError("missing actual-device qualification measurements")


def terminal_request(root, folder):
    """Preserve every terminal path even when scientific tables were never produced."""
    from .contracts import load

    result = load(Path(folder) / "result.json")
    if seal(result) != result:
        raise ValueError("terminal result seal differs")

    def error(e):
        if not e:
            return None
        return dict(
            type=e.get("type", "CollectorError"),
            code=e.get("code", "COLLECTOR_FAILURE"),
            producer=e.get("producer", "collector"),
            local_traceback=e.get("local_traceback"),
        )

    # Detailed errors remain local; the public references preserve their originals.
    return seal(
        dict(
            version="progressive-public-report-input/v1",
            campaign_record=None,
            scope_record=None,
            terminal=dict(
                identity=result["identity"],
                mechanical_status=result.get("mechanical_status", "BLOCKED"),
                scientific_status=result.get("scientific_status", "UNMEASURED"),
                qualification_status=result.get("qualification_status", "PENDING"),
                claim_status=result.get("claim_status", "UNREVIEWED"),
            ),
            campaign=result.get("campaign") or digest("unbound-terminal"),
            source=digest(result["identity"]),
            seeds=[],
            design=dict(claim="capability", bootstrap_seed=0, comparisons=1, links=[], costs=None),
            scope=dict(required=list(KINDS), qualification_scopes=[], tasks=[]),
            tables={},
            decisions=[],
            producer_error=error(result.get("producer_error")),
            collector_errors=[error(e) for e in result.get("collector_errors", [])],
        )
    )
