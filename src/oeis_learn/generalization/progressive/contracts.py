"""Closed authority, exact values and confined immutable artifact references.

The small stdlib schema interpreter supports only the committed schema vocabulary.
It fails on unsupported validation keywords; installed jsonschema tests cross-check
this boundary. A content seal proves binding, never producer or role permission.
"""

from datetime import datetime
from functools import lru_cache
from importlib.resources import files
import json
import math
from pathlib import Path
import re

from oeis_learn.pipeline_validation.identity import digest, seal, file_hash, confined
from oeis_learn.generalization.contracts import verify_artifact


# Full finite tuning campaigns contain up to 512 tasks and 648 choices per task.
# Keep their containing authority documents bounded independently of small receipts.
CAMPAIGN_DOCUMENT_BYTES = 128 << 20


@lru_cache(maxsize=32)
def schema(name):
    path = files("oeis_learn.generalization").joinpath(
        "schemas", "progressive-" + name + ".schema.json"
    )
    return json.loads(path.read_text())


def _finite(value):
    if type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite authority value")
    if isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise ValueError("string keys required")
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError("JSON authority required")


def _check(rule, value, document):
    allowed = {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "oneOf",
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "const",
        "enum",
        "description",
        "title",
    }
    if set(rule) - allowed:
        raise ValueError("unsupported installed schema vocabulary")
    if "$ref" in rule:
        target, fragment = rule["$ref"].split("#")
        doc = (
            schema(target.removeprefix("progressive-").removesuffix(".schema.json"))
            if target
            else document
        )
        resolved = doc
        for part in fragment.lstrip("/").split("/"):
            resolved = resolved[part]
        _check(resolved, value, doc)
        return
    if "oneOf" in rule:
        matched = 0
        for item in rule["oneOf"]:
            try:
                _check(item, value, document)
                matched += 1
            except ValueError:
                pass
        if matched != 1:
            raise ValueError("one closed variant required")
    if "const" in rule and (type(value) is not type(rule["const"]) or value != rule["const"]):
        raise ValueError("constant mismatch")
    if "enum" in rule and not any(type(value) is type(x) and value == x for x in rule["enum"]):
        raise ValueError("unregistered value")
    kind = rule.get("type")
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if kind and (
        type(value) not in (types[kind] if isinstance(types[kind], tuple) else (types[kind],))
    ):
        raise ValueError("schema type mismatch")
    if kind == "object":
        props = rule["properties"]
        if rule.get("additionalProperties") is not False:
            raise ValueError("open authority schema forbidden")
        if set(value) - set(props) or not set(rule.get("required", [])) <= set(value):
            raise ValueError("unknown or missing keys")
        for key, child in value.items():
            _check(props[key], child, document)
    if kind == "array":
        if not rule.get("minItems", 0) <= len(value) <= rule.get("maxItems", 2**63):
            raise ValueError("array bound")
        if rule.get("uniqueItems") and len({digest(x) for x in value}) != len(value):
            raise ValueError("duplicate array member")
        for child in value:
            _check(rule["items"], child, document)
    if kind in {"integer", "number"}:
        if not rule.get("minimum", -math.inf) <= value <= rule.get("maximum", math.inf):
            raise ValueError("numeric bound")
    if kind == "string":
        if not rule.get("minLength", 0) <= len(value) <= rule.get("maxLength", 2**63):
            raise ValueError("string bound")
        if (
            "pattern" in rule
            and re.fullmatch(rule["pattern"].removeprefix("^").removesuffix("$"), value) is None
        ):
            raise ValueError("noncanonical string")
        if rule.get("format") == "date-time":
            try:
                timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                raise ValueError("invalid timestamp") from None
            if timestamp.utcoffset() is None:
                raise ValueError("timestamp timezone required")


def shape(kind, record):
    _finite(record)
    document = schema(kind)
    _check(document, record, document)
    return record


def validate(kind, record):
    shape(kind, record)
    if seal(record) != record:
        raise ValueError("canonical identity mismatch")
    return json.loads(json.dumps(record, allow_nan=False))


def record(kind, payload, *, created_at, parents=(), purpose, track):
    return validate(
        kind,
        seal(
            dict(
                version="discovery-" + kind + "/v1",
                created_at=created_at,
                parents=list(parents),
                purpose=purpose,
                track=track,
                payload=payload,
            )
        ),
    )


def load(path, *, maximum=1 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("bounded regular JSON required")

    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result

    return json.loads(
        path.read_text(),
        object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")),
    )


def reference(root, path):
    candidate = Path(path)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(Path(root))
        except ValueError:
            raise ValueError("reference outside root") from None
    name = str(candidate)
    # Path objects normalize aliases, so callers supplying strings are checked first.
    if isinstance(path, str) and not Path(path).is_absolute():
        name = path
    _check(schema("common")["$defs"]["path"], name, schema("common"))
    target = confined(root, name)
    if not target.is_file():
        raise ValueError("regular artifact required")
    return dict(path=name, bytes=target.stat().st_size, sha256=file_hash(target))


def verify_refs(root, refs, *, max_bytes=512 << 20):
    paths = set()
    inodes = set()
    total = 0
    result = []
    for ref in refs:
        _check(schema("common")["$defs"]["artifact"], ref, schema("common"))
        if ref["path"] in paths:
            raise ValueError("duplicate or conflicting artifact path")
        paths.add(ref["path"])
        total += ref["bytes"]
        if total > max_bytes:
            raise ValueError("artifact inventory cap")
        payload = verify_artifact(root, ref, max_bytes=max_bytes)
        info = confined(root, ref["path"]).stat()
        inode = (info.st_dev, info.st_ino)
        if inode in inodes:
            raise ValueError("same-file artifact alias")
        inodes.add(inode)
        result.append(payload)
    return result


def _atomic_write(path, value):
    """Immutable, durable publication: never replace a prior valid object."""
    import os
    import uuid

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ValueError("fresh publication required")
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(
                (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write(path, value):
    from .storage import writing

    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with writing(path, len(raw)):
        return _atomic_write(path, value)
