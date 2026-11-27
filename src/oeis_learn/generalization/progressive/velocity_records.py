"""Round029 wire records. Authorization requires a separately retained operator decision."""

from pathlib import Path

from .contracts import _check, _finite, digest, schema, seal

SESSION = "016-research-round-029"
ACTIONS = {
    "qualification": {"qualify"},
    "cpu": {"cpu"},
    "exploration": {"run", "cpu"},
    "confirmation": {"run"},
}


def checked(kind, value):
    document = schema("round029")
    _check(document["$defs"][kind], value, document)
    return value


def open_stored(root, value, *, owner, format):
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file

    checked("stored_reference", value)
    if (value["owner"], value["format"]) != (owner, format):
        raise ValueError("stored artifact owner or format differs")
    return checked_file(root, value["artifact"])


def grant_terms(grant):
    return {k: v for k, v in grant.items() if k not in {"identity", "authorizing_record_ref"}}


def validate_grant(grant, authorization, *, root, parent, action):
    checked("grant", grant)
    allowed = set(grant["allowed_operations"])
    # These are fundable operations. The prospective study policy lists possible
    # future decisions, including confirmation, before any child has authority.
    branches = {
        "qualification": set(),
        "cpu": {"structure_search", "same_bank_selection", "resolve_missing"},
        "exploration": {"repair", "resolve_missing", "oeis_measurement", "paired_extension",
                        "structure_search", "same_bank_selection", "conditioning", "retain_D"},
        "confirmation": {"confirmation"},
    }
    if (
        seal(grant) != grant
        or root != grant["state_root_id"]
        or parent != grant["registration_or_parent_ref"]
        or action not in ACTIONS[grant["scope"]]
        or action not in allowed
        or not allowed <= ACTIONS[grant["scope"]] | branches[grant["scope"]]
        or grant["holdback_us"] > grant["total_us"]
        or grant["resources"]["elapsed_us"] != grant["total_us"]
    ):
        raise ValueError("grant phase, physical owner, parent or allowance differs")
    expected = seal(dict(version="round029-authorization/v1", terms=digest(grant_terms(grant))))
    if authorization != expected or grant["authorizing_record_ref"] != expected["identity"]:
        raise ValueError("independent operator authorization differs")
    return grant


def physical_root_id(root):
    path = Path(root).resolve(strict=True)
    stat = path.stat()
    return digest([SESSION, str(path), stat.st_dev, stat.st_ino])


def operation_key(coordinates, attempt):
    checked("operation", coordinates)
    _check(schema("common")["$defs"]["integer"], attempt, schema("common"))
    logical = digest(["round029-operation/v1", coordinates])
    return dict(logical=logical, physical=digest([logical, attempt]))


def validate_result(result):
    _finite(result)
    checked("result", result)
    if "progress" in result:
        saved = result["progress"]
        if saved != seal(saved) or saved.get("action") not in {"qualify", "cpu", "run"}:
            raise ValueError("invalid saved action result")
        validate_result({k: v for k, v in saved.items() if k != "identity"})
    return result


def visible_prompt(prompt):
    from .cohorts import validate_prompt

    if set(prompt) != {"indices", "values"}:
        raise ValueError("closed visible-only prompt required")
    validate_prompt(dict(prompt, id="round029"))
    return prompt


def registered_worker(expected, current):
    import re

    if (
        expected != current
        or set(current) != {"source", "image", "device", "role_policy"}
        or not re.fullmatch("[0-9a-f]{40}", current["source"])
        or not current["device"]
    ):
        raise ValueError("registered worker identity differs")
    for k in ("image", "role_policy"):
        _check(schema("common")["$defs"]["hash"], current[k], schema("common"))
    return current
