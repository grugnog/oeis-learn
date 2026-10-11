"""Reconcile all original review inputs; this is traceability, not qualification."""

from pathlib import Path
import re

from oeis_learn.experiments.artifacts import compute_file_hash, load_json


def governance_path(name):
    root = Path(__file__).resolve().parents[3]
    path = root / "specs/007-experiment-foundation" / name
    return path if path.is_file() else Path(__file__).parent / "governance" / Path(name).name


def original_inputs_path():
    return governance_path("validation/review-inputs.json")


def require_dispositions(record):
    from oeis_learn.evaluation.experiment_runner import implementation_identity

    path = original_inputs_path()
    original = load_json(path.read_bytes())
    expected = {row["id"]: row for row in original["items"]}
    tasks_path = governance_path("tasks.md")
    task_states = dict(
        (task, marker)
        for marker, task in re.findall(
            r"^- \[([X/\- ])\] (T\d+)\b", tasks_path.read_text(), re.MULTILINE
        )
    )
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != "review-disposition/v1"
        or record.get("implementation_sha256") != implementation_identity()
        or record.get("source_sha256") != compute_file_hash(path)
        or record.get("task_inventory_sha256") != compute_file_hash(tasks_path)
        or record.get("full_007_complete") is not False
        or type(record.get("original_inputs")) is not int
        or record["original_inputs"] != len(expected)
        or not isinstance(record.get("items"), list)
        or len(record["items"]) != len(expected)
    ):
        raise ValueError("current original-input disposition binding required")
    seen = set()
    for item in record["items"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("malformed review disposition")
        key = item["id"]
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate original review input")
        seen.add(key)
        source = expected[key]
        states = item.get("task_states")
        if (
            item.get("title") != source["title"]
            or item.get("design_disposition") != source["status"]
            or not isinstance(item.get("rationale"), str)
            or not item["rationale"].strip()
            or not isinstance(states, dict)
            or set(states) != set(source["tasks"])
            or any(not isinstance(s, str) or s not in ("X", "/", "-", " ") for s in states.values())
            or any(task_states.get(task) != marker for task, marker in states.items())
        ):
            raise ValueError("original review disposition/task mapping incomplete: " + key)
        wanted = (
            "deferred_loda"
            if source["status"] == "deferred_loda"
            else "task_checklist_complete_not_release_authorization"
            if states and all(state == "X" for state in states.values())
            else "partial_or_pending"
        )
        if item.get("implementation_disposition") != wanted:
            raise ValueError("review completion inconsistent with pending tasks: " + key)
    return len(seen)
