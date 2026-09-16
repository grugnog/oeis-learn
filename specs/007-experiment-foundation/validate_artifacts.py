#!/usr/bin/env python3
"""Validate 007 documentation/contracts, not the future application implementation.

Requires jsonschema==4.26.0 in a venv. This is project-specific validation,
not an upstream Spec Kit command or a substitute for semantic review.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote

from jsonschema import Draft202012Validator, FormatChecker


ROOT = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> dict:
    required = [
        "spec.md", "plan.md", "tasks.md", "research.md", "data-model.md",
        "quickstart.md", "constitution-rfc.md", "review-coverage.md", "contracts/execution.md",
        "contracts/repairs.md", "contracts/learning.md", "contracts/experiments.md",
        "validation/review-inputs.json",
        "contracts/evaluation.md", "contracts/cli.md", "contracts/artifacts.schema.json",
        "contracts/schema-examples.json", "checklists/requirements.md",
        "checklists/design.md", "validation/report.md", "validation/task-dependencies.json",
    ]
    for relative in required:
        require((ROOT / relative).is_file(), f"Missing required artifact: {relative}")

    spec = (ROOT / "spec.md").read_text()
    tasks = (ROOT / "tasks.md").read_text()
    requirements = re.findall(r"^- \*\*((?:FR|SC)-\d{3})\*\*:", spec, re.M)
    expected = [f"FR-{n:03}" for n in range(1, 39)] + [f"SC-{n:03}" for n in range(1, 15)]
    require(requirements == expected, "Requirements must be unique, ordered FR-001..038 and SC-001..014")
    stories = re.findall(r"^### User Story (\d+) .+\(Priority: P[123]\)", spec, re.M)
    require(stories == [str(n) for n in range(1, 9)], "Expected eight prioritized user stories")
    for heading in ["User Scenarios & Testing", "Requirements", "Success Criteria", "Assumptions"]:
        require(f"## {heading}" in spec, f"Missing spec section: {heading}")
    for heading in ["Summary", "Technical Context", "Constitution Check", "Project Structure", "Complexity Tracking"]:
        require(f"## {heading}" in (ROOT / "plan.md").read_text(), f"Missing plan section: {heading}")

    rows = re.findall(r"^- \[([ x])\] (T\d{3}) (?:\[P\] )?(?:\[US(\d)\] )?(.+)$", tasks, re.M)
    ids = [row[1] for row in rows]
    require(ids == [f"T{n:03}" for n in range(1, 106)], "Expected sequential unique T001..T105")
    require(all(state == " " for state, *_ in rows), "Planning deliverable must not mark implementation complete")
    require(len(re.findall(r"^- \[[ x]\] T", tasks, re.M)) == len(rows), "Malformed task checkbox line")
    counts = {"setup_and_foundation": 0, **{f"US{n}": 0 for n in range(1, 9)}, "cross_cutting": 0}
    for _, task_id, story, description in rows:
        num = int(task_id[1:])
        wanted = "1" if 9 <= num <= 19 else "2" if 20 <= num <= 29 else "3" if 30 <= num <= 37 else "4" if 38 <= num <= 49 else "5" if 50 <= num <= 61 else "6" if 62 <= num <= 79 else "7" if 80 <= num <= 92 else "8" if 93 <= num <= 102 else ""
        require(story == wanted, f"Incorrect/missing story label on {task_id}")
        require(re.search(r"`[^`]*(?:src/|tests/|configs/|docker/|scripts/|specs/|docs/|README|\.specify/|\.github/)[^`]*`", description) is not None, f"Task lacks concrete path: {task_id}")
        counts[f"US{story}" if story else "setup_and_foundation" if num <= 8 else "cross_cutting"] += 1

    coverage = {}
    for key, cell in re.findall(r"^\| ((?:FR|SC)-\d{3}) \| ([^|]+) \|", tasks, re.M):
        require(key not in coverage, f"Duplicate coverage row: {key}")
        covered = re.findall(r"T\d{3}", cell)
        require(bool(covered) and set(covered) <= set(ids), f"Invalid/empty coverage: {key}")
        coverage[key] = covered
    require(set(coverage) == set(expected), "Requirement coverage incomplete or contains unknown requirement")
    require(set().union(*(set(v) for v in coverage.values())) == set(ids), "Unmapped implementation task")

    dependencies = json.loads((ROOT / "validation/task-dependencies.json").read_text())
    require(set(dependencies) == set(ids), "Dependency graph must contain every task exactly once")
    for task, prerequisites in dependencies.items():
        require(isinstance(prerequisites, list) and len(set(prerequisites)) == len(prerequisites), f"Invalid dependencies for {task}")
        require(all(p in ids and ids.index(p) < ids.index(task) for p in prerequisites), f"Forward/cyclic/missing dependency for {task}")

    # External review coverage is a separate inventory, not inferred from FR presence.
    review = json.loads((ROOT / "validation/review-inputs.json").read_text())
    items = review["items"]
    wanted_inputs = [f"R{n:02}" for n in range(1, 21)] + [f"A{n:02}" for n in range(1, 20)] + [f"P{n:02}" for n in range(1, 18)] + [f"D{n:02}" for n in range(1, 13)]
    require([r["id"] for r in items] == wanted_inputs, "Incomplete or reordered review/decision inventory")
    allowed = {"included", "corrected", "already_present", "not_adopted", "deferred_loda"}
    for row in items:
        require(row["status"] in allowed and len(row["disposition"]) >= 25, f"Invalid disposition: {row['id']}")
        require(set(row["tasks"]) <= set(ids), f"Unknown review task: {row['id']}")
        if row["status"] in {"included", "corrected", "already_present"}:
            require(bool(row["tasks"]) and any(int(t[1:]) <= (105 if row["id"] == "D04" else 102) for t in row["tasks"]), f"No substantive task for {row['id']}")
        if row["status"] == "deferred_loda":
            require(row["id"] in {"D06", "D07", "D08"} and not row["tasks"], "Non-LODA deferral or ambiguous task mapping")
        require(row["id"] in (ROOT / "review-coverage.md").read_text(), f"Missing human-readable coverage: {row['id']}")
    require({r["id"] for r in items if r["status"] == "not_adopted"} == {"A01", "A11"}, "Unexpected rejected review item")
    require(len(review["sources"]) == 2 and all(re.fullmatch(r"[0-9a-f]{64}", x["sha256"]) for x in review["sources"]), "Missing source provenance")
    # Ensure each substantive FR has implementation/test coverage before final aggregate checks.
    require(all(any(int(t[1:]) <= 102 for t in ts) for ts in coverage.values()), "Only aggregate task coverage")

    links = 0
    for doc in ROOT.rglob("*.md"):
        content = doc.read_text()
        require(not re.search(r"\[NEEDS CLARIFICATION:|TODO\(|TKTK|\[FEATURE NAME\]|\[###-", content), f"Unresolved template marker in {doc.relative_to(ROOT)}")
        for target in re.findall(r"\[[^\]\n]+\]\(([^)]+)\)", content):
            if re.match(r"(?:https?://|mailto:|#)", target):
                continue
            path = unquote(target.split("#", 1)[0])
            require((doc.parent / path).exists(), f"Broken local document link in {doc.relative_to(ROOT)}: {target}")
            links += 1

    schema = json.loads((ROOT / "contracts/artifacts.schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    examples = json.loads((ROOT / "contracts/schema-examples.json").read_text())
    require(len(examples) == 3, "Expected three explicitly synthetic schema fixtures")
    for example in examples:
        validator.validate(example)

    # Negative shapes exercise the actual trust-boundary schema, not future runtime behavior.
    mutations = [
        (0, lambda x: x.update(hidden_terms=["1"]), "hidden field in visible prompt"),
        (0, lambda x: x.update(observed_terms=x["observed_terms"][:-1]), "short prefix"),
        (0, lambda x: x["observed_terms"].__setitem__(0, "01"), "noncanonical integer"),
        (0, lambda x: x["observed_terms"].__setitem__(0, 1.0), "floating-point integer"),
        (1, lambda x: x.update(proof_status="PROVEN"), "false proof promotion"),
        (1, lambda x: x.update(purpose="model"), "model result missing checkpoint"),
        (1, lambda x: x.update(outputs=x["outputs"][:20]), "short full-horizon match"),
        (1, lambda x: x.update(verified_terms=99), "incomplete verified horizon"),
        (1, lambda x: x.update(outcome="numeric_limit"), "failure without reason"),
        (1, lambda x: x["usage"]["fuel"].update(state="measured", value=None), "fake measured null"),
        (2, lambda x: x.update(checkpoint_state="writing"), "partial checkpoint"),
        (2, lambda x: x.update(payload_state_keys=x["payload_state_keys"][:-1]), "missing active state key"),
        (2, lambda x: x.update(blob_sha256="sha256:bad"), "bad digest"),
        (2, lambda x: x.update(blob_path="../outside.pt"), "escaping blob path"),
    ]
    for index, mutate, label in mutations:
        invalid = copy.deepcopy(examples[index])
        mutate(invalid)
        require(not validator.is_valid(invalid), f"Negative schema case was accepted: {label}")

    return {
        "status": "pass", "scope": "document structure, links, coverage, dependency graph and schema shapes only",
        "requirements": len(requirements), "coverage_percent": 100, "tasks": len(ids),
        "tasks_by_story": counts, "review_decision_items": len(items),
        "deferred_items": [r["id"] for r in items if r["status"] == "deferred_loda"],
        "local_links_checked": links,
        "positive_schema_fixtures": len(examples), "negative_schema_cases": len(mutations),
        "runtime_gates_executed": False, "constitution_adoption_claimed": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(main(), indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"Artifact validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
