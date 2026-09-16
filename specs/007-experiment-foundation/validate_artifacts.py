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
        "quickstart.md", "constitution-rfc.md", "contracts/execution.md",
        "contracts/evaluation.md", "contracts/cli.md", "contracts/artifacts.schema.json",
        "contracts/schema-examples.json", "checklists/requirements.md",
        "checklists/design.md", "validation/report.md", "validation/task-dependencies.json",
    ]
    for relative in required:
        require((ROOT / relative).is_file(), f"Missing required artifact: {relative}")

    spec = (ROOT / "spec.md").read_text()
    tasks = (ROOT / "tasks.md").read_text()
    requirements = re.findall(r"^- \*\*((?:FR|SC)-\d{3})\*\*:", spec, re.M)
    expected = [f"FR-{n:03}" for n in range(1, 21)] + [f"SC-{n:03}" for n in range(1, 7)]
    require(requirements == expected, "Requirements must be unique, ordered FR-001..020 and SC-001..006")
    stories = re.findall(r"^### User Story (\d+) .+\(Priority: P[123]\)", spec, re.M)
    require(stories == ["1", "2", "3", "4"], "Expected four prioritized user stories")
    for heading in ["User Scenarios & Testing", "Requirements", "Success Criteria", "Assumptions"]:
        require(f"## {heading}" in spec, f"Missing spec section: {heading}")
    for heading in ["Summary", "Technical Context", "Constitution Check", "Project Structure", "Complexity Tracking"]:
        require(f"## {heading}" in (ROOT / "plan.md").read_text(), f"Missing plan section: {heading}")

    rows = re.findall(r"^- \[([ x])\] (T\d{3}) (?:\[P\] )?(?:\[US(\d)\] )?(.+)$", tasks, re.M)
    ids = [row[1] for row in rows]
    require(ids == [f"T{n:03}" for n in range(1, 53)], "Expected sequential unique T001..T052")
    require(all(state == " " for state, *_ in rows), "Planning deliverable must not mark implementation complete")
    require(len(re.findall(r"^- \[[ x]\] T", tasks, re.M)) == len(rows), "Malformed task checkbox line")
    counts = {"setup_and_foundation": 0, "US1": 0, "US2": 0, "US3": 0, "US4": 0, "cross_cutting": 0}
    for _, task_id, story, description in rows:
        num = int(task_id[1:])
        wanted = "1" if 9 <= num <= 19 else "2" if 20 <= num <= 29 else "3" if 30 <= num <= 37 else "4" if 38 <= num <= 49 else ""
        require(story == wanted, f"Incorrect/missing story label on {task_id}")
        require(re.search(r"`[^`]*(?:src/|tests/|configs/|docker/|scripts/|specs/|docs/|README|\.specify/)[^`]*`", description) is not None, f"Task lacks concrete path: {task_id}")
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
        "tasks_by_story": counts, "local_links_checked": links,
        "positive_schema_fixtures": len(examples), "negative_schema_cases": len(mutations),
        "runtime_gates_executed": False, "constitution_adoption_claimed": False,
    }


if __name__ == "__main__":
    try:
        print(json.dumps(main(), indent=2))
    except (ValueError, OSError, json.JSONDecodeError) as error:
        print(f"Artifact validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
