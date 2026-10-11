"""Convergence extends traceability without accepting gaps or stale evidence."""

import hashlib
import json
from pathlib import Path
import re
import runpy

import pytest


ROOT = Path(__file__).resolve().parents[2] / "specs/007-experiment-foundation"
VALIDATOR = runpy.run_path(str(ROOT / "validate_artifacts.py"))
EXPECTED = [f"FR-{n:03}" for n in range(1, 39)] + [f"SC-{n:03}" for n in range(1, 15)]


def inventory():
    return (ROOT / "tasks.md").read_text(), json.loads(
        (ROOT / "validation/task-dependencies.json").read_text()
    )


def test_current_convergence_inventory_passes_all_artifact_checks():
    report = VALIDATOR["main"]()
    assert report["tasks"] == 121
    assert report["coverage_percent"] == 100
    assert report["runtime_gates_executed"] is False


def test_another_mapped_convergence_task_does_not_need_a_new_count_limit():
    tasks, dependencies = inventory()
    tasks += (
        "\n## Phase 15: Convergence\n\n"
        "- [ ] T122 Check `tests/unit/test_foundation_task_inventory.py` per FR-038 (partial)\n"
    )
    tasks = re.sub(r"^(\| FR-038 \| [^|]+)( \|)", r"\1, T122\2", tasks, flags=re.M)
    dependencies["T122"] = ["T121"]
    _, ids, _, coverage = VALIDATOR["validate_task_inventory"](tasks, EXPECTED, dependencies)
    assert ids[-1] == "T122" and "T122" in coverage["FR-038"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("duplicate", "sequential"),
        ("gap", "sequential"),
        ("malformed", "Malformed"),
        ("unmapped", "Unmapped convergence source"),
        ("unknown_source", "source reference"),
        ("missing_source", "source reference"),
        ("missing_gap_type", "gap type"),
        ("missing_dependency", "every task"),
        ("forward_dependency", "Forward/cyclic"),
    ],
)
def test_appended_tasks_keep_strict_inventory_checks(mutation, message):
    tasks, dependencies = inventory()
    if mutation == "duplicate":
        tasks = tasks.replace("- [ ] T121 ", "- [ ] T120 ")
    elif mutation == "gap":
        tasks = tasks.replace("- [ ] T121 ", "- [ ] T122 ")
    elif mutation == "malformed":
        tasks = tasks.replace("- [ ] T121 ", "- [ ] T121\n")
    elif mutation == "unmapped":
        # T109 remains mapped elsewhere, but one of its declared sources is missing.
        tasks = re.sub(r"^(\| FR-012 \| [^\n]+)", lambda m: m[0].replace(", T109", ""), tasks, flags=re.M)
    elif mutation in ("unknown_source", "missing_source", "missing_gap_type"):
        lines = tasks.splitlines()
        for index, line in enumerate(lines):
            if line.startswith("- [ ] T121 "):
                if mutation == "unknown_source":
                    lines[index] = line.replace("FR-038", "FR-999")
                elif mutation == "missing_source":
                    lines[index] = re.sub(r"\b(?:FR|SC)-\d{3}\b", "unspecified", line)
                else:
                    lines[index] = line.removesuffix("(partial)")
        tasks = "\n".join(lines)
    elif mutation == "missing_dependency":
        dependencies.pop("T121")
    else:
        dependencies["T109"] = ["T110"]
    with pytest.raises(ValueError, match=message):
        VALIDATOR["validate_task_inventory"](tasks, EXPECTED, dependencies)


def test_current_dispositions_bind_new_tasks_without_rebinding_historical_evidence():
    current = json.loads((ROOT / "validation/current-dispositions.json").read_text())
    historical_path = ROOT / "validation/phase-11-dispositions.json"
    historical = json.loads(historical_path.read_text())
    assert current["task_inventory_sha256"] == "sha256:" + hashlib.sha256(
        (ROOT / "tasks.md").read_bytes()
    ).hexdigest()
    assert historical["task_inventory_sha256"] != current["task_inventory_sha256"]
    assert current["historical_template"]["sha256"] == "sha256:" + hashlib.sha256(
        historical_path.read_bytes()
    ).hexdigest()
    assert "prior_validation" not in current
    assert current["full_007_complete"] is False
    assert current["items"] == historical["items"]
