"""Closed recipes and durable identity/isolation precede production integration."""

from copy import deepcopy
import pytest
from oeis_learn.fresh.bigint_config import resolve, defaults


def test_recipe_defaults_are_closed_explicit_and_identity_bound():
    a = resolve(dict(version="bigint-bootstrap/v1"))
    assert a["learning"]["max_updates"] == 3000 and a["evaluation"]["final_access"] == "sealed"
    assert set(a) - {"identity"} == set(defaults())
    assert a == resolve(dict(version="bigint-bootstrap/v1"))
    changed = deepcopy(a)
    changed.pop("identity")
    changed["model"]["seed"] = 20261017
    assert resolve(changed)["identity"] != a["identity"]
    for path in (("oops",), ("model", "oops"), ("fitting", "service_limits", "oops")):
        bad = dict(version="bigint-bootstrap/v1")
        node = bad
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = True
        with pytest.raises(ValueError):
            resolve(bad)


def test_assisted_requires_review_and_no_final_access_in_training_recipe():
    with pytest.raises(ValueError, match="assisted"):
        resolve(dict(version="bigint-bootstrap/v1", track=dict(kind="assisted")))
    with pytest.raises(ValueError, match="sealed"):
        resolve(dict(version="bigint-bootstrap/v1", evaluation=dict(final_access="authorized")))


def test_public_commands_and_explicit_recipe_conflicts():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from foundation_local import parser
    from foundation_bigint import validate_options

    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    args = parser().parse_args(
        [
            "train",
            "--profile",
            "bigint-bootstrap",
            "--config",
            "recipe.yaml",
            "--inputs",
            "inputs",
            "--run",
            "run",
            "--seed",
            "20261016",
        ]
    )
    args.explicit_options = {"--seed"}
    validate_options(args, cfg)
    args.seed += 1
    with pytest.raises(ValueError, match="seed"):
        validate_options(args, cfg)
    args = parser().parse_args(
        [
            "learnability",
            "--config",
            "recipe.yaml",
            "--inputs",
            "inputs",
            "--run",
            "run",
            "--decoder-position-scale",
            "1",
        ]
    )
    args.explicit_options = {"--decoder-position-scale"}
    with pytest.raises(ValueError, match="conflicting"):
        validate_options(args, cfg)
    for suite in ("bigint-cpu", "bigint-gpu", "bigint-retirement"):
        assert parser().parse_args(["validate", "--suite", suite]).suite == suite
    final = parser().parse_args(
        [
            "evaluate",
            "--run",
            "run",
            "--split",
            "final",
            "--final-plan",
            "plan.json",
            "--output",
            "child",
            "--fitting",
            "paired",
        ]
    )
    assert final.final_plan == "plan.json"
    assert parser().parse_args(["compare", "--runs", "a", "b", "--output", "comparison"]).runs == [
        "a",
        "b",
    ]


def test_committed_recipes_are_explicit_and_assisted_placeholder_fails():
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "configs/foundation"
    for name in ("smoke", "literal_screen", "parameter_screen", "encoder_study"):
        value = yaml.safe_load((root / ("bigint_" + name + ".yaml")).read_text())
        cfg = resolve(value, seed=20261017)
        assert cfg["model"]["seed"] == 20261017 and cfg["evaluation"]["paired_seeds"] == [
            20261016,
            20261017,
            20261018,
        ]
        assert cfg["model"]["position_scale"] == 16 and cfg["learning"]["cost_preflight"] is None
        assert set(value["fitting"]["service_limits"]) == set(
            defaults()["fitting"]["service_limits"]
        )
    with pytest.raises(ValueError, match="assisted"):
        resolve(yaml.safe_load((root / "bigint_assisted_screen.yaml").read_text()))


def test_recipe_rejects_unregistered_beam_and_unbounded_reporting():
    for value in (
        dict(search=dict(beam=4)),
        dict(reporting=dict(max_examples=9)),
        dict(learning=dict(cost_preflight={"fake": True})),
    ):
        with pytest.raises(ValueError):
            resolve(dict(version="bigint-bootstrap/v1", **value))


def test_compare_lists_missing_seeds_and_rejects_profile_drift(tmp_path):
    from oeis_learn.fresh.comparison import compare_runs
    from oeis_learn.fresh.common import write
    from oeis_learn.fresh.bigint_state import DurableRun

    cfg = resolve(dict(version="bigint-bootstrap/v1"))
    h = "sha256:" + "1" * 64

    def arm(name, recipe):
        run = DurableRun.create(
            tmp_path / name, recipe, dict(identity=h, split_sha256=h), image=h, device="cpu"
        )
        run.finalize("completed", {})
        write(run.root / "comparison.json", dict(data=h, splits={}))
        return run.root

    first = arm("first", cfg)
    report = compare_runs([first], tmp_path / "report")
    assert (
        len(report["missing_seeds"]) == 5
        and report["decision"] is None
        and report["automatic_run_authorized"] is False
    )
    changed = resolve(dict(version="bigint-bootstrap/v1", fitting=dict(disabled=True)))
    with pytest.raises(ValueError, match="mismatched"):
        compare_runs([first, arm("changed", changed)], tmp_path / "different")


def test_analysis_child_is_immutable_and_does_not_edit_parent(tmp_path):
    from oeis_learn.fresh.bigint_state import child_analysis

    h = "sha256:" + "1" * 64
    j = "sha256:" + "2" * 64
    child_analysis(tmp_path / "child", h, h, h)
    assert child_analysis(tmp_path / "child", h, h, h)["parent"] == h
    with pytest.raises(ValueError, match="immutable"):
        child_analysis(tmp_path / "child", h, j, h)


def test_portable_bigint_host_imports_without_site_packages():
    import os, subprocess, sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    environment = dict(
        os.environ, PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "scripts")])
    )
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import foundation_bigint,foundation_local; from oeis_learn.fresh.bigint_inputs import check; from oeis_learn.fresh.provenance import validate_ancestry; foundation_local.parser().parse_args(['validate','--suite','bigint-cpu'])",
        ],
        env=environment,
        check=True,
        timeout=10,
    )
