"""T095: admission and reservations must describe the run actually executed."""

import pytest

from oeis_learn.fresh.cost_policy import screen_cost


def measured():
    return dict(
        training_p95=1,
        teacher_forcing_p95=0.01,
        generation_p95=1,
        generation_batch_p95=2,
        generation_batch_size=8,
        fit_p95=0.1,
        reference_p95=0.1,
        hidden_scoring_p95=0.2,
        checkpoint_p95=0.5,
    )


def test_diverse32_and_transfer_screen_have_distinct_complete_workloads():
    tiny = screen_cost(**measured(), scope="diverse32", updates=200, cadence=100, budget=14400)
    screen = screen_cost(**measured(), scope="transfer-screen", updates=200, cadence=100)
    assert tiny["scope"] == "diverse32"
    assert tiny["workload"]["periodic_prompts"] == 32
    assert tiny["end_screen_prompts"] == 32
    assert screen["workload"]["periodic_prompts"] == 96
    assert screen["end_screen_prompts"] == 256
    assert tiny["projected_seconds"] < screen["projected_seconds"]
    for result in (tiny, screen):
        assert result["workload"]["generation_rounds"] == 24
        assert result["workload"]["fitting_rounds"] == 32
        assert result["workload"]["hidden_rounds"] == 6
        assert result["omitted_prompts"] == 0
        assert result["reserve_fraction"] == 0.2


def test_scoped_tail_validation_and_role_reservations_match_preflight():
    from oeis_learn.fresh.cost_policy import training_role_seconds, end_role_seconds

    costs = measured()
    result = screen_cost(**costs, scope="diverse32", updates=150, cadence=100, budget=14400)
    assert result["scheduled_validations"] == 2
    reserved = sum(training_role_seconds(costs, "diverse32", n) for n in (100, 50))
    reserved += end_role_seconds(costs, "diverse32", 32, "generation")
    reserved += end_role_seconds(costs, "diverse32", 32, "score")
    assert result["reserved_seconds"] == reserved
    assert reserved <= result["budget"]


@pytest.mark.parametrize("scope", ["tiny", "full", "", False])
def test_unknown_scope_cannot_authorize_training(scope):
    with pytest.raises(ValueError, match="scope"):
        screen_cost(**measured(), scope=scope)


def admitted_recipe():
    from oeis_learn.fresh.bigint_config import defaults, resolve
    from oeis_learn.fresh.common import digest
    from oeis_learn.fresh.cost_policy import schedule_identity

    recipe = defaults()
    recipe["budgets"]["training_seconds"] = 14400
    recipe["learning"]["validation_every"] = 100
    receipt = dict(
        image="sha256:" + "a" * 64,
        source_sha256="sha256:" + "b" * 64,
        shape_sha256="sha256:" + "c" * 64,
        fixture_sha256="sha256:" + "d" * 64,
        scope="diverse32",
        scoring_data=False,
        costs=measured(),
        profile_sha256=digest(
            {k: recipe[k] for k in ("execution", "inputs", "model", "fitting", "search")}
        ),
        schedule_sha256=schedule_identity(recipe, "diverse32"),
    )
    # Cheap costs allow the full3000-update registration, with unchanged rules.
    receipt["costs"] = {
        k: v / 100 if k != "generation_batch_size" else v for k, v in receipt["costs"].items()
    }
    recipe["learning"]["cost_preflight"] = receipt
    return resolve(recipe)


@pytest.mark.parametrize("changed", [None, "scope", "image", "source_sha256", "shape_sha256"])
def test_scoped_admission_binds_authority_source_image_and_prepared_data(changed):
    from oeis_learn.fresh.cost_policy import validate_preflight

    recipe = admitted_recipe()
    receipt = recipe["learning"]["cost_preflight"]
    args = {k: receipt[k] for k in ("scope", "image", "source_sha256", "shape_sha256")}
    if changed:
        args[changed] = "transfer-screen" if changed == "scope" else "sha256:" + "e" * 64
        with pytest.raises(ValueError, match="scope|mismatch"):
            validate_preflight(recipe, **args)
    else:
        assert validate_preflight(recipe, **args)["within_budget"]


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("learning", "validation_every", 500),
        ("learning", "max_updates", 2000),
        ("budgets", "training_seconds", 12000),
        ("evaluation", "hidden_seconds", 20),
    ],
)
def test_schedule_changes_require_new_receipt(section, key, value):
    from oeis_learn.fresh.bigint_config import resolve

    recipe = admitted_recipe()
    recipe.pop("identity")
    recipe[section][key] = value
    with pytest.raises(ValueError, match="schedule"):
        resolve(recipe)


def test_legacy_receipt_is_readable_without_new_training_authority():
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.cost_policy import validate_preflight

    recipe = admitted_recipe()
    recipe.pop("identity")
    receipt = recipe["learning"]["cost_preflight"]
    for key in ("scope", "source_sha256", "shape_sha256", "schedule_sha256"):
        receipt.pop(key)
    recipe = resolve(recipe)
    with pytest.raises(ValueError, match="scope"):
        validate_preflight(
            recipe,
            scope="diverse32",
            image=receipt["image"],
            source_sha256="sha256:" + "b" * 64,
            shape_sha256="sha256:" + "c" * 64,
        )


def test_budget_revision_preserves_finite_four_hour_resource_cap():
    from oeis_learn.fresh.bigint_config import defaults, resolve

    value = defaults()
    value["budgets"]["training_seconds"] = 14400
    assert resolve(value)["budgets"]["training_seconds"] == 14400
    value["budgets"]["training_seconds"] = 14401
    with pytest.raises(ValueError):
        resolve(value)


def test_preregistered_smoke_revision_changes_only_the_resource_budget():
    from pathlib import Path
    from oeis_learn.fresh.bigint_config import load

    root = Path(__file__).resolve().parents[2] / "configs/foundation"
    original = load(root / "bigint_smoke.yaml")
    revised = load(root / "bigint_smoke_phase5.yaml")
    original.pop("identity")
    revised.pop("identity")
    assert revised["budgets"]["training_seconds"] == 14400
    revised["budgets"]["training_seconds"] = original["budgets"]["training_seconds"]
    assert revised == original


def test_memorization_periodic_panel_is_exactly_the_frozen32():
    from oeis_learn.fresh.bigint_worker import periodic_panels

    examples = {
        split: [dict(id=f"{split}-{i}", stage="ABCD"[i % 4]) for i in range(n)]
        for split, n in (
            ("training", 32),
            ("parameter_development", 128),
            ("structural_development", 128),
        )
    }
    tiny = periodic_panels(examples, "diverse32")
    assert tiny == {"training": [r["id"] for r in examples["training"]]}
    assert sum(map(len, periodic_panels(examples, "transfer-screen").values())) == 96
    examples["training"].pop()
    with pytest.raises(ValueError, match="complete32"):
        periodic_panels(examples, "diverse32")
