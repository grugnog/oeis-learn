"""Installed assets bind contracts, schemas, policies, tests and dependencies."""

import json
from pathlib import Path
from oeis_learn.experiments.source_snapshot import collect_snapshot


def test_source_inventory_covers_all_discovery_owners():
    root = Path(__file__).resolve().parents[2]
    snapshot = collect_snapshot(root)
    for folder, pattern in [
        ("src/oeis_learn/generalization/schemas", "progressive-*.json"),
        ("src/oeis_learn/generalization/policies", "progressive-*.json"),
        ("specs/013-progressive-mathematical-discovery/contracts", "*.md"),
    ]:
        paths = list((root / folder).glob(pattern))
        assert paths
        assert all(str(p.relative_to(root)) in snapshot for p in paths)
    assert "tests/contract/test_progressive_assets.py" in snapshot
    assert "uv.lock" in snapshot


def test_every_installed_schema_is_valid_json_schema():
    from jsonschema import Draft202012Validator
    from importlib.resources import files

    schemas = files("oeis_learn.generalization").joinpath("schemas")
    for path in schemas.iterdir():
        if path.name.startswith("progressive-"):
            Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_installed_contract_names_match_source():
    # The existing full wheel test calls source_snapshot() outside the checkout,
    # which checks the copied markdown resources against their bound originals.
    root = Path(__file__).resolve().parents[2]
    snapshot = collect_snapshot(root)
    for name in ("cli", "execution", "experiment", "learning-discovery", "contingencies"):
        assert f"src/oeis_learn/generalization/progressive/contracts/{name}.md" in snapshot
