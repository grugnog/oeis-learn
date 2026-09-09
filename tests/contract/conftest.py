"""Shared contract test fixtures and JSON schema validators with referencing support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict
import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

CONTRACTS_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "specs"
    / "005-trustworthy-synthesis-readiness"
    / "contracts"
)

CONTRACTS_006_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "specs"
    / "006-multilimb-curriculum-scaling"
    / "contracts"
)


@pytest.fixture(scope="session")
def contracts_dir() -> Path:
    return CONTRACTS_DIR


@pytest.fixture(scope="session")
def contracts_006_dir() -> Path:
    return CONTRACTS_006_DIR


@pytest.fixture(scope="session")
def schema_registry(contracts_dir: Path, contracts_006_dir: Path) -> Registry:
    registry = Registry()
    for search_dir in [contracts_dir, contracts_006_dir]:
        if search_dir.exists():
            for schema_file in search_dir.glob("*.schema.json"):
                with open(schema_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                resource = Resource.from_contents(data)
                if "$id" in data:
                    registry = registry.with_resource(data["$id"], resource)
                registry = registry.with_resource(schema_file.name, resource)
                registry = registry.with_resource(str(schema_file), resource)
    return registry


@pytest.fixture(scope="session")
def load_schema(contracts_dir: Path, contracts_006_dir: Path) -> Callable[[str], Dict[str, Any]]:
    def _loader(name: str) -> Dict[str, Any]:
        if not name.endswith(".schema.json"):
            name = f"{name}.schema.json"
        path = contracts_dir / name
        if not path.exists():
            path = contracts_006_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Schema not found at {path} (checked {contracts_dir} and {contracts_006_dir})")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    return _loader


@pytest.fixture(scope="session")
def multi_limb_preamble_wat(contracts_006_dir: Path) -> str:
    path = contracts_006_dir / "multi-limb-preamble.wat"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="session")
def macro_instruction_set_md(contracts_006_dir: Path) -> str:
    path = contracts_006_dir / "macro-instruction-set.md"
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


@pytest.fixture(scope="session")
def get_validator(
    load_schema: Callable[[str], Dict[str, Any]], schema_registry: Registry
) -> Callable[[str], Draft202012Validator]:
    def _validator_factory(name: str) -> Draft202012Validator:
        schema_data = load_schema(name)
        return Draft202012Validator(schema_data, registry=schema_registry)

    return _validator_factory


@pytest.fixture(scope="session")
def validate_contract(
    get_validator: Callable[[str], Draft202012Validator]
) -> Callable[[Dict[str, Any], str], None]:
    def _validate(instance: Dict[str, Any], schema_name: str) -> None:
        validator = get_validator(schema_name)
        validator.validate(instance)

    return _validate
