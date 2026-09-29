"""Contract test validating Constant Solver schema conformance against contracts/constant-solver.contract.json."""

from __future__ import annotations

import json
import os
import pytest
from oeis_learn.data.models import ASTSkeleton, ConstantSolverResult
from oeis_learn.decoder.constant_solver import (
    parse_ast_placeholders,
    solve_linear_diophantine,
    solve_smt_constants,
    splice_constants_into_wat,
)
from oeis_learn.sandbox.runner import WasmRunner


def test_constant_solver_schema_contract():
    contract_path = "specs/004-decoupled-grounding-and-symple-engine/contracts/constant-solver.contract.json"
    assert os.path.exists(contract_path), f"Contract file missing: {contract_path}"

    with open(contract_path, "r") as f:
        schema = json.load(f)

    assert schema["title"] == "ConstantSolverContract"
    assert "SolverRequest" in schema["definitions"]
    assert "SolverResponse" in schema["definitions"]

    # Verify field types
    resp_props = schema["definitions"]["SolverResponse"]["properties"]
    assert "is_sat" in resp_props
    assert "solver_type" in resp_props
    assert "constants" in resp_props
    assert "grounded_wat" in resp_props
    assert "solve_duration_ms" in resp_props


def test_constant_solver_result_dataclass_contract():
    res = ConstantSolverResult(
        solver_type="DIOPHANTINE_HNF",
        constants=[2, 5],
        solve_duration_ms=0.5,
        is_sat=True,
        grounded_wat="(module (func (export \"compute\") (param $n i32) (result i64) (i64.add (i64.mul (i64.extend_i32_u (local.get $n)) (i64.const 5)) (i64.const 2))))",
        error_message=None,
    )
    d = res.to_dict()
    assert d["is_sat"] is True
    assert d["solver_type"] == "DIOPHANTINE_HNF"
    assert d["constants"] == [2, 5]
    assert d["solve_duration_ms"] == 0.5
    assert d["error_message"] is None
