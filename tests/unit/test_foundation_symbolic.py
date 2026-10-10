import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from oeis_learn.data.models import LatentDiscoveryCandidate
from oeis_learn.data.symbolic_definitions import SymbolicDefinitionRegistry
from oeis_learn.discovery.formula_checker import check_certificate
from oeis_learn.discovery.formula_syntax import parse_formula
from oeis_learn.discovery.symbolic_prover import SymbolicProver


def test_formula_identity_has_independent_certificate_and_external_assumption():
    status, evidence = SymbolicProver()._run(["n*(n+1)/2", "(n*n+n)/2"], [1, -1])
    assert status == "FORMULA_IDENTITY"
    assert evidence["independently_checked"] and check_certificate(evidence["certificate"])
    assert evidence["assumptions"]
    mutated = deepcopy(evidence["certificate"])
    mutated["coefficients"] = [1, 1]
    assert not check_certificate(mutated)


def test_exact_certificate_proposal_does_not_import_sympy(monkeypatch):
    import builtins

    from oeis_learn.discovery.symbolic_prover import _proof

    original_import = builtins.__import__

    def reject_sympy(name, *args, **kwargs):
        if name == "sympy" or name.startswith("sympy."):
            raise AssertionError("contained proof must not depend on a cold CAS import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_sympy)
    status, evidence = _proof(
        ["n*(n+1)/2", "(n*n+n)/2"],
        [1, -1],
        [(1, 0), (1, 0)],
        [{"integer_only": True, "lower_bound": 0, "upper_bound": None}] * 2,
        ["Formula-to-OEIS correspondence is an external assumption."],
    )
    assert status == "FORMULA_IDENTITY"
    assert evidence["backend_version"] == "python-fractions/v1"
    assert check_certificate(evidence["certificate"])


def test_everywhere_undefined_expression_cannot_prove_identity():
    status, evidence = SymbolicProver()._run(["1/(n-n)", "1/(n-n)"], [1, -1])
    assert status == "UNKNOWN"
    assert evidence["independently_checked"] is False
    assert "certificate" not in evidence
    domain = [{"integer_only": True, "lower_bound": 0, "upper_bound": 3}] * 2
    assert (
        SymbolicProver()._run(["1/(n-n)", "1/(n-n)"], [1, -1], domains=domain)[0] == "EMPTY_DOMAIN"
    )


def test_original_pole_survives_cancellation():
    status, evidence = SymbolicProver()._run(["(n-1)/(n-1)", "1"], [1, -1])
    assert status == "FORMULA_IDENTITY"
    assert evidence["domain"]["excluded_poles"]
    assert 1 not in evidence["certificate"]["points"]


def test_false_shift_has_exact_witness():
    status, evidence = SymbolicProver()._run(["n", "n"], [1, -1], [(1, 0), (1, 1)])
    assert status == "COUNTEREXAMPLE"
    assert evidence["witness"] == {"index": 0, "values": ["0", "1"], "residual": "-1"}


def test_negative_scale_and_constant_index_domains():
    status, evidence = SymbolicProver()._run(["n", "2-n"], [1, -1], [(-1, 2), (1, 0)])
    assert status == "FORMULA_IDENTITY"
    assert evidence["domain"]["lower_bound"] == 0
    assert evidence["domain"]["upper_bound"] == 2
    assert SymbolicProver()._run(["n"], [1], [(0, -1)])[0] == "EMPTY_DOMAIN"


def test_unknown_without_concrete_witness_in_domain():
    domain = [{"integer_only": True, "lower_bound": 0, "upper_bound": 0}]
    assert SymbolicProver()._run(["n"], [1], domains=domain)[0] == "UNKNOWN"


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('true')",
        "sin(n)",
        "m+n",
        "n**100",
        "1.0",
        "n.__class__",
    ],
)
def test_allowlist(source):
    with pytest.raises(ValueError):
        parse_formula(source)


def test_both_public_prover_apis_are_scoped():
    prover = SymbolicProver()
    candidate = LatentDiscoveryCandidate("test", "linear", ("A000027", "A000027"), 0.0, [1, -1])
    assert prover.prove_relation(candidate, ["n", "n"]).status == "FORMULA_IDENTITY"
    registry = SymbolicDefinitionRegistry()
    relation = SimpleNamespace(
        operands=[
            SimpleNamespace(oeis_id="A000027", index_scale=1, index_shift=0),
            SimpleNamespace(oeis_id="A000027", index_scale=1, index_shift=1),
        ],
        coefficients=["1", "-1"],
        canonical_expression="n-(n+1)",
    )
    assert prover.prove_canonical_relation(relation, registry)[0] == "COUNTEREXAMPLE"
    entry = registry.get_definition("A000027")
    entry["expression"] = "99"
    assert registry.get_definition("A000027")["expression"] == "n"


def test_registry_hash_is_checked(tmp_path):
    from pathlib import Path

    data = json.loads(Path("data/benchmarks/symbolic_definitions_v1.json").read_text())
    data["definitions"][0]["expression"] = "99"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        SymbolicDefinitionRegistry(path)


def test_legacy_report_is_not_a_proof(tmp_path):
    from oeis_learn.cli.reporting import export_discovered_proofs_markdown

    candidate = LatentDiscoveryCandidate(
        "test", "linear", ("A000027",), 0.0, [1], symbolic_proof="Q.E.D.", status="PROVEN"
    )
    report = export_discovered_proofs_markdown([candidate], str(tmp_path / "report.md"))
    assert "UNVERIFIED_LEGACY_LABEL" in report and "Q.E.D." not in report


def test_reporting_replays_scope_and_marks_legacy_labels(tmp_path):
    import copy
    import json

    from oeis_learn.cli.reporting import (
        export_discovered_proofs_markdown,
        project_discovery_markdown,
    )
    from oeis_learn.data.models import LatentDiscoveryCandidate

    evidence = SymbolicProver()._run(["(n-1)/(n-1)", "1"], [1, -1])[1]
    claim = {
        "status": "FORMULA_IDENTITY",
        "relation": {"canonical_expression": "identity", "claim_id": "x"},
        "symbolic_evidence": evidence,
    }
    report = {"claims": [claim]}
    rendered = project_discovery_markdown(report)
    assert "external assumption" in rendered and "excluded_poles" in rendered
    bad = copy.deepcopy(report)
    bad["claims"][0]["symbolic_evidence"]["domain"]["lower_bound"] = 3
    assert "Checked Conditional Formula Identities**: `0`" in project_discovery_markdown(bad)
    candidate = LatentDiscoveryCandidate(
        candidate_id="c",
        relation_type="LINEAR_SUM",
        sequences=("A000001", "A000002"),
        vector_distance=0,
        pslq_vector=[1, -1],
        status="FORMULA_IDENTITY",
        symbolic_proof=json.dumps(bad["claims"][0]["symbolic_evidence"]),
    )
    assert "UNVERIFIED_EVIDENCE" in export_discovered_proofs_markdown(
        [candidate], str(tmp_path / "proofs.md")
    )
    candidate.status = "PROVEN"
    assert "UNVERIFIED_LEGACY_LABEL" in export_discovered_proofs_markdown(
        [candidate], str(tmp_path / "legacy.md")
    )
