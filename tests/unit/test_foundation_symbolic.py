from copy import deepcopy
import json
from types import SimpleNamespace
import pytest
from oeis_learn.data.models import LatentDiscoveryCandidate
from oeis_learn.data.symbolic_definitions import SymbolicDefinitionRegistry
from oeis_learn.discovery.symbolic_prover import SymbolicProver
from oeis_learn.discovery.formula_checker import check_certificate
from oeis_learn.discovery.formula_syntax import parse_formula


def test_formula_identity_has_independent_certificate_and_external_assumption():
    status, evidence = SymbolicProver()._run(["n*(n+1)/2", "(n*n+n)/2"], [1, -1])
    assert status == "FORMULA_IDENTITY"
    assert evidence["independently_checked"] and check_certificate(evidence["certificate"])
    assert evidence["assumptions"]
    mutated = deepcopy(evidence["certificate"])
    mutated["coefficients"] = [1, 1]
    assert not check_certificate(mutated)


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


@pytest.mark.parametrize("source", [
    "__import__('os').system('true')", "sin(n)", "m+n", "n**100", "1.0", "n.__class__",
])
def test_allowlist(source):
    with pytest.raises(ValueError):
        parse_formula(source)


def test_both_public_prover_apis_are_scoped():
    prover = SymbolicProver()
    candidate = LatentDiscoveryCandidate("test", "linear", ("A000027", "A000027"), 0.0, [1, -1])
    assert prover.prove_relation(candidate, ["n", "n"]).status == "FORMULA_IDENTITY"
    registry = SymbolicDefinitionRegistry()
    relation = SimpleNamespace(
        operands=[SimpleNamespace(oeis_id="A000027", index_scale=1, index_shift=0),
                  SimpleNamespace(oeis_id="A000027", index_scale=1, index_shift=1)],
        coefficients=["1", "-1"], canonical_expression="n-(n+1)",
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
    candidate = LatentDiscoveryCandidate("test", "linear", ("A000027",), 0.0, [1],
                                         symbolic_proof="Q.E.D.", status="PROVEN")
    report = export_discovered_proofs_markdown([candidate], str(tmp_path / "report.md"))
    assert "UNVERIFIED_LEGACY_LABEL" in report and "Q.E.D." not in report
