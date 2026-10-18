"""One-shot independent expectation generator for spec 008.

Uses only Python standard-library integers/math, never production parsing,
lowering or arithmetic. Regeneration changes qualification fixture identities.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "tests/fixtures/bigint"


def write(name, value):
    (DEST / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def freeze():
    cases = [{"id": "factorial-0-99", "operation": "factorial", "indices": list(map(str, range(100))),
              "expected": [str(math.factorial(n)) for n in range(100)]}]
    for bits in (256, 512, 1024, 4096):
        x = 2**(bits - 1)
        cases.append({"id": f"neighbors-{bits}", "operation": "compare",
                      "values": list(map(str, (-x - 1, -x, -x + 1, x - 1, x, x + 1))),
                      "adjacent_comparisons": [-1] * 5})
        cases.append({"id": f"difference-{bits}-plus-one", "operation": "subtract",
                      "operands": [str(2**bits - 1), str(-(2**bits - 1))],
                      "expected": str(2**(bits + 1) - 2)})
    x, y = 2**512 + 17, -(2**300 + 9)
    cases.extend([
        {"id": "two-large-operands", "operation": "multiply", "operands": [str(x), str(y)], "expected": str(x*y)},
        {"id": "large-cancellation", "operation": "multiply_then_subtract",
         "operands": [str(x), str(x), str(x*x - 7)], "intermediate": str(x*x), "expected": "7",
         "with_intermediate_bits_1024": "integer_limit"},
        {"id": "zero-product", "operation": "multiply", "operands": [str(2**4096 - 1), "0"], "expected": "0"},
        {"id": "borderline-product", "operation": "multiply", "operands": [str(2**255), "1"],
         "expected": str(2**255), "magnitude_bits": 256},
        {"id": "actual-indices", "operation": "affine", "indices": ["-7", "-3", "2", "11"],
         "coefficient": "3", "offset": "-5", "expected": ["-26", "-14", "1", "28"]},
    ])
    write("arithmetic.json", {"version": "bigint-fixtures/v1", "cases": cases})
    write("signed-256.json", {"version": "signed-256-control-fixtures/v1", "profile": "i256x4_v1", "cases": [
        {"id": "max-signed", "source": f"i256.const {2**255-1}", "expected": str(2**255-1)},
        {"id": "min-signed", "source": f"i256.const {-2**255}", "expected": str(-2**255)},
        {"id": "checked-overflow", "source": f"i256.const {2**255-1} i256.const 1 i256.add", "failure": "arithmetic_overflow"},
        {"id": "negative-limbs", "limbs": [-1, -1, -1, -1], "expected": "-1"},
        {"id": "wide-limb", "limbs": [0, 1, 0, 0], "expected": str(2**64)},
    ]})
    write("false-fits.json", {"version": "false-fit-fixtures/v1", "cases": [
        {"id": "square-16-versus-4", "structure": "p0*p0", "provisional_assignment": {"p0": "4"},
         "production_output": "16", "visible_value": "4", "expected_admission": False,
         "valid_assignments": [{"p0": "-2"}, {"p0": "2"}],
         "regression": "tests/unit/test_foundation_grounding.py::test_nonlinear_parameter_local_square"},
        {"id": "integer-free-variable", "equation": "2*p0+p1=1", "assignment": {"p0": "0", "p1": "1"},
         "expected": "integer-feasible", "note": "Do not reject after choosing the free variable p1=0."},
        {"id": "nonintegral-unique", "equation": "2*p0=1", "expected": "unsat_in_scope"},
        {"id": "modular-singular-pivot", "equation": "2305843009213693951*p0=16140901064495857657", "assignment": {"p0": "7"}},
        {"id": "large-coefficient", "equation": "p0*n+p1", "assignment": {"p0": str(2**300+1), "p1": str(-2**100-13)}}
    ]})
    fixtures = []
    for name, origin in [
        ("arithmetic.json", "Python math.factorial, exact integer exponentiation/add/sub/mul; no project imports"),
        ("signed-256.json", "Independent two's-complement bounds/limb arithmetic; old profile identity retained"),
        ("false-fits.json", "Hand-derived equations and existing historical regression locations"),
        ("request.json", "Phase 1 disabled request shape"),
        ("record-stubs.json", "Phase 1 disabled record shape catalogue"),
    ]:
        raw = (DEST / name).read_bytes()
        fixtures.append({"path": name, "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "provenance": origin})
    write("manifest.json", {"version": "bigint-fixture-manifest/v1", "baseline": "b96b92ff680a1509d3cede89e9abf0ddb65d2665",
                            "generator": "scripts/freeze_bigint_fixtures.py", "qualification": "expectations only; production algorithms not implemented", "fixtures": fixtures})


if __name__ == "__main__":
    freeze()
