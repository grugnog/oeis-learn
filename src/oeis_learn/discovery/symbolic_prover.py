"""Symbolic Proof Generator and Computer Algebra Verifier using SymPy."""

from __future__ import annotations

import logging
from typing import Optional, Sequence
import sympy as sp
from oeis_learn.data.models import LatentDiscoveryCandidate

logger = logging.getLogger(__name__)


class SymbolicProver:
    """Proves conjectured and PSLQ-verified sequence identities symbolically via SymPy."""

    def prove_relation(
        self,
        candidate: LatentDiscoveryCandidate,
        formulas: Optional[Sequence[str]] = None,
    ) -> LatentDiscoveryCandidate:
        """Constructs and checks formal symbolic proof for a relation vector sum_i c_i * a_i(n) = 0."""
        if not candidate.pslq_vector or len(candidate.pslq_vector) != len(candidate.sequences):
            candidate.status = "REJECTED"
            return candidate

        n = sp.Symbol("n", integer=True)
        coeffs = candidate.pslq_vector
        seq_ids = candidate.sequences

        # If explicit formulas provided, verify sum_i c_i * f_i(n) == 0
        if formulas and len(formulas) == len(seq_ids):
            try:
                sym_terms = []
                for f_str in formulas:
                    # Clean formula
                    clean_f = f_str.replace("a(n) =", "").replace("^", "**").strip()
                    expr = sp.sympify(clean_f)
                    sym_terms.append(expr)

                total_expr = sum(c * expr for c, expr in zip(coeffs, sym_terms))
                simplified = sp.simplify(total_expr)

                if simplified == 0:
                    proof_script = (
                        f"Theorem: For sequences {', '.join(seq_ids)},\n"
                        f"Identity holds: {' + '.join(f'({c})*{s}' for c, s in zip(coeffs, seq_ids))} = 0\n"
                        f"Symbolic proof: sum_i c_i * f_i(n) = {sp.expand(total_expr)} = 0 identically (Q.E.D.)."
                    )
                    candidate.symbolic_proof = proof_script
                    candidate.status = "PROVEN"
                    return candidate
            except Exception as e:
                logger.warning(f"Symbolic verification exception: {e}")

        # When symbolic proof cannot be established, retain numerical conjecture status
        candidate.status = "NUMERICALLY_VERIFIED_CONJECTURE"
        candidate.symbolic_proof = None
        return candidate

    def prove_canonical_relation(
        self,
        relation: Any,
        registry: Any,
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Proves a CanonicalRelation using verified closed-form definitions from registry."""
        import datetime
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        definition_ids = []
        definition_hashes = []
        expr_terms = []

        n = sp.Symbol("n", integer=True)

        for op in relation.operands:
            entry = registry.get_definition(op.oeis_id) if hasattr(registry, "get_definition") else None
            if not entry:
                return "MISSING_DEFINITION", {
                    "outcome": "MISSING_DEFINITION",
                    "definition_ids": [],
                    "definition_hashes": [],
                    "normalized_identity": relation.canonical_expression,
                    "proof_method": "SYMPY_SIMPLIFY",
                    "reduced_expression": "None",
                    "domain": {"integer_only": True, "lower_bound": 0, "upper_bound": None},
                    "verifier_version": f"sympy-{sp.__version__}",
                    "verified_at": now_utc,
                    "diagnostic": f"Missing definition for operand {op.oeis_id}",
                }

            definition_ids.append(entry["definition_id"])
            definition_hashes.append(entry["source"]["content_sha256"])

            # Substitute scaled/shifted n if needed
            base_expr = sp.sympify(entry["expression"])
            if op.index_scale != 1 or op.index_shift != 0:
                sub_n = op.index_scale * n + op.index_shift
                base_expr = base_expr.subs(n, sub_n)
            expr_terms.append(base_expr)

        int_coeffs = [int(c) for c in relation.coefficients]
        total_expr = sum(c * t for c, t in zip(int_coeffs, expr_terms))
        simplified = sp.simplify(total_expr)

        outcome = "PROVEN" if simplified == 0 else "COUNTEREXAMPLE"
        evidence = {
            "outcome": outcome,
            "definition_ids": definition_ids,
            "definition_hashes": definition_hashes,
            "normalized_identity": relation.canonical_expression,
            "proof_method": "SYMPY_SIMPLIFY",
            "reduced_expression": str(simplified),
            "domain": {"integer_only": True, "lower_bound": 0, "upper_bound": None},
            "verifier_version": f"sympy-{sp.__version__}",
            "verified_at": now_utc,
            "diagnostic": None if simplified == 0 else f"Non-zero reduction: {simplified}",
        }
        return outcome, evidence
