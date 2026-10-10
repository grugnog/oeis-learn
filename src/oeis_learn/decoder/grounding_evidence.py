"""Portable scoped grounding replay. Never asserts all-index generalization."""

from oeis_learn.sandbox.pipeline import digest_bytes


def unsat_statement(source, terms, bound, semantics):
    return {
        "version": "grounding-unsat/v1",
        "source": source,
        "source_sha256": digest_bytes(source.encode()),
        "terms": [str(t) for t in terms],
        "fit_indices": list(range(len(terms))),
        "visible_prompt_sha256": digest_bytes(repr(tuple(terms)).encode()),
        "coefficient_bound": bound,
        "numeric_semantics": semantics,
        "scope": "complete_bounded_visible_fit",
    }


def check_unsat(statement):
    """Reconstruct complete supported constraints; solver replay is scoped evidence."""
    import z3
    from oeis_learn.decoder.grounding import instructions, translate
    from oeis_learn.decoder.wat_grammar import tokenize_wat
    from oeis_learn.decoder.recurrence import recognize, lag_system

    try:
        terms = tuple(int(t) for t in statement["terms"])
        source, bound, semantics = (
            statement[k] for k in ("source", "coefficient_bound", "numeric_semantics")
        )
        if statement != unsat_statement(source, terms, bound, semantics):
            return False
        if not 1 <= len(terms) <= 20 or type(bound) is not int or not 0 <= bound <= 1000:
            return False
        if any(not -(1 << 255) <= t < (1 << 255) for t in terms):
            return False
        count = sum(w in ("i64.const_?", "i256.const_?") for w in tokenize_wat(source))
        if count > 8:
            return False
        rec = recognize(source)
        if rec is not None:
            if len(terms) < 3 or tuple(terms[:2]) != rec:
                return False
            matrix, rhs = lag_system(terms)
            cs = [z3.Int(f"c{i}") for i in range(2)]
            constraints = [sum(a * c for a, c in zip(row, cs)) == b for row, b in zip(matrix, rhs)]
            legacy = False
        elif count == 0:
            from oeis_learn.decoder.grounding_reference import scalar_value
            from oeis_learn.sandbox.reference import execute_term
            from oeis_learn.sandbox.wat_ast import parse

            try:
                _, _, legacy = instructions(source)
            except ValueError:
                legacy = False
            if legacy:
                return semantics == "modular_i64" and any(
                    scalar_value(source, i) != t for i, t in enumerate(terms)
                )
            values = [execute_term(parse(source), i) for i in range(len(terms))]
            return (
                semantics == "checked_i256"
                and all(v.outcome is None for v in values)
                and any(v.value != t for v, t in zip(values, terms))
            )
        else:
            _, _, legacy = instructions(source)
            cs = [z3.BitVec(f"c{i}", 64) if legacy else z3.Int(f"c{i}") for i in range(count)]
            constraints = []
            if legacy and any(not -(1 << 63) <= t < (1 << 63) for t in terms):
                return semantics == "modular_i64"
            for i, t in enumerate(terms):
                expr, _, guards, _ = translate(source, i, cs)
                constraints.extend(guards)
                constraints.append(expr == t)
        if semantics != ("modular_i64" if legacy else "checked_i256"):
            return False
        solver = z3.Solver()
        solver.set(timeout=1000)
        solver.add(*constraints, *[g for c in cs for g in (c >= -bound, c <= bound)])
        return solver.check() == z3.unsat
    except (ValueError, KeyError, TypeError, IndexError, z3.Z3Exception):
        return False


def check_grounding_record(record):
    try:
        if record["outcome"] == "proved_unsat_in_scope":
            statement = record["unsat_evidence"]
            return (
                statement["source"] == record["source"]
                and statement["visible_prompt_sha256"] == record["visible_prompt_sha256"]
                and statement["fit_indices"] == record["fit_indices"]
                and statement["coefficient_bound"] == record["coefficient_bound"]
                and statement["numeric_semantics"] == record["numeric_semantics"]
                and check_unsat(statement)
            )
        if record["outcome"] != "verified_solution":
            return False
        evidence = record["execution"]
        source = record["source"]
        indices = record["fit_indices"]
        terms = tuple(int(t) for t in evidence["fit_terms"])
        if indices != list(range(len(terms))) or evidence["final_source_sha256"] != digest_bytes(
            source.encode()
        ):
            return False
        if record["visible_prompt_sha256"] != digest_bytes(repr(terms).encode()):
            return False
        from oeis_learn.decoder.grounding_reference import scalar_value
        from oeis_learn.decoder.recurrence import recognize_grounded, reference
        from oeis_learn.sandbox.pipeline import Runtime

        rec = recognize_grounded(source)
        if rec is not None:
            initial, coeffs = rec
            from oeis_learn.sandbox.runner import WasmRunner

            production = WasmRunner(fuel_budget=1000000).run_single(
                source, terms_to_generate=len(terms), result_profile="i256x4_v1"
            )
            return (
                record["numeric_semantics"] == "checked_i256"
                and production.status == "SUCCESS"
                and production.output == list(terms)
                and reference(initial, coeffs, len(terms)) == list(terms)
                and evidence["production"] == evidence["reference"] == [str(t) for t in terms]
            )
        if record["numeric_semantics"] == "modular_i64":
            from oeis_learn.sandbox.runner import WasmRunner

            production = WasmRunner(fuel_budget=1000000).run_single(
                source, terms_to_generate=len(terms), result_profile="i64_scalar_v1"
            )
            if production.status != "SUCCESS" or production.output != list(terms):
                return False
            values = [str(scalar_value(source, i)) for i in indices]
        else:
            run = Runtime().evaluate(source, indices)
            if run.outcome is not None or run.outputs != run.reference_outputs:
                return False
            values = run.outputs
        return (
            values == [str(t) for t in terms]
            and evidence["production"] == evidence["reference"] == values
        )
    except (ValueError, KeyError, TypeError, OverflowError):
        return False


def qualified_result(result):
    from oeis_learn.experiments.qualified_models import seal_record

    evidence = result.execution
    if result.outcome == "verified_solution":
        evidence = {**evidence, "final_source_sha256": digest_bytes(result.source.encode())}
    return seal_record(
        "grounding_result",
        source=result.source,
        parameter_values=[str(c) for c in result.constants],
        fit_indices=list(result.fit_indices),
        visible_prompt_sha256=result.visible_prompt_sha256,
        solver_method=result.method,
        solver_version=result.solver_version,
        coefficient_bound=result.bound,
        elapsed={
            "value": round(result.elapsed_ms * 1000000),
            "unit": "ns",
            "source": "monotonic clock",
        },
        outcome=result.outcome,
        reason=result.reason,
        numeric_semantics=result.semantics if result.semantics != "not_run" else "checked_i256",
        execution=evidence if result.outcome == "verified_solution" else None,
        unsat_evidence=result.unsat_evidence,
    )
