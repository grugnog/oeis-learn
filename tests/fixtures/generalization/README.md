# Generalization contract fixtures

Run `python3 tests/fixtures/generalization/generate.py` to reproduce the JSON payloads and inventory. The generator uses only standard-library arithmetic and formatting; it never imports project admission, fitting, grouping or selection code.

Authority examples have synthetic hashes and flags solely to exercise JSON shape. They are not actual qualified/admitted records. Mechanical source cases have independent closed-form expected terms. Policy expectations are golden counterexamples for Phase B, not a Phase A implementation of those policies. See `specs/011-controlled-generalization/contracts/algorithm-handoff.md` for ownership and exit criteria.
