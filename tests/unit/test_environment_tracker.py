"""Unit tests for Environment Tracker and No-Ghost Soundness."""

from oeis_learn.decoder.environment_tracker import EnvironmentTracker
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID


def test_environment_tracker_no_ghost_soundness():
    tracker = EnvironmentTracker()
    tracker.reset()

    # Feed module and func header declaring $n and $temp
    tokens = ["(", "module", "(", "func", "(", "export", '"compute"', ")",
              "(", "param", "$n", "i32", ")", "(", "result", "i64", ")",
              "(", "local", "$temp", "i64", ")"]

    for tok in tokens:
        tracker.update(tok)

    assert "$n" in tracker.declared_vars
    assert "$temp" in tracker.declared_vars
    assert "$unbound" not in tracker.declared_vars

    # Test variable access instruction: local.get
    tracker.update("local.get")
    valid_ids = tracker.get_valid_next_tokens()

    # Allowed tokens MUST only be declared variables ($n, $temp)
    assert TOKEN_TO_ID["$n"] in valid_ids
    assert TOKEN_TO_ID["$temp"] in valid_ids
    if "$unbound" in TOKEN_TO_ID:
        assert TOKEN_TO_ID["$unbound"] not in valid_ids


def test_environment_tracker_paren_nesting():
    tracker = EnvironmentTracker()
    tracker.reset()

    assert tracker.paren_depth == 0
    tracker.update("(")
    assert tracker.paren_depth == 1
    tracker.update("module")
    tracker.update("(")
    assert tracker.paren_depth == 2
    tracker.update(")")
    assert tracker.paren_depth == 1
    tracker.update(")")
    assert tracker.paren_depth == 0


def test_environment_tracker_mandatory_header_sequencing():
    tracker = EnvironmentTracker()
    tracker.reset()

    # Step 1: Start module
    tracker.update("(")
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["module"] in valid

    tracker.update("module")
    tracker.update("(")
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["func"] in valid

    tracker.update("func")
    tracker.update("(")
    valid = tracker.get_valid_next_tokens()
    # Must declare export next!
    assert TOKEN_TO_ID["export"] in valid
    assert TOKEN_TO_ID["i64.add"] not in valid

    tracker.update("export")
    tracker.update('"compute"')
    tracker.update(")")

    # Must declare param next!
    tracker.update("(")
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["param"] in valid

    tracker.update("param")


def test_environment_tracker_min_locals_enforcement():
    tracker = EnvironmentTracker(min_locals=3)
    tracker.reset()

    header = ["(", "module", "(", "func", "(", "export", '"compute"', ")", "(", "param", "$n", "i32", ")", "(", "result", "i64", ")"]
    for tok in header:
        tracker.update(tok)

    # With min_locals=3, body instructions cannot start yet
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["("] in valid
    assert TOKEN_TO_ID["local.get"] not in valid
    assert TOKEN_TO_ID["i64.const"] not in valid

    # Declare local 1: $a
    for tok in ["(", "local", "$a", "i64", ")"]:
        tracker.update(tok)

    # 1 local declared, still need 2 more
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["local.get"] not in valid

    # Declare local 2: $b
    for tok in ["(", "local", "$b", "i64", ")"]:
        tracker.update(tok)

    # Declare local 3: $temp
    for tok in ["(", "local", "$temp", "i64", ")"]:
        tracker.update(tok)

    # Now 3 locals declared, body instructions become valid
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["local.get"] in valid
    assert TOKEN_TO_ID["i64.const"] in valid
    assert "$a" in tracker.declared_vars
    assert "$b" in tracker.declared_vars
    assert "$temp" in tracker.declared_vars


def test_environment_tracker_stack_type_soundness():
    tracker = EnvironmentTracker()
    tracker.reset()

    # Feed header
    for tok in ["(", "module", "(", "func", "(", "export", '"compute"', ")",
                "(", "param", "$n", "i32", ")", "(", "result", "i64", ")"]:
        tracker.update(tok)

    # Empty stack: i64.add should NOT be valid yet
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["i64.add"] not in valid

    # Push first constant
    tracker.update("i64.const")
    tracker.update("10")
    assert tracker.stack_depth == 1
    assert tracker.operand_stack == ["i64"]

    # Still only 1 i64 on stack: binary add not valid
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["i64.add"] not in valid

    # Push second constant
    tracker.update("i64.const")
    tracker.update("20")
    assert tracker.stack_depth == 2
    assert tracker.operand_stack == ["i64", "i64"]

    # Now binary add IS valid!
    valid = tracker.get_valid_next_tokens()
    assert TOKEN_TO_ID["i64.add"] in valid

    # Execute i64.add
    tracker.update("i64.add")
    assert tracker.stack_depth == 1
    assert tracker.operand_stack == ["i64"]
