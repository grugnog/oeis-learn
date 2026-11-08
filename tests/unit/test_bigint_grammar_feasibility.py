import random

from oeis_learn.decoder.bigint_grammar import Reader, support


def test_nested_statement_cannot_enter_a_body_with_no_completable_statement():
    prefix = ['(', 'program'] + ['(', 'if', '(', 'true', ')', '(', 'then'] * 15
    assert support(prefix) == {')'}


def test_depth_limit_keeps_bigint_leaves_and_filters_required_nested_children():
    reader = Reader([])
    reader.depth = 32
    reader.locals = {'$b0': 'bigint', '$c0': 'control', '$q0': 'bool'}
    assert reader.statement_ops() == {'set'}
    assert reader.writable() == {'$b0'}
    reader.depth = 31
    assert 'loop' in reader.statement_ops()
    del reader.locals['$b0']
    assert 'loop' not in reader.statement_ops()


def test_many_legal_prefix_walks_never_reach_a_grammar_dead_end():
    # Exercise declarations, protected loop counters and all typed operators.
    # Token-budget exhaustion remains a separate termination condition.
    for seed in range(40):
        rng = random.Random(seed)
        tokens = []
        for _ in range(300):
            choices = sorted(support(tokens))
            assert choices, (seed, tokens)
            token = rng.choice(choices)
            tokens.append(token)
            if token == '<eos>':
                Reader(tokens).program()
                break
