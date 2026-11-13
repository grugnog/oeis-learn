"""Independent source syntax and original reference-work measurements.

Nodes/depth cover expression and statement ASTs, with roots at depth one;
program/return wrappers and declarations are excluded. Work counts reference
evaluation steps over the original declared indices, never native time or fuel.
"""

FIELDS = ('depth', 'nodes', 'work')
SCOPE = 'reference-ast-and-declared-index-steps/v1'


def validate(value):
    if set(value) not in (set(FIELDS), {*FIELDS, 'unavailable', 'scope'}):
        raise ValueError('closed difficulty measurement required')
    missing = value.get('unavailable', dict.fromkeys(FIELDS))
    if set(missing) != set(FIELDS) or value.get('scope', SCOPE) != SCOPE:
        raise ValueError('explicit difficulty availability and scope required')
    for key in FIELDS:
        number, reason = value[key], missing[key]
        if number is None:
            if not isinstance(reason, str) or not 1 <= len(reason) <= 256:
                raise ValueError('unavailable difficulty needs an explicit reason')
        elif type(number) is not int or not 0 <= number <= 2**63-1 or reason is not None:
            raise ValueError('nonnegative measured difficulty required')
    return value


def measure(source, *, reference_steps=None):
    values = dict.fromkeys(FIELDS)
    reasons = dict.fromkeys(FIELDS, 'No original source program is available')
    if source is not None:
        from oeis_learn.synthesis.reference_parser import parse
        program = parse(source)
        nodes, depth = 0, 0

        def visit(node, level):
            nonlocal nodes, depth
            nodes += 1
            depth = max(depth, level)
            if isinstance(node, tuple):
                # Literal operands and local names are attributes of their AST
                # node. Index/local reads themselves are scalar AST leaves.
                children = () if node[0] in {'int', 'control', 'get', 'param.get'} else node[1:]
                if node[0] in {'set', 'loop'}:
                    children = node[2:]  # Assignment/counter names are attributes.
                for child in children:
                    if isinstance(child, tuple) or isinstance(child, str) and child.startswith('$'):
                        visit(child, level + 1)

        for root in (*program.statements, program.result):
            visit(root, 1)
        values.update(depth=depth, nodes=nodes, work=reference_steps)
        reasons.update(depth=None, nodes=None,
                       work=None if reference_steps is not None else 'Original reference step measurement was not retained')
    elif reference_steps is not None:
        raise ValueError('work cannot be bound without its original source')
    return validate(dict(values, unavailable=reasons, scope=SCOPE))


def summarize(rows):
    """Keep unavailable members in denominators; never invent a zero band."""
    for row in rows:
        validate(row['difficulty'])
    result = {}
    for key in FIELDS:
        numbers = [r['difficulty'][key] for r in rows if r['difficulty'][key] is not None]
        result[key] = dict(measured=len(numbers), unavailable=len(rows)-len(numbers),
                           minimum=min(numbers) if numbers else None,
                           maximum=max(numbers) if numbers else None)
    return result
