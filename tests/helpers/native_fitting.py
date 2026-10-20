"""Independent fixed equation/source fixtures; no fitting implementation."""
from oeis_learn.synthesis.fitting import fit_program


def fit(client, source, terms, **kwargs):
    return fit_program(source, terms, client=client, profile='bigint-body/v1', **kwargs)


def equations(matrix):
    def row(coefficients):
        parts = [f'(mul (int {a}) (param.get p{i}))' for i,a in enumerate(coefficients)]
        value = parts[0]
        for term in parts[1:]: value = f'(add {value} {term})'
        return value
    value = row(matrix[-1])
    for i in reversed(range(len(matrix)-1)):
        value = f'(select (eq $index (int {i})) {row(matrix[i])} {value})'
    return f'(program (return {value}))'
