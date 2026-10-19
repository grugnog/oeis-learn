"""Eight predeclared generic templates; no prompt values or metadata accepted."""

from oeis_learn.synthesis.contracts import identity


def template_bank():
    sources = [
        "(program (return (param.get p0)))",
        "(program (return (add $index (param.get p0))))",
        "(program (return (add (mul (param.get p0) $index) (param.get p1))))",
        "(program (return (add (mul (param.get p0) (mul $index $index)) (param.get p1))))",
        "(program (return (add (mul (param.get p0) (mul $index (mul $index $index))) (param.get p1))))",
        "(program (local $b0 bigint) (local $c0 control) (set $b0 (param.get p0)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (param.get p1)))) (return (get $b0)))",
        "(program (local $b0 bigint) (local $c0 control) (set $b0 (param.get p0)) (loop $c0 (to_control $index) (set $b0 (mul (get $b0) (add (to_bigint (increment (get $c0))) (param.get p1))))) (return (get $b0)))",
        "(program (return (select (lt $index (int 10)) (add $index (param.get p0)) (mul $index (param.get p1)))))",
    ]
    return dict(
        version="generic-template-bank/v1",
        sources=sources,
        sha256=identity(sources),
        priors="constant, offset, affine, quadratic, cubic, carried product, shifted product, threshold composition; no output-selected expansion",
    )
