"""Task-independent bounded transform priors for sealed program proposals."""

from dataclasses import dataclass
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.discovery.pslq_solver import exact_integer

PRIORS = {
    "shift": 1,
    "reindex": 1,
    "difference": 1,
    "sum": 1,
    "product": 1,
    "interleave": 1,
    "convolution": 1,
    "composition": 1,
}


@dataclass(frozen=True)
class Transform:
    op: str
    operands: tuple[str, ...]
    parameters: tuple[int, ...] = ()

    def identity(self):
        return digest(
            {
                "version": "program-transform/v1",
                "op": self.op,
                "operands": self.operands,
                "parameters": self.parameters,
                "priors": PRIORS,
            }
        )

    def evaluate(self, index, programs, *, budget=4096):
        if (
            self.op not in PRIORS
            or not self.operands
            or len(self.operands) > 2
            or len(self.parameters) > 2
            or any(type(v) is not int for v in self.parameters)
            or type(index) is not int
            or not 0 <= index <= 4095
            or not 1 <= budget <= 4096
        ):
            raise ValueError("bounded typed transform required")
        cost = 0

        def read(key, i):
            nonlocal cost
            cost += 1
            if cost > budget:
                raise TimeoutError("transform cost cap")
            if key not in programs or not 0 <= i < len(programs[key]):
                raise ValueError("transform outside supplied indexed domain")
            return exact_integer(programs[key][i])

        a = self.operands[0]
        unary = self.op in ("shift", "reindex", "difference")
        if len(self.operands) != (1 if unary else 2):
            raise ValueError("transform arity")
        b = None if unary else self.operands[1]
        if self.op == "shift" and len(self.parameters) == 1 and abs(self.parameters[0]) <= 120:
            v = read(a, index + self.parameters[0])
        elif (
            self.op == "reindex"
            and len(self.parameters) == 2
            and abs(self.parameters[0]) <= 8
            and abs(self.parameters[1]) <= 120
        ):
            v = read(a, self.parameters[0] * index + self.parameters[1])
        elif self.parameters:
            raise ValueError("transform parameters")
        elif self.op == "difference":
            v = read(a, index + 1) - read(a, index)
        elif self.op == "sum":
            v = read(a, index) + read(b, index)
        elif self.op == "product":
            v = read(a, index) * read(b, index)
        elif self.op == "interleave":
            v = read(a if index % 2 == 0 else b, index // 2)
        elif self.op == "convolution":
            v = sum(read(a, i) * read(b, index - i) for i in range(index + 1))
        elif self.op == "composition":
            v = read(a, read(b, index))
        else:
            raise ValueError("transform parameters")
        if not -(1 << 255) <= v < (1 << 255):
            raise OverflowError("checked transform signed256 output")
        return v, cost


def discover_transforms(
    programs, *, fit_indices, validation_indices, final_indices, max_proposals=64
):
    """Bounded program operator hypotheses sealed before validation/final tests."""
    from itertools import combinations
    from oeis_learn.discovery.novelty import seal_proposal

    parts = [tuple(p) for p in (fit_indices, validation_indices, final_indices)]
    if (
        not 1 <= max_proposals <= 64
        or any(
            not p or len(set(p)) != len(p) or any(type(i) is not int or i < 0 for i in p)
            for p in parts
        )
        or any(set(parts[i]) & set(parts[j]) for i in range(3) for j in range(i))
    ):
        raise ValueError("bounded disjoint transform partitions required")
    ids = sorted(programs)[:128]
    hypotheses = []
    for a in ids:
        hypotheses.extend(
            Transform(op, (a,), params)
            for op, params in [
                ("shift", (1,)),
                ("shift", (-1,)),
                ("reindex", (2, 0)),
                ("difference", ()),
            ]
        )
    for a, b in list(combinations(ids, 2))[:32]:
        hypotheses.extend(
            Transform(op, (a, b))
            for op in ("sum", "product", "interleave", "convolution", "composition")
        )
    retained = []
    attempts = cost = 0
    for transform in hypotheses:
        for target in ids:
            if target in transform.operands:
                continue
            if attempts >= max_proposals:
                return {
                    "claims": retained,
                    "attempts": attempts,
                    "cost": cost,
                    "search_status": "UNKNOWN",
                    "priors": PRIORS,
                }
            attempts += 1
            try:
                fitting = []
                for n in parts[0]:
                    value, charged = transform.evaluate(n, programs)
                    cost += charged
                    fitting.append(value == exact_integer(programs[target][n]))
                if not all(fitting):
                    continue
                proposal = seal_proposal(
                    {
                        "version": "transformed-program-relation/v1",
                        "transform": {
                            "op": transform.op,
                            "operands": transform.operands,
                            "parameters": transform.parameters,
                            "identity": transform.identity(),
                        },
                        "target": target,
                        "partitions": parts,
                        "source_sha256": digest(
                            {
                                k: [str(v) for v in programs[k]]
                                for k in (*transform.operands, target)
                            }
                        ),
                        "coefficients": ["1", "-1"],
                        "priors": PRIORS,
                        "evidence_kind": "empirical_relation",
                    }
                )
                residuals = []
                for n in parts[1] + parts[2]:
                    value, charged = transform.evaluate(n, programs)
                    cost += charged
                    residuals.append(str(value - exact_integer(programs[target][n])))
                retained.append(
                    {
                        "proposal": proposal,
                        "residuals": residuals,
                        "status": "FINITE_VALIDATED"
                        if all(v == "0" for v in residuals)
                        else "COUNTEREXAMPLE",
                        "proof_status": "not_claimed",
                    }
                )
            except (ValueError, OverflowError, TimeoutError, IndexError):
                continue
    return {
        "claims": retained,
        "attempts": attempts,
        "cost": cost,
        "search_status": "COMPLETE",
        "priors": PRIORS,
    }
