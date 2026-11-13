from oeis_learn.generalization.progressive.contracts import digest, seal


def evidence(value=0.3, elapsed=100, completion=True, retention=False):
    from fractions import Fraction

    rate = Fraction(str(value)).limit_denominator(100)
    rows = []
    for stratum in ("composition", "unseen_structure"):
        for g in range(16):
            for i in range(rate.denominator):
                rows.append(
                    dict(
                        task=f"{stratum}/{g}/{i}",
                        group=f"{stratum}/{g}",
                        stratum=stratum,
                        outcome="success"
                        if i < rate.numerator
                        else "failure"
                        if completion
                        else "unresolved",
                    )
                )
    keep = [
        dict(task=str(i), baseline="success", candidate="failure" if retention else "success")
        for i in range(20)
    ]
    return seal(
        dict(
            version="progressive-trial-evidence/v1",
            role="development",
            checkpoint=digest("model"),
            qualification=digest("qualified-scope"),
            scope="DIAGNOSTIC_ONLY",
            roster=[{k: r[k] for k in ("task", "group", "stratum")} for r in rows],
            outcomes=rows,
            retention=keep,
            generator=digest("generator"),
            scorer=digest("scorer"),
            costs=dict(startup=0, training=elapsed, inference=0, search=0, verification=0),
        )
    )
