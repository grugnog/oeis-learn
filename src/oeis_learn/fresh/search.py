"""Visible-only selection and the single native fitter route."""

def select(attempts, *, verified):
    from .common import digest

    return next(
        (
            digest(r["source"])
            for r in attempts
            if r.get("complete")
            and r.get("source")
            and (not verified or r.get("prefix_fit") is True)
        ),
        None,
    )


def native_variants(source, visible, client, *, deadline, profile="wat_i256_checked_v1"):
    """Single fitter route. Unsupported retained candidates remain unsupported."""
    from oeis_learn.synthesis.admission import VisiblePrompt

    prompt = VisiblePrompt(
        tuple(str(i) for i in range(len(visible))), tuple(str(v) for v in visible)
    )
    return client.fit(
        source, prompt, profile=profile, policy="eligible-literals/v1", deadline=deadline
    )
