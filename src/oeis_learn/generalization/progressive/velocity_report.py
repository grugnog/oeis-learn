"""Public sufficient statistics for a round029 paired comparison."""

import hmac


def public_family_cells(result, *, key):
    """Use the same private study key to link families across looks, seeds and strata."""
    if not isinstance(key, bytes) or not key:
        raise ValueError("private study key required for opaque family labels")
    coordinates = [{k: c[k] for k in ("registration", "method", "seed", "look", "profile", "input_mode")}
                   for c in result["coordinates"]]
    families = []
    for family in result["families"]:
        label = hmac.digest(key, family["family"].encode("utf-8"), "sha256").hex()
        patterns = [
            dict(count=p["count"], outcomes=[
                {k: o[k] for k in ("status", "complete", "closed", "equality")}
                for o in p["outcomes"]])
            for p in family["patterns"]
        ]
        families.append(dict(family=label, stratum=family["stratum"],
                             frozen_size=family["frozen_size"], patterns=patterns))
    families.sort(key=lambda f: (f["stratum"], f["family"]))
    return dict(coordinates=coordinates, families=families, denominator=result["denominator"])
