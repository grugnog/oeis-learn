"""Independent pre-score grant, operation schedule and anonymized family denominators."""

from collections import Counter
import hmac

from oeis_learn.generalization.progressive import velocity_donation_report as public
from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.velocity_records import operation_key, validate_grant
from .progressive_velocity_donation import retained_qualification
from . import progressive_velocity_conditioning as conditioning

MEMBER = "comparative-prescore.json"


def coordinate(grant, look=None, *, role, ordinal=0, mode="true", stage="exploration"):
    return dict(
        study=grant["identity"], stage=stage, method=look["method"] if look else "wave",
        seed=look["seed"] if look else grant.get("seed", 0),
        look=look["look"] if look else (4_000_000 if role == "publish" else 0),
        panel="development", profile=look["profile"] if look else "full",
        input_mode=mode, role=role, ordinal=ordinal,
    )


def prescore(accepted, design, grant, approval, marker, secret, sample=None):
    graph, looks, crossings, ops = design
    expected = {}
    def add(coords, work):
        key = operation_key(coords, 0)["logical"]
        if key in expected:
            raise ValueError("duplicate prospective operation")
        expected[key] = dict(coordinates=coords, work=work)
    for role, stage in (("prepare", "preparation"), ("publish", "publication")):
        add(dict(coordinate(grant, role=role, stage=stage), seed=accepted["seed"]), {})
    for i, op in enumerate(ops):
        if op["cost"] or op["role"] == "train":
            add(coordinate(grant, looks[op["roster"]], role=op["role"], ordinal=i,
                           stage=op.get("stage", "exploration"), mode=op.get("mode", "true")), op["cost"])
    families = {}
    for profile, series in graph["00"].items():
        counts = Counter((row["stratum"], hmac.digest(secret, row["group"].encode(), "sha256").hex())
                         for row in series[0]["members"])
        families[profile] = [dict(stratum=s, family=f, frozen_size=n) for (s, f), n in sorted(counts.items())]
    paired = {}
    if sample is not None:
        groups = {}
        for op in ops:
            if op.get("stage") == conditioning.STAGE and op["role"] == "search":
                ref = op["roster"]
                look = looks[ref]
                groups.setdefault(look["method"] + "-" + look["profile"], {})[op["mode"]] = ref
        paired["conditioning"] = dict(
            sample=sample["identity"], prefixes=sample["prefixes"], donor_map=digest(sample["donors"]),
            missing=sample["missing"], price=sample["price"], groups=groups,
            families=sorted((dict(stratum=r["stratum"], frozen_size=1,
                                  family=hmac.digest(secret, r["group"].encode(), "sha256").hex())
                             for r in sample["members"]), key=lambda r: (r["stratum"], r["family"])),
        )
    return seal(dict(version="round029-comparative-prescore/v1", preparation=accepted, **paired,
                     authorization=approval, grant=grant, controller=marker, operations=expected,
                     families=families, crossings={m: {str(k): v for k,v in cuts.items()}
                                                  for m,cuts in crossings.items()}))


def install(root, grant, record):
    public.install(retained_qualification(root, grant), MEMBER, record)


def read(path):
    record = public.read_member(path, MEMBER)
    fields = "version preparation authorization grant controller operations families crossings"
    public.closed(record, fields + (" conditioning" if "conditioning" in record else ""), sealed=True)
    q, g = record["preparation"], record["grant"]
    validate_grant(g, record["authorization"], root=q["owner"], parent=q["identity"], action="run")
    if (record["version"] != "round029-comparative-prescore/v1" or q != seal(q)
            or g["total_us"] < q["proposed_total_us"] or g["holdback_us"] != q["followup_us"]
            or record["controller"]["grant"] != g["identity"]
            or record["controller"]["accepted"] != q["identity"]):
        raise ValueError("independent pre-score grant differs from the accepted wave")
    for logical, entry in record["operations"].items():
        if entry["coordinates"]["study"] != g["identity"] or operation_key(entry["coordinates"], 0)["logical"] != logical:
            raise ValueError("independent dispatch schedule differs")
    return record
