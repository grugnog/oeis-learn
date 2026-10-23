"""Deterministic structure/cost alternatives, separate from training selection."""

import math
from oeis_learn.synthesis.state import require_hash


def retain_alternatives(rows, cap=8, *, policy='structure-pareto/v1'):
    if policy == 'structure-pareto/v2':
        return _retain_v2(rows, cap)
    if policy != 'structure-pareto/v1':
        raise ValueError('unknown archive policy')
    if type(cap) is not int or not 1 <= cap <= 8:
        raise ValueError("archive cap must be 1..8")
    seen = {}
    fields = {
        "id",
        "structure",
        "source",
        "cost",
        "verified",
        "verification",
        "track",
        "profile",
        "assignment",
    }
    for row in rows:
        if (
            set(row) != fields
            or row["verified"] is not True
            or row["track"] not in {"generic_only", "assisted"}
            or row["profile"] != "bigint-body/v1"
        ):
            raise ValueError("independently verified archive record required")
        require_hash(row["verification"])
        if (
            type(row["cost"]) not in (int, float)
            or not math.isfinite(row["cost"])
            or row["cost"] < 0
        ):
            raise ValueError("measured cost required")
        if row["id"] in seen and seen[row["id"]] != row:
            raise ValueError("conflicting archive identity")
        seen[row["id"]] = row
    key = lambda r: (len(r["source"].encode()), r["cost"], r["id"])
    groups = {}
    for row in seen.values():
        groups.setdefault(row["structure"], []).append(row)
    front = []
    dominated = []
    for rows in groups.values():
        for r in rows:
            if any(
                len(q["source"].encode()) <= len(r["source"].encode())
                and q["cost"] <= r["cost"]
                and (len(q["source"].encode()) < len(r["source"].encode()) or q["cost"] < r["cost"])
                for q in rows
            ):
                dominated.append(r)
            else:
                front.append(r)
    representatives = sorted(
        (min((r for r in front if r["structure"] == g), key=key) for g in groups), key=key
    )
    retained = representatives[:cap]
    selected = {r["id"] for r in retained}
    for r in sorted(front, key=key):
        if len(retained) == cap:
            break
        if r["id"] not in selected:
            retained.append(r)
            selected.add(r["id"])
    omissions = [
        dict(
            id=r["id"],
            reason="dominated_within_structure" if r in dominated else "frozen_capacity",
            structure=r["structure"],
            verification=r["verification"],
        )
        for r in sorted(seen.values(), key=key)
        if r["id"] not in selected
    ]
    import copy

    return dict(
        policy="structure-pareto/v1",
        retained=copy.deepcopy(sorted(retained, key=key)),
        omitted=omissions,
    )


def training_representative(archive):
    if not archive["retained"]:
        return None
    import copy

    if archive.get('policy') == 'structure-pareto/v2':
        return copy.deepcopy(min(archive['retained'], key=v2_key))
    if archive.get('policy') != 'structure-pareto/v1':
        raise ValueError('unknown archive policy')
    return copy.deepcopy(min(archive["retained"], key=lambda r: (len(r["source"].encode()), r["cost"], r["id"])))



def assignment_bits(assignment):
    from oeis_learn.synthesis.contracts import canonical_integer
    if not isinstance(assignment,dict) or len(assignment)>8:
        raise ValueError('bounded canonical assignment required')
    return sum(max(1,abs(canonical_integer(v)).bit_length()) for v in assignment.values())


def v2_key(row):
    from oeis_learn.synthesis.admission import source_identity
    return (len(row['source'].encode()),assignment_bits(row['assignment']),source_identity(row['source']),row['attempt_ordinal'],row['job_id'])


def _retain_v2(rows, cap):
    import copy
    if type(cap) is not int or not 1<=cap<=8:
        raise ValueError('archive cap must be 1..8')
    seen={}
    for row in rows:
        if set(row) != {'id','structure','source','cost','verified','verification','track','profile','assignment','attempt_ordinal','job_id'}:
            raise ValueError('closed v2 archive record required')
        # Reuse v1 validation without changing its identity/schema.
        retain_alternatives([{k:v for k,v in row.items() if k not in {'attempt_ordinal','job_id'}}])
        require_hash(row['job_id'])
        if type(row['attempt_ordinal']) is not int or not 0<=row['attempt_ordinal']<8:
            raise ValueError('stable attempt ordinal required')
        key=v2_key(row)[2]
        if key in seen:
            previous=seen[key]
            for field in ['structure','verification','track','profile','assignment']:
                if previous[field]!=row[field]:
                    raise ValueError('conflicting equal-source archive evidence')
            if v2_key(row)<v2_key(previous):
                seen[key]=row
        else:
            seen[key]=row
    groups={}
    for row in seen.values():
        groups.setdefault(row['structure'],[]).append(row)
    front=[];dominated=[]
    for group in groups.values():
        for row in group:
            a=v2_key(row)[:2]
            if any(all(x<=y for x,y in zip(v2_key(q)[:2],a)) and v2_key(q)[:2]!=a for q in group):
                dominated.append(row)
            else:
                front.append(row)
    representatives=sorted((min((r for r in front if r['structure']==g),key=v2_key) for g in groups),key=v2_key)
    retained=representatives[:cap]
    for row in sorted(front,key=v2_key):
        if len(retained)>=cap:
            break
        if row not in retained:
            retained.append(row)
    return dict(policy='structure-pareto/v2',retained=copy.deepcopy(sorted(retained,key=v2_key)),
        omitted=[dict(id=r['id'],reason='dominated_within_structure' if r in dominated else 'frozen_capacity',structure=r['structure'],verification=r['verification']) for r in sorted(seen.values(),key=v2_key) if r not in retained])
