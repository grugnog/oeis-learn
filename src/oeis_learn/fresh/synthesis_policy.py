"""Visible-only portfolios and sealed full-horizon scoring; no fitter implementation."""

import math
from oeis_learn.synthesis.contracts import identity, canonical_integer


class Portfolio:
    def __init__(self, *, start, seed, beam=None, seconds=60):
        if not math.isfinite(start) or type(seed) is not int:
            raise ValueError("invalid portfolio clock/seed")
        if seconds not in (60,180):
            raise ValueError('registered production60/fixed-work180 portfolio required')
        self.start = start
        self.last_now = start
        self.deadline = start + seconds
        self.seed = seed
        self.records = {}
        self.beam = beam
        if beam is not None and (
            set(beam) != {"attempts", "max_expansions"}
            or type(beam["attempts"]) is not int
            or not 1 <= beam["attempts"] <= 8
            or type(beam["max_expansions"]) is not int
            or not 1 <= beam["max_expansions"] <= 8192
        ):
            raise ValueError("beam needs explicit replacement attempt/expansion allocation")
        self.max_attempts = 8 if beam is None else beam["attempts"]

    def remaining(self, *, now):
        if not math.isfinite(now) or now < self.last_now:
            raise ValueError("nonmonotone portfolio clock")
        self.last_now = now
        return max(0, self.deadline - now)

    def begin(self, *, now):
        if len(self.records) >= self.max_attempts or self.remaining(now=now) <= 0:
            raise ValueError("portfolio exhausted")
        n = len(self.records)
        key = identity([self.seed, n, self.start, self.beam])
        record = dict(
            id=key,
            method="beam" if self.beam else "greedy" if n == 0 else "stochastic",
            seed=self.seed + n,
            token_cap=1024,
            status="generating",
            started=now,
        )
        self.records[key] = record
        return dict(record)

    def finish_generation(self, key, *, tokens, complete, source, now):
        record = self.records[key]
        if (
            record["status"] != "generating"
            or now < record["started"]
            or len(tokens) > 1024
            or type(complete) is not bool
        ):
            raise ValueError("invalid generation result")
        if complete:
            from oeis_learn.decoder.bigint_codec import decode_body

            if decode_body(tokens) != source:
                raise ValueError("complete EOS/source identity mismatch")
        elif source:
            raise ValueError("incomplete candidates cannot be repaired")
        record.update(
            tokens=list(tokens),
            source=source,
            complete=complete,
            status="decoded" if complete else "incomplete",
            generation_seconds=now - record["started"],
            within_budget=self.remaining(now=now) > 0,
        )

    def fit_allowance(self, *, now):
        remaining = self.remaining(now=now)
        reference = min(2, remaining)
        return min(2, max(0, remaining - reference)), reference


def rescue_identity(raw_identity, fit_original_identity):
    if raw_identity != fit_original_identity:
        raise ValueError("rescue must fit exactly the same original candidate")
    return raw_identity


def select_visible(rows, *, policy='visible-only/v1'):
    if policy == 'visible-only/v2':
        return _select_v2(rows)
    if policy != 'visible-only/v1':
        raise ValueError('unknown selection policy')
    allowed = {"source", "source_sha256", "verified", "visible_count", "cost"}
    for row in rows:
        if (
            set(row) != allowed
            or type(row["verified"]) is not bool
            or type(row["visible_count"]) is not int
        ):
            raise ValueError("visible-only candidate record required")
        from oeis_learn.synthesis.admission import source_identity

        if not isinstance(row["source"], str) or row["source_sha256"] != source_identity(
            row["source"]
        ):
            raise ValueError("candidate source identity mismatch")
        if (
            type(row["cost"]) not in (float, int)
            or not math.isfinite(row["cost"])
            or row["cost"] < 0
        ):
            raise ValueError("measured nonnegative cost required")
    admitted = [r for r in rows if r["verified"] and r["visible_count"] == 20]
    if not admitted:
        return None
    chosen = min(admitted, key=lambda r: (len(r["source"].encode()), r["cost"], r["source_sha256"]))
    return dict(chosen, seal_sha256=identity(chosen), selection="visible-only/v1")


def transfer_success(selected, indices, expected, production, reference):
    if selected is None:
        return False
    record = {k: v for k, v in selected.items() if k not in {"seal_sha256", "selection"}}
    if selected.get("selection") not in {"visible-only/v1", "visible-only/v2"} or selected.get("seal_sha256") != identity(record):
        raise ValueError("changed sealed candidate")
    if not selected["verified"] or len(indices) != 100 or len(expected) != 100:
        return False
    for record in (production, reference):
        if (
            not isinstance(record, dict)
            or set(record) != {"source_sha256", "indices", "values", "outcome"}
            or record["outcome"] != "ok"
            or record["source_sha256"] != selected["source_sha256"]
            or record["indices"] != list(indices)
            or len(record["values"]) != 100
        ):
            return False
    try:
        ns = [canonical_integer(n) for n in indices]
        if any(a >= b for a, b in zip(ns, ns[1:])):
            return False
        values = [
            [canonical_integer(n, bits=16384) for n in xs]
            for xs in (expected, production["values"], reference["values"])
        ]
    except ValueError:
        return False
    return values[0] == values[1] == values[2]



def _select_v2(rows):
    from oeis_learn.synthesis.admission import source_identity
    from oeis_learn.synthesis.state import require_hash
    fields = {'source','source_sha256','verified','visible_count','cost','attempt_ordinal','job_id','provenance'}
    admitted = []
    for row in rows:
        if set(row) != fields or type(row['verified']) is not bool or type(row['visible_count']) is not int:
            raise ValueError('closed visible-only/v2 record required')
        if not isinstance(row['source'], str) or source_identity(row['source']) != row['source_sha256']:
            raise ValueError('candidate source identity mismatch')
        require_hash(row['job_id'])
        if type(row['attempt_ordinal']) is not int or not 0 <= row['attempt_ordinal'] < 8:
            raise ValueError('stable attempt ordinal required')
        if type(row['cost']) not in (int,float) or not math.isfinite(row['cost']) or row['cost'] < 0:
            raise ValueError('finite nonnegative observation required')
        provenance = row['provenance']
        if not isinstance(provenance,dict) or set(provenance) != {'prompt','snapshot','policies','original_source_sha256','transformed_source_sha256','assignment','verification'}:
            raise ValueError('complete selected provenance required')
        if row['verified'] and row['visible_count'] == 20:
            admitted.append(row)
    if not admitted:
        return None
    selected = min(admitted, key=lambda r: (len(r['source'].encode()),r['source_sha256'],r['attempt_ordinal'],r['job_id']))
    import copy
    semantic = copy.deepcopy({k:v for k,v in selected.items() if k != 'cost'})
    return dict(semantic, selection='visible-only/v2', seal_sha256=identity(semantic))


def validate_selection_envelope(envelope):
    """Bind v2 scoring provenance to the visible-only seal before hidden execution."""
    from oeis_learn.synthesis.admission import source_identity
    selected=envelope['selected']
    record={k:v for k,v in selected.items() if k not in {'seal_sha256','selection'}}
    if selected.get('selection') not in {'visible-only/v1','visible-only/v2'} or selected.get('seal_sha256')!=identity(record):
        raise ValueError('changed sealed candidate')
    if selected['selection']=='visible-only/v1':
        return selected
    if set(envelope)!={'selected','verification','assignment','original_source','transformed_source','original_source_sha256','prompt_sha256'}:
        raise ValueError('closed selected provenance envelope required')
    provenance=selected['provenance']
    expected=dict(prompt=envelope['prompt_sha256'],assignment=envelope['assignment'],
        original_source_sha256=envelope['original_source_sha256'],
        transformed_source_sha256=source_identity(envelope['transformed_source']),
        verification=identity(envelope['verification']))
    if (any(provenance[key]!=value for key,value in expected.items())
            or source_identity(envelope['original_source'])!=envelope['original_source_sha256']
            or source_identity(selected['source'])!=selected['source_sha256']
            or envelope['verification']['prompt_sha256']!=envelope['prompt_sha256']):
        raise ValueError('changed sealed candidate provenance')
    return selected
