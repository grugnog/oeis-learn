"""Experimental visible-prefix ranking; never accepts scores or hidden terms.

This is not registered as a production selection policy. All execution is delegated
to the existing independent bounded reference interpreter.
"""
from dataclasses import asdict, replace

from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.synthesis.contracts import ResourceProfile, canonical_integer
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.reference_parser import parse

LIMITS = replace(ResourceProfile(), evaluation_seconds=2.0, term_seconds=0.25,
                 reference_steps=10000, aggregate_reference_steps=100000)
POLICY = seal(dict(version='visible-branch-coverage/v1', visible_terms=20,
                   ranking=['missing_branch_edges', 'ast_nodes', 'literal_bits', 'source_sha256'],
                   unavailable='retain_original_mark_incomplete', limits=asdict(LIMITS)))


def observe(source, prompt, *, limits=LIMITS, deadline=None):
    if not isinstance(prompt, VisiblePrompt) or len(prompt.indices) != 20:
        raise ValueError('exactly 20 visible terms required')
    sites = {}
    def event(site, value):
        seen = sites.setdefault(site, set())
        if value is not None:
            seen.add(value)
    result = evaluate(source, prompt.indices, limits, deadline=deadline, branch_observer=event)
    complete = result.outcome == 'ok' and result.coverage == 20
    reason = result.reason
    if complete and result.values != prompt.values:
        complete, reason = False, 'visible_mismatch'
    nodes = literal_bits = None
    if complete:
        program = parse(source, limits)
        nodes, literal_bits = 0, 0
        def walk(node):
            nonlocal nodes, literal_bits
            nodes += 1
            if not isinstance(node, tuple):
                return
            if node[0] in {'int', 'control'}:
                literal_bits += max(1, abs(canonical_integer(node[1], bits=limits.literal_bits)).bit_length())
            for child in node[1:]:
                walk(child)
        for node in (*program.statements, program.result):
            walk(node)
        nodes += len(program.declarations)
    return dict(complete=complete, reason=reason, steps=result.steps,
                branch_sites=len(sites), missing_branch_edges=sum(2-len(v) for v in sites.values()) if complete else None,
                ast_nodes=nodes, literal_bits=literal_bits)


def rank(candidates, prompt, *, original, limits=LIMITS, deadline=None):
    """Closed candidate fields intentionally exclude evaluation metadata."""
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 64:
        raise ValueError('one to 64 original eligible candidates required')
    by_id = {}
    for row in candidates:
        if set(row) != {'source', 'source_sha256'}:
            raise ValueError('closed candidate fields required')
        if not isinstance(row['source'], str) or len(row['source'].encode()) > limits.source_bytes:
            raise ValueError('candidate source limit')
        if source_identity(row['source']) != row['source_sha256']:
            raise ValueError('candidate source hash mismatch')
        if row['source_sha256'] in by_id:
            raise ValueError('duplicate candidate')
        by_id[row['source_sha256']] = row['source']
    if original not in by_id:
        raise ValueError('original candidate absent')
    measurements = {sha: observe(source, prompt, limits=limits, deadline=deadline)
                    for sha, source in sorted(by_id.items())}
    complete = all(v['complete'] for v in measurements.values())
    selected = min(measurements, key=lambda sha: (
        measurements[sha]['missing_branch_edges'], measurements[sha]['ast_nodes'],
        measurements[sha]['literal_bits'], sha)) if complete else original
    return dict(complete=complete, selected=selected, candidates=measurements)
