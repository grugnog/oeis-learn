"""Outcome-blind projection of one original indexed OEIS source cut."""
from collections import Counter
from pathlib import Path
import json
import time

from .identity import confined, digest, seal
from .progressive_research_inputs import _json, checked_file
from .progressive_workstation_state import write
from oeis_learn.generalization.progressive.contracts import reference


def project(root, snapshot, output, *, limit=4096, seconds=600, maximum_bytes=1 << 30):
    from oeis_learn.fresh.bulk import indexed_terms
    root, output = Path(root), Path(output)
    path = confined(root, snapshot)
    if path.stat().st_size > 128 << 20:
        raise ValueError('bounded original source manifest required')
    original = reference(root, path)
    data = _json(checked_file(root, original), maximum=128 << 20)
    if (set(data) != {'version', 'release', 'records'} or data['version'] != 'indexed-source/v1'
        or not isinstance(data['release'], str) or not data['release']
        or not isinstance(data['records'], list) or len(data['records']) > 1_000_000):
        raise ValueError('original immutable indexed-source/v1 snapshot required')
    if type(limit) is not int or not 1 <= limit <= 8192:
        raise ValueError('finite outcome-blind source reservoir required')
    if any(not isinstance(r, dict) or not isinstance(r.get('record_id'), str)
           or not r['record_id'] or len(r['record_id']) > 1024 for r in data['records']):
        raise ValueError('bounded unique original source member identities required')
    if len({r['record_id'] for r in data['records']}) != len(data['records']):
        raise ValueError('duplicate source member')
    if not 0 < seconds <= 3600 or not 0 < maximum_bytes <= 8 << 30:
        raise ValueError('finite source projection allowance required')
    deadline = time.monotonic() + seconds
    consumed = candidate_bytes = 0
    selected = sorted(data['records'], key=lambda r: digest([20261110, r['record_id']]))[:limit]
    rows, rejected, failures, originals, outcomes = [], Counter(), [], [], []
    for ordinal, item in enumerate(selected):
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('source projection allowance exhausted')
            if set(item) != {'record_id', 'terms', 'metadata'}:
                raise ValueError('closed indexed source member required')
            files, member_refs = {}, []
            for key in ('terms', 'metadata'):
                ref = item[key]
                if set(ref) != {'path', 'sha256'}:
                    raise ValueError('original source path/hash required')
                file = confined(path.parent, ref['path'])
                size = file.stat().st_size
                if consumed + 3*size > maximum_bytes:
                    raise ValueError('aggregate source byte allowance exhausted')
                # Reserve the initial hash, parse and post-parse integrity hash.
                consumed += 3*size
                checked_file(path.parent, dict(ref, bytes=size), maximum=4 << 20)
                files[key] = file
                member_refs.append(dict(path=str(file.relative_to(root)), bytes=size, sha256=ref['sha256']))
            originals.extend(member_refs)
            meta = _json(files['metadata'])
            offset = meta.get('offset')
            if type(offset) is not int or meta.get('finite') or meta.get('table'):
                rejected['offset_or_task_profile'] += 1
                outcomes.append(dict(member=item['record_id'], status='INELIGIBLE', reason='offset_or_task_profile'))
                continue
            values = indexed_terms(files['terms'], offset, horizon=100)
            for ref in member_refs:
                checked_file(root, ref, maximum=4 << 20)
            if len(values) != 100:
                rejected['incomplete_horizon'] += 1
                outcomes.append(dict(member=item['record_id'], status='INELIGIBLE', reason='incomplete_horizon'))
                continue
            row = dict(id=item['record_id'], indices=[str(offset+i) for i in range(100)],
                       values=[values[offset+i] for i in range(100)])
            size = len(json.dumps(row, indent=2).encode()) + 4*200 + 128
            if candidate_bytes + size > 128 << 20:
                raise ValueError('private candidate projection byte allowance exhausted')
            candidate_bytes += size
            rows.append(row)
            outcomes.append(dict(member=item['record_id'], status='PROJECTED'))
        except Exception as exc:
            # Missing/tampered selected bytes are integrity deficits, not a reason
            # to substitute later source members or silently shrink denominators.
            failures.append(dict(member=item['record_id'], error=type(exc).__name__, reason=str(exc)))
            outcomes.append(dict(member=item['record_id'], status='BLOCKED', reason=str(exc)))
        finally:
            if ordinal % 64 == 0 or ordinal == len(selected) - 1:
                write(output / 'source-progress.json', seal(dict(
                    version='progressive-research-source-progress/v1', completed=ordinal+1,
                    selected=len(selected), eligible=len(rows), rejected=dict(rejected),
                    failures=failures, outcomes=outcomes, source_bytes=consumed, candidate_bytes=candidate_bytes)))
    checked_file(root, original, maximum=128 << 20)
    write(output / 'private/oeis-candidates.json', rows)
    result = seal(dict(version='progressive-research-source-projection/v1',
        status='BLOCKED' if failures else 'COMPLETE', snapshot=original, release=data['release'],
        selection=dict(seed=20261110, limit=limit, selected=len(selected), ordering='sha256(seed,record_id)'),
        eligible=len(rows), rejected=dict(rejected), failures=failures, originals=originals, outcomes=outcomes,
        limits=dict(seconds=seconds, source_bytes=maximum_bytes, candidate_bytes=128 << 20),
        consumed_source_bytes=consumed,
        candidates=reference(output, output / 'private/oeis-candidates.json'),
        purpose='fresh evaluation only; formulas and metadata never become training targets'))
    write(output / 'source-projection.json', result)
    return result
