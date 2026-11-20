"""Read existing training-only timing records; never execute or resume a role."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import stat
import sys
import uuid
import zipfile

MAX_FILE = 64 << 20
STATUS_KEYS = (
    'complete', 'outcome', 'reason', 'phase', 'completed_bounds',
    'coefficient_bits', 'uniqueness', 'coverage', 'elapsed_seconds',
    'clock', 'job_id', 'expected', 'attempted', 'completed', 'required', 'outcomes',
)


def pick(value, keys):
    return {k: value[k] for k in keys if k in value}


def status(value):
    return pick(value, STATUS_KEYS) if isinstance(value, dict) else value


def candidates(value):
    """Export completion/cost diagnostics, excluding programs and assignments."""
    if value.get('version') != 'discovery-candidates/v1':
        raise ValueError('unsupported candidates version')
    result = pick(value, ('version', 'identity', 'model', 'checkpoint', 'model_control',
                         'recipe', 'methods', 'prompts', 'clock', 'counts', 'elapsed_us'))
    result['records'] = {}
    for key, row in value['records'].items():
        arms = {}
        for method, arm in row['arm_evidence'].items():
            arms[method] = status(arm)
            for phase in ('generation', 'fitting'):
                if phase in arm:
                    arms[method][phase] = status(arm[phase])
        costs = {}
        for method, cost in row['costs'].items():
            costs[method] = pick(cost, ('complete', 'elapsed_seconds', 'total_elapsed_us',
                                       'cache_parity'))
            details = []
            for detail in cost.get('details', []):
                item = pick(detail, ('complete', 'generation_seconds', 'generation_batch_size',
                                    'generation_batch_id', 'position', 'seed', 'source_sha256',
                                    'outcome', 'native_seconds', 'native_kind'))
                item['token_count'] = len(detail.get('tokens', []))
                for phase in ('generation', 'raw', 'fitting', 'native'):
                    if phase in detail:
                        item[phase] = status(detail[phase])
                details.append(item)
            costs[method]['details'] = details
        result['records'][key] = dict(arm_evidence=arms, costs=costs,
            selected={m: selection is not None for m, selection in row['selections'].items()})
    return result


def scores(value):
    if value.get('version') != 'discovery-scoring/v1':
        raise ValueError('unsupported scoring version')
    result = pick(value, ('version', 'identity', 'candidates', 'denominator',
                         'scientific_status', 'claim'))
    result['outcomes'] = {}
    for key, row in value['outcomes'].items():
        result['outcomes'][key] = {}
        for method, outcome in row.items():
            item = pick(outcome, ('complete', 'success', 'outcome', 'reason', 'candidate_sha256'))
            for phase in ('generation', 'fitting', 'scoring'):
                if isinstance(outcome.get(phase), dict):
                    item[phase] = status(outcome[phase])
                    for executor in ('production', 'reference'):
                        if executor in outcome[phase]:
                            item[phase][executor] = status(outcome[phase][executor])
            result['outcomes'][key][method] = item
    return result


def read_json(root, relative):
    path = root / relative
    # Reject symlinks at every component, including those still inside the session.
    current = root
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink is not original evidence')
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE:
        raise ValueError('not a regular evidence file within the 64 MiB read limit')
    with path.open('rb') as stream:
        raw = stream.read(MAX_FILE + 1)
    if len(raw) > MAX_FILE or len(raw) != info.st_size:
        raise ValueError('oversized or changing evidence file')
    return json.loads(raw), dict(path=relative, bytes=len(raw),
                               sha256='sha256:' + hashlib.sha256(raw).hexdigest())


def collect(session):
    session = Path(session).resolve(strict=True)
    timing = session / 'prepare-research/evaluation-timing'
    request, request_ref = read_json(session, 'prepare-research/evaluation-timing/request.json')
    if (request.get('version') != 'progressive-research-evaluation-request/v1'
            or request.get('scope') != 'training-only prospective throughput; no panel exposure'
            or len(request.get('cases', [])) != 14):
        raise ValueError('expected original fourteen-case training-only timing request')
    share = session / 'share'
    if share.is_symlink():
        raise ValueError('share directory must not be a symlink')
    share.mkdir(exist_ok=True)
    output = share / ('evaluation-timing-diagnostics-' +
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8] + '.zip')
    report = dict(version='research-timing-observer/v1', scope=request['scope'],
        authority='diagnostic projection only; cannot qualify or resume research',
        session=str(session), request=request_ref, source=request['source'],
        command=sys.argv, collector_sha256='sha256:' + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        files=[], errors=[], cases=[])
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        try:
            for index, case in enumerate(request['cases']):
                report['cases'].append(dict(index=index, case=case))
                for name, project in (('generation/candidates.json', candidates),
                                      ('scoring/scores.json', scores)):
                    relative = str((timing / 'cases' / str(index) / name).relative_to(session))
                    try:
                        value, original = read_json(session, relative)
                        projected = project(value)
                        member = f'cases/{index}/{name}'
                        raw = json.dumps(projected, sort_keys=True, indent=2).encode()
                        archive.writestr(member, raw)
                        report['files'].append(dict(original=original, projection=member,
                            bytes=len(raw), sha256='sha256:' + hashlib.sha256(raw).hexdigest()))
                    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
                        report['errors'].append(dict(path=relative, type=type(error).__name__,
                            message=str(error)[:2048]))
        finally:
            report['status'] = 'PARTIAL' if report['errors'] or len(report['files']) != 28 else 'COMPLETE'
            archive.writestr('manifest.json', json.dumps(report, sort_keys=True, indent=2))
    return output, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    args = parser.parse_args()
    output, report = collect(args.session)
    print(f"{report['status']}: {len(report['files'])}/28 existing records collected")
    print(output)
    if report['errors']:
        print('Share this ZIP even if records are missing; collection errors are included.')
    return 1 if report['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
