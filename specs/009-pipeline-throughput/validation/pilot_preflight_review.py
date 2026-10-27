"""Read-only public preflight audit. Never extracts private data or admits a run."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def review(path):
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or len(names) > 4097:
            raise ValueError('duplicate or excessive ZIP members')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):
            raise ValueError('unsafe ZIP path')
        if sum(i.file_size for i in z.infolist()) > 512 << 20:
            raise ValueError('oversized public evidence')
        manifest = json.loads(z.read('manifest.json'))
        entries = manifest['members']
        if manifest['omissions'] or len({e['path'] for e in entries}) != len(entries):
            raise ValueError('omitted or duplicate inventory entries')
        if {e['path'] for e in entries} != set(names) - {'manifest.json'}:
            raise ValueError('inventory does not cover exactly the public ZIP')
        for e in entries:
            b = z.read(e['path'])
            if len(b) != e['bytes'] or 'sha256:' + hashlib.sha256(b).hexdigest() != e['sha256']:
                raise ValueError('payload hash/size mismatch: ' + e['path'])
        read = lambda n: json.loads(z.read(n))
        invocation = read('invocation.json')
        source = invocation['binding']['source']
        receipts = {n.split('/')[1]: read(n) for n in names
                    if n.startswith('stages/') and n.endswith('/receipt.json') and n.count('/') == 2}
        observations = []
        cost = read('stages/pilot.report/cost.json')
        for i in range(3):
            root = f'stages/pilot.session/generate-{i}/'
            index = read(root + 'attempt-index.json')
            jobs = index['jobs']
            if len({(j['prompt'], j['mode'], j['attempt']) for j in jobs}) != index['expected']:
                raise ValueError('duplicate/missing registered jobs')
            if dict(Counter(j['state'] for j in jobs)) != index['states']:
                raise ValueError('attempt counts differ from registered jobs')
            selections = read(root + 'selections.json')
            roster = {(p, mode, a) for p in selections for mode, n in
                      [('correct', 8), ('shuffled', 1), ('absent', 1), ('template', 8)] for a in range(n)}
            if len(selections) != 296 or roster != {(j['prompt'], j['mode'], j['attempt']) for j in jobs}:
                raise ValueError('full frozen prompt/attempt roster missing')
            by_mode = defaultdict(Counter)
            by_split = defaultdict(Counter)
            reasons = Counter()
            for j in jobs:
                by_mode[j['mode']][j['state']] += 1
                by_split[selections[j['prompt']]['strata']['split']][j['state']] += 1
            for r in selections.values():
                for arm in r['arm_evidence'].values():
                    if not arm['complete']:
                        reasons[arm['reason']] += 1
            complete = index['states'].get('COMMITTED', 0) + index['states'].get('INCOMPLETE', 0) == index['expected']
            if cost['observations'][i]['complete'] != complete:
                raise ValueError('reported calibration completeness differs from attempt ledger')
            observations.append(dict(ordinal=i,states=index['states'],by_mode=dict(by_mode),by_split=dict(by_split),
                unresolved_arm_reasons=dict(reasons),cost_complete=complete,
                active=read(root + 'boundary.json')['active'],reserved=read(root + 'boundary.json')['reserved']))
        prep = read('stages/pilot.prepare/prepare/preparation.json')
        learning = read('stages/pilot.report/learning.json')
        diagnostic = 6 * (max(30, __import__('math').ceil(2 * max(r['generation_seconds'] for r in cost['observations'])))
                          + max(30, __import__('math').ceil(2 * max(r['projected_all_selected_scoring_seconds'] for r in cost['observations']))))
        summary = read('summary.json')
        with path.open('rb') as f:
            h = hashlib.sha256()
            for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
        return dict(version='pipeline-pilot-offline-review/v1',archive_sha256=h.hexdigest(),source=source,
            inventory=dict(payloads=len(entries),omissions=0,hash_mismatches=0),
            elapsed_seconds=summary['elapsed_seconds'],engineering=summary['engineering'],portability=summary['portability'],
            receipts={k:dict(state=v['state'],reason=v['reason'],identity=v['identity']) for k,v in receipts.items()},
            preparation=prep,observations=observations,cost_status=cost['status'],
            diagnostic_only_incomplete_evaluation_envelope_seconds=diagnostic,
            diagnostic_is_admissible_forecast=False,
            snapshots=[{k:r[k] for k in ('ordinal','updates','tokens','unique_examples','learner_role_seconds','effective_token_epochs')}
                       for r in learning['actual_snapshots']],
            long_run_admitted=False,automatic_run_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.archive)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print('Verified public archive; engineering:', result['engineering'], '; cost:', result['cost_status'])


if __name__ == '__main__': main()
