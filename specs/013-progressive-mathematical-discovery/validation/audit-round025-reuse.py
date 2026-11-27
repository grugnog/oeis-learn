"""Read the stopped round025 originals; export only recovery metadata and hashes."""
import fcntl
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'src'))
from oeis_learn.pipeline_validation.identity import confined, digest, seal
from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
from oeis_learn.generalization.progressive.contracts import load, reference, CAMPAIGN_DOCUMENT_BYTES

SESSION = 'run-state/discovery/013-research-round-025'
INVOCATION = SESSION + '/run-research/execution-0000'
CAMPAIGN = 'sha256:eaf72f0eb07d1459970c41c43f277d917c752948ef29b89c55099a5fb1bf0757'
FAILURE = 'sha256:9ebfdc5b7dcbf429a06c5446f24c43d83d9ec1af2e44e719f666c3d0e8755395'
CAPABILITY_CUTS = 'sha256:72b1ffc6fab3ff5ab729eb85b27ab829b06999795639ac04375562db95302e7c'
CONTROL = 'run-state/discovery-control/' + CAMPAIGN[7:]
ACCOUNT = 'run-state/discovery-accounting/' + CAMPAIGN[7:]


def require(ok, why):
    if not ok:
        raise ValueError(why)


def original(path):
    value = load(confined(ROOT, path), maximum=CAMPAIGN_DOCUMENT_BYTES)
    require(seal(value) == value, 'document identity differs: ' + str(path))
    return value


def audit():
    failure = original(INVOCATION + '/result.json')
    require(failure['identity'] == FAILURE, 'expected original round025 failure')
    campaign = original(SESSION + '/prepare-research/frozen/campaign.json')
    require(campaign['identity'] == CAMPAIGN, 'expected original round025 campaign')
    plans = {p['id']:p for p in campaign['payload']['tasks']}
    expected = {k for k,p in plans.items() if p['stage'] == 'capability'}
    journal = original(SESSION + '/journal.json')['steps']
    require(journal['run-research/invocation-0000']['status'] == 'FAILED', 'expected stopped invocation')
    control = confined(ROOT, CONTROL)
    account = confined(ROOT, ACCOUNT)
    previous = CAMPAIGN
    transitions = sorted(control.glob('transition-*.json'))
    for index, path in enumerate(transitions):
        transition = original(path)
        require(path.name == f'transition-{index:04d}.json' and transition['previous'] == previous,
                'control chain differs')
        previous = transition['identity']
    state = transition['state']
    outputs = state.get('outputs', {})
    require(set(outputs) == expected and all(state['tasks'].get(k) == 'COMPLETE' for k in expected),
            'capability output roster differs')
    require(digest(outputs) == CAPABILITY_CUTS, 'saved capability references differ from the diagnostic bundle')
    verified = {}
    for name, cut in sorted(outputs.items()):
        result_path = checked_file(ROOT, cut['result'])
        result = original(result_path)
        refs = [r for r in result.get('metrics', {}).values()
                if isinstance(r,dict) and set(r) == {'bytes','path','sha256'}]
        for ref in refs:
            checked_file(result_path.parent,ref)
        verified[name] = dict(cut=cut, metric_files=refs)
    policies = {r['artifact']['sha256']:r['artifact'] for r in campaign['payload']['references']
                if r['kind'] == 'policy'}
    reservations = {}
    for name, plan in plans.items():
        if plan['kind'] == 'training':
            path = checked_file(ROOT, policies[plan['policy']])
            request = load(path)
            reservations[name] = dict(policy=plan['policy'], reservation=request['reservation'],
                                      seconds=request['seconds'], operation=request['operation'])
    binding_path = account/'budget-binding.json'
    binding = original(binding_path)
    ledger_path = confined(ROOT, account/'budget.jsonl')
    events = [json.loads(line) for line in ledger_path.read_bytes().splitlines()]
    require(binding == seal(dict(campaign=CAMPAIGN, caps=campaign['payload']['caps'])), 'budget binding differs')
    previous = binding['identity']
    pending = set()
    for i, event in enumerate(events):
        require(seal(event) == event and event['index'] == i and event['previous'] == previous,
                'budget chain differs')
        key = event['operation']
        if event['event'] == 'RESERVE':
            require(key not in pending, 'duplicate pending reservation')
            pending.add(key)
        else:
            require(event['event'] == 'COMMIT' and key in pending, 'unmatched budget event')
            pending.remove(key)
        previous = event['identity']
    metadata = dict(campaign=CAMPAIGN, original_failure=FAILURE, outputs=verified, reservations=reservations,
                    transition_tip=transitions[-1].name, transition=transition, ledger_tip=previous,
                    pending=sorted(pending), budget_binding=binding,
                    original_run=original(INVOCATION+'/run.json'),
                    task_bindings=[reference(ROOT,p) for p in sorted(account.glob('task-binding-*.json'))],
                    storage_inventories=[reference(ROOT,p) for p in sorted((account/'storage-inventory').glob('*.json'))],
                    assessment_exposed=(control/'assessment-exposure.json').exists())
    return metadata, ledger_path


if __name__ == '__main__':
    out = confined(ROOT, 'run-state/discovery-tools')
    try:
        with confined(ROOT, CONTROL + '/run.lock').open('rb') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            metadata, ledger = audit()
            with tempfile.NamedTemporaryFile(dir=out,prefix='round025-reuse-',suffix='.zip',delete=False) as f:
                with zipfile.ZipFile(f,'w',compression=zipfile.ZIP_DEFLATED) as z:
                    z.writestr('reuse.json',json.dumps(seal(metadata),sort_keys=True))
                    z.write(ledger,'budget.jsonl')
    except (ValueError, OSError, KeyError) as exc:
        print(f"Reuse audit stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Verified {len(metadata['outputs'])} saved capability results and their referenced metric files.")
    print(f"Outstanding reservations: {len(metadata['pending'])}. Share: {f.name}")
