"""Replay the stopped round027 diversity publication off-line, without changing its campaign."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
from unittest.mock import patch
import zipfile

ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / 'src'))
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES, load, record, reference, seal
from oeis_learn.generalization.progressive.decisions import verify_gate_producers
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.projections import project
from oeis_learn.pipeline_validation import progressive_outputs, progressive_publication, progressive_role
from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
from oeis_learn.pipeline_validation.progressive_workstation_state import read

SESSION = Path('run-state/discovery/013-research-round-027')
CAMPAIGN = 'sha256:b35444dcbdc68693917a61d05ac3dc11efbdebee568a074b80f2596191035304'
FAILURE = 'sha256:91170db7a5d83e4449424adc60d852afe6c766a62f1fbdca53fc560f7db73885'
NAME = 'diversity-comparison'
MODULE = 'src/oeis_learn/pipeline_validation/progressive_publication.py'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def corrected(revision):
    raw = subprocess.check_output(['git', 'show', revision + ':' + MODULE], cwd=ROOT)
    require(hashlib.sha256(raw).hexdigest() == 'effe3ec15ce90935f99a90450fa04c03de5ee467e61b94c926d837903812d3ce',
            'expected the reviewed publication fix')
    module = ModuleType('round027_diagnostic_publisher')
    exec(compile(raw, MODULE, 'exec'), module.__dict__)
    return module.decision_from_outputs


def audit(root, control, account, note):
    session = root / SESSION
    campaign = load(session/'prepare-research/frozen/campaign.json', maximum=CAMPAIGN_DOCUMENT_BYTES)
    require(seal(campaign) == campaign and campaign['identity'] == CAMPAIGN, 'frozen round027 campaign differs')
    result = read(session/'run-research/execution-0000/result.json')
    require(result['identity'] == FAILURE, 'original research failure differs')
    require(not (control/'assessment-exposure.json').exists() and not (session/'assess').exists(),
            'original unassessed run required')
    for path in (control/'campaign.json', account/'budget-binding.json', account/'budget.lock', account/'budget.jsonl'):
        require(path.is_file(), 'missing original ' + str(path))
    store = RunStore(control, campaign)
    tasks = store.state['tasks']
    completed = {k for k, status in tasks.items() if status == 'COMPLETE'}
    require(tasks == result['metrics']['tasks'] and set(store.state['outputs']) == completed
            and len(completed) == 47 and tasks[NAME] == 'FAILED', 'original task state differs')
    note.update(transition=reference(root,max(control.glob('transition-*.json'))),
                outputs=store.state['outputs'], campaign=CAMPAIGN, failure=FAILURE)
    for cut in store.state['outputs'].values():
        path = checked_file(root, cut['result'])
        for ref in read(path)['metrics'].values():
            if isinstance(ref, dict) and set(ref) == {'path','bytes','sha256'}:
                checked_file(path.parent,ref)
    from oeis_learn.generalization.progressive.import_state import inspect
    note['training_checkpoints'] = checkpoints = {}
    for name, cut in store.state['outputs'].items():
        if cut['operation'] != 'train':
            continue
        base = (root / cut['result']['path']).parent
        result = read(base/'result.json')
        try:
            checkpoint = checked_file(base,result['metrics']['checkpoint_record'])
            spec = dict(artifact=reference(root,checkpoint), accounting=reference(root,base/'accounting-final.json'),
                        state_root=f"run-state/discovery-learning/{CAMPAIGN[7:]}/{cut['task'][7:]}/checkpoints")
            state, cost, _, _, refs = inspect(root,spec)
            require(state['payload']['campaign']==CAMPAIGN and cost['task']==cut['task']
                    and cost['useful_tokens']==result['metrics']['useful_tokens'], 'training checkpoint or charge differs')
            checkpoints[name] = dict(source=spec, files=refs, useful_tokens=cost['useful_tokens'])
        except (KeyError, ValueError, OSError) as exc:
            checkpoints[name] = dict(error=f'{type(exc).__name__}: {exc}')
    require(len(checkpoints)==4, 'four original full-state training checkpoints required')
    ledger = BudgetJournal(account, CAMPAIGN, campaign['payload']['caps'])
    note.update(budget=ledger.snapshot(), ledger=reference(root,account/'budget.jsonl'))
    for key, paths in (('task_bindings',account.glob('task-binding-*.json')),
                       ('storage_inventories',(account/'storage-inventory').glob('*.json'))):
        note[key] = [reference(root,p) for p in sorted(paths)]
    require(not note['budget']['pending'], 'unsettled research charges')
    verified = sum('error' not in item for item in checkpoints.values())
    print(f'Verified 47 saved results, {verified}/4 full-state checkpoints and the original budget journal.', flush=True)
    return store


def replay(root, store, temp, fixed, note):
    campaign = store.campaign
    plan = next(p for p in campaign['payload']['tasks'] if p['id'] == NAME)
    task = record('task',dict(plan,campaign=CAMPAIGN),created_at=campaign['created_at'],
                  purpose=campaign['purpose'],track=campaign['track'])
    ref, = [r['artifact'] for r in campaign['payload']['references'] if r['name'] == 'requests/'+NAME]
    require(ref['sha256'] == plan['policy'], 'comparison policy differs')
    request = load(checked_file(root,ref))
    views, _ = progressive_outputs.resolve(root,store,task,request,temp/'resolved')
    project(root,temp/'inputs','publish',views,registry=progressive_role.REGISTRY,
            authorized=[d.ref for d in views['public']])
    def counted(values):
        built = fixed(values)
        rows = built[0]['data']['rows']
        note['comparison_rows'] = len(rows)
        note['verified_success_in_incomplete_search'] = sum(r['outcome']=='success' and not r['complete'] for r in rows)
        return built
    with patch.object(progressive_publication, 'decision_from_outputs', counted):
        result = progressive_role.execute('publish',inputs=temp/'inputs',output=temp/'role')
    require(result['mechanical_status']=='COMPLETE', 'diagnostic publication did not complete')
    bundle = read(temp/'role/decision-bundle.json')
    note['gate_registration'] = verify_gate_producers(root,campaign,bundle,store=store)
    note.update(decision=bundle['decision'], decision_bundle=bundle['identity'])
    print('Comparison replay:',result['scientific_status'],result['next_action'],flush=True)


def main(revision):
    fixed = corrected(revision)
    tools = ROOT/'run-state/discovery-tools'
    control = ROOT/'run-state/discovery-control'/CAMPAIGN[7:]
    account = ROOT/'run-state/discovery-accounting'/CAMPAIGN[7:]
    note = dict(version='round027-diagnostic-replay/v1',stage='audit')
    with (ROOT/SESSION/'session.lock').open('rb') as session_lock, (control/'run.lock').open('rb') as run_lock:
        for lock in (session_lock,run_lock):
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            store = audit(ROOT,control,account,note)
            note['stage'] = 'comparison'
            with tempfile.TemporaryDirectory(prefix='comparison-replay-',dir=tools) as tmp:
                replay(ROOT,store,Path(tmp),fixed,note)
            note['stage'] = ('PARTIAL_CHECKPOINTS' if any('error' in c for c in note['training_checkpoints'].values())
                             else 'PASS')
        except (KeyError, OSError, ValueError, RuntimeError) as exc:
            note['error'] = f'{type(exc).__name__}: {exc}'
            print(note['error'],file=sys.stderr)
        with tempfile.NamedTemporaryFile(dir=tools,prefix='round027-comparison-',suffix='.zip',delete=False) as out:
            with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr('replay.json',json.dumps(seal(note),sort_keys=True))
        print('Share diagnostics:',out.name)
        return int(note['stage']!='PASS')


if __name__ == '__main__':
    if len(sys.argv)!=2:
        raise SystemExit('Pass the fetched publication-fix commit.')
    raise SystemExit(main(sys.argv[1]))
