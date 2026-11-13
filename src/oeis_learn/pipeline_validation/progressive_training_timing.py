"""Training-only timing diagnostics, never scientific selection or qualification."""
import os
from pathlib import Path
import time
import traceback

from .identity import confined, digest, seal
from .progressive_workstation_state import read
from .preparation_recovery import retain, revise
from .progressive_research_inputs import checked_file, _json
from oeis_learn.generalization.progressive.contracts import reference

VERSION = 'progressive-training-timing/v1'
SECONDS = 1800
HOST_SECONDS = SECONDS + 120
UPDATES = 16
CHECKPOINT_BYTES = 16 << 30
HORIZONS = (500000, 1000000, 2000000, 4000000, 25000000)


def gather(root, donor, output):
    """Reuse successful service publications and original gradient-only support."""
    import fcntl
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.prepared_support import training_input
    from .progressive_training import resolve_update_policy
    root, output = Path(root), Path(output)
    folder = confined(root, donor)
    if not folder.is_relative_to(root/'run-state') or output.resolve().is_relative_to(folder):
        raise ValueError('separate workspace diagnostic session required')
    with confined(root, folder/'session.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = read(folder/'journal.json')
        refs = {}
        for name in ('original-inputs', 'service-plan', 'service-measurements', 'budget-review'):
            step = journal['steps'].get('measure-services/'+name, {})
            if step.get('status') != 'COMPLETE':
                raise ValueError('complete original service diagnostic required: '+name)
            for ref in step['outputs']:
                checked_file(root, ref)
                refs[ref['path']] = ref
        def owned(name):
            path = folder/'measure-services/services'/name
            if refs.get(str(path.relative_to(root))) != reference(root, path):
                raise ValueError('original service publication is not journal-bound')
            return read(path)
        inputs = owned('service-inputs.json')
        review = owned('service-budget-review.json')
        req = owned('request.json')
        if req['inputs'] != inputs['identity'] or review['service_request'] != req['identity']:
            raise ValueError('original service request differs')
        materials = _json(checked_file(root, inputs['materials']))
        if materials['support'] != inputs['support']:
            raise ValueError('original support reference differs')
        support = _json(checked_file(root, inputs['support']), maximum=256 << 20)
        summary = _json(checked_file(root, materials['support_summary']))
        from oeis_learn.generalization.progressive.training_rows import validate_row
        for shard in support['shards']:
            for row in shard:
                validate_row(row, native_verified=True)
        profiles = [m['case']['profile'] for m in inputs['historical_measurements']
            if m['case']['profile']['shape']=='reference' and m['case']['profile']['model']['dropout']==0]
        if not profiles or any(p != profiles[0] for p in profiles):
            raise ValueError('one original reference profile required')
        policy = resolve_update_policy(profiles[0], schedule_horizon_tokens=400000000)
        package = training_input(support, summary, profiles[0], policy)
        retain(output/'private/training-input.json', package)
        value = seal(dict(version=VERSION, source=digest(collect_snapshot(root)),
            donor=reference(root, folder/'journal.json'), original_inputs=inputs['identity'],
            original_review=reference(root, folder/'measure-services/services/service-budget-review.json'),
            package=reference(output, output/'private/training-input.json'),
            support_summary=summary, scope='DIAGNOSTIC_ONLY; original training data; no panel access'))
        retain(output/'training-timing-inputs.json', value)
        return value


def plan(root, output, inputs, image, recipient):
    root, output = Path(root), Path(output)
    req = seal(dict(version=VERSION, source=inputs['source'], image=image,
        inputs=inputs['identity'], package=inputs['package'], seconds=SECONDS,
        updates=UPDATES, checkpoint_bytes=CHECKPOINT_BYTES,
        scope='DIAGNOSTIC_ONLY; reference recipe; no scientific checkpoint or learning claim'))
    claim = confined(root, root/'run-state/discovery-training-timing'/
        (inputs['original_inputs'][7:]+'.json'), exists=False)
    claim.parent.mkdir(parents=True, exist_ok=True)
    owner = seal(dict(request=req['identity'], recipient=str(Path(recipient).relative_to(root))))
    if claim.exists():
        if read(claim) != owner:raise ValueError('original diagnostic already owned; no sibling allowance reset')
    else:
        import json
        with claim.open('x') as stream:
            json.dump(owner, stream);stream.flush();os.fsync(stream.fileno())
        fd=os.open(claim.parent, os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    retain(output/'request.json', req)
    return req


def validate_request(req):
    if (set(req) != {'version','identity','source','image','inputs','package','seconds','updates','checkpoint_bytes','scope'}
        or seal(req)!=req or req['version']!=VERSION or req['seconds']!=SECONDS
        or req['updates']!=UPDATES or req['checkpoint_bytes']!=CHECKPOINT_BYTES
        or req['scope']!='DIAGNOSTIC_ONLY; reference recipe; no scientific checkpoint or learning claim'):
        raise ValueError('closed finite training timing request required')
    return req


def batches(package):
    """Fixed coverage sample, not a simulation of learned curriculum allocation."""
    from oeis_learn.generalization.progressive.prepared_support import bin_name
    excluded={p['id'] for p in package['layout']['probes'].values()}
    shards=[[r for r in shard if r['draw_identity'] not in excluded] for shard in package['shards']]
    groups=[('initial-familiar',[r for r in shards[0] if bin_name(r)=='familiar']),
            ('initial-frontier',[r for r in shards[0] if bin_name(r)=='frontier']),
            ('later-frontier',[r for shard in shards[1:] for r in shard if bin_name(r)=='frontier'])]
    if any(not rows for _,rows in groups):raise ValueError('initial and refreshed gradient bins required')
    count=package['update_policy']['effective_batch']
    result=[]
    for i in range(UPDATES):
        name, rows=groups[i%len(groups)]
        offset=(i//len(groups))*count
        result.append((name,[rows[(offset+j)%len(rows)] for j in range(count)]))
    allrows=[r for shard in shards for r in shard]
    result.extend([('stress-target',[max(allrows,key=lambda r:(len(r['tokens']),r['identity']))]*count),
                   ('stress-visible',[max(allrows,key=lambda r:(max(len(v) for v in r['values'][:20]),r['identity']))]*count)])
    return result


def measure(package, output, *, device='cuda', seconds=SECONDS):
    """Real canonical updates; component timings retain cold/steady separation."""
    import torch
    from oeis_learn.decoder.bigint_codec import BOS_ID, PAD_ID
    from oeis_learn.generalization.budgets import Ledger
    from oeis_learn.fresh.bigint_state import ProgressiveCheckpointStore
    from .progressive_training import PolicyUpdate, RecoverableUpdate
    from .progressive_learner import compatibility
    from .generalization_curriculum import observe_targets
    from oeis_learn.generalization.progressive.storage import preflight
    if device not in {'cpu','cuda'} or not 0<seconds<=SECONDS or seal(package)!=package:
        raise ValueError('bounded original training timing input required')
    if device=='cuda':
        from .progressive_qualification import require_hip
        from .generalization_execution import verify_execution
        require_hip();verify_execution()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();deadline=began+seconds
    def now():
        if device=='cuda':torch.cuda.synchronize()
        if time.monotonic()>=deadline:raise TimeoutError('training timing diagnostic allowance exhausted')
        return time.monotonic_ns()//1000
    result=dict(version=VERSION,device='actual_HIP' if device=='cuda' else 'cpu',status='RUNNING',
        package=package['identity'],updates=[],components={},scope='DIAGNOSTIC_ONLY; no learned capability',
        interpretation='fixed coverage of initial/future gradient bins; curriculum-dependent throughput remains uncertain')
    def publish():revise(output/'training-timing-progress.json',seal(result))
    try:
        preflight(CHECKPOINT_BYTES,64)
        sample=batches(package)
        cap=sum(sum(len(r['tokens']) for r in rows) for _,rows in sample)
        ledger=Ledger(cap)
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        start=now();runner=PolicyUpdate(package['profile'],package['update_policy'],device=device)
        result['components']['initialization_us']=now()-start;publish()
        for i,(kind,rows) in enumerate(sample):
            start=now()
            tokens=torch.full((len(rows),max(len(r['tokens']) for r in rows)+1),PAD_ID,dtype=torch.long,device=device)
            for j,row in enumerate(rows):
                tokens[j,:len(row['tokens'])+1]=torch.tensor([BOS_ID,*row['tokens']],device=device)
            sequences=[[int(v) for v in r['values'][:20]] for r in rows]
            charge=int(tokens[:,1:].ne(PAD_ID).sum())
            ledger=ledger.reserve(str(i),charge)
            retain(output/'debits'/str(i)/'started.json',seal(dict(index=i,kind=kind,ledger=ledger.snapshot())))
            packing=now()-start;start=now()
            metrics,ledger=runner.update(sequences,tokens,ledger=ledger)
            elapsed=now()-start
            result['updates'].append(dict(index=i,kind=kind,cold=i==0,charged_tokens=charge,
                packing_us=packing,update_us=elapsed,maximum_target_tokens=tokens.shape[1]-1,
                loss=float(metrics['loss']),ledger=ledger.snapshot()))
            publish()
        lookup={r['draw_identity']:r for shard in package['shards'] for r in shard}
        probes=[dict(lookup[p['id']],values=lookup[p['id']]['values'][:20]) for p in package['layout']['probes'].values()]
        start=now()
        # Production observes old/new allocation-bin probes separately.
        for row in probes:observe_targets(runner.encoder,runner.decoder,[row],device=device)
        result['components']['two_probe_us']=now()-start;publish()
        start=now()
        states=dict(weights=dict(encoder=runner.encoder.state_dict(),decoder=runner.decoder.state_dict()),
            optimizer=runner.state_dict(),scheduler={},data=dict(cursor=ledger.updates,pending_batch=None),
            replay={},history={},pins={},selection=dict(scope='DIAGNOSTIC_ONLY'),
            rng=dict(python=__import__('random').getstate(),numpy=__import__('numpy').random.get_state(),
                     torch=torch.get_rng_state(),device=torch.cuda.get_rng_state_all() if device=='cuda' else []))
        from oeis_learn.generalization.progressive.state import CUTS
        scope=digest(['DIAGNOSTIC_ONLY',package['identity']])
        store=ProgressiveCheckpointStore(output/'checkpoints',compatibility=compatibility(package),
            campaign=scope,task=scope,token_cap=cap)
        store.store.MAX_BYTES=CHECKPOINT_BYTES
        checkpoint=store.save(states,ledger,cuts={k:digest(['diagnostic',k]) for k in CUTS},
            created_at=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            purpose='acceptance',track='synthetic_strict',maximum_bytes=CHECKPOINT_BYTES)
        result['components']['capture_and_save_us']=now()-start;publish()
        del states
        # Retain the historical five-read comparator so original measurements
        # remain interpretable. Hourly learners do not perform these per update.
        result['checkpoint_read_us']=[]
        for _ in range(5):
            start=now();restored=store.load(checkpoint)
            result['checkpoint_read_us'].append(now()-start)
            del restored
            publish()
        del runner
        if device=='cuda':
            from .progressive_qualification import _cleanup_memory
            _cleanup_memory(torch)
        result['checkpoint_io']=dict(store.store.io_stats)
        # Use a fresh store/model as a restarted process would. load() always
        # verifies original hashes; the operating-system page cache is not flushed.
        store=ProgressiveCheckpointStore(output/'checkpoints',compatibility=compatibility(package),
            campaign=scope,task=scope,token_cap=cap)
        start=now();restored=store.load(checkpoint)
        runner=PolicyUpdate(package['profile'],package['update_policy'],device=device)
        runner.encoder.load_state_dict(restored['states']['weights']['encoder'])
        runner.decoder.load_state_dict(restored['states']['weights']['decoder'])
        runner.load_state_dict(restored['states']['optimizer'])
        RecoverableUpdate._rng_restore(restored['states']['rng'])
        if restored['ledger'].snapshot()!=ledger.snapshot():raise ValueError('restored timing ledger differs')
        result['components']['restore_into_fresh_model_us']=now()-start
        result['checkpoint']=checkpoint
        result['checkpoint_files']=[reference(output,p) for p in sorted((output/'checkpoints').rglob('*')) if p.is_file()]
        if device=='cuda':result['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        start=now();del restored,runner,tokens
        if device=='cuda':
            from .progressive_qualification import _cleanup_memory
            result['cleanup']=_cleanup_memory(torch)
            if result['cleanup']['allocated_bytes']!=0:raise ValueError('owned HIP tensors remain after diagnostic cleanup')
        result['components']['cleanup_us']=now()-start
        result.update(status='PASS',actual_elapsed_seconds=time.monotonic()-began)
        publish();retain(output/'producer.json',seal(result))
        return seal(result)
    except BaseException as error:
        result.update(status='FAIL',error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                      actual_elapsed_seconds=time.monotonic()-began)
        publish();raise


def run_inside(root):
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .progressive_qualification import hardware_facts
    root=Path(root);req=validate_request(read(root/'request.json'))
    if req['source']!=digest(source_snapshot()) or req['image']!=os.environ.get('OEIS_QUALIFICATION_IMAGE'):
        raise ValueError('current installed timing source/image required')
    retain(root/'hardware.json',hardware_facts())
    package=_json(checked_file(root,req['package']),maximum=256<<20)
    return measure(package,root/'measurement')


def report(root, inputs):
    root=Path(root);req=validate_request(read(root/'request.json'))
    if inputs['identity']!=req['inputs']:raise ValueError('original timing inputs required')
    path=root/'measurement/producer.json'
    producer=read(path) if path.exists() else None
    package=_json(checked_file(root,req['package']),maximum=256<<20)
    value=dict(version=VERSION,status='INCOMPLETE',expected_total_seconds=None,
        request=req['identity'],source=req['source'],image=req['image'],executable=False,
        scientific_budgets_changed=False,assessment_exposed=False,
        refresh_updates=package['refresh_updates'],support_bounds=inputs['support_summary']['bounds'],
        note='Wall-time sizing diagnostic only. Learning speed/useful horizon needs a separate prospectively registered pilot.')
    if producer:
        if producer['package']!=package['identity'] or producer['status']!='PASS':raise ValueError('original complete timing producer required')
        rows=producer['updates']
        if len(rows)!=UPDATES+2:raise ValueError('complete timing update roster required')
        ordinary=[r for r in rows[:UPDATES] if not r['cold']]
        elapsed=sum(r['packing_us']+r['update_us'] for r in ordinary)/1e6
        tokens=sum(r['charged_tokens'] for r in ordinary)
        rates=[r['charged_tokens']*1e6/(r['packing_us']+r['update_us']) for r in ordinary]
        mean_charge=tokens/len(ordinary)
        initial=[r for r in ordinary if r['kind'].startswith('initial-')]
        initial_charge=sum(r['charged_tokens'] for r in initial)/len(initial)
        components=producer['components']
        value.update(status='MEASURED',device=producer['device'],producer=reference(root,path),
            steady_tokens_per_second=tokens/elapsed,observed_update_rate_range=[min(rates),max(rates)],
            measured_components_seconds={k:v/1e6 for k,v in components.items()},
            mean_charged_tokens_per_update=mean_charge,
            first_refresh_tokens_projection=package['refresh_updates']*initial_charge,
            checkpoint_reads_per_update=5,
            checkpoint_projection_policy='historical per-update comparator; hourly estimate is separate',
            hourly_horizons=[hourly_projection(producer,n) for n in HORIZONS],
            checkpoint_read_seconds=[v/1e6 for v in producer['checkpoint_read_us']],
            horizons=[dict(tokens=n,update_only_seconds=n*elapsed/tokens,
                observed_rate_seconds_range=[n/max(rates),n/min(rates)],
                two_probes_every_eight_updates_seconds=(n/mean_charge/8)*components['two_probe_us']/1e6,
                per_update_checkpoint_seconds=(n/mean_charge)*(components['capture_and_save_us']+sum(producer['checkpoint_read_us']))/1e6,
                training_component_projection_seconds=n*elapsed/tokens+(n/mean_charge)*
                    (components['two_probe_us']/8+components['capture_and_save_us']+sum(producer['checkpoint_read_us']))/1e6)
                for n in HORIZONS],
            caveats=['No safety multiplier included; observed rate range is not a confidence interval.',
                'Initial/future bin coverage is fixed, not a prediction of learned allocation.',
                'Legacy horizons retain the original five-read per-update comparator; hourly_horizons describe the new policy without relabeling old measurements.',
                'Journal/quarantine growth, retention scans and long-run filesystem pressure remain unmeasured; this is a component projection, not full-learner throughput.',
                'Fresh-model restore cost applies to startup/recovery, not every ordinary update.',
                'Restore uses a fresh store/model; the operating-system file cache is not flushed.',
                'Evaluation, qualification and setup are separate; shorter training does not reduce fixed evaluation work.',
                'First-refresh projection is approximate; use original bounds to guarantee intervention exposure.',
                'This brief diagnostic provides no evidence that 500k, 1M or 2M tokens are scientifically sufficient.'])
    revise(root/'training-runtime-review.json',seal(value))
    return seal(value)


def hourly_projection(producer, tokens):
    """Unpadded component arithmetic, explicitly not a new HIP measurement."""
    from .generalization_persistence import INTERVAL
    from oeis_learn.generalization.progressive.milestones import MILESTONES
    if type(tokens) is not int or tokens<1:raise ValueError('positive token horizon required')
    rows=[r for r in producer['updates'][:UPDATES] if not r['cold']]
    charge=sum(r['charged_tokens'] for r in rows)
    mean_charge=charge/len(rows)
    update_seconds=tokens*sum(r['packing_us']+r['update_us'] for r in rows)/charge/1e6
    m=producer['components']
    probes=(tokens/mean_charge/8)*m['two_probe_us']/1e6
    # The hourly timer restarts after each successful snapshot. Initial and final
    # cuts are included; a final milestone is not counted twice.
    snapshots=2+int((update_seconds+probes)//INTERVAL)+sum(0<n<tokens for n in MILESTONES)
    verified_snapshot=(m['capture_and_save_us']+sum(producer['checkpoint_read_us'])/len(producer['checkpoint_read_us']))/1e6
    return dict(tokens=tokens,checkpoint_policy='hourly/v1',projected_snapshots=snapshots,
        update_only_seconds=update_seconds,probe_seconds=probes,checkpoint_seconds=snapshots*verified_snapshot,
        training_component_projection_seconds=update_seconds+probes+snapshots*verified_snapshot+m['initialization_us']/1e6,
        measured_full_learner=False,
        caveat='Original component timings only; journal/quarantine/retention and allocation-dependent throughput need current-source HIP measurement. Startup restore, evaluation, images and qualification are separate.')
