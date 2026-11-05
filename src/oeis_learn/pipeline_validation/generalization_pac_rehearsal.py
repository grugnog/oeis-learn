"""Bounded R0 composition of existing learners/proofs; never a research initializer."""
import copy
import math
import re
import time
from pathlib import Path
from .identity import seal,digest,write

UPDATES=256
BOUNDARY=129


def verify_rehearsal(record,*,source=None,environment=None):
    """Stdlib-only reconciliation of the public diagnostic disposition/counts."""
    fields={'version','identity','status','scientific_status','purpose','device','source','environment','study','inventory','initializer',
            'seed','batch_size','updates','target_tokens','cadence','runs','proofs','resume','production_retention_validated'}
    if not isinstance(record,dict) or set(record)!=fields or seal(record)!=record:raise ValueError('closed sealed rehearsal required')
    constants=dict(version='pac-rehearsal/v1',status='PASS',scientific_status='UNMEASURED',purpose='diagnostic',device='hip',
                   seed=20261016,batch_size=32,updates=UPDATES,target_tokens=25000000,cadence='pac-exp3s-leitner/v1',production_retention_validated=False)
    if any(type(record[k]) is not type(v) or record[k]!=v for k,v in constants.items()):raise ValueError('R0 scope/cadence differs')
    for key in ('source','environment','study','inventory','initializer'):
        if not isinstance(record[key],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',record[key]):raise ValueError('R0 producer identity')
    if source is not None and record['source']!=source or environment is not None and record['environment']!=environment:raise ValueError('R0 producer changed')
    if set(record['runs'])!=set('BCD'):raise ValueError('all three diagnostic arms required')
    keys={'training_status','updates','channels','tokens','charged_tokens','discarded_tokens','valid_rewards','initial_weights',
          'checkpoint','finite','changed','forward_loss_sum','forward_tokens','pauses'}
    for arm,r in record['runs'].items():
        if set(r)!=keys or r['training_status']!='PARTIAL' or r['finite'] is not True or r['changed'] is not True or r['pauses']!=[]:raise ValueError('diagnostic learner/proof incomplete')
        for key in ('updates','tokens','charged_tokens','discarded_tokens','valid_rewards','forward_tokens'):
            if type(r[key]) is not int or r[key]<0:raise ValueError('integer diagnostic counts')
        if r['updates']!=UPDATES or r['discarded_tokens'] or not 0<r['tokens']==r['charged_tokens']<25000000 or r['forward_tokens']<1:raise ValueError('R0 token/update accounting')
        for key in ('initial_weights','checkpoint'):
            if not isinstance(r[key],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',r[key]):raise ValueError('R0 tensor identities')
        if r['initial_weights']!=record['initializer']:raise ValueError('fresh identical initial tensor content required')
        channels=r['channels']
        if any(type(n) is not int or n<0 for n in channels.values()) or sum(channels.values())!=UPDATES:raise ValueError('R0 channels do not reconcile')
        if arm=='B':expected=channels=={'fixed':256} and r['valid_rewards']==0
        elif arm=='C':expected=set(channels)=={'bandit','review','fallback'} and channels['bandit']==192 and channels['review']+channels['fallback']==64 and r['valid_rewards']==24
        else:expected=channels=={'bandit':256,'review':0,'fallback':0} and r['valid_rewards']==31
        # D's deliberate stop is after gradient256, before its 32nd post-probe.
        if not expected:raise ValueError('production R0 channel/feedback clock differs')
        if type(r['forward_loss_sum']) not in (int,float) or not math.isfinite(r['forward_loss_sum']) or r['forward_loss_sum']<0:raise ValueError('finite loaded forward required')
    expected=dict(arm='C',boundary=BOUNDARY,updates=UPDATES,equal=True,pac_equal=True)
    if record['resume']!=expected or any(type(record['resume'][k]) is not type(v) for k,v in expected.items()):raise ValueError('exact in-block resume proof required')
    proofs=record['proofs']
    if set(proofs)!=set('BCD')|{'resume'}:raise ValueError('all installed tensor proofs required')
    for arm,p in proofs.items():
        expected_keys={'identity','status','runs','resume_equal'}|({'pac_equal'} if arm=='resume' else set())
        if set(p)!=expected_keys or seal(p)!=p or p['status']!='PASS' or len(p['runs'])!=(2 if arm=='resume' else 1):raise ValueError('sealed tensor proof required')
        if arm!='resume':
            r=record['runs'][arm];run=p['runs'][0]
            if set(run)!={'checkpoint','weights','initial','updates','tokens','finite','changed'} or run['checkpoint']!=r['checkpoint'] or run['updates']!=r['updates'] or run['tokens']!=r['tokens'] or run['finite'] is not True or run['changed'] is not True or p['resume_equal'] is not None:raise ValueError('tensor proof differs from diagnostic measurements')
            for k in ('weights','initial'):
                if not re.fullmatch(r'sha256:[0-9a-f]{64}',run[k]):raise ValueError('tensor proof hash')
    p=proofs['resume']
    if p['resume_equal'] is not True or p['pac_equal'] is not True or p['runs'][0]!=proofs['C']['runs'][0]:raise ValueError('resume proof is not bound to C trace')
    if any(p['runs'][0][k]!=p['runs'][1][k] for k in ('weights','initial','updates','tokens','finite','changed')):raise ValueError('resumed tensor/debit proof differs')
    return record


def rehearsal_envelope(envelope):
    """Enclose all five invocations and four full-state proofs, plus final loads."""
    from .generalization_execution import pac_role_envelope
    for key in ('learner','proof'):pac_role_envelope(envelope[key])
    value=dict(envelope['learner'])
    # C paused+resumed has a cumulative allowance, but reserve both conservatively.
    value['work']=5*envelope['learner']['outer']+4*envelope['proof']['outer']
    value['outer']=sum(value[k] for k in ('work','startup','publication','projection','cleanup'))
    return pac_role_envelope(value)


def production_batches(root,bundle,arm):
    from .generalization_training import PacPreparedBatches
    batches=PacPreparedBatches(root,bundle,arm=arm,batch_size=32,updates=25000000)
    from oeis_learn.generalization.contracts import pac_record
    pac_record('inventory',batches.inventory)
    if len(batches.families)!=32 or len(batches.rows)!=32768:raise ValueError('original full production pool required')
    return batches


def rehearse(inputs,output,settings,*,seconds):
    """Real production-shape diagnostic runs, clean in-block resume and final loads."""
    import json
    import torch
    from .generalization_execution import verify_execution
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .generalization_role import learner
    from .generalization_acceptance import check_training
    from .generalization_evaluation import load_retained
    from .generalization_curriculum import observe_targets
    from .generalization_runtime import learner_config
    from .generalization_preparation import reference
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import remaining_seconds
    fields={'recipe','source','environment','bundle','study','runtime'}
    if set(settings)!=fields:raise ValueError('closed rehearsal request')
    study=settings['study'];bundle=settings['bundle'];recipe=settings['recipe']
    if study['arms']!=['B','C','D'] or study['seed']!=20261016 or study['target_tokens']!=25000000 or settings['runtime']['envelope']['batch_size']!=32:raise ValueError('frozen BCD production request required')
    if settings['source']!=study['source'] or settings['source']!=digest(source_snapshot()):raise ValueError('installed rehearsal source mismatch')
    if digest({k:v for k,v in recipe['model'].items() if k!='seed'})!=study['model'] or digest(recipe['learning'])!=study['optimizer']:raise ValueError('production model/optimizer changed')
    from .generalization_execution import verify_pac_runtime
    verify_pac_runtime(study,settings['runtime'])
    if settings['environment']!=digest(dict(images=settings['runtime']['images'],engine=settings['runtime']['engine'],backend=study['backend'])):raise ValueError('R0 environment changed')
    minimum=5*(settings['runtime']['envelope']['learner']['work']+settings['runtime']['envelope']['learner']['publication'])+4*settings['runtime']['envelope']['proof']['work']
    if seconds<minimum:raise ValueError('rehearsal parent clips cumulative child budgets')
    require_hip()
    from .generalization_training import storage_preflight
    storage_preflight(recipe,study,settings['runtime'],extra_generations=16)
    root=inputs['training_targets'];output=Path(output);output.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+seconds;runs={};proofs={};resume_proof=None
    def proof_run(folder,value):
        return dict(directory=str(folder.relative_to(output)),compatibility=value['compatibility'],cap=25000000,
                    checkpoint=value['checkpoint'],roster=reference(folder/'retained.json',folder))
    for arm in 'BCD':
        batches=production_batches(root,bundle,arm)
        if batches.inventory['identity']!=study['inventory']:raise ValueError('rehearsal pool inventory changed')
        config=learner_config(study,settings['runtime'],recipe,arm,study['seed'],prepared=bundle)
        # Distinct diagnostic compatibility; never usable as a research initializer.
        config=copy.deepcopy(config);config['study']=digest(['R0',study['identity']]);config['pac']['purpose']='acceptance'
        def run(name,stop,prior=None):
            target=output/'private'/name
            value=learner(dict(training_targets=root,verified_training_replay=prior[0] if prior else root),target,
                dict(config=config,resume=prior[1]['checkpoint'] if prior else None,stop_after_updates=stop),
                seconds=min(config['seconds'],remaining_seconds(deadline)))
            if value['status']!='PARTIAL' or value['reason']!='PARTIAL' or value['updates']!=stop or value['device']!='cuda' or value['discarded_tokens'] or value['pac']['pauses']:
                raise ValueError('intentional real diagnostic boundary not reached')
            folder=target/'private/learner/checkpoints'
            loaded=GeneralizationCheckpointStore(folder,value['compatibility'],25000000).load(value['checkpoint'])
            require_device_rng(loaded)
            return folder,value,loaded
        print('rehearsal: '+arm+' full',flush=True)
        full=run(arm+'-full',UPDATES)
        if full[1]['initial_weights']!=study['initializer']:raise ValueError('diagnostic initial weights differ from frozen research initializer')
        proof=check_training({'verified_training_replay':output},output/('proof-'+arm),dict(runs=[proof_run(*full[:2])],compare=False),seconds=remaining_seconds(deadline))
        proofs[arm]=proof
        if arm=='C':
            paused=run('C-paused',BOUNDARY)
            state=paused[2]['states']['scheduler']['pac']['policy']
            if state['block']['phase']!='updates' or state['block']['completed']!=1:raise ValueError('R0 missed in-block pause')
            resumed=run('C-resumed',UPDATES,paused)
            resume_proof=check_training({'verified_training_replay':output},output/'proof-resume',
                dict(runs=[proof_run(*full[:2]),proof_run(*resumed[:2])],compare=True),seconds=remaining_seconds(deadline))
            proofs['resume']=resume_proof
            del paused,resumed
        roster=json.loads((full[0]/'retained.json').read_text());descriptor=next(c for c in roster['checkpoints'] if c['state']==full[1]['checkpoint'])
        encoder,decoder,_=load_retained(full[0],descriptor['weights'],recipe,device='cuda')
        forward=observe_targets(encoder,decoder,[batches.probe_rows[ids[0]] for ids in batches.probes.values()],device='cuda')
        del encoder,decoder
        value=full[1]
        runs[arm]=dict(training_status=value['status'],updates=value['updates'],channels=value['pac']['channels'],tokens=value['tokens'],
            charged_tokens=value['charged_tokens'],discarded_tokens=value['discarded_tokens'],valid_rewards=value['pac']['valid_rewards'],
            initial_weights=value['initial_weights'],checkpoint=proof['runs'][0]['checkpoint'],finite=proof['runs'][0]['finite'],changed=proof['runs'][0]['changed'],
            forward_loss_sum=forward['loss_sum'],forward_tokens=forward['tokens'],pauses=value['pac']['pauses'])
    result=seal(dict(version='pac-rehearsal/v1',status='PASS',scientific_status='UNMEASURED',purpose='diagnostic',device='hip',source=settings['source'],
        environment=settings['environment'],study=study['identity'],inventory=study['inventory'],initializer=study['initializer'],seed=study['seed'],
        batch_size=32,updates=UPDATES,target_tokens=25000000,cadence='pac-exp3s-leitner/v1',runs=runs,proofs=proofs,
        resume=dict(arm='C',boundary=BOUNDARY,updates=UPDATES,equal=resume_proof['resume_equal'],pac_equal=resume_proof['pac_equal']),production_retention_validated=False))
    verify_rehearsal(result,source=settings['source'],environment=settings['environment'])
    write(output/'rehearsal.json',result);return result


def require_hip():
    import torch
    from .generalization_execution import verify_execution
    verify_execution()
    if not torch.version.hip or not torch.cuda.is_available():raise RuntimeError('actual HIP required for R0')


def require_device_rng(loaded):
    if not loaded['states']['rng']['device']:raise ValueError('actual HIP RNG required')
