"""Deterministic mechanical examples; expected values use Python formulae, not project code.

Authority hashes are intentionally synthetic, well-shaped placeholders. These examples
are NOT admitted programs, resumable checkpoints or qualification evidence.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
H = 'sha256:' + '0' * 64
R = dict(path='fixture/payload.json', sha256=H, bytes=1)
B = dict(target_tokens=20000, attempts=8, seconds=60, draws=10000,
         memory_bytes=1048576, storage_bytes=1048576, log_bytes=1024)
M = dict(profile='fixture', initialization='random', architecture=H)
C = dict(milestones=[4000, 8000, 16000, 25000], retain=['best', 'latest', 'selected'])
SELECTION = dict(cohort=R, completion_floor=.95, objective='structural_family_macro_confirmed',
                 tie_breaks=['parameter_family_macro_confirmed', 'earlier_tokens', 'identity'])
ASSESSMENT = dict(cohort=R, authorization=R, exposure_ledger=R, feedback_to_selection=False)
INDICES = [str(i) for i in range(100)]
VALUES = [str(2*i+1) for i in range(100)]
SOURCE = '(program (return (add (mul (int 2) $index) (int 1))))'


def write(name, value):
    (ROOT/name).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def main():
    authority = {
        'study': dict(recipe='P1-refresh-screen/v1', track='generic_only', hypothesis='fixture',
                      changing_factors=['refresh'], arms=['static', 'refreshed'], seeds=[20261016],
                      source_closure=R, model=M, precision='fp32',
                      data=dict(generator=R, descriptors=R, manifests=[R], target_format='literal',
                                sharing_policy='occurrence', exclusions=R),
                      selection=SELECTION, assessment=ASSESSMENT,
                      inference=dict(visible_terms=20, total_terms=100, attempts=8, seconds=60,
                                     policy='portfolio-clock60-matched-drain/v2'),
                      budgets=B, checkpoints=C,
                      progression=dict(stage='P1-screen', advance_rule='selection-only',
                                       stop_rule='budget', assessment_feedback=False)),
        'preparation': dict(generator=R, numeric_law='uniform[-16,16]', descriptors=R,
                            roles=['training','selection','assessment'], quotas={'fixture':1},
                            exclusions=R, seeds=[20261016], budgets=B,
                            study_settings=dict(recipe='fixture', arms=['static','refreshed'],
                                                model=M, precision='fp32', checkpoints=C)),
        'cohort': dict(role='assessment', family_rule='fixture/v1', sources=[H], variants=['v0'],
                       groups=[H], quotas={'fixture':1}, coverage={'fixture':1}, exclusions=R,
                       indices=INDICES, public_payloads=[R], private_payloads=[R], preparation_ledger=R),
        'refresh': dict(descriptor_set=H, shards=[R], numeric_law='uniform[-16,16]', seed=20261016,
                        draws=1, rejections={}, cursor=dict(shard=0,row=0), exhaustion='stop'),
        'checkpoint': dict(kind='full_state', weights=R, state=R, initialization=H, parent=None,
                           tokens=0, updates=0, pins=['latest'], compatibility=H),
        'replay': dict(task=H, role='training', track='generic_only', origin_checkpoint=R,
                       original_source=SOURCE, final_source=SOURCE, assignment={}, tokens=[1],
                       verification=dict(oracle=R, production=R, reference=R,
                                         indices=INDICES, values=VALUES),
                       admission_order=1, task_cap=2, capacity=4096, batch_fraction=.25,
                       sampler_cursor=0, eviction_history=[]),
        'qualification': dict(suite_selection=['fixture'], covered_capabilities=['fixture'],
                              source_closure=R, build_dependencies=[R], images=[H],
                              runtime=dict(hip='synthetic',driver='synthetic',python='synthetic',torch='synthetic'),
                              hardware=dict(hip_available=True,device='synthetic-not-real',memory_bytes=1),
                              precision='fp32',resources=B,
                              engine_security=dict(engine='docker',runtime_socket=False),
                              producer=dict(receipt=R,status='PASS',artifacts=[R]),
                              artifact_inventory=[R],status='PASS'),
        'evidence': dict(study=R,effective_invocation=R,qualification_producers=[R],stage_ledger=R,
                         identities=[H],coverage=R,attempts=R,families=R,seeds=R,summary=R,selection=R,
                         exposure=R,costs=R,bounds=R,interpretation='shape fixture only',inventory=[R],
                         omissions=['no real evidence'],software_status='BLOCKED',scientific_status='UNMEASURED'),
    }
    for name, record in authority.items():
        record.update(version=f'generalization-{name}/v1',identity=H)
    write('authority.json',authority)
    write('payloads.json',{
        'visible-prompt':dict(id='affine',indices=INDICES[:20],values=VALUES[:20]),
        'training-target':dict(id='affine',role='training',source=SOURCE,tokens=[1],indices=INDICES,values=VALUES),
        'scorer-target':dict(id='affine',role='assessment',source=SOURCE,family=H,stratum='fixture',indices=INDICES,values=VALUES),
    })
    write('role-expectations.json',{
        'preparation':['mechanical_sources','isolation_truth'],
        'learner':['training_targets','verified_training_replay'],
        'generator':['visible_prompts','frozen_weights'],
        'fitter':['visible_prompts','generated_candidates'],
        'selection':['selection_supervision','selection_candidates'],
        'assessment':['assessment_truth','frozen_assessment_authorization','frozen_assessment_candidates'],
        'report':['public_evidence'],
    })
    # Sources come from a generic affine template and a bounded counter loop, not named sequences.
    cases=[]
    for a,b in [(2,1),(-3,4),(0,7)]:
        cases.append(dict(id=f'affine-{a}-{b}',
            source=f'(program (return (add (mul (int {a}) $index) (int {b}))))',
            parameter_source='(program (return (add (mul (param.get p0) $index) (param.get p1))))',
            assignment={'p0':str(a),'p1':str(b)},indices=[str(n) for n in range(6)],
            expected=[str(a*n+b) for n in range(6)]))
    for bound in ['$index','(int 0)']:
        cases.append(dict(id='active-loop' if bound=='$index' else 'inactive-loop',
            source=f'(program (local $b0 bigint) (local $c0 control) (set $b0 (int 0)) (loop $c0 (to_control {bound}) (set $b0 (add (get $b0) (int 1)))) (return (get $b0)))',
            indices=['0','1','2','3'],expected=['0','1','2','3'] if bound=='$index' else ['0']*4,
            expected_loop_iterations=[0,1,2,3] if bound=='$index' else [0]*4))
    cases.append(dict(id='branch',source='(program (local $b0 bigint) (if (lt $index (int 2)) (then (set $b0 (int 3))) (else (set $b0 (int 5)))) (return (get $b0)))',
                      indices=['0','1','2','3'],expected=['3','3','5','5'],expected_branches=['then','then','else','else']))
    write('mechanical-cases.json',cases)
    write('policy-expectations.json',{
        'metrics':dict(families={'small':{'successes':1,'unresolved':0,'total':1},'large':{'successes':1,'unresolved':1,'total':3}},
                       row_lower='1/2',row_upper='3/4',macro_lower='2/3',macro_upper='5/6'),
        'replay':dict(target_indices=INDICES,target_values=['0']*100,candidate_values=['0']*20+['1']*80,
                      expected_admitted=False,reason='training_audit_disagreement'),
        'isolation':dict(training_group='duplicate-1',assessment_group='duplicate-1',expected_admitted=False),
        'resume':dict(batch_ids=['a','b','c'],checkpoint_after=1,pending_batch='b',expected_remaining=['b','c'],
                      tokens_consumed=4,budget=12,expected_remaining_tokens=8,weights_only_resume=False),
        'coverage':dict(inactive_control_flow_satisfies_quota=False,impossible_quota_outcome='SHORTFALL'),
    })
    names=['authority.json','payloads.json','role-expectations.json','mechanical-cases.json','policy-expectations.json']
    write('inventory.json',{'version':'generalization-fixtures/v1','scope':'synthetic-contract-and-reference-only',
                           'files':{name:{'bytes':(ROOT/name).stat().st_size,
                                          'sha256':'sha256:'+hashlib.sha256((ROOT/name).read_bytes()).hexdigest()} for name in names}})


if __name__ == '__main__':
    main()
