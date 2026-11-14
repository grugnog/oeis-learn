"""Replay complete original calibration under an explicit host-only revision."""
import copy
import json
from functools import lru_cache
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES, digest, reference, seal, record, validate, verify_refs,
)
from .identity import confined
from .progressive_research_inputs import _json, checked_file
from .preparation_recovery import retain

NAME = 'calibration_reuse'
VERSION = 'progressive-calibration-reuse/v1'


def named(document, name):
    refs = [r['artifact'] for r in document['payload']['references'] if r['name'] == name]
    if len(refs) != 1:
        raise ValueError('one original '+name+' required')
    return refs[0]


def source_authority(root, proof, qualification):
    """Current Q authorizes execution; original Q only authenticates measurements."""
    from .progressive_handoff import original_authority, validate_device_proofs
    from .progressive_calibration_equivalence import validate as equivalent
    fields={'version','identity','qualification','original_qualification','source','reviewed_host_changes','blueprint','prepared','request','batch','original_costs_retained'}
    if (set(proof)!=fields or seal(proof) != proof or proof.get('version') != VERSION or proof['qualification'] != qualification
            or proof['original_costs_retained'] is not True):
        raise ValueError('closed original calibration reuse binding required')
    oldq, oldroot = original_authority(root, proof['original_qualification'])
    q, qroot = original_authority(root, qualification)
    oldsource = _json(checked_file(oldroot, oldq['source_closure']))
    source = _json(checked_file(qroot, q['source_closure']))
    changes = equivalent(oldsource, source)
    if proof['reviewed_host_changes'] != changes or proof['source'] != digest(source):
        raise ValueError('reviewed calibration source migration differs')
    req = _json(checked_file(root, proof['request']))
    matrix, _ = validate_device_proofs(qroot, q, source=digest(source), image=q['images'][1], engine=req['engine'])
    if (req['source'] != digest(oldsource) or req['image'] != oldq['images'][1]
            or req['hardware'] != matrix['hardware'] or req['execution_policy'] != q['runtime']['execution_policy']
            or oldq['runtime'] != q['runtime'] or oldq['resources'] != q['resources']
            or oldq['engine_security'] != q['engine_security']):
        raise ValueError('same original hardware, numerical/runtime and engine policy required')
    return req


def build_proof(root, original_base, qualification):
    from .progressive_handoff import original_authority
    from .progressive_calibration_equivalence import validate as equivalent
    base = Path(original_base)
    oldref = reference(root, base/'qualification/qualification.json')
    oldq, oldroot = original_authority(root, oldref)
    q, qroot = original_authority(root, qualification)
    source = _json(checked_file(qroot, q['source_closure']))
    proof = seal(dict(version=VERSION, qualification=qualification, original_qualification=oldref,
        source=digest(source), reviewed_host_changes=equivalent(_json(checked_file(oldroot, oldq['source_closure'])), source),
        blueprint=reference(root, base/'construction/blueprint.json'),
        prepared=reference(root, base/'prepared/prepared.json'),
        request=reference(root, base/'prepared/calibration/request.json'),
        batch=reference(root, base/'prepared/calibration/producer.json'), original_costs_retained=True))
    source_authority(root, proof, qualification)
    return proof


def raw_prepared(root, blueprint):
    plan = _json(checked_file(root, named(blueprint, 'preparation_plan')), maximum=CAMPAIGN_DOCUMENT_BYTES)
    if set(plan) != {'version', 'payload'} or plan['version'] != 'progressive-preparation-plan/v1':
        raise ValueError('original prospective preparation plan required')
    return record('prepared', dict(plan['payload'], blueprint=blueprint['identity']),
        created_at=blueprint['created_at'], purpose=blueprint['purpose'], track=blueprint['track'])


def assemble(root, blueprint, original, output):
    from .progressive_completed_calibration_recovery import originals
    from .progressive_calibration import selected_inputs
    from .progressive_research_resources import finalize
    from .research_limits import padded_us
    from .progressive_handoff import named_plan
    proof = _json(checked_file(root, named(original, NAME)))
    runtime = _json(checked_file(root, original['payload']['runtime']))
    request = source_authority(root, proof, runtime['payload']['qualification'][0])
    base = confined(root, proof['blueprint']['path']).parent.parent
    for name, path in [('blueprint', base/'construction/blueprint.json'), ('prepared', base/'prepared/prepared.json'),
                       ('request', base/'prepared/calibration/request.json'), ('batch', base/'prepared/calibration/producer.json')]:
        if proof[name] != reference(root, path):
            raise ValueError('original calibration proof path differs')
    _, previous, old_request, producers = originals(root, base)
    if old_request != request:
        raise ValueError('original complete calibration request differs')
    reservation = named_plan(root, blueprint, 'iteration_calibration_plan', 'progressive-calibration-plan/v1',
        {'seconds', 'caps', 'checkpoint_bytes', 'maximum_inodes'}, optional={'checkpoint_retention'})
    if not reservation or reservation['caps'] != request['caps']:
        raise ValueError('full original calibration debit must remain reserved')
    payload = copy.deepcopy(original['payload'])
    cases, bindings = selected_inputs(root, original)
    if {c['compatibility'] for c in cases} != {p['compatibility'] for p in producers}:
        raise ValueError('exact complete original input scope required; no partial calibration reuse')
    oldrefs = previous['payload']['references']
    refs = {r['name'].removeprefix('iteration:'): r['artifact'] for r in oldrefs if r['name'].startswith('iteration:')}
    budgets = {p['compatibility']: dict(producer=refs[p['compatibility']],
        **{k: padded_us(v) for k, v in p['measurements'].items()}) for p in producers}
    for b in bindings:
        task = payload['task_plans'][b['task']]
        if b['choice'] is not None: task = task['trial_choices'][b['choice']]
        task['iteration_budget'] = budgets[b['case']]
    payload['references'].extend(r for r in oldrefs if r['name'].startswith('iteration:') or r['name'] in {
        'iteration_calibration', 'iteration_calibration_request', 'iteration_calibration_input', 'iteration_calibration_cost'})
    # Producers and cases are independently sorted by compatibility before the
    # estimator associates per-setting measurements with training inputs.
    by_scope = {p['compatibility']: p for p in producers}
    resources = finalize(root, payload, cases, [by_scope[c['compatibility']] for c in cases])
    if resources is None:
        raise ValueError('prospective trajectory resource finalization required')
    return seal(dict(original, payload=payload)), resources


def prepare(root, blueprint, prepared, output, args, host_factory):
    from .generalization_qualification_io import current_facts
    from .progressive_handoff import original_authority
    from oeis_learn.generalization.progressive.campaign import validate_runtime
    from oeis_learn.generalization.progressive.freeze import trial_runtime
    from oeis_learn.generalization.progressive.trial_scopes import variants
    completed=Path(output)/'prepared.json'
    if completed.exists():
        value=_json(completed,maximum=CAMPAIGN_DOCUMENT_BYTES)
        validate_prepared(root,blueprint,value)
        return value
    qref = reference(root, confined(root, args.qualification)/'qualification.json')
    q, qroot = original_authority(root, qref)
    inspection = Path(output)/'reuse-inspection'
    # A fresh preparation invocation owns this inspection; an incomplete old
    # invocation is never silently accepted as a fresh device observation.
    if inspection.exists():
        raise ValueError('existing calibration reuse inspection requires explicit recovery')
    facts = current_facts(host_factory(args), root, inspection, images=q['images'],
        source_ref=q['source_closure'], build_refs=q['build_dependencies'], producer_root=qroot)
    runtime = _json(checked_file(root, prepared['payload']['runtime']))
    for task in prepared['payload']['task_plans']:
        for variant in variants(task):
            if variant['kind'] == 'training':
                validate_runtime(trial_runtime(root, variant, runtime), root=root, current_facts=facts, inspected=True)
    result, resources = assemble(root, blueprint, prepared, output)
    retain(Path(output)/'resource-estimate.json', resources)
    result['payload']['references'].append(dict(name='measured_research_resources', role='custodian', kind='source',
        artifact=reference(root, Path(output)/'resource-estimate.json')))
    return seal(result)


def validate_prepared(root, blueprint, prepared):
    original = raw_prepared(root, blueprint)
    output = confined(root, named(prepared, 'measured_research_resources')['path']).parent
    expected, resources = assemble(root, blueprint, original, output)
    rref = named(prepared, 'measured_research_resources')
    if _json(checked_file(root, rref)) != resources:
        raise ValueError('replayed trajectory resource estimate differs')
    expected['payload']['references'].append(dict(name='measured_research_resources', role='custodian', kind='source', artifact=rref))
    if seal(expected) != prepared:
        raise ValueError('reused prepared publication differs from original replay')


def producer_authority(root, references, qualification, producer_ref):
    """Called after full preparation replay; still bind each original producer."""
    holder = dict(payload=dict(references=references))
    proof = _json(checked_file(root, named(holder, NAME)))
    # Qualification/source migration is replayed once by assemble(), then every
    # case must be in the original hash-bound successful batch at freeze.
    if seal(proof) != proof or proof.get('version') != VERSION or proof['qualification'] != qualification:
        raise ValueError('current qualification-bound original calibration required')
    def stamp(ref):
        p=confined(root,ref['path']); st=p.stat()
        return (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
    source,image,inventory=_inventory(str(root),json.dumps(proof,sort_keys=True),
        json.dumps(qualification,sort_keys=True),stamp(proof['request']),stamp(proof['batch']),stamp(qualification))
    base = confined(root, proof['batch']['path']).parent
    relative = dict(producer_ref, path=str(confined(root, producer_ref['path']).relative_to(base)))
    if (relative['path'],relative['bytes'],relative['sha256']) not in inventory:
        raise ValueError('producer not in original calibrated inventory')
    return source, [image]


@lru_cache(maxsize=4)
def _inventory(root, encoded, qualified, request_stamp, batch_stamp, qualification_stamp):
    # Cache only small authenticated immutable records within one invocation;
    # changed file metadata or references force full source/authority replay.
    root=Path(root);proof=json.loads(encoded);qualification=json.loads(qualified)
    request=source_authority(root,proof,qualification)
    batch=_json(checked_file(root,proof['batch']))
    if seal(batch)!=batch or batch.get('status')!='PASS' or batch['request']!=request['identity']:
        raise ValueError('original complete calibration batch required')
    inventory=frozenset((r['path'],r['bytes'],r['sha256']) for r in batch['producers'].values())
    return request['source'],request['image'],inventory
