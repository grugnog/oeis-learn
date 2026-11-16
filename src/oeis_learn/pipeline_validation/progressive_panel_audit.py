"""Read-only diagnosis of the value-derived cross-panel freeze guard.

This stdlib module can be copied to run-state and executed against an unchanged
qualified checkout. It uses that checkout's detector, never patches its validator,
and cannot launch model work, modify panels, or authorize a campaign. Only hashed
member/group identities and aggregate counts leave the custodian boundary.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import sys


def audit(root, panels, exposures, *, progress=None):
    from oeis_learn.generalization.progressive.contracts import digest, seal, validate, verify_refs
    from oeis_learn.generalization.progressive.cohorts import validate_cohort, validate_prompt, _linked
    from oeis_learn.generalization.progressive.exposure import detect

    for value in exposures.values():
        validate('exposure', value)
    if not exposures or any(k != v['identity'] for k, v in exposures.items()):
        raise ValueError('original identity-bound exposure cuts required')
    for panel in panels:
        validate('cohort', panel)
        if panel['payload']['role'] == 'assessment':
            raise ValueError('this diagnostic does not open reserved assessment panels')
    if len({p['payload']['panel'] for p in panels}) != len(panels):
        raise ValueError('distinct original panels required')
    buckets = defaultdict(lambda: defaultdict(list))
    summaries, errors = [], []
    total = sum(len(p['payload']['rows']) for p in panels)
    processed = 0
    for panel in panels:
        p = panel['payload']; name = p['panel']
        if p['exposure'] not in exposures:
            raise ValueError('original panel exposure cut missing')
        try:
            validate_cohort(panel, exposure=exposures[p['exposure']], other_cohorts=panels)
        except ValueError as exc:
            # No private values or arbitrary producer text in the public report.
            errors.append(dict(panel=name, check='declared_metadata', type=type(exc).__name__))
        good = 0
        for row in p['rows']:
            try:
                prompt, truth = [json.loads(b) for b in verify_refs(root, [row['visible'], row['truth']], max_bytes=4 << 20)]
                validate_prompt(prompt)
                expected = [str(i) for i in range(int(row['offset']), int(row['offset']) + 100)]
                if (set(truth) != {'id','indices','values'} or truth['id'] != row['id']
                        or prompt['id'] != row['id'] or truth['indices'] != expected
                        or len(truth['values']) != 100 or prompt['indices'] != expected[:20]
                        or prompt['values'] != truth['values'][:20]):
                    raise ValueError('original prompt/truth binding differs')
                detected = detect([dict(truth, structure=row['structure'], composition=row['composition'],
                                        relations=[row['source'] + ':' + row['id']])])[0]['groups']
                if p['group_policy'] == 'parameter_structure_sharing' and row['stratum'] in {'parameter','retention'}:
                    detected = [g for g in detected if not g.startswith(('structure:', 'composition:'))]
                effective = dict(row, groups=sorted(set(row['groups']) | set(detected)))
                for group in effective['groups']:
                    buckets[group][name].append(effective)
                good += 1
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(dict(panel=name, member=digest(row['id']), check='original_truth', type=type(exc).__name__))
            processed += 1
            if progress and (processed % 128 == 0 or processed == total):
                progress(dict(processed=processed, total=total, errors=len(errors)))
        summaries.append(dict(panel=name, role=p['role'], cohort=panel['identity'],
                              rows=len(p['rows']), checked=good))
    by_name = {p['payload']['panel']:p['payload'] for p in panels}
    pairs = {}
    for group, owners in sorted(buckets.items()):
        for a, b in combinations(sorted(owners), 2):
            # Every shared group is a canonical rejection. Check against the
            # installed implementation rather than treating this report as authority.
            left, right = owners[a], owners[b]
            if not all(_linked(left[0], right[0], e, by_name[a]['group_policy']) for e in exposures.values()):
                raise ValueError('installed cross-panel detector differs')
            pair = pairs.setdefault((a,b), dict(panels=[a,b], reasons={}))
            category = group.split(':',1)[0]
            if category not in {'behavior','window','structure','composition','source'}:
                category = 'declared_group'
            reason = pair['reasons'].setdefault(category, dict(shared_keys=0, row_pair_occurrences=0,
                                                              strata={}, examples=[]))
            reason['shared_keys'] += 1
            reason['row_pair_occurrences'] += len(left)*len(right)
            lc, rc = defaultdict(int), defaultdict(int)
            for row in left: lc[row['stratum']] += 1
            for row in right: rc[row['stratum']] += 1
            for x, n in lc.items():
                for y, m in rc.items():
                    key=x+'/'+y
                    reason['strata'][key]=reason['strata'].get(key,0)+n*m
            if len(reason['examples']) < 3:
                reason['examples'].append(dict(group=digest(group), left=digest(left[0]['id']),
                                              right=digest(right[0]['id'])))
    return seal(dict(version='progressive-panel-audit/v1',
        status='INCOMPLETE' if errors else 'OVERLAPS_FOUND' if pairs else 'NO_DIRECT_OVERLAPS_FOUND',
        panels=summaries, panel_pairs=list(pairs.values()), errors=errors,
        exposure_events=sum(len(e['payload']['events']) for e in exposures.values()),
        count_scope='row pair occurrences per shared key; counts may overlap across keys/reasons',
        scope='direct augmented detector group intersections; not a complete freshness/qualification check',
        not_repeated=['qualification','calibration','training','indexed_history_scan'],
        assessment_opened=False, original_inputs_modified=False, scientific_authority=False))


def prepared_publication(session):
    """Follow the real wrapper journal -> result -> prepared artifact chain."""
    from oeis_learn.generalization.progressive.contracts import seal, validate, reference, CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file, _json
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    row = session.journal['steps'].get('prepare-research/prepare', {})
    value = row.get('value') or {}
    if row.get('status') != 'COMPLETE' or value.get('operation') != 'prepare' or value.get('exit_code') != 0:
        raise ValueError('one completed original preparation publication required')
    output = confined(session.root,value['output'])
    binding = row.get('binding')
    if binding is not None and (seal(binding) != binding or binding.get('operation') != 'prepare'
                               or binding.get('arguments',{}).get('output') != value['output']):
        raise ValueError('original preparation output binding differs')
    refs = [r for r in row.get('outputs',[]) if r['path'] == str((output/'result.json').relative_to(session.root))]
    if len(refs) != 1:
        raise ValueError('original journal-bound preparation result required')
    result = read(checked_file(session.root,refs[0]))
    if (result.get('operation') != 'prepare' or result.get('mechanical_status') != 'COMPLETE'
            or result.get('producer_error') or result.get('missing_evidence')):
        raise ValueError('completed original preparation result required')
    prepared_ref = result.get('metrics',{}).get('prepared',{})
    if prepared_ref.get('path') != 'prepared.json':
        raise ValueError('original result-bound prepared publication required')
    path = checked_file(output,prepared_ref)
    return validate('prepared',_json(path,maximum=CAMPAIGN_DOCUMENT_BYTES)),reference(session.root,path),refs[0]


def run(session, script):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import digest, seal, validate, reference, CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file, _json
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    if digest(collect_snapshot(session.root)) != session.binding['source']:
        raise ValueError('keep the original qualified checkout; fetch the audit file without pulling source')
    prepared, prepared_ref, result_ref = prepared_publication(session)
    panels, exposures, inputs = [], {}, []
    errors=[]
    for item in prepared['payload']['references']:
        if item['kind'] != 'groups': continue
        ref=item['artifact']; inputs.append(ref)
        try:
            value=_json(checked_file(session.root, ref),maximum=CAMPAIGN_DOCUMENT_BYTES)
            if value['version']=='discovery-exposure/v1':
                validate('exposure',value); exposures[value['identity']]=value
            elif value['version']=='discovery-cohort/v1':
                validate('cohort',value); panels.append(value)
            else: raise ValueError('closed original group artifact required')
        except (OSError,ValueError,KeyError,TypeError) as exc:
            errors.append(dict(artifact=digest(ref),check='group_artifact',type=type(exc).__name__))
    if not panels or not exposures:
        raise ValueError('original panel and exposure artifacts unavailable')
    output=session.folder/'panel-audit'
    def progress(value):
        write(output/'panel-audit-progress.json',seal(value))
        print('Panel audit: '+str(value['processed'])+'/'+str(value['total'])+' rows; '+str(value['errors'])+' read errors',flush=True)
    result=audit(session.root,panels,exposures,progress=progress)
    result.update(source=session.binding['source'], auditor=reference(session.root,script),
                  prepared=prepared_ref, preparation_result=result_ref, inputs=inputs)
    if errors:
        result['errors'].extend(errors); result['status']='INCOMPLETE'
    result=seal(result)
    write(output/'panel-audit.json',result)
    print('Panel audit: '+result['status']+'; '+str(len(result['panel_pairs']))+' panel pairs with direct overlaps',flush=True)
    return result,[output/'panel-audit.json']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--session',default='run-state/discovery/013-research-round-016')
    args=p.parse_args(argv)
    root=Path(args.root).resolve()
    sys.path.insert(0,str(root/'src'))
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,read,NAMES
    from oeis_learn.generalization.progressive.contracts import digest, reference
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    folder=confined(root,args.session)
    binding=read(folder/'session.json')['binding']
    s=Session(root,args.session,binding,'panel-audit')
    NAMES.update({'panel-audit.json','panel-audit-progress.json'})
    with s.own():
        if digest(collect_snapshot(root)) != binding['source']:
            raise ValueError('keep the original qualified checkout; fetch the audit file without pulling source')
        # A new diagnostic step retains the failed v1 lookup and its binding.
        # It grants no retry/allowance to preparation or any scientific operation.
        result=s.step('inspect-v2',lambda:run(s,Path(__file__).resolve()),
                      binding=reference(root,Path(__file__).resolve()))
        return 1 if result is None else 3 if result['status']!='NO_DIRECT_OVERLAPS_FOUND' else 0


if __name__=='__main__':
    raise SystemExit(main())
