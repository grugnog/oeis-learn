"""Stdlib custodian conversion of original native census artifacts.

No original generation or exposure import is repeated. The host-side operation
only hashes and projects existing bytes; native execution remains in the installed
Podman/Docker census producer and all model work remains in isolated roles.
"""
from pathlib import Path
import time

from .identity import confined, seal
from .progressive_research_inputs import checked_file, _json
from .progressive_research_census import validate_request
from .progressive_workstation_state import write
from oeis_learn.generalization.progressive.contracts import record, reference
from oeis_learn.generalization.progressive import prepared_support, prepared_panels
from oeis_learn.generalization.progressive.cohorts import publish_many


def build(root, census_path, request_path, output, rule):
    root=Path(root)
    output=confined(root,output,exists=False)
    if output.exists():
        raise ValueError('original materials retained; no automatic replacement or budget reset')
    census_path=confined(root,census_path); request_path=confined(root,request_path)
    census=_json(census_path,maximum=32<<20)
    request=validate_request(_json(request_path,maximum=32<<20))
    if (seal(census)!=census or census.get('version')!='progressive-research-census/v1'
            or census['request']!=request['identity'] or census['diagnostic']!=request['diagnostic']):
        raise ValueError('original request-bound native census required')
    if rule['seed']!=request['seed'] or rule['target_tokens']!=request['token_target']:
        raise ValueError('support projection differs from original prospective census domain')
    began=time.monotonic()
    refs={Path(r['path']).name:r for r in census['artifacts']}
    expected={'training-reservoir.json','panel-reservations.json','census.sqlite','training-exposure.sqlite'}
    if len(refs)!=len(census['artifacts']) or set(refs)!=expected:
        raise ValueError('complete original native census artifacts required')
    files={name:checked_file(census_path.parent,ref) for name,ref in refs.items()}
    training=_json(files['training-reservoir.json'],maximum=128<<20)
    roster=_json(files['panel-reservations.json'],maximum=128<<20)
    if len(training)!=census['counts'].get('TRAINING',0) or set(roster)!={*request['panels'],*request['retention_panels']}:
        raise ValueError('original reservoir/count binding differs')
    for panel,rows in roster.items():
        for stratum in (['parameter'] if panel in request['retention_panels'] else ['parameter','composition','unseen_structure','oeis']):
            if sum(r['stratum']==stratum for r in rows)!=census['panels'][panel+'/'+stratum]['tasks']:
                raise ValueError('original panel census/count binding differs')
    # The external step binds the original producer receipt; the native identity
    # is retained here, never replaced by a projection-generated PASS.
    private=output/'private';private.mkdir(parents=True,mode=0o700)
    write(output/'support-policy.json',rule)
    support,summary=prepared_support.project(training,rule,diagnostic=request['diagnostic'])
    write(private/'support.json',support);write(output/'support-summary.json',summary)
    screening,screening_summary=prepared_panels.screening(roster['development'],seed=request['seed'])
    write(private/'screening.json',screening);write(output/'screening-summary.json',screening_summary)
    exposure=record('exposure',dict(previous=None,detector_limits=prepared_panels.LIMITS,
        events=[],indexes=[request['exposure'],reference(root,files['training-exposure.sqlite'])]),
        created_at='2026-11-11T00:00:00Z',purpose='acceptance' if request['diagnostic'] else 'research',
        track='synthetic_strict',parents=[census['identity']])
    write(private/'exposure.json',exposure)
    publications=[]
    for panel,rows in roster.items():
        role=('assessment' if panel=='assessment' else
              'confirmation' if panel.startswith('confirmation') or panel in {'retention-diversity','retention-tuning'} else 'development')
        decision=('tuning' if 'tuning' in panel else 'diversity' if 'diversity' in panel
                  else 'assessment' if role=='assessment' else 'capability')
        value=prepared_panels.cohort(root,private/'panels'/panel,rows,panel=panel,role=role,
            decision=decision,exposure=exposure,required=request['required_per_stratum'],
            minimum_groups=request['minimum_groups'],purpose=exposure['purpose'],created_at=exposure['created_at'])
        publications.append((str((private/'published'/panel).relative_to(root)),value))
    outcomes=publish_many(root,publications,exposure=exposure)
    deficits=list(summary['deficits'])
    deficits.extend(screening_summary['deficits'])
    deficits.extend(d for d in census['deficits'] if d!='refresh_cadence_requires_prospective_revision')
    for (_,panel),outcome in zip(publications,outcomes):
        if outcome['status']!='COMPLETE':
            deficits.append('cohort_publication:'+panel['payload']['panel'])
    result=seal(dict(version='progressive-research-materials/v1',
        status='INCOMPLETE' if deficits else 'COMPLETE',research_ready=False,
        diagnostic=request['diagnostic'],census=reference(root,census_path),
        request=reference(root,request_path),native=census['native'],
        support=reference(root,private/'support.json'),support_summary=reference(root,output/'support-summary.json'),
        screening=reference(root,private/'screening.json'),screening_summary=reference(root,output/'screening-summary.json'),
        exposure=reference(root,private/'exposure.json'),
        panel_censuses=outcomes,
        panels=[dict(panel=p['payload']['panel'],role=p['payload']['role'],status=o['status'],
            census=reference(root,Path(root)/path/'census.json'),
            cohort=reference(root,Path(root)/path/'cohort.json') if o['status']=='COMPLETE' else None)
            for (path,p),o in zip(publications,outcomes)],
        deficits=sorted(set(deficits)),elapsed_seconds=time.monotonic()-began,
        original_generation_repeated=False,assessment_exposed=False,
        cost_scope='projection only; original native census and exposure costs remain charged'))
    write(output/'materials.json',result)
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser()
    for name in ('root','census','request','output','policy'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    build(args.root,args.census,args.request,args.output,_json(confined(args.root,args.policy)))


if __name__=='__main__':
    main()
