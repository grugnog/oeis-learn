"""Revalidate an admission archive against the exact public completed-run decision."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from oeis_learn.pipeline_validation.bundle import verify
from oeis_learn.pipeline_validation.identity import seal,file_hash,confined,digest
from oeis_learn.pipeline_validation.longrun_contract import parse_decision,PROFILE
from oeis_learn.pipeline_validation.longrun_import import environment
from oeis_learn.experiments.source_snapshot import collect_snapshot


def require(value,reason):
    if not value:raise ValueError(reason)


def review(admission,run,producer,calibration):
    paths=list(map(Path,(admission,run,producer,calibration)))
    manifests=[verify(p) for p in paths]
    require(all(not m['omissions'] for m in manifests),'omitted evidence')
    with zipfile.ZipFile(paths[0]) as a,zipfile.ZipFile(paths[1]) as r,zipfile.ZipFile(paths[2]) as p,zipfile.ZipFile(paths[3]) as c:
        read=lambda z,n:json.loads(z.read(n))
        summary=read(a,'summary.json');decision=read(a,'stages/study.report/decision.json')
        require(decision==read(r,'contract.json')['decision']==read(r,'stages/study.report/debits.json')['decision'],'admission/run/ledger decision mismatch')
        require(decision['profile']==PROFILE and decision['source']==digest(collect_snapshot(Path(__file__).resolve().parents[3])),'qualified source/profile changed')
        source=decision['source'];inv=read(a,'invocation.json')
        require(seal(inv)==inv and inv['binding']['source']==source,'admission invocation source mismatch')
        inventory={m['path']:m for m in manifests[0]['members']}
        for key,row in summary['stages'].items():
            require(row==read(a,'stages/'+key+'/receipt.json') and seal(row)==row and row['state']=='PASS','failed/changed admission stage')
            require(row['binding']['source']==source,'stage source mismatch')
            for name,sha in row['files'].items():require(inventory['stages/'+key+'/'+name]['sha256']==sha,'unbound receipt file')
            for parent,sha in row.get('prerequisites',{}).items():require(summary['stages'][parent]['identity']==sha,'broken prerequisite')
        q=read(a,'stages/study.report/qualification.json');cost=read(a,'stages/study.report/cost.json')
        proof=read(a,'stages/pilot.compatibility/compatibility.json');auth=read(a,'stages/study.report/authority.json')
        require(cost['calibration']==read(c,'stages/calibration.report/candidate-cost.json'),'retained calibration differs')
        require(proof['calibration_manifest']=='sha256:'+hashlib.sha256(c.read('manifest.json')).hexdigest(),'calibration manifest changed')
        require(proof['producer_manifest']=='sha256:'+hashlib.sha256(p.read('manifest.json')).hexdigest(),'producer manifest changed')
        prefix=decision['authority']['path'].split('/stages/')[0]
        require(prefix!=decision['authority']['path'],'registered relative admission root required')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);base=confined(root,prefix,exists=False);base.mkdir(parents=True)
            for member in manifests[0]['members']:
                target=confined(base,member['path'],exists=False);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(a.read(member['path']))
            # Only the public prepared manifest is required by public admission.
            # No examples, hidden values, weights or private checkpoint is resolved.
            prepared=confined(root,decision['prepared']['path'],exists=False)
            prepared.parent.mkdir(parents=True,exist_ok=True);prepared.write_bytes(p.read('stages/pilot.prepare/prepared-index.json'))
            require(file_hash(prepared)==decision['prepared']['sha256'],'prepared manifest binding changed')
            verdict=parse_decision(decision,root=root,source=source)
            require(verdict['status']=='ADMITTED','public authority failed to admit')
            require(environment(base,q['engine'])==proof['environment']==read(a,'stages/study.report/environment.json'),'hardware/runtime compatibility mismatch')
        counts={};fixture_names=[]
        for name in a.namelist():
            if not name.endswith('/results.xml'):continue
            cases=list(ET.fromstring(a.read(name)).iter('testcase'))
            count=dict(tests=len(cases),failures=sum(x.find('failure') is not None for x in cases),errors=sum(x.find('error') is not None for x in cases),skipped=sum(x.find('skipped') is not None for x in cases))
            require(count['tests']>0 and not count['failures'] and not count['errors'],'empty/failed JUnit')
            if '/operations.' not in name:require(count['skipped']==0,'mandatory qualification skipped')
            for x in cases:
                if 'full_clock_pilot_actual_worker_state_and_metric_resume' in x.attrib.get('name',''):
                    require(x.find('skipped') is None,'new full-state fixture skipped');fixture_names.append(x.attrib['name'])
            count['reported_suite_tests']=sum(int(t.attrib.get('tests',0)) for t in ET.fromstring(a.read(name)).iter('testsuite'))
            counts[name]=count
        require(any('cpu' in n for n in fixture_names) and any('cuda' in n for n in fixture_names),'both actual full-state purpose fixtures required')
        for target in ('gpu','qualification'):
            inspection=read(a,f'stages/image.{q["engine"]}.{target}/inspection/logs/stdout.log')
            require(inspection['source']==proof['current_snapshot'],'installed package/source mismatch')
            command=read(r,f'stages/pilot.current-images/{target}/logs/command.json')['argv']
            require(q['image'][target] in command,'completed run re-inspected different image')
        for i in range(6):
            for mode,role,target in [('train','train','gpu'),('evaluate','generate','gpu'),('evaluate','score','qualification')]:
                command=read(r,f'stages/pilot.{mode}-{i}/{role}-{i}/command.json')['argv']
                require(q['image'][target] in command,'completed run used a different qualified image')
        admitted=read(a,'stages/study.report/admission.json')
        require(admitted['status']=='ADMITTED' and admitted['long_run_admitted'] is True and seal(admitted)==admitted,'admission publication contradiction')
        require(admitted['decision']['sha256']=='sha256:'+hashlib.sha256(a.read('stages/study.report/decision.json')).hexdigest(),'published decision file mismatch')
        return dict(version='pipeline-admission-public-review/v1',source=source,study=decision['identity'],
            archive_sha256=file_hash(paths[0]),inventory=dict(members=len(manifests[0]['members'])+1,omissions=[]),
            stages={k:v['state'] for k,v in summary['stages'].items()},junit=counts,new_full_state_fixtures=fixture_names,
            admission_status=verdict['status'],authority=auth['identity'],qualification=q['identity'],compatibility=proof['identity'],
            exact_completed_run_binding=True,current_images_and_native_environment=True,
            projected_seconds=cost['projected_seconds'],admission_elapsed_seconds=summary['elapsed_seconds'],
            portability='INDEPENDENT_DOCKER_PENDING',new_execution_authorized=False,
            summary_note='Generic engineering summary says NOT_ADMITTED; explicit sealed admission/authority/decision revalidate as ADMITTED. No execution authority is inferred from summary wording.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('admission','run','producer','calibration','output'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();result=review(a.admission,a.run,a.producer,a.calibration)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['admission_status'],result['inventory'],result['junit'])

if __name__=='__main__':main()
