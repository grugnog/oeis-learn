"""Versioned long-run authority and unchanged historical cost semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from oeis_learn.pipeline_validation.identity import digest,seal,write,file_hash


def candidate(source=None):
    from oeis_learn.pipeline_validation.pilot_calibration_cost import forecast
    s=source or digest('producer');w=digest('work')
    rows=[seal(dict(complete=True,clock_complete=True,evaluation_policy='portfolio-clock60/v1',portfolios=1184,
        scientific_complete=False,seconds=2242.05,generation_seconds=2232.02,scoring_seconds=10.03,
        projected_all_selected_scoring_seconds=382.4,rows=296,source=s,work=w,checkpoint=digest(i))) for i in range(3)]
    archives=[seal(dict(source=s,complete=True,seconds=1.,bytes=16<<20,members=4096,filesystem_device=1,ordinal=i)) for i in range(3)]
    return forecast(rows,source=s,work=w,archives=archives)


class Longrun(unittest.TestCase):
    def test_new_profile_preserves_old_profile_and_candidate_is_not_authority(self):
        from oeis_learn.pipeline_validation.longrun_contract import PROFILE,SPEC,cost_authority,valid_cost
        from oeis_learn.pipeline_validation.pilot_contract import PROFILE as old,SPEC as old_spec,parse_decision
        from oeis_learn.pipeline_validation.pilot_cost import validate_forecast
        c=candidate();current=digest('current');compat=digest('compatibility')
        self.assertEqual(old['version'],'pipeline-learning-pilot-profile/v2')
        self.assertEqual(old_spec['reservations']['evaluation'],28800)
        self.assertEqual(PROFILE['version'],'pipeline-learning-pilot-profile/v3')
        self.assertEqual(SPEC['reservations']['evaluation'],36000)
        self.assertEqual(SPEC['reservations']['safety'],41400)
        self.assertEqual(sum(SPEC['reservations'].values()),172800)
        self.assertFalse(validate_forecast(c,source=c['source']))
        cost=cost_authority(c,source=current,compatibility=compat)
        self.assertEqual(cost['projected_seconds'],126780)
        self.assertTrue(valid_cost(cost,source=current,compatibility=compat))
        self.assertFalse(valid_cost(c,source=current,compatibility=compat))
        self.assertFalse(valid_cost(cost,source=digest('other'),compatibility=compat))
        for field,value in [('projected_seconds',1),('generator_role_seconds',1),('evaluation_policy','legacy')]:
            with self.subTest(field=field):self.assertFalse(valid_cost(seal(dict(cost,**{field:value})),source=current,compatibility=compat))
        for broken in [dict(c,within_budget=True),seal(dict(c,observations=c['observations'][:2]))]:
            if broken==c:continue
            with self.assertRaises(ValueError):cost_authority(broken,source=current,compatibility=compat)
        with self.assertRaises(ValueError):parse_decision(seal(dict(version='pipeline-pilot-study/v2')))

    def test_ledger_uses_decision_profile_and_recovery_keeps_debits(self):
        from oeis_learn.pipeline_validation.longrun_contract import PROFILE
        from oeis_learn.pipeline_validation.pilot_ledger import PilotLedger
        with tempfile.TemporaryDirectory() as folder:
            d=seal(dict(profile=PROFILE));clock=[10.]
            l=PilotLedger(Path(folder),d,clock=lambda:clock[0],boot='fixture')
            for i in range(6):l.reserve('evaluation-'+str(i),seconds=5230,category='evaluation',output=Path(folder)/str(i))
            with self.assertRaises(ValueError):l.reserve('excess',seconds=6000,category='evaluation',output=folder)
            l2=PilotLedger(Path(folder),d,clock=lambda:clock[0],boot='fixture')
            self.assertEqual(l2.value,l.value)
            with self.assertRaises(ValueError):l2.reserve('evaluation-0',seconds=1,category='evaluation',output=folder)

    def test_worker_new_purpose_is_profile_bound_and_fresh(self):
        from oeis_learn.pipeline_validation.longrun_contract import PROFILE,PURPOSE
        from oeis_learn.fresh.pilot_control import configuration,parse,Control
        cfg=configuration(source=digest('current'),work=digest('work'),purpose=PURPOSE,ordinal=0,seconds=0,cadence_tokens=128)
        self.assertEqual(parse(cfg)['profile'],PROFILE['identity'])
        with tempfile.TemporaryDirectory() as folder:
            Control({'pilot':cfg},{},Path(folder))
            with self.assertRaises(ValueError):Control({'pilot':cfg},{'pilot':{}},Path(folder))
        with self.assertRaises(ValueError):parse(seal(dict(cfg,purpose='longrun')))

    def test_full_clock_generation_is_independent_of_weights_loader(self):
        from oeis_learn.pipeline_validation.longrun_contract import full_clock,PURPOSE
        self.assertTrue(full_clock(PURPOSE));self.assertTrue(full_clock('calibration'))
        self.assertFalse(full_clock('longrun'));self.assertFalse(full_clock('preflight'))

    def test_main_cli_admission_requires_explicit_review_for_approval(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('foundation_local',Path(__file__).resolve().parents[2]/'scripts/foundation_local.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        a=m.parser().parse_args(['pilot-admit','--inputs','producer','--calibration','calibration','--output','out'])
        self.assertEqual(a.command,'pilot-admit');self.assertFalse(a.approve)
        a=m.parser().parse_args(['pilot-admit','--inputs','p','--calibration','c','--output','o','--approve','--reviewed-by','operator'])
        self.assertTrue(a.approve)


class Authority(unittest.TestCase):
    def fixture(self,root):
        from oeis_learn.pipeline_validation.longrun_contract import PROFILE,requirements,decision,cost_authority
        from oeis_learn.pipeline_validation.longrun_import import MIGRATION_PATHS
        from oeis_learn.pipeline_validation.pilot_calibration_import import PRODUCER_SOURCE
        from oeis_learn.pipeline_validation.pilot_contract import decision_payload
        old={'unchanged/model.py':digest('model')};current=dict(old)
        current['src/oeis_learn/pipeline_validation/longrun_contract.py']=digest('fixture migration')
        source=digest(current);c=candidate(digest(old));work=c['work'];engine='podman'
        proof=seal(dict(version='pipeline-pilot-compatibility/v1',source=source,calibration_source=digest(old),
            producer_source=PRODUCER_SOURCE,work=work,engine=engine,cpus=16,resources=digest('resources'),
            calibration_manifest=digest('manifest'),producer_manifest=digest('producer manifest'),old_snapshot=old,current_snapshot=current,
            changes={'src/oeis_learn/pipeline_validation/longrun_contract.py':dict(producer=None,current=current['src/oeis_learn/pipeline_validation/longrun_contract.py'])},
            environment=dict(engine={},hardware={},gpu={},cpu={},artifacts={}),
            authority='retained data/native/cost only; fresh learner; current qualification required'))
        ref=lambda n:dict(path=n,sha256=file_hash(root/n))
        write(root/'compatibility.json',proof);write(root/'cost.json',cost_authority(c,source=source,compatibility=proof['identity']))
        required=sorted(requirements(engine));receipts={};refs={}
        for k in required:
            write(root/(k+'-proof.json'),dict(fixture=True))
            image=digest('gpu') if k.endswith('.gpu') else digest('cpu')
            receipt=seal(dict(stage=k,state='PASS',binding={'source':source},prerequisites={},
                files={k+'-proof.json':file_hash(root/(k+'-proof.json'))},observations={'result':{'image':image}}));name=k+'.json'
            write(root/name,receipt);receipts[k]={f:receipt[f] for f in ('state','identity')};refs[k]=ref(name)
        q=seal(dict(version='pipeline-pilot-qualification/v2',source=source,profile=PROFILE['identity'],work=work,
            image=dict(gpu=digest('gpu'),qualification=digest('cpu')),engine=engine,resources=proof['resources'],cpus=16,
            required=required,receipts=receipts,receipt_refs=refs,state='PASS',compatibility=proof['identity']))
        write(root/'q.json',q);write(root/'prepared.json',dict(identity=work,source=PRODUCER_SOURCE))
        d=decision(source=source,prepared=ref('prepared.json'),qualification=ref('q.json'),cost=ref('cost.json'),compatibility=ref('compatibility.json'))
        def approve(record):
            write(root/'authority.json',seal(dict(version='pipeline-study-authority/v2',decision_payload=decision_payload(record),approved=True,reviewed_by='fixture operator')),exclusive=False)
            return seal(dict(record,state='ADMITTED',authority=ref('authority.json')))
        return approve(d),proof,q,ref,approve

    def test_public_gates_precede_private_data_and_require_current_receipts(self):
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.longrun_contract import parse_decision
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);d,proof,q,ref,approve=self.fixture(root)
            with patch('oeis_learn.pipeline_validation.longrun_import.CALIBRATION_SOURCE',proof['calibration_source']):
                self.assertEqual(parse_decision(d,root=root,source=d['source'])['status'],'ADMITTED')
                no_private=dict(path='private/never-read',sha256=digest('absent'))
                for mutation in ('receipt','cost','authority','source','resource','work'):
                    with self.subTest(mutation=mutation):
                        bad=seal(dict(d,prepared=no_private))
                        badq=copy.deepcopy(q)
                        if mutation=='receipt':badq['receipts'][badq['required'][0]]['state']='PARTIAL'
                        if mutation=='resource':badq['resources']=digest('other resources')
                        if mutation=='work':badq['work']=digest('other work')
                        write(root/'badq.json',seal(badq),exclusive=False);bad=seal(dict(bad,qualification=ref('badq.json')))
                        if mutation=='cost':bad=seal(dict(bad,cost=ref('compatibility.json')))
                        bad=approve(bad)
                        if mutation=='authority':
                            a=json.loads((root/'authority.json').read_text());a['decision_payload']=digest('wrong')
                            write(root/'authority.json',seal(a),exclusive=False);bad=seal(dict(bad,authority=ref('authority.json')))
                        verdict=parse_decision(bad,root=root,source=digest('changed') if mutation=='source' else d['source'])
                        self.assertEqual(verdict['status'],'BLOCKED')
                with self.assertRaises(ValueError):parse_decision(seal(dict(d,unknown=True)),root=root,source=d['source'])

    def test_migration_rejects_unmapped_model_native_build_and_missing_dependencies(self):
        from oeis_learn.pipeline_validation.longrun_import import changes
        for path in ('src/oeis_learn/decoder/sampler.py','Dockerfile','uv.lock','crates/native.rs','unknown.py'):
            with self.subTest(path=path),self.assertRaises(ValueError):changes({path:digest('old')},{path:digest('new')})
        self.assertEqual(changes({'old':digest(1)},{'old':digest(1)}),{})
        with self.assertRaises(ValueError):changes({'scripts/foundation_local.py':digest('old')},{})

    def test_registry_and_failed_import_never_touch_engine_or_launch_research(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.longrun_admission import main,registry
        from oeis_learn.pipeline_validation.bundle import verify
        stages=registry('podman')
        self.assertFalse(any(s.adapter in ('pilot-prepare','pilot-session','calibration-session','pilot-study-train') for s in stages))
        self.assertIn('pilot.compatibility',next(s.prerequisites for s in stages if s.adapter=='engine'))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('.specify/memory/constitution.md','specs/007-experiment-foundation/validation/adoption.md','specs/007-experiment-foundation/validation/review-inputs.json','specs/007-experiment-foundation/tasks.md'):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
            args=SimpleNamespace(engine='podman',output='out',approve=False,reviewed_by=None,
                inputs='missing-producer',calibration='missing-calibration',max_seconds=None,scope_report=False)
            def fake(self,stage,out):
                if stage.adapter!='host':raise AssertionError('failed import must precede all engine access')
                return {}
            with patch('oeis_learn.pipeline_validation.adapters.Adapters.__call__',fake):
                code=main(args,root,lambda *_:self.fail('engine called'))
            self.assertEqual(code,1)
            summary=json.loads((root/'out/summary.json').read_text())
            self.assertEqual(summary['stages']['engine.podman']['state'],'BLOCKED')
            self.assertFalse((root/'out/stages/study.report/decision.json').exists())
            self.assertFalse(verify(root/'out.zip')['omissions'])

    def test_main_entrypoint_forwards_without_host_model_dependencies(self):
        import subprocess,sys
        root=Path(__file__).resolve().parents[2]
        result=subprocess.run([sys.executable,'-S','-m','oeis_learn.cli.main','pipeline','--workspace',str(root),
            '--engine','podman','pilot-admit','--inputs','unused','--calibration','unused','--output','tmp-unused','--scope-report'],
            cwd=root,text=True,capture_output=True,env=dict(__import__('os').environ,PYTHONPATH=str(root/'src')))
        self.assertEqual(result.returncode,0,result.stderr)
        scope=json.loads(result.stdout)
        self.assertEqual(scope['profile']['version'],'pipeline-learning-pilot-profile/v3')
        self.assertFalse(scope['training']);self.assertFalse(scope['calibration'])

class GenerationRoute(unittest.TestCase):
    def test_full_clock_runtime_uses_ordinary_loader_and_frozen_parent_on_all_splits(self):
        from types import SimpleNamespace,ModuleType
        from contextlib import nullcontext
        from unittest.mock import patch,Mock
        from oeis_learn.pipeline_validation.longrun_contract import PURPOSE
        from oeis_learn.pipeline_validation.pilot_runtime import generate
        from oeis_learn.pipeline_validation.model_sizes import PROFILES,parameters
        from oeis_learn.pipeline_validation.pilot_contract import SPLITS,SPEC
        from oeis_learn.fresh.pipeline_jobs import JobLedger
        import sys
        torch=ModuleType('torch');torch.set_num_threads=lambda *_:None;torch.set_num_interop_threads=lambda *_:None
        worker=ModuleType('oeis_learn.fresh.bigint_worker');worker.client=lambda *_:nullcontext(None)
        calls=[];worker._load_models=Mock(return_value=(None,None,{'actual':parameters(PROFILES['m100'])}))
        def evaluate(*args,**kw):
            calls.append(kw)
            return dict(records={r['id']:dict(selections={},arm_evidence={},complete=False,family=r['family'],strata=r['strata']) for r in args[2]},
                pipeline_boundary=JobLedger().boundary({}),controls={})
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs=root/'in';inputs.mkdir();out=root/'out';out.mkdir()
            rows=[dict(id=split+str(i),values=['0']*20,indices=list(map(str,range(20))),horizon_indices=list(map(str,range(100))),
                family='f',strata={'split':split}) for split in SPLITS for i in range(SPEC[split])]
            write(inputs/'prompts.json',rows)
            from oeis_learn.fresh.pilot_control import configuration
            cfg=dict(pilot_prompts_sha256=file_hash(inputs/'prompts.json'),checkpoint={'path':'own/checkpoint.json'},recipe={'pipeline':{}},pilot=configuration(source=digest('source'),work=digest('work'),purpose=PURPOSE,ordinal=0,seconds=0,cadence_tokens=128))
            write(inputs/'settings.json',cfg)
            config=dict(source=digest('source'),options=dict(seconds=5000,memory_ceiling=1<<30,render_node=None,purpose=PURPOSE,ordinal=0,worker_seconds=0,image=digest('image'),interrupted=[]))
            with patch.dict(sys.modules,{'torch':torch,worker.__name__:worker}),patch('oeis_learn.fresh.bigint_diagnostics.generate',evaluate),patch('oeis_learn.pipeline_validation.pilot_runtime.execution_observation',lambda:{}):
                result=generate.__wrapped__(inputs,out,config)
            worker._load_models.assert_called_once()
            self.assertEqual(len(calls),4)
            self.assertEqual(len({c['deadline'] for c in calls}),1)
            self.assertTrue(all(c['evaluation_policy']=='portfolio-clock60/v1' for c in calls))
            self.assertFalse(result['clock_complete']) # mock did no actual portfolio work
            self.assertEqual(result['portfolios'],1184)


class AdmissionPublication(unittest.TestCase):
    def test_failed_or_missing_approval_cannot_publish_admitted_decision(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from oeis_learn.pipeline_validation.longrun_admission import Admission
        from oeis_learn.pipeline_validation.longrun_contract import parse_decision,requirements
        from oeis_learn.pipeline_validation.pilot_contract import preflight_manifest
        from oeis_learn.pipeline_validation.resources import execution_identity
        for approve in (False,True):
            with self.subTest(approve=approve),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);helper=Authority();d,proof,q,ref,sign=helper.fixture(root)
                # Use actual sealed fixture receipts; only platform inspection is synthetic.
                receipts={k:json.loads((root/(k+'.json')).read_text()) for k in requirements('podman')}
                invocation=root/'invocation';invocation.mkdir()
                for k,r in receipts.items():
                    out=invocation/'stages'/k;out.mkdir(parents=True)
                    for name in r['files']:(out/name).write_bytes((root/name).read_bytes())
                    write(out/'receipt.json',r)
                write(invocation/'stages/pilot.compatibility/compatibility.json',proof)
                write(root/'prepared-folder/prepared.json',json.loads((root/'prepared.json').read_text()))
                policy={'fixture':'resources'}
                proof=seal(dict(proof,resources=execution_identity(policy)))
                write(invocation/'stages/pilot.compatibility/compatibility.json',proof,exclusive=False)
                host=SimpleNamespace(policy={'cpus':16,'pipeline_resources':policy})
                args=SimpleNamespace(engine='podman',approve=approve,reviewed_by='fixture operator' if approve else None)
                a=SimpleNamespace(root=root,args=args,host=lambda _:host,images=SimpleNamespace(images={('podman',k):v for k,v in q['image'].items()}))
                adapter=Admission(a);adapter.imported=dict(compatibility=proof,candidate=candidate(proof['calibration_source']),prepared=root/'prepared-folder')
                output=invocation/'stages/study.report'
                with patch('oeis_learn.pipeline_validation.longrun_import.CALIBRATION_SOURCE',proof['calibration_source']),patch('oeis_learn.pipeline_validation.longrun_import.environment',lambda *_:proof['environment']):
                    result=adapter.finalize(output,invocation,dict(receipts,**{'pilot.compatibility':{'state':'PASS'}}))
                    self.assertEqual(result['long_run_admitted'],approve)
                    record=json.loads((output/'decision.json').read_text())
                    self.assertEqual(parse_decision(record,root=root,source=d['source'])['status'],'ADMITTED' if approve else 'BLOCKED')
                self.assertEqual((output/'authority.json').exists(),approve)
                self.assertFalse((root/'run-state').exists())

if __name__=='__main__':unittest.main()
