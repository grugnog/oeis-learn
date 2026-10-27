"""Saved-snapshot calibration rejects cut clocks and changed lineage (stdlib)."""
import copy
import tempfile
from pathlib import Path
import unittest
from oeis_learn.pipeline_validation.identity import seal,digest


class Calibration(unittest.TestCase):
    def test_original_clock_and_complete_attempts_are_distinct_from_science(self):
        from oeis_learn.pipeline_validation.pilot_clocks import clock_record
        kw=dict(prompt='a',mode='correct',snapshot=digest('weights'),start=100.,scope_deadline=200.,finish=160.,expected=8,started=2,terminal=2)
        r=clock_record(**kw)
        self.assertTrue(r['cost_complete']);self.assertEqual(r['unstarted'],6)
        self.assertEqual(r['termination'],'original_clock_exhausted')
        early=clock_record(**dict(kw,finish=140.,scope_deadline=140.))
        self.assertFalse(early['cost_complete']);self.assertEqual(early['termination'],'parent_cut')
        done=clock_record(**dict(kw,finish=110.,started=8,terminal=8))
        self.assertTrue(done['cost_complete'])
        self.assertEqual(done['termination'],'attempt_roster_terminal')
        self.assertFalse(clock_record(**dict(kw,terminal=1))['cost_complete'])
        for bad in ({'finish':99.},{'started':9},{'expected':7},{'start':float('nan')}):
            with self.assertRaises(ValueError):clock_record(**dict(kw,**bad))

    def test_clock_ledger_binds_registered_jobs_and_missing_portfolios(self):
        from oeis_learn.pipeline_validation.pilot_clocks import FullClockEvidence,clock_record
        with tempfile.TemporaryDirectory() as tmp:
            e=FullClockEvidence(Path(tmp),[dict(id='a')],source=digest('source'),work=digest('work'))
            for i in range(2):
                for event in ('STARTED','GENERATED','INCOMPLETE'):e('a','correct',i,event,{})
            r=clock_record(prompt='a',mode='correct',snapshot=digest('weights'),start=1.,scope_deadline=100.,finish=61.,expected=8,started=2,terminal=2)
            e.portfolio('a','correct',r)
            with self.assertRaises(ValueError):e.portfolio('a','correct',r)
            with self.assertRaises(ValueError):e.portfolio('b','correct',r)
            e.finish();self.assertFalse(e.clock_summary['cost_complete'])
            self.assertEqual(e.clock_summary['missing_portfolios'],3)

    def test_candidate_cost_cannot_be_used_as_legacy_long_run_authority(self):
        from oeis_learn.pipeline_validation.pilot_calibration_cost import forecast
        from oeis_learn.pipeline_validation.pilot_cost import validate_forecast
        r=forecast([],source=digest('s'),work=digest('w'))
        self.assertEqual(r['status'],'UNKNOWN');self.assertFalse(r['within_budget'])
        self.assertFalse(validate_forecast(r,source=digest('s')))
        self.assertFalse(r['long_run_admitted'])

    def test_manifest_and_registry_have_no_training_or_preparation(self):
        from oeis_learn.pipeline_validation.pilot_calibration import manifest,registry
        from oeis_learn.pipeline_validation.contracts import parse
        m=manifest();self.assertEqual(parse(m),m)
        stages=registry(['podman'],'podman')
        self.assertFalse(any(s.id in ('pilot.prepare','pilot.session') for s in stages))
        self.assertTrue(any(s.id=='calibration.import' for s in stages))
        self.assertEqual(next(s.seconds for s in stages if s.id=='calibration.session'),31000)

if __name__=='__main__':unittest.main()

class SnapshotBindings(unittest.TestCase):
    def fixture(self,root):
        import hashlib,json
        from oeis_learn.pipeline_validation.identity import file_hash
        root=Path(root);(root/'weights.pt').write_bytes(b'fixture tensor payload; host never loads it')
        weights=file_hash(root/'weights.pt')
        record=dict(version='bigint-checkpoint/v1',files={'weights':dict(path='state-abc/weights.pt',sha256=weights)},
            binding=dict(contract=digest('contract'),weights=weights))
        raw=json.dumps(record,sort_keys=True,separators=(',',':')).encode()+b'\n'
        pointer=dict(path='state-abc/checkpoint.json',sha256='sha256:'+hashlib.sha256(raw).hexdigest())
        desc=dict(pointer=pointer,record=record,authority='weights-only diagnostic; nonresumable')
        (root/'checkpoint.json').write_text(json.dumps(desc))
        return dict(checkpoint=pointer),dict(checkpoint=pointer,payload_hashes=dict(weights=weights)),dict(checkpoint=pointer,contract_sha256=digest('contract'))

    def test_snapshot_checks_actual_weight_bytes_and_original_record(self):
        from oeis_learn.pipeline_validation.pilot_calibration_import import snapshot
        with tempfile.TemporaryDirectory() as root:
            trainer,audit,settings=self.fixture(root)
            s=snapshot(root,0,trainer,audit,settings)
            self.assertEqual(s['ordinal'],0)
            Path(root,'weights.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'weights changed'):snapshot(root,0,trainer,audit,settings)

    def test_snapshot_rejects_swapped_pointer_or_record_or_contract(self):
        import json
        from oeis_learn.pipeline_validation.pilot_calibration_import import snapshot
        for kind in ('pointer','record','contract','missing','symlink'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as root:
                trainer,audit,settings=self.fixture(root)
                if kind=='pointer':settings=copy.deepcopy(settings);settings['checkpoint']['sha256']=digest('other')
                elif kind=='record':
                    p=Path(root,'checkpoint.json');d=json.loads(p.read_text());d['record']['binding']['contract']=digest('other');p.write_text(json.dumps(d))
                elif kind=='contract':settings['contract_sha256']=digest('other')
                elif kind=='missing':Path(root,'weights.pt').unlink()
                else:
                    p=Path(root,'weights.pt');p.rename(Path(root,'other'));p.symlink_to(Path(root,'other'))
                with self.assertRaises((ValueError,FileNotFoundError)):snapshot(root,0,trainer,audit,settings)

    def test_legacy_forecast_cannot_count_missing_clocks(self):
        from oeis_learn.pipeline_validation.pilot_calibration_cost import forecast
        from oeis_learn.pipeline_validation.pilot_clocks import POLICY
        source=digest('s');work=digest('w')
        rows=[seal(dict(complete=True,rows=296,source=source,work=work,seconds=11,generation_seconds=10,
            scoring_seconds=1,projected_all_selected_scoring_seconds=1,checkpoint=digest(i))) for i in range(3)]
        archives=[seal(dict(source=source,complete=True,members=4096,bytes=16<<20,filesystem_device=1,seconds=1,ordinal=i)) for i in range(3)]
        self.assertEqual(forecast(rows,source=source,work=work,archives=archives)['status'],'UNKNOWN')
        rows=[seal(dict(r,evaluation_policy=POLICY,clock_complete=True,portfolios=1184)) for r in rows]
        r=forecast(rows,source=source,work=work,archives=archives)
        self.assertEqual(r['status'],'CANDIDATE_COST_ONLY');self.assertFalse(r['long_run_admitted'])
        rows[0]=seal(dict(rows[0],generation_seconds=4000,seconds=4001))
        self.assertEqual(forecast(rows,source=source,work=work,archives=archives)['status'],'OVER_BUDGET')

class FailureRouting(unittest.TestCase):
    def test_missing_private_producer_blocks_engines_and_still_archives(self):
        import json
        from types import SimpleNamespace
        from oeis_learn.pipeline_validation.pilot_calibration import manifest,registry
        from oeis_learn.pipeline_validation.adapters import Adapters
        from oeis_learn.pipeline_validation.runner import execute
        from oeis_learn.pipeline_validation.bundle import verify
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(prefix='tmp-calibration-',dir=root) as tmp:
            tmp=Path(tmp);(tmp/'producer').mkdir();calls=[]
            def host(*a):calls.append(a);raise AssertionError('engine touched before private audit')
            args=SimpleNamespace(engine='podman',inputs=str(tmp/'producer'),pipeline_engines=['podman'])
            a=Adapters(args,root,host,manifest())
            class Gate:
                receipts={}
                def __call__(self,stage,out):
                    a.receipts=self.receipts
                    if stage.adapter=='host':return {'fixture':'already checked host contracts'}
                    return a(stage,out)
                def finalize(self,*args):return a.calibration.finalize(*args)
            out=tmp/'out'
            code=execute(registry(['podman'],'podman'),out,adapter=Gate(),binding={'source':digest('fixture')},
                cap=86400,reserve=1200,contract=manifest())
            self.assertEqual(code,1);self.assertEqual(calls,[])
            s=json.loads((out/'summary.json').read_text())
            self.assertEqual(s['stages']['engine.podman']['state'],'BLOCKED')
            self.assertEqual(s['stages']['calibration.session']['state'],'BLOCKED')
            self.assertFalse(verify(out.with_suffix('.zip'))['omissions'])
