"""Versioned long-run authority and unchanged historical cost semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from oeis_learn.pipeline_validation.identity import digest,seal,write,file_hash


def candidate():
    from oeis_learn.pipeline_validation.pilot_calibration_cost import forecast
    s=digest('producer');w=digest('work')
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

if __name__=='__main__':unittest.main()
