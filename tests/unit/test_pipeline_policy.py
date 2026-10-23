import copy
import itertools
import unittest
from oeis_learn.fresh.synthesis_policy import select_visible, transfer_success
from oeis_learn.fresh.candidate_archive import retain_alternatives, training_representative
from oeis_learn.synthesis.admission import source_identity


class Policy(unittest.TestCase):
    def test_v2_archive_capacity_omissions_and_conflicting_duplicates(self):
        rows=[dict(id=source_identity(str(i)),source=str(i),structure=str(i),cost=i,
            verified=True,verification=source_identity('verification'),track='generic_only',
            profile='bigint-body/v1',assignment={},attempt_ordinal=0,job_id=source_identity(str(i))) for i in range(10)]
        result=retain_alternatives(rows,policy='structure-pareto/v2')
        self.assertEqual(len(result['retained']),8)
        self.assertEqual(len(result['omitted']),2)
        self.assertTrue(all(r['reason']=='frozen_capacity' for r in result['omitted']))
        self.assertEqual(result,retain_alternatives(rows[::-1],policy='structure-pareto/v2'))
        for field,value in [('verification',source_identity('different')),('track','assisted')]:
            with self.assertRaises(ValueError):
                retain_alternatives([rows[0],dict(rows[0],**{field:value})],policy='structure-pareto/v2')

    def test_serial_policy_report_retains_both_provenance_decisions(self):
        from oeis_learn.pipeline_validation.policies import fixture_report
        report=fixture_report()
        self.assertTrue(report['fixtures_only'])
        comparison=report['records'][0]
        self.assertEqual(comparison['selected_provenance']['v1']['attempt_ordinal'],1)
        self.assertEqual(comparison['selections']['v2']['attempt_ordinal'],0)
        self.assertTrue(comparison['differences']['selection'])
        self.assertEqual(comparison['archives']['v2']['retained'][0]['attempt_ordinal'],0)

    def visible(self, ordinal, cost):
        return dict(source='x',source_sha256=source_identity('x'),verified=True,visible_count=20,
            cost=cost,attempt_ordinal=ordinal,job_id=source_identity(str(ordinal)),
            provenance=dict(prompt='p',snapshot='s',policies={'selection':'v2'},original_source_sha256='o',
                            transformed_source_sha256='t',assignment={},verification='v'))

    def test_v2_timing_arrival_and_tie_provenance(self):
        rows=[self.visible(1,0),self.visible(0,99)]
        selected=select_visible(rows,policy='visible-only/v2')
        self.assertEqual(selected['attempt_ordinal'],0)
        for order in itertools.permutations(rows):
            self.assertEqual(selected,select_visible(order,policy='visible-only/v2'))
        rows[0]['cost']=999;rows[1]['cost']=0
        self.assertEqual(selected,select_visible(rows,policy='visible-only/v2'))
        self.assertNotIn('cost',selected)
        with self.assertRaises(ValueError):
            select_visible(rows,policy='unknown')
        selected['source']='changed'
        with self.assertRaises(ValueError):
            transfer_success(selected,[],[],{}, {})

    def test_v2_frontier_uses_assignment_bits_not_duration(self):
        def row(i,source,assignment):
            return dict(id=source_identity(source),source=source,structure='s',cost=100-i,
                verified=True,verification='sha256:'+'1'*64,track='generic_only',profile='bigint-body/v1',
                assignment=assignment,attempt_ordinal=i,job_id=source_identity(str(i)))
        rows=[row(0,'x',{'p0':'1024'}),row(1,'long',{})]
        a=retain_alternatives(rows,policy='structure-pareto/v2')
        self.assertEqual(len(a['retained']),2)
        selected=training_representative(a)
        rows[0]['cost']=0;rows[1]['cost']=999
        b=retain_alternatives(rows[::-1],policy='structure-pareto/v2')
        self.assertEqual([r['id'] for r in a['retained']],[r['id'] for r in b['retained']])
        self.assertEqual(selected['id'],training_representative(b)['id'])

    def test_selected_provenance_is_detached(self):
        row = self.visible(0, 1)
        selected = select_visible([row], policy='visible-only/v2')
        row['provenance']['assignment']['p0'] = '2'
        self.assertEqual(selected['provenance']['assignment'], {})

    def test_equal_source_retains_earliest_provenance_before_assignment_size(self):
        base = dict(id=source_identity('x'), source='x', structure='s', cost=10,
            verified=True, verification='sha256:'+'1'*64, track='generic_only',
            profile='bigint-body/v1', assignment={'p0':'1024'}, attempt_ordinal=0,
            job_id=source_identity('0'))
        later = dict(base, assignment={}, attempt_ordinal=1, job_id=source_identity('1'))
        for rows in ([base, later], [later, base]):
            result = retain_alternatives(rows, policy='structure-pareto/v2')
            self.assertEqual(len(result['retained']), 1)
            self.assertEqual(result['retained'][0]['attempt_ordinal'], 0)

    def test_v2_envelope_provenance_cannot_be_changed_after_selection(self):
        from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
        from oeis_learn.synthesis.contracts import identity
        row=self.visible(0,1)
        verification={'prompt_sha256':source_identity('p')}
        row['provenance'].update(prompt=source_identity('p'),snapshot=source_identity('s'),
            original_source_sha256=source_identity('original'),transformed_source_sha256=source_identity('transformed'),
            verification=identity(verification))
        selected=select_visible([row],policy='visible-only/v2')
        envelope=dict(selected=selected,verification=verification,assignment={},original_source='original',
            transformed_source='transformed',original_source_sha256=source_identity('original'),prompt_sha256=source_identity('p'))
        validate_selection_envelope(envelope)
        for change in [dict(assignment={'p0':'1'}),dict(original_source='changed'),dict(transformed_source='changed'),dict(prompt_sha256=source_identity('q'))]:
            with self.assertRaises(ValueError):validate_selection_envelope(dict(envelope,**change))

    def test_v1_cost_selection_and_seal_are_preserved(self):
        from oeis_learn.synthesis.contracts import identity
        rows=[dict(source='x',source_sha256=source_identity('x'),verified=True,visible_count=20,cost=c) for c in (9,1)]
        selected=select_visible(rows)
        self.assertEqual(selected,dict(rows[1],selection='visible-only/v1',seal_sha256=identity(rows[1])))
