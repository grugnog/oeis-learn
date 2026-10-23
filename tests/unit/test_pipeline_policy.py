import copy
import itertools
import unittest
from oeis_learn.fresh.synthesis_policy import select_visible, transfer_success
from oeis_learn.fresh.candidate_archive import retain_alternatives, training_representative
from oeis_learn.synthesis.admission import source_identity


class Policy(unittest.TestCase):
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
