"""Strict selected-population overlap and durable corpus preparation evidence."""
import unittest
import tempfile
import json
import shutil
from unittest.mock import patch
from pathlib import Path
from oeis_learn.pipeline_validation.identity import digest


class CorpusSelection(unittest.TestCase):
    def test_failed_preparation_keeps_specific_reason_and_reusable_reference_prefix(self):
        from oeis_learn.pipeline_validation import pilot_data
        from oeis_learn.pipeline_validation.pilot_selection import Shortfall
        from oeis_learn.pipeline_validation.identity import seal,file_hash,write
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inputs=root/'inputs';inputs.mkdir();output=root/'first';output.mkdir()
            truth={str(i):dict(indices=list(map(str,range(100))),values=[str(n+1000000*i) for n in range(100)]) for i in range(8)}
            prompts=[dict(id=k,indices=v['indices'][:20],values=v['values'][:20],horizon_indices=v['indices']) for k,v in truth.items()]
            write(inputs/'oeis-prompts.json',prompts);write(inputs/'oeis-truth.json',truth)
            write(inputs/'oeis-lineage.json',seal(dict(version='pipeline-pilot-exposed-development/v1',prepared=digest('approved'),count=8,exposure='fixture',
                files={n:dict(sha256=file_hash(inputs/n),bytes=(inputs/n).stat().st_size) for n in ('oeis-prompts.json','oeis-truth.json')})))
            failure=Shortfall(dict(counts={'training':0},required={'training':32768},reason='explicit fixture shortage'))
            config=dict(source=digest('source'),options=dict(seconds=120))
            with patch.dict(pilot_data.SPEC,max_source_counters=1),patch.object(pilot_data,'CHECKPOINTS',(1,)),patch.object(pilot_data,'choose',side_effect=failure):
                with self.assertRaisesRegex(ValueError,'training'):pilot_data.prepare(inputs,output,config)
                report=json.loads((output/'preparation.json').read_text())
                self.assertEqual(report['status'],'FAILED')
                self.assertEqual(report['selection_attempts'][0]['audit']['reason'],'explicit fixture shortage')
                self.assertEqual(report['reference_cache']['count'],1)
                shutil.copytree(output/'private/reference-cache',inputs/'reference-cache')
                second=root/'second';second.mkdir()
                with patch('oeis_learn.data.bigint_programs.materialize',side_effect=AssertionError('must reuse original reference')):
                    with self.assertRaisesRegex(ValueError,'training'):pilot_data.prepare(inputs,second,config)
                self.assertEqual(json.loads((second/'preparation.json').read_text())['reused_reference_counters'],1)
                from oeis_learn.pipeline_validation.pilot_cache import ReferenceCache
                original=list(ReferenceCache(output/'private/reference-cache',report['reference_cache']['binding']).records())[0]['row']
                for label,change,pattern in [
                    ('overflow',dict(values=[str(1<<4096)]+original['values'][1:]),'input integer'),
                    ('oracle',dict(oracle='unknown-reference'),'oracle'),
                    ('fields',dict(unexpected=True),'fields')]:
                    row=dict(original,**change);seed=ReferenceCache(root/label,report['reference_cache']['binding'])
                    seed.append(0,row);seed.flush()
                    shutil.rmtree(inputs/'reference-cache');shutil.copytree(root/label,inputs/'reference-cache')
                    destination=root/(label+'-result');destination.mkdir()
                    with self.assertRaisesRegex(ValueError,pattern):pilot_data.prepare(inputs,destination,config)


    def test_selection_deadline_reports_partial_counts_before_stopping(self):
        from oeis_learn.pipeline_validation.pilot_selection import select
        observed=[]
        with self.assertRaisesRegex(TimeoutError,'selection deadline'):
            select([],training=8,parameter=1,structural=1,deadline=0,progress=observed.append)
        self.assertEqual(observed[-1]['counts'],dict(training=0,parameter_development=0,structural_development=0))
        self.assertEqual(observed[-1]['required']['training'],8)
        self.assertEqual(observed[-1]['stages']['training'],{})
        self.assertEqual(observed[-1]['selected_families']['training'],0)

    def test_reference_compaction_preserves_all_bytes_and_shares_repeated_storage(self):
        from oeis_learn.pipeline_validation.pilot_data import compact_reference
        from oeis_learn.data.bigint_programs import BigintSampler,materialize
        row=materialize(BigintSampler().sample(0));before=json.dumps(row,sort_keys=True)
        a=compact_reference(json.loads(before));b=compact_reference(json.loads(before))
        self.assertEqual(json.dumps(a,sort_keys=True),before)
        self.assertIs(a['indices'],b['indices'])
        self.assertIs(a['source'],b['source'])
        for x,y in zip(a['values'],b['values']):self.assertIs(x,y)
        bad=json.loads(before);bad['indices'][-1]='100'
        with self.assertRaisesRegex(ValueError,'horizon'):compact_reference(bad)

    def test_reference_cache_preserves_prefix_and_rejects_corruption_or_changed_semantics(self):
        from oeis_learn.pipeline_validation.pilot_cache import ReferenceCache
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);first=ReferenceCache(root/'first',digest('math'))
            for i in range(3):first.append(i,{'value':i})
            first.flush()
            resumed=ReferenceCache(root/'second',digest('math'),seed=root/'first')
            self.assertEqual([r['counter'] for r in resumed.records()],[0,1,2])
            resumed.append(3,None,'original reference exclusion');resumed.flush()
            self.assertEqual(resumed.count,4)
            with self.assertRaises(ValueError):resumed.append(3,{})
            with self.assertRaises(ValueError):ReferenceCache(root/'third',digest('changed'),seed=root/'first')
            next((root/'second').glob('*.gz')).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):list(resumed.records())

    def test_unused_bridge_does_not_merge_two_disjoint_selected_examples(self):
        from oeis_learn.pipeline_validation.pilot_selection import Windows
        from oeis_learn.fresh.bigint_splits import overlap_groups
        ns=list(map(str,range(100)))
        a=dict(id='a',indices=ns,values=[str(i*i) for i in range(100)])
        b=dict(id='b',indices=ns,values=[str(i**3+1000000) for i in range(100)])
        bridge=dict(id='unused',indices=ns,values=a['values'][:50]+b['values'][50:])
        groups=overlap_groups([a,b,bridge]);self.assertEqual(groups['a'],groups['b'])
        guard=Windows();self.assertTrue(guard.admit(a));self.assertTrue(guard.admit(b))
        self.assertFalse(guard.admit(bridge))
        self.assertNotEqual(overlap_groups([a,b])['a'],overlap_groups([a,b])['b'])

    def test_actual_codec_selection_has_exact_counts_and_no_shared_windows(self):
        from oeis_learn.pipeline_validation.pilot_selection import select
        from oeis_learn.fresh.bigint_splits import overlap_groups
        from oeis_learn.data.bigint_programs import canonical_family
        rows=[]
        for family in range(4):
            term='(add $index (int 1))';power=term
            for _ in range(family+1):power=f'(mul {power} {term})'
            template=f'(program (return (add {power} (param.get p0))))'
            for variant in range(16):
                constant=variant*1000000
                rows.append(dict(id=digest([family,variant]),family=canonical_family(template),variant=variant,
                    stage='ABCD'[family],source=template.replace('(param.get p0)',f'(int {constant})'),
                    parameter_source=template,assignment={'p0':str(constant)},indices=list(map(str,range(100))),
                    values=[str((i+1)**(family+2)+constant) for i in range(100)],origin='generic_random'))
        result=select(rows,training=8,parameter=1,structural=1)
        self.assertEqual({k:len(v) for k,v in result['splits'].items()},dict(training=8,parameter_development=1,structural_development=1))
        flat=[r for rs in result['splits'].values() for r in rs]
        self.assertEqual(len(set(overlap_groups(flat).values())),10)
        training={r['family'] for r in result['splits']['training']}
        self.assertTrue({r['family'] for r in result['splits']['parameter_development']}<=training)
        self.assertFalse({r['family'] for r in result['splits']['structural_development']}&training)



class CompleteCodecValidation(unittest.TestCase):
    def test_complete_parser_matches_incremental_grammar_on_valid_and_invalid_streams(self):
        from oeis_learn.decoder.bigint_codec import encode_body,validate_body,PAD_ID,EOS_ID
        from oeis_learn.decoder.bigint_grammar import BodyState,CodecError
        from oeis_learn.data.bigint_programs import BigintSampler
        def incremental(ids):
            state=BodyState()
            try:
                for token in ids:state.consume(token)
                return state.finished
            except CodecError:return False
        sampler=BigintSampler()
        for counter in range(0,512,16):
            ids=encode_body(sampler.sample(counter)['source'])
            cases=[ids,ids[:-1],ids[:len(ids)//2],ids+[EOS_ID],[],[True],[1.0],[-1],[999999]]
            for offset in (0,len(ids)//2,len(ids)-2):
                bad=list(ids);bad[offset]=PAD_ID;cases.append(bad)
            for case in cases:
                try:validate_body(case);actual=True
                except CodecError:actual=False
                self.assertEqual(actual,incremental(case),(counter,case))
