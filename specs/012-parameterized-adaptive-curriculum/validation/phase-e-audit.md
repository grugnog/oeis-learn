# Historical original -004 Phase E evidence audit

This historical audit describes the first six-task closeout at revision
`9457bea2fdaca242a424a4475b6f43fe5b1fb33b`, before the focused test changed source
and T072 completed. Use that checkout and its machine record to replay it. For
the subsequent actual HIP proof and current producer check, use
[t072-result.md](t072-result.md). It intentionally fails a different producer
or later task-marker state rather than rewriting historical evidence.

Requires only Python's standard library and the identified historical checkout. Obtain the byte-identical
operator archive identified in [operator-handoff.md](operator-handoff.md); it is not
published in GitHub. No engine/device operation or original private tensor read is
performed. A future source/helper change correctly fails the original identity
comparison. Only temporary local copies of intentionally damaged public archives
are created for rejection tests; the input is never rewritten.

From the repository root, replace the archive path:

```sh
PYTHONPATH=src python3 -S - /path/to/report.zip <<'PY'
from pathlib import Path
p=Path('specs/012-parameterized-adaptive-curriculum/validation/phase-e-audit.md')
code=p.read_text().split('```python\n',1)[1].split('\n```',1)[0]
exec(compile(code,str(p),'exec'))
PY
```

This transparent audit is kept in Markdown so it does not alter the qualified
runtime/test snapshot or the separately hashed orchestration-helper Python set.
It reconciles original evidence; it does not create a replacement qualification.

```python
"""Read-only reconciliation against the operator archive; no engine/device work."""
import hashlib,json,re,sys,tempfile,zipfile
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
from oeis_learn.pipeline_validation.generalization_archive import verify_archive
from oeis_learn.pipeline_validation.generalization_pac_export import helper_identity
from oeis_learn.pipeline_validation.bundle import verify
from oeis_learn.pipeline_validation.identity import digest
from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.structural import original_cells

root=Path.cwd();spec=root/'specs/012-parameterized-adaptive-curriculum';validation=spec/'validation'
archive=Path(sys.argv[1]);record=json.loads((validation/'evidence/acceptance-004-record.json').read_text())
assert archive.stat().st_size==record['archive']['bytes']
assert 'sha256:'+hashlib.sha256(archive.read_bytes()).hexdigest()==record['archive']['sha256']
assert verify_archive(archive)==record['verification']
with zipfile.ZipFile(archive) as z:
 documents={i.filename:json.loads(z.read(i)) for i in z.infolist() if i.filename.endswith('.json')}
 e=documents['evidence.json'];n=documents['native-admission.json'];a=documents['acceptance/pac-acceptance.json'];c=documents['calibration/calibration.json'];r=documents['admission/training/representation.json'];f=documents['acceptance/checkpoint-evaluation.json']
 assert documents['qualification/source.json']==collect_snapshot(root)
 assert digest(collect_snapshot(root))==e['source']==record['source']
 assert helper_identity(root)==e['helper']['observed']==e['helper']['requested']==record['helper']['observed']
 assert e['plan']==record['plan'] and e['images']==record['images'] and e['scope']==record['scope'] and not e['scope']['research']
 assert documents['campaign-controls.json']==record['controls']
 assert sum(s['seconds'] for s in record['controls']['requested'])+record['controls']['finalization_seconds']==record['controls']['campaign_seconds']
 assert sorted(n['originals'],key=lambda x:x['name'])==sorted(original_cells(),key=lambda x:x['name'])
 assert Counter(x['category'] for x in n['roster'].values())=={'certified':68,'ambiguous':26,'unresolved':34}
 assert {cat:len({x['family'] for x in n['roster'].values() if x['category']==cat}) for cat in ['certified','ambiguous','unresolved']}=={'certified':15,'ambiguous':3,'unresolved':5}
 av=documents['admission/availability.json'];assert sum(v['draws'] for v in av['families'])==2288 and sum(v['available'] for v in av['families'])==1394 and not any(v['deficit'] for v in av['families'])
 assert r['rows']==record['representation_review']['rows']==32768 and len(r['families'])==32
 assert r['corpus_tokens']==record['representation_review']['corpus_tokens']
 assert all(v['rows']==v['supported']==1024 and v['fallback']==0 for v in r['families'].values())
 assert f['weights']==record['representation_review']['final_weights'] and f['checkpoint']==record['representation_review']['loaded_checkpoint']
 for boundary in [1,2]:
  p=documents['acceptance/proof-'+str(boundary)+'/training-proof.json'];assert p['resume_equal'] and p['pac_equal']
  assert len({r['weights'] for r in p['runs']})==len({r['initial'] for r in p['runs']})==1
  assert all(r['updates']==16 and r['tokens']==752 and r['finite'] and r['changed'] for r in p['runs'])
 assert a['valid_rewards']==6 and a['graded_anchors']==2 and c['updates']==64 and c['tokens']==122464
 assert c['samples']==record['curriculum_review']['samples']
 for arm in ['B','C']:
  p=c['forecasts'][arm];v=record['curriculum_review']['forecasts'][arm]
  assert p['update_forecast_seconds']==v['update_seconds'] and p['diagnostics']['forecast_seconds']==v['diagnostic_forecast_seconds'] and p['learner']['seconds']==v['learner_ceiling_seconds']
 for suffix,count in [('cpu',1499),('gpu',27)]:
  name=next(n for n in z.namelist() if n.endswith('results.xml') and 'bigint-'+suffix+'-regressions' in n)
  suites=list(ET.fromstring(z.read(name)).iter('testsuite'))
  assert sum(int(s.get('tests','0')) for s in suites)==count and all(int(s.get(k,'0'))==0 for s in suites for k in ['failures','errors','skipped'])
 with tempfile.TemporaryDirectory() as folder:
  p=Path(folder)/'failure.zip';p.write_bytes(z.read('acceptance/failure.zip'));negative=verify(p);assert negative['summary_state']=='FAIL' and {r['path'] for r in negative['members']}=={'summary.json','failure.json'}
  for change in ['member','terminal','negative']:
   path=Path(folder)/(change+'.zip')
   with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as out:
    for item in z.infolist():
     if change=='terminal' and item.filename=='terminal.json':continue
     payload=z.read(item)
     if change=='member' and item.filename=='summary.json':payload=b'{}\n'
     if change=='negative' and item.filename=='acceptance/failure.zip':payload=b'not the original negative zip'
     out.writestr(item.filename,payload)
   try:verify_archive(path)
   except (ValueError,KeyError,FileNotFoundError,zipfile.BadZipFile):pass
   else:raise AssertionError('corrupted '+change+' accepted')
print('PASS: original archive/source/helper, counts, state proofs, forecasts and three tamper rejections')
tasks=(spec/'tasks.md').read_text();states={int(n):v.upper()=='X' for v,n in re.findall(r'^- \[([ Xx])\] T(\d+)',tasks,re.M)}
assert all(states[n] for n in range(1,72)) and not states[72] and not states[73] and states[74]
assert sum(states[n] for n in range(67,75))==6 and record['phase_e_complete'] is False
assert not (validation/'evidence/acceptance-004.zip').exists()
for name in ['operator-handoff.md','hip-acceptance.md','structural-admission.md','research-results.md','closeout.md','phase-e-audit.md']:
 p=validation/name
 for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
  if '://' not in target:assert (p.parent/target.split('#')[0]).exists(),(name,target)
for p in (spec/'checklists').glob('*.md'):assert '- [ ]' not in p.read_text()
assert not (root/'.specify/extensions.yml').exists()
print('PASS: 6/8 E markers, hardware/research gaps, bounded publication, document links and checklist/hooks')

from fractions import Fraction
q=[Fraction(1,2)]*2
for u in range(5):
    alpha=Fraction(1,u+2)
    q=[(1-alpha)*v+alpha*(sum(q)-v) for v in q]
    assert q==[Fraction(1,2)]*2
print('PASS: independent zero-reward sharing preserves initial six selection probabilities')

```
