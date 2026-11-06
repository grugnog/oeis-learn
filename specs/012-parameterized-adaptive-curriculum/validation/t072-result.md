# T072 — Actual HIP crash/recovery PASS

The operator supplied the small [original summary](evidence/t072-summary.json) on
2026-11-05. It is 1,653 bytes, SHA-256
`9b8df22a522e4e93786cbbfe0e16a692d99788adfc70f72a1070b3ccf7a96cf9`.
No ZIP, raw log or private checkpoint upload is required.

## Verified original result

| Check | Actual result |
| --- | --- |
| Dedicated test | 1 passed, 0 skipped; Podman |
| Device / scope | HIP, device RNG present; diagnostic only |
| Abrupt process death | Exit 86 after 19 actual optimizer calls |
| Last complete checkpoint | Initial generation, 0 committed updates |
| Newer unsaved state | 7 valid rewards and 2 graded anchors |
| Restoration | Old weights and complete old policy restored |
| Recovery segment | `recovery-1` |
| Recovered optimizer updates | 16 |
| Useful / discarded tokens | 752 / 910 |
| Charged / cap | 1,662 / 4,096 |
| Crashed process time | 4.875498110 seconds |
| Charged learner time | 9.404323475 seconds |

The exact arithmetic is **752 + 910 = 1,662 <= 4,096**. Crash time remains charged.
The restored checkpoint is
`sha256:fa9a2d31baf8473bb1b35d85cfd5f8b73c256ec9a3d4d03ee7ed8ae3a11dd508`.
The subprocess test additionally checks the complete saved policy and observer,
allowing only cost/segment fields to change, and compares host/device torch RNG
before the first recovered gradient. Thus the device proof does not restore newer
reward/scaler/review events onto the old model. Sparse semantic generation is
explicitly disabled for this accounting fixture; real target-loss/reward/grade,
optimizer work and checkpoint recovery remain enabled.

## Source, helper, image and proof binding

| Binding | Original value |
| --- | --- |
| Source | `sha256:a38fa6acab09e0b978ce559f928a3270a4a45842c265a4236070e8d7ccd6b4af` |
| Immutable image | `sha256:cb977acdc053e9f18362f53f8b396045641c04b3470fafa0dd1fe4ce0fabb776` |
| Helper bytes | `sha256:c2a02788efff1397678ca1a53724311543f5c0d2dfa9967d0e2a9712e0014356` |
| Proof bytes | `sha256:ae1970867faa2a4b3b8d5e85cf041189fed06a6c7da84bfe07c27d2576e18687` |
| Inner proof identity | `sha256:6d91eacbe877057db995131eaf812028f7b715521bc9edf74a5e5e1d78d54928` |
| Summary identity | `sha256:3950c0d0b0e87c873e6ee66d59043eb070a631adc6dd810d257bae73e026890a` |
| No-Lt execution policy | `sha256:7e8b6150696841d883936a6599c05d80b7055cb6d8c518846c29bc0897f5d71d` |

The outer and inner seals, strict fields, exact live source snapshot at producer
revision `425fff95249fd32eb426fd4a1e4e4d4ef98a0c15`, helper file hash and
reconstructed original proof-byte hash all verify. The summary attests that the
source-bound host helper checked original JUnit, stage receipt and proof-file
hashes on the workstation. Those raw receipts were not separately uploaded and
were not re-read here. This follows the declared small-summary evidence path;
no extra private-state or unprovided raw-log audit is claimed.

The original -004 clean-pause and success/failure archive proofs remain tied to
their original source/image pair. Comparing original producer source revision
`9bcd598ef407467d0bf9921d37e6789a9b185c20` with the focused producer shows changes
only to the portable parser, container validation, the added crash-proof validator
and its two test files. The learner, persistence, curriculum, checkpoint, model/
codec algorithms, Dockerfile and dependency locks are byte-unchanged. No old
qualification was rebound to a different global snapshot. This supplemental
actual HIP proof closes the missing T072 crash-debit sub-proof alongside the
already verified clean-resume/negative-export evidence. It is not a replacement
full qualification, research result or production throughput measurement.

**T072 is COMPLETE.** All mandatory Phase E software/evidence work is now complete;
only conditional T073 research analysis remains unchecked/DEFERRED. No new run,
research campaign, seed, assessment execution, winner or archive publication is
authorized or needed by this result.

## Reproduce the summary check

Run from the producer checkout or the current documentation-only successor:

```sh
PYTHONPATH=src python3 -S - <<'PY'
import hashlib,json
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal,digest,file_hash
from oeis_learn.pipeline_validation.generalization_crash_proof import verify
from oeis_learn.experiments.source_snapshot import collect_snapshot
p=Path('specs/012-parameterized-adaptive-curriculum/validation/evidence/t072-summary.json')
v=json.loads(p.read_text())
assert p.stat().st_size==1653
assert hashlib.sha256(p.read_bytes()).hexdigest()=='9b8df22a522e4e93786cbbfe0e16a692d99788adfc70f72a1070b3ccf7a96cf9'
assert set(v)==set('version identity status diagnostic_only engine image helper_sha256 source proof_sha256 proof tests skipped zip_upload'.split())
assert seal(v)==v and v['version']=='pac-t072-summary/v1' and v['status']=='PASS'
assert v['diagnostic_only'] is True and v['tests']==1 and v['skipped']==0
assert v['engine']=='podman' and v['zip_upload']=='NOT_REQUESTED'
assert v['source']==digest(collect_snapshot(Path.cwd()))
assert v['helper_sha256']==file_hash(Path('specs/012-parameterized-adaptive-curriculum/validation/t072.py'))
verify(v['proof'],source=v['source'])
raw=(json.dumps(v['proof'],sort_keys=True,indent=2)+'\n').encode()
assert 'sha256:'+hashlib.sha256(raw).hexdigest()==v['proof_sha256']
print('PASS: original summary, producer bindings, actual HIP crash recovery and retained debits')
PY
```

A later runtime/test/helper change should fail the producer identity comparison;
use the identified producer revision to replay the historical proof rather than
rewriting the uploaded result.
