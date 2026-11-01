# E0 readiness review after acceptance-016

Reviewed 2026-11-01 Pacific against production ef886a7 using the Spec Kit implement
workflow. Requirements checklist16/16 and design15/15 pass; implementation checklist
has no items. No extension hooks exist. Existing Python/Rust/workspace ignore rules
are present. No production code, locks, authoritative contracts or image pin changed.

## What already exists

| Requirement | Existing implementation / evidence |
| --- | --- |
| E0, zero training, frozen choices | generalization/execution.py validate_policy/bind; generalization_inputs.py assemble_study; E0 recipe template |
| Original uniform weights provenance | generalization/legacy.py verify_legacy; inputs import-uniform; weights-only descriptors cannot claim full-state resume |
| Isolated exact preparation | generalization_preparation.py; role views and native/reference admission; explicit quota shortfall |
| Original qualification and feature gate | generalization_commands._qualification; qualification_io.consume_original with research=True |
| Real model loading and matched controls | generalization_workflow.evaluate; generalization_evaluation.evaluate_retained; fresh/bigint_diagnostics.generate |
| Shared absent-input bank | build_bank/apply_bank/bank_cost, frozen seeds and explicit CLI bank identity |
| Reproducible prospective HIP execution | generalization_execution startup flags; role launcher before Torch; installed generator checks; qualification policy binding |
| Deadline containment | projection/copy and model setup charged; external role supervisor; separate startup/exit/cleanup reserves; bounded native calls |
| Research report verification | reporting.package_evidence and generalization.bundles.verify_bundle independently reconcile an explicit aggregate evidence root |
| Restart | fresh output roots; lifecycle invocation identity/verified reuse; original producer closure; no unchecked skip flags |

Existing source is not a reason to implement another evaluator or bypass canonical
qualification. Acceptance-016 demonstrates installed operation on the tested small
fixtures. The current local regression command passes387 tests, skips1 for absent
local Wasmtime and deselects1 GPU test. Actual CPU/native/HIP results are in the
separate acceptance review; this local run does not replace hardware evidence.

## Launch gaps found in the review

1. **Research authority is not assembled.** generalization-E0.json explicitly says
   PREPARATION_REQUIRED. The uploaded acceptance archive has no original recipe,
   prepared cohort inputs, research study/runtime or checkpoint provenance tree.
   The imported uniform best/latest are available on the operator machine; availability
   of the earlier3-hour/selected snapshots is not established here. No substitute
   weights or reduced acceptance rows can stand in for those research inputs.
2. **The old quickstart skipped preparation.** Model work rejects a runtime whose
   preparation is non-null. The corrected example explicitly prepares and consumes
   the returned resolved manifest/runtime. Preflight alone does not resolve them.
3. **The old quickstart omitted the shared bank.** The bank is generated only when
   an explicit proposal-bank argument is supplied. Freeze eight unique integer
   seeds, pass their canonical digest and include the bank in the measured roster.
4. **Time budgets need the actual roster.** A single evaluate scope is capped at
   21600 seconds, and allocates seconds/(2*checkpoint_count) to each generator.
   That includes projection/loading/startup and cleanup. The E0 campaign's24-hour
   ceiling is not a per-call override. Count every prompt/control portfolio and
   bank application, plus scoring/publication and final reporting; don't transplant
   the acceptance720-second setting or simply fill every slot up to the ceiling.
   Any batching must be frozen up front with shared-bank/accounting semantics
   preserved, not invented after inspecting outcomes. Record censoring explicitly.
5. **Research export needs an explicit adapter/assembly step.** The evaluator
   writes per-checkpoint generation, selection and command records. It does not
   automatically write the aggregate evidence.json required by report_main.
   The generic report verifier is implemented; the one-command E0 delivery still
   needs its frozen measurement roster, public attempt/cost projections, original
   stage ledger and complete qualification closure assembled and tested together.
   A thin command ZIP must not be presented as a scientific report. This wiring
   can live in a temporary validation helper using the existing policy functions;
   new reconciliation algorithms, if required, must reopen the single Phase B.

The concrete E0 launch remains BLOCKED on research input identification and the
resulting orchestration/export handoff. This review does not certify that a full
research campaign is already turnkey. Software acceptance and research-delivery
readiness are separate findings. No new full qualification run is requested now;
any later production/source-closure change must be checked by canonical consumers.

## Small metadata handoff, no HIP

The command was exercised with the existing uniform-import fixture: successful
stdlib execution, exact allowlist/hash verification, rejection of an existing
output without modification, and missing-input rejection before creating output.

Run this from the existing project checkout. It reads eleven specific public
import JSON files, with a4MiB per-file bound; it does not enumerate the tree or read
private weights/targets. The original prepared metadata identifies the historical
data inventory so the next input step can target the correct corpus instead of
asking for a large file listing. Missing files stop before creating the ZIP.
This is an input-discovery package, never a qualification or research result.

```bash
python3 - <<'PY'
import hashlib, json, zipfile
from pathlib import Path

root = Path('run-state/generalization/uniform-import').resolve(strict=True)
names = ['import.json', 'best-checkpoint.json', 'latest-checkpoint.json',
         'best-provenance.json', 'latest-provenance.json']
names += ['provenance/' + name + '.json'
          for name in ('report', 'producer', 'history', 'prepared', 'recipe', 'training')]
payloads = {}
for name in names:
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise SystemExit('Missing regular metadata file: ' + str(path))
    path.resolve(strict=True).relative_to(root)
    with path.open('rb') as stream:
        data = stream.read((4 << 20) + 1)
    if len(data) > 4 << 20:
        raise SystemExit('Metadata exceeds size bound: ' + name)
    json.loads(data)
    payloads[name] = data
manifest = {'purpose': 'E0 input discovery only; no qualification claim',
            'members': [{'path': name, 'bytes': len(data),
                         'sha256': hashlib.sha256(data).hexdigest()}
                        for name, data in payloads.items()]}
output = Path('run-state/generalization/e0-input-metadata-001.zip')
with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
    for name, data in payloads.items():
        archive.writestr(name, data)
    archive.writestr('metadata-manifest.json', json.dumps(manifest, indent=2))
print(output)
PY
```

Upload e0-input-metadata-001.zip. Keep all private weights and acceptance-016 on
the operator workstation. The next deliverable uses these exact metadata identities
to identify/freeze the E0 research roster and produce one bounded serial command
through assembly, preflight, prepare, evaluation, report and canonical verification.
The original raw source corpus or additional snapshot metadata may still be needed;
their precise references must come from this inventory, not guessed paths.


## Metadata-001 received: exact historical corpus identified

All11 metadata members verify by length and SHA256. Both descriptors pass the
canonical `verify_legacy` audit under python -S, without loading tensors.
Uniform best:17,885,681 tokens /18,433 updates; latest:24,999,970 tokens /
25,764 updates. Both use the original m100 FP32 recipe and remain weights-only.
The earlier3-hour and behavior-selected snapshots are not in this import.

The exact original prepared identity is
`sha256:8a8ddc9b4f3160c5d6577126dc688eac3f555b4b7d58e29e01d6792f9c85609e`.
Its roster has32 training-panel,128 parameter-development,128 structural-development
and8 already-exposed OEIS-development rows (296 total). All128 structural rows
are stage A and span18 families; these are historical limitations, not a new
loop/branch structural benchmark. No sealed final cohort is requested or exported.

Only two additional payloads are required to inspect that actual evaluation data:

| Original file | Bytes | SHA256 |
| --- | --- | --- |
| prompts.json | 648182 | b76452d12425b0b525491c38e7ed6f9767ff56b4d5c3635d482c388b384ceaed |
| private/truth.json | 878378 | a01b7d9487e79bfc63caf0b914248f1d6418123aad75d950d1a6a6998fb331ef |

The temporary collector below searches at most100000 entries,12 descendant
levels and60 seconds, skips symlinks/caches/checkpoints, and matches both file
hashes plus the byte-identical prepared.json. It reads no weight file, no large
examples/admissions file and no sealed final file. It exports the two **historical
already-exposed** payloads with the11 audited metadata documents and a discovery
inventory containing the matched source directory. It neither invokes HIP nor
modifies the original run. It fails before export if the exact triple is absent.
If the bounded search exhausts, retry with --search-root pointing at the original
input parent; do not relax hash matching or substitute a different corpus.

```bash
(
  set -e
  test "$(git branch --show-current)" = "011-controlled-generalization"
  git pull --ff-only origin 011-controlled-generalization
  python3 -S specs/011-controlled-generalization/validation/collect_e0_inputs.py
)
```

Upload `run-state/generalization/e0-historical-inputs-001.zip`. This resolves the
specific remaining local-data blocker without a broad file listing. The next E0
launch must preserve the historical/exposed label and measured denominator; it
must not relabel these rows as fresh assessment or fabricate new cohort coverage.

Validation:19 discovery/legacy/input tests pass. Collector tests cover matched
input export, exclusion of extra private files, rejection of each corrupted input,
fresh-output protection, search bounds and symlink roots. The actual uploaded
metadata independently passes canonical provenance verification. No production
source closure changed and no HIP or CI wait ran.


## Superseded next step: operator-approved fresh start

The historical prompt/truth archive arrived and its exact metadata binding was
verified. Those rows include source-free OEIS targets; the prospective selection
path requires source programs. A historical scoring adapter would therefore be
additional work, not merely input assembly. On2026-11-01 Pacific the operator
approved deferring E0/P0 and executing fresh P1 instead. Preserve this review and
historical files; do not interpret deferral as completion. Current research-scale
repairs and delivery tasks are T132–T137.
