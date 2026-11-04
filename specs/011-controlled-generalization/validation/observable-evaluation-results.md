# Observable continuation follow-up — 2026-11-04

T168–T170 are implemented. The certified portion still gives a near tie:
**109/153 static versus110/153 refreshed**. This is a post-hoc projection of the
existing pilot, not a new experiment or replacement primary score. The confirmed
bank-original provenance defect is also repaired for future evaluations.

## Implemented policy and evidence

[The plan](observable-evaluation-plan.md) preceded implementation. The temporary
stdlib-only helper produces a distinct [sealed diagnostic protocol](observable-evaluation-protocol.json),
a [score projection](observable-evaluation-projection.json), and a [run summary](observable-evaluation-summary.json).
The protocol identity is
`sha256:725d54628ef32a36a1c315143287475700c7f2b6db3e14a5d12b9e58731cf537`.
It binds the original verified report, selection roster, family sources, checkpoint
weights, classification policy and analysis implementation. Runtime was5.63 seconds
locally with a600-second allowance. No native/HIP execution or training occurred.

The classifier receives no model outcomes. For a fixed registered family it
uses a conservative polynomial degree bound on each continuation-relevant path.
If a path's degree is at most d and at least d+1 distinct visible indices execute
that path, matching the visible values determines that path's polynomial
continuation. A parameter-independent output needs no observations to distinguish
parameter assignments. Unsupported constructs and insufficient evidence remain
unresolved. Constructed same-prefix/different-continuation witnesses independently
establish ambiguity; failure to find a witness never establishes certification.

This certifies **within-family mathematical continuation**, not identification of
the correct family from the prefix, uniqueness across all programs, or runtime
qualification. Family metadata remains analyst-only and is not supplied to the
model. Actual hidden coefficients and assessment rows were not accessed.

| Classification | Tasks | Static successes | Refreshed successes | Static evaluation unresolved | Refreshed evaluation unresolved |
| --- | ---: | ---: | ---: | ---: | ---: |
| Certified within-family | 153 | 109 | 110 | 3 | 2 |
| Demonstrated ambiguity | 55 | 0 | 0 | 2 | 6 |
| Classification unresolved | 48 | 19 | 18 | 11 | 13 |
| Original total | 256 | 128 | 128 | 16 | 21 |

The153 certified tasks include44 stage-A,42 stage-B and67 stage-D tasks. The55
ambiguous tasks are stage D. The48 unresolved tasks are42 loops and six rows from
one family with a cancelled hidden parameter. No task disappears and no original
outcome changes. The certified subset's completion rates are150/153 and151/153;
this does not retrospectively satisfy the original whole-benchmark progression
rule or justify a new significance claim.

## Representative certificate review

Full source, paths, indices, degrees and witnesses are in the protocol. Family
prefixes below identify those records, not newly generated target rows.

- `0aa27908fe64`: `n*p0`, degree1, has20 visible observations. Any two assignments
  agreeing there produce the same continuation within this family.
- `04986e9882b8`: a polynomial expression with degree bound2 and20 observations.
  Individual parameters need not be uniquely identifiable for its output
  polynomial to be determined.
- `174a9a2071d5`: the branch after index3 is quadratic and has16 visible points.
  The other path is parameter-independent and has no later indices. Certified.
- `16b2050e13f4`: the continuation uses the branch after index12. Its degree bound1
  has seven observed points, sufficient for the certificate.
- `073880ed82c4`: a branch beginning at49 contains a coefficient absent from the
  visible path. Changing that coefficient from2 to3 preserves the prefix and
  changes index49. Ambiguous, with an independently executed witness.
- `07876d525cb4`: a special case at59 has its own coefficient. The witness agrees
  through19 and first differs at59. Ambiguous.
- `023c94524b41` and `54cda52c5b9c6`: additive and multiplicative loops. The bounded
  certificate deliberately does not prove recurrence identifiability; unresolved
  is not a claim that these loops are unobservable or unsolvable.
- `4d74cb7070ca`: the hidden term is `(n-n)*p0`, which cancels. The conservative
  degree/dependency analysis does not simplify this identity, and no differing
  continuation witness exists in the bounded check. Its six tasks remain
  unresolved rather than incorrectly called ambiguous or silently certified.

Independent tests cover linear/quadratic cases, repeated parameters, branch
boundaries18/19/20, late equality guards, insufficient point counts,
parameter-independent hidden paths, degree32, unsupported loops/guards, invalid
sources, cancellation, expired witness analysis, original scoring/source joins,
paired-family disagreement and outcome-swapped projections. Classifier tests
were observed failing before implementation.

## Bank provenance repair and reliability

The real retained-evaluation path now copies every bank admission with the exact
proposal passed to the fitter in `_original_source`, after checking its hash.
A mismatched original hash raises outside the unsupported-source handler. The
real sealer therefore preserves the literal original independently of the
parameterized transformation and fitted final program. Fitter/cache-owned records
remain unchanged. No fitting budget, proposal order, completion rule or selected
program ranking changed.

The regression exercised the actual bank callback and sealer, with a fake native
fitter at the service boundary. Complete/incomplete results and wrong hashes
failed on the original code; all five cases now pass, including empty/unsupported
results, multiple admissions and different recipient prompts. Legacy evidence
is not rewritten: the new projection explicitly records98 static and109 refreshed
historical bank metadata recoveries by original proposal hash.

The [source transition](bank-provenance-source-transition.json) records the two
changed entries in the911-file qualification inventory: the bank integration and
its existing owning test. The build-generated installed snapshot will reflect
those bytes. All68 review dispositions validate with the updated implementation
binding. Old qualification has not been relabeled; execution of changed production
code requires compatible current-source images/evidence under the existing gates.
The offline classification does not require that hardware step.

## Validation and limits

- 66 focused evaluation, qualification, provenance and classification/audit tests
  passed. After adding the export regression, all14 classification tests passed.
- A broader run passed81 checks; three unrelated foundation handoff tests could
  not run because this host environment lacks `wasmtime`. These are recorded as
  environment-blocked checks, not passes or repaired production failures.
- The real archive was verified and projected successfully. All candidate-file,
  task/control, final selected-source and envelope bindings were checked before
  projection; bank legacy recovery is explicit. Output is a fresh verified zip.
- Missing-input failure creates a verified failure archive. Existing output or zip
  causes rejection before overwrite. Source/candidate disagreement fails closed.
- No HIP commands, new scores, model updates, assessment access or CI wait.

## Next scientific step

Do not repeat the completed full P1 pilot to repair the metadata or remove the
ambiguity from its headline. Preserve it as exploratory evidence. The original
153-row certified subset is useful for diagnosis but has already been inspected.

For a prospective comparison, freeze this policy, a new selection roster, frozen
checkpoint choices and completion/decision criteria **before** evaluating new
rows. Apply the same distribution, family support and classification rules to
both arms, publish the ambiguous/unresolved strata too, and give the changed
benchmark a new identity. Reuse retained checkpoints and compatible prepared
inputs where the existing contracts allow; training is not needed merely to
compare them on a new evaluation cohort. Do not open assessment for development.

Explicit-parameter P3 remains a plausible separate learning intervention. The
existing pilot did not test it. It should use the same fitter and a prospectively
frozen evaluation, retain required literal control constants, and report examples
and updates as well as tokens. Neither placeholders nor extra training can recover
unseen independent coefficients from an unchanged prefix. A larger search-budget
experiment, if needed for loop timeouts, is another separately labeled diagnostic.
No such new experiment was started here.

Optional local reproduction, from the repo root with the original attachment:

```bash
python3 -S specs/011-controlled-generalization/validation/observable_evaluation.py \
  --bundle 'report(5).zip' \
  --output "run-state/generalization/observable-diagnostic-$(date -u +%Y%m%dT%H%M%SZ)"
```

This only reads public evidence and writes a new diagnostic archive; it does not
launch containers or require a new HIP run.
