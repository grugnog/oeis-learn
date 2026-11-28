# Round029 quickstart and validation

Run commands from an OEIS Learn checkout. The historical commands below are available at the handoff revision. The round029 commands describe the release interface the implementation must deliver. A release handoff must supply its actual full candidate commit and pre-score evidence.

## Check today's baseline

The host help uses the standard library and does not need a running engine.

```bash
python3 -S -B scripts/foundation_run.py --engine podman discovery --help
python3 -S -B scripts/foundation_run.py --engine podman discovery round028 --help
```

At `4a13ec34560e7cbeffcd2dad2b24a25b07ee0dd5` the first command lists the 013 operations and `round028`. In a development installation of the locked environment, these existing checks exercise the integration seams.

```bash
uv sync --locked
uv run pytest -q \
  tests/unit/test_progressive_metrics.py \
  tests/unit/test_progressive_trial_eligibility.py \
  tests/unit/test_progressive_freeze_progress.py \
  tests/unit/test_progressive_budgets.py \
  tests/integration/test_round028_pilot_resume.py \
  tests/contract/test_progressive_roles.py
```

To verify historical copies, set `OEIS_ARCHIVES` to the absolute directory containing the three returned ZIPs. This block uses isolated reader checkouts and leaves the qualified donor untouched.

```bash
(
  set -eu
  : "${OEIS_ARCHIVES:?absolute directory containing the round028 archives}"
  readers=$(mktemp -d)
  for rev in 04e0fd9c30ce64d835c91fb06e5014799bceccee 3ecac2fdca8613dc1601864d51f610254308a58a; do
    git init -q "$readers/$rev"
    git -C "$readers/$rev" fetch -q --depth=1 https://github.com/grugnog/oeis-learn.git "$rev"
    git -C "$readers/$rev" checkout -q --detach "$rev"
  done
  python3 -S -B "$readers/04e0fd9c30ce64d835c91fb06e5014799bceccee/scripts/foundation_run.py" \
    --engine podman discovery round028 --action verify \
    --archive "$OEIS_ARCHIVES/round028-pilot.zip" \
    --qualification-archive "$OEIS_ARCHIVES/round028-pilot-qualification.zip"
  python3 -S -B "$readers/3ecac2fdca8613dc1601864d51f610254308a58a/scripts/foundation_run.py" \
    --engine podman discovery round028 --action verify \
    --archive "$OEIS_ARCHIVES/round028-release.zip"
)
```

The historical pilot should reproduce its one-seed 4M result retaining D. The release reader checks its independently verified package. These receipts establish historical identity. Current probe costs, untouched family supply and continuation eligibility require the round029 preflight. [Historical operator guide](../015-learned-program-search/quickstart.md)

## Qualify the learner on the detached donor

The implementation supports read-only inventory, status and verification, plus separate disposable learner and donor-continuation qualifications. Run the following from the physical root of the detached round027 donor at `77667e67176ae32fa16457d52565bae1dbe41afc`. For a fresh qualification, set `ROUND029_REV` to the full implementation SHA you intend to test. The prompt sets a cumulative wall-clock budget for this qualification. The funded clock starts after preflight and includes image acquisition, the learner and cleanup. The supervisor reserves the last two minutes for cleanup. A verified early finish charges only elapsed time. If termination cannot be verified, the full reservation stays charged. A retry with the same revision and minutes uses the same grant. Changing the minutes creates a separate authorization.

The current four-recipe launch estimate and the command for the smaller funded ordinary
initialization are in the [workstation record](validation/checkpoints.md#qualify-the-ordinary-zero-token-boundary).


The donor completed both batch sizes in **3 min 8 s** under a **30-minute** grant. To review that run, reopen its saved receipt with the command in the [workstation record](validation/round029-release.md#reopen-the-saved-qualification). For a fresh qualification on the same donor with cached image layers, **10–15 minutes** is a planning range based on this one run and the two-minute cleanup reserve. Add the usual image-build time for a cold build.

```bash
(
  set -euo pipefail
  : "${ROUND029_REV:?full implementation commit from the handoff}"
  git fetch --no-tags origin "$ROUND029_REV"
  read -r -p 'Maximum workstation minutes approved for learner qualification? ' minutes
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify-learner.sh" |
    bash -s -- "$ROUND029_REV" "$minutes"
)
```

The helper verifies the candidate source and prints preflight. It opens an operator grant for the two registered batch sizes and starts the installed worker under the donor's GPU lock. It uses the original campaign's reference architecture with seed 29, zero dropout and exposed generated prefixes. Batch 16 exercises recipe 04 and batch 32 exercises recipe 02, including matching and its training-only calibration. For each, it computes an uninterrupted reference, kills a process during backward, resumes from the complete cut and reopens the final cut. The worker compares model weights, AdamW moments, the scheduler and the next rows. Two additional native checks report the disposition of a refittable literal and a constant program on varying terms.

The command prints the qualification and final status. The qualification receipt records the device peak alongside the historical 4,852,388,352-byte peak, checkpoint bytes/inodes, discarded token charges, GPU lease time and sampled activity coverage. `receipt_path` identifies the private receipt. On failure, `first_error` retains the first reason and attempt path. A capacity block reports the byte/inode deficit before issuing tokens. Repeating a completed qualification verifies its saved receipt. The grant covers this rehearsal only. The donor-continuation check below has its own grant. Comparative study qualification and scientific reporting remain open.

## Check the donated ordinary 4M continuation

From the same detached workstation, use the full candidate SHA from the handoff. The helper authenticates the original pilot archive, pre-score package, frozen stream and complete checkpoint. It runs one batch twice on HIP under the retained round028 ordinary policy and the registered round029 recipe 00. Both updates restore the original AdamW moments, schedule, cursor and RNGs. Their weights, moments, RNGs, cumulative exposure and next rows must agree. The donor checkpoint is mounted read-only.

```bash
(
  set -euo pipefail
  : "${ROUND029_REV:?full candidate commit from the handoff}"
  git fetch --no-tags origin "$ROUND029_REV"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" 30
)
```

The last argument sets a cumulative qualification budget in minutes. On the qualified donor, the continuation took 182.685033 s of a 30-minute grant. For a future run with cached image layers, 10–15 minutes allows for that observed workload and the two-minute cleanup window. Allow additional time for a cold image build. Funded preparation and HIP dispatch share the allowance, including image acquisition and cleanup. The initial metadata read and subsequent receipt replays use no training tokens or GPU lease. [Measured continuation and receipt](validation/round029-release.md#ordinary-4m-continuation-on-hip)

The helper retains a pre-update commitment, reopens the same funded operation twice, verifies the private evidence, and checks copied public ZIPs from a separate directory. Its final JSON gives the candidate, parent, receipt, report, measured times and charged tokens. This diagnostic publishes a null scientific verdict.

## Rehearse a separately funded child

The 4M donor comparison is recorded in the [workstation results](validation/round029-release.md#ordinary-4m-continuation-on-hip). The following qualification uses disposable training prefixes and a fresh small parent at each batch size. It checks the transition to an 8M schedule horizon with a separately accounted child grant. Each child performs one update, then reopens its complete cut twice. Its cumulative token clock includes the parent and its interrupted update. Weights, AdamW state, RNG state and next rows must match an uninterrupted third update.

Run from the physical root of the detached donor at `77667e67176ae32fa16457d52565bae1dbe41afc`. Use the candidate SHA from the handoff.

```bash
(
  set -euo pipefail
  : "${ROUND029_REV:?full candidate commit from the handoff}"
  git fetch --no-tags origin "$ROUND029_REV"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" 20 children
)
```

The grant funds 4,182 training tokens across the references, interrupted parents and children. The children charge 209 and 400 incremental tokens at logical batches 16 and 32. The runtime measures both parent and child checkpoint storage while they coexist. The qualified donor used 6 minutes 18.529 seconds of its 20-minute grant for this child rehearsal. A 15–20 minute budget allows for variation with cached layers. [Measured result](validation/round029-release.md#separately-funded-child-qualification) The helper reopens the committed qualification twice, verifies the private receipt and prints the measured charge, lease and child checkpoint identities. It issues zero research dispatches.

### Recover the child receipt as a file

Use this after losing the console output or seeing a JSON decoding error during qualification. Run it from the same detached donor. The reader checks the saved worker evidence and settled journal twice under a shared lock. It finds the original grant even if another qualification has since updated session status. The commit and minutes identify the original run. This review uses zero GPU time and creates no grant.

```bash
(
  set -euo pipefail
  reader=1d31e0a80d8c159b5ccf4d22a6c8a8da6a4dea7d
  candidate=a89af7fe820138a925090e340bbfeef45094adb7
  git fetch --no-tags origin "$reader" "$candidate"
  out=$(mktemp -d "$HOME/round029-child.XXXXXX")
  git show "${reader}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$candidate" 20 review-children --output "$out/qualification.json"
)
```

Attach the printed file. It contains the compact verified summary on success or an `UNVERIFIED` diagnostic if the receipt is missing, busy or inconsistent. The reader preserves existing files and returns a nonzero exit on failure. The updated helper also accepts `--output FILE` for funded runs.

## Inspect prices on the donor

Run the following from the detached donor with `ROUND029_REV` set to the full preflight implementation commit. Attach the printed file. This reads the historical pilot and its independently retained qualification, the exposed one- and eight-attempt timing blocks, and any registered round029 family census. It acquires no device and requires no grant.

```bash
(
  set -eu
  : "${ROUND029_REV:?full preflight commit from the handoff}"
  umask 077
  out=$(mktemp -d "$HOME/round029-preflight.XXXXXX")
  git fetch --no-tags origin "$ROUND029_REV"
  git show "$ROUND029_REV:specs/016-measure-learning-velocity/validation/launch-round029.sh" > "$out/launch.sh"
  bash "$out/launch.sh" "$ROUND029_REV" preflight > "$out/preflight.json"
  printf '%s\n' "$out/preflight.json"
)
```

The common-probe estimate covers ordinary and one challenger at one look. The follow-up reserve covers a matched full-profile comparison on one OEIS family with existing weights. Each price includes cold role startup and selected-choice scoring. Image acquisition is shown separately for an uncached host. Expected and tail estimates come from the historical exposed sample. The wave lists the sums of the priced items and each outstanding measurement. A null total means the listed work still needs pricing. Research funding requires a separate decision.

After the fresh-start qualification, one read-only donor command bundles the verified timing receipt, preflight, grant balance, child recovery and reducer checks into a ZIP. [Collect the completed qualification and readiness checks](validation/update-timings.md#collect-the-completed-qualification-and-readiness-checks)

## Measure donor family and matching-pair supply

Run this block from the physical root of the detached round027 donor. One CPU preparation grant covers the two measurements. The helper then reopens the private receipts and runs read-only preflight to reprice the common probe and the follow-up. It saves the results and any first failure in one file.

```bash
(
  set -euo pipefail
  umask 077
  candidate=fbb3f9db8afd83c37b2cbe367f1ef5f401989ad1
  out=$(mktemp -d "$HOME/round029-supply.XXXXXX")
  git fetch --no-tags origin "$candidate"
  git show "$candidate:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$candidate" 30 supply --output "$out/supply.json"
)
```

The 30-minute argument is a cumulative workstation spending cap for the two workers. A planning range of 30–45 minutes allows for reopening the original exposure index, the family scan and cleanup. The earlier round028 panel collection took 457.07 seconds. Round029 also checks later exposures and reconstructs links from the original source records. Completed operations release their unused time. Running the same block again uses the original grant and writes a new attachment. Changing the candidate or cap creates a different grant. [Historical CPU collection](../../015-learned-program-search/validation/round028-panel-results.md)

The first step checks the ordered ordinary training stream. It reports proven fixed-target matching pairs and counts rows that still require native coefficient fitting. The second step proposes 32 cases per transfer stratum for development, replication and assessment, 32 for each retention role and one case per synthetic stratum for each of practice, teacher and calibration. These are provisional, outcome-blind reservation requests. The file records the actual cases, independent families and deficits by role. The command acquires no device, trains no learner and opens no comparative scores. Attach the resulting `supply.json`.

## Complete preparation for the supported trial wave

Run this from the detached donor root. Set `ROUND029_REV` to the full commit in the
handoff. The second argument to `--prepare-trials` authorizes a separate maximum
of 15 workstation minutes for the host-only trial freeze and publication. The
initialization step uses its existing 30-minute grant and original worker. On an
already completed initialization it reopens the paid result. Preparation issues
zero comparative training tokens. The research tranche has its own authorization.

```bash
(
  set -euo pipefail
  rev=${ROUND029_REV:?full trial preparation commit}
  git fetch --no-tags origin "$rev"
  git show "$rev:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - --prepare-trials "$rev" 15
)
```

The command finishes the zero-token HIP checkpoint and two restores, reopens the
paid training, checkpoint, control, bank, conditioning and accounting evidence,
and checks full-wave capacity after image acquisition. It freezes the actual
learner inputs and linked targets at 0, 1M and 4M for 00, 04, 05 and 06. The common
and full 4M evaluations have separate rosters. D and the conventional controls
keep their full-profile endpoints. Recipes 01–03 keep the exclusions measured on
the fresh stream. The grant records the preparation spend. The prospective
exploration price includes a separately named follow-up reserve.

The donor stores the verified inputs privately. It installs the supported-wave
qualification and its independent copy before accepting the schedule. The public
`trial-preparation.zip` contains the original initialization ZIP and, on completion,
the pre-score qualification, its verification report and the private replay
reference. Return the printed file on either `COMPLETE` or `PARTIAL`. Rerunning
the same command uses the same grants and resumes at the first unfinished step.
Verify a copied result offline with

```bash
python3 -I -S -B specs/016-measure-learning-velocity/validation/qualify_continuation.py \
  --verify-trial-preparation-zip /path/to/trial-preparation.zip
```

## Start or resume after the round029 release

Use the original private workstation, a local Podman engine with resource controls and a qualified HIP device for learner work. Start in the **physical root of the detached round027 donor**. Set `ROUND029_REV` from the implementation release, and `ROUND029_GRANT` to the absolute path of the operator-authorized grant. If all you need is inventory, run through the preflight line. The same `run` line starts or resumes an authorized campaign.

```bash
(
  set -eu
  : "${ROUND029_REV:?full round029 implementation commit from the release}"
  test "$(pwd -P)" = "$(git rev-parse --show-toplevel)"
  test "$(git rev-parse HEAD)" = 77667e67176ae32fa16457d52565bae1dbe41afc
  if git symbolic-ref -q HEAD >/dev/null; then exit 2; fi
  git fetch --no-tags origin "$ROUND029_REV"
  test "$(git rev-parse --verify "${ROUND029_REV}^{commit}")" = "$ROUND029_REV"
  launcher_dir=$(mktemp -d)
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/launch-round029.sh" \
    > "$launcher_dir/launch-round029.sh"
  bash "$launcher_dir/launch-round029.sh" "$ROUND029_REV" preflight
  : "${ROUND029_GRANT:?absolute path of the operator grant for run}"
  bash "$launcher_dir/launch-round029.sh" "$ROUND029_REV" run "$ROUND029_GRANT"
)
```

A healthy launch prints the last authenticated boundary, measured and outstanding work, the next eligible action and eventually the verified report path. A missing donor or insufficient mount reports the affected stage and numeric shortfall where measurable. Independent CPU work can continue with a grant that names it. The launcher also supports `status`, and `qualify` or `cpu` with an appropriate grant as specified in the [operator contract](contracts/operations.md). Prospective qualification and pre-score freeze precede comparative outcomes. An independently funded confirmation uses the same command with its own named child grant.

## Acceptance on the candidate

| Exercise | Observable result |
| --- | --- |
| Present a valid 00 checkpoint, a changed probe and a weights-only continuation | Credit the known cut, schedule only the changed inference/scoring layer and identify the missing restore state. No baseline retraining from a score mismatch. |
| Inject a shifted duplicate, a regroundable matching “negative” and an unequal batch cursor | Reject cross-panel leakage, omit false auxiliary label while keeping its positive row, and reject a mismatched checkpoint before updates. |
| Run a disposable 0/1M/4M observation fixture with a shared unresolved zero, hidden-term canaries and a verified solve with unfinished search | Shared unresolved contrast cancels. All reported rates use one explicit probe, and that solve retains its separate incomplete-search flag. Private canaries remain inside their role. |
| Exercise known-structure, fixed, guided, shuffled and D controls on registered cases | Only live sealed choices earn solves. Inspect the first correct admission after sealing and reproduce the fixed-order or guidance decision. |
| Run a real HIP update at batches 16 and 32, kill one backward pass, and interrupt freeze/report publication | Repeat the identical launcher twice. Check uninterrupted final learner identity and next rows, one final result, retained discarded charges and one publication transition. |
| Feed largest valid current inputs and both report encodings to source and installed readers, then cause a genuine byte or inode shortage in disposable storage | Valid inputs reach independent verification. The shortage trace retains its first error, deficit, previous work and restart point. |
| Open a copied report and independently retained qualification on a different checkout or machine | Reproduce all joint counts, contrasts/bounds, cost and decision. Altering one count, multiplicity or parent causes rejection. |

After release, run the candidate’s copied-report verifier with the donor unmounted. `OEIS_ARCHIVES` here names the directory containing the copies for this registration. For a confirmation child supply its pre-score evidence chain as described in the receipt.

```bash
python3 -S -B scripts/foundation_run.py --engine podman discovery round029 \
  --action verify --archive "$OEIS_ARCHIVES/round029-report.zip" \
  --qualification-archive "$OEIS_ARCHIVES/round029-qualification.zip"
```

A verified inconclusive result or decision to retain D completes the scientific handoff. Final assessment remains sealed.
