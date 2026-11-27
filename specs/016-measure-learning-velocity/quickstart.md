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

The current implementation supports read-only inventory and status, plus a disposable learner qualification. Run the following from the physical root of the detached round027 donor at `77667e67176ae32fa16457d52565bae1dbe41afc`. Set `ROUND029_REV` to the full implementation SHA in the PR handoff. The prompt records the maximum workstation minutes you authorize for this qualification, including an image build if needed. The existing stage supervisor reserves 120 seconds for cleanup.

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

The command prints the qualification and final status. The qualification receipt records the device peak alongside the historical 4,852,388,352-byte peak, checkpoint bytes/inodes, discarded token charges, GPU lease time and sampled activity coverage. `receipt_path` identifies the private receipt. On failure, `first_error` retains the first reason and attempt path. A capacity block reports the byte/inode deficit before issuing tokens. Repeating the command with the same revision and minutes reopens the original allowance and verifies any completed receipt. The grant covers this rehearsal only. Historical continuation parity, full study qualification, comparative runs and report verification remain open.

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
