# Run the optimizer timing qualification

Run from the qualified detached checkout after completing the supply step. Set `ROUND029_REV` to the public implementation commit and `ROUND029_MINUTES` to the maximum cumulative minutes approved for this qualification.

```bash
(
  set -euo pipefail
  umask 077
  : "${ROUND029_REV:?full implementation commit}"
  : "${ROUND029_MINUTES:?maximum qualification minutes}"
  output=$(mktemp -d "${HOME}/round029-timings.XXXXXX")
  git fetch --no-tags origin "$ROUND029_REV"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" "$ROUND029_MINUTES" timings --output "$output/timings.json"
)
```

The command searches completed learner qualifications for a compatible local ROCm image. It checks the original image receipt, settled grant and locked build inputs. A CPU-only container in that image builds and checks the candidate wheel offline. The timed learner receives the wheel and its exposed training projection. When the local runtime is unavailable or incompatible, preparation uses the root Dockerfile and retains the build logs.

One GPU lease covers matching calibration and five ordinary-stream optimizer configurations. The command then privately verifies the measurements and updates the forecast. On success, the file includes acquisition mode, available bytes and inodes around preparation, and the retained wheel transport allocation when applicable.

The minutes are a cumulative ceiling for funded work, including cleanup. Repeating a completed revision and allowance reopens its measurements. A failed attempt with proven termination keeps the elapsed debit. A lost or unconfirmed build keeps its reservation. Use a new output directory on each invocation. If preparation fails, return its JSON file. It includes bounded log tails, termination evidence and available storage observations.


## Broader-composition timings

After the ordinary timings and family census have settled, use the same detached checkout and the next published implementation revision. One cumulative grant covers the custodian audit, native preparation, and eligible optimizer samples for recipes 01, 02 and 03.

```bash
(
  set -euo pipefail
  umask 077
  : "${ROUND029_REV:?full implementation commit}"
  : "${ROUND029_MINUTES:?maximum qualification minutes}"
  output=$(mktemp -d "${HOME}/round029-broader.XXXXXX")
  git fetch --no-tags origin "$ROUND029_REV"
  git show "${ROUND029_REV}:specs/016-measure-learning-velocity/validation/qualify_continuation.py" |
    python3 -I -S -B - "$ROUND029_REV" "$ROUND029_MINUTES" broader --output "$output/broader.json"
)
```

The custodian reopens the original family reservations, difficulty map, positive stream and paid ordinary calibration. It checks the original slots first, then audits the next block of up to 512 mechanical draws against protected family links. A collision in the original slots records the need for a fresh matched start and ends the qualification before acquiring a device.

The CPU role replays the training witnesses and measures diversity at matched token lengths and difficulty. Repeated draws retain every occupied slot. If breadth increases, the native worker applies the trial stream seed to both streams, checks adjustable targets and retains each donor map and its proof identities. One HIP lease measures the ordinary matching weight on that selection, samples recipe 01, and samples 02 and 03 wherever the measured support covers their learner inputs. The learner reopens those private manifests. Fits that time out contribute only ordinary positive rows.

The earlier ordinary receipt retains its measured updates, tokens and preparation cost. When its donor selection differs, its recipe 02 sample and calibration remain visible as historical measurements. The forecast carries forward eligible ordinary optimizer prices and admits each matching price after verifying its registered scope.

The command verifies the receipt and refreshes the forecast. Return the saved `broader.json` file. It includes the disposition and charges if the exposure or diversity check ends the run before HIP. A physical capacity block reports the affected mount, byte and inode deficits and an affordable draw count where one exists. A retry against the same grant reuses settled stages and its remaining allowance.
