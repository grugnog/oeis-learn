# Fresh training from the VM

This workflow starts a new random-initialized FP32 SFT diagnostic using the
foundation codec and exact integer-byte model. It does not promote historical
checkpoints, change the frozen three-update foundation smoke, or complete the
spec-007 release gates. The historical vocabulary failure remains outstanding.

## Run directly from the lab host

For sustained preparation/training, use `scripts/foundation_local.py` from the
registered OEIS checkout on the lab host. It reuses the GPU Dockerfile target
and the existing rootless Podman image store. Fresh CPU containers bind the host
checkout directly at `/workspace`, bypassing SmolVM's shared-filesystem layer.
GPU operations call the installed host broker locally with bounded JSON; there
is no SSH or VM hop. The broker still owns inference shutdown/restoration,
GPU locking, private roles, resource limits and pinned run images.

The existing host integration below must be deployed, including `foundation =
true`. The host entrypoint needs only Python's standard library and your host
sudo access; model/native dependencies remain in the image. Run as your usual
host user, not with `sudo python` against checkout code:

```bash
# On the lab host, from the registered OEIS checkout:
sudo -v
python3 scripts/foundation_local.py image ensure
sudo -v  # renew credentials after a potentially long first image build
python3 scripts/foundation_local.py data fetch
python3 scripts/foundation_local.py prepare --output tmp-fresh-inputs-host
python3 scripts/foundation_local.py check --inputs tmp-fresh-inputs-host
python3 scripts/foundation_local.py smoke \
  --inputs tmp-fresh-inputs-host --run runs/fresh-host-smoke
python3 scripts/foundation_local.py train \
  --inputs tmp-fresh-inputs-host --run runs/fresh-host-001
```

The default bulk cache remains `tmp-oeis-bulk` in this same checkout, so existing
downloads are reused. A new preparation directory is recommended when switching
images/runtimes: an incomplete admission plan freezes its runtime identity.
New containers are created for each CPU command and reclaimed on exit; the
image and bind-mounted artifacts persist. Only bulk acquisition gets outbound
networking; preparation/check containers receive no GPU devices or runtime
socket. GPU training/evaluation remain offline in separate private containers,
rather than running inside the workspace-mounted preparation container.

```bash
python3 scripts/foundation_local.py status --run runs/fresh-host-001
python3 scripts/foundation_local.py cancel --run runs/fresh-host-001
python3 scripts/foundation_local.py resume --run runs/fresh-host-001
python3 scripts/foundation_local.py evaluate \
  --run runs/fresh-host-001 --split development
python3 scripts/foundation_local.py pack \
  --run runs/fresh-host-001 --output tmp-analysis/fresh-host-001.zip
```

Packaging runs directly as your host user using the standard-library bundle
writer, so it needs no container, GPU image, or sudo credentials. CPU commands
enter the project account once and perform cleanup there; an expired sudo ticket
cannot prevent cleanup at the end of a long preparation run. Run `sudo -v` again
before starting a new privileged command if the ticket has expired.

Status/cancel bypass image rebuilding and GPU acquisition, using the existing
host broker's control path. The final pilot remains capped at one hour and
automatically publishes an analysis ZIP. `foundation_run.py` and `lab-gpu`
remain available inside the VM for testing. Never invoke their SSH-based
training transport from inside a `lab-gpu exec` container: that would nest
requests behind the same project/GPU lease.

### Recover from a failed smoke

An interrupted run retains its issued budget. Repeating `smoke` against that
same output requires `resume`; it cannot issue another allowance. Resume also
retains the original image, so rebuilding the image does not repair an existing
run that failed because of a code defect. Keep that run and its analysis ZIP as
failure evidence, then use a new output directory with the repaired image:

```bash
sudo -v
python3 scripts/foundation_local.py image ensure
sudo -v
python3 scripts/foundation_local.py smoke \
  --inputs tmp-fresh-inputs-host --run runs/fresh-host-smoke-002
```

For the encoder device-alias and deterministic token-loss repairs, completed
prepared inputs and bulk downloads can be reused: these changes affect learner
math/device handling, not source admission. The token-loss repair keeps strict
determinism enabled and retains the existing per-program/padding/EOS reductions.
For a recoverable interruption using the same image and remaining budget, use
`sudo -v` followed by `python3 scripts/foundation_local.py resume --run RUN`.
The published `RUN/roles/*/stderr.log` contains the underlying role error, and
`tmp-analysis/BASENAME.zip` includes it for analysis. Packaging a failed run is
successful evidence capture; the original command still exits with the failure.

## Install the host integration once

Apply the companion [lab host integration PR #29](https://github.com/grugnog/lab/pull/29)
and rebuild the host
using that repository's normal NixOS deployment procedure. The OEIS project
policy enables `foundation = true` and `foundationTimeout = 4200`: project runs
still have their own 1,800-second smoke or 3,600-second pilot lifetime cap. The
extra host allowance covers staging/transport/reclamation; it cannot extend a
run's ledger. GPU inference locking/restoration remains host-owned.

Refresh the VM client without recreating the existing VM. On the host, from the
OEIS checkout:

```bash
cp /etc/nixos/packages/lab-gpu/lab-gpu* system/
smolvm machine exec -i --name oeis -- sudo install -m 555 \
  /workspace/system/lab-gpu /workspace/system/lab-gpu.py /usr/local/bin/
```

The Dockerfile installer now includes Git LFS. Rebuild the VM image with your
usual workflow when it lacks `git lfs`; `start.sh` deletes/recreates the VM, so
do not run it merely to refresh the client. The updated Smolfile permits
`oeis.org`, which hosts the official Git LFS endpoint. Apply the VM network
configuration using the owning SmolVM workflow when updating an existing VM.

The GPU image must include these changes. From `/workspace` in the VM:

```bash
lab-gpu image ensure
```

The VM does not need Docker or Podman. The broker extracts the coordinator from
the pinned image and executes it as the project account. Roles use rootless
Podman with private enumerated mounts, read-only roots/inputs, offline networking
and no runtime socket. Learner/generator receive GPU devices; native candidate
workers and the scorer do not. The host keeps private role views under its
project state directory, outside the whole-workspace diagnostics mount.

## Bulk source and admitted training data

```bash
cd /workspace
python scripts/foundation_run.py data fetch
python scripts/foundation_run.py prepare --output tmp-fresh-inputs
python scripts/foundation_run.py check --inputs tmp-fresh-inputs
```

`configs/foundation/fresh_pilot.yaml` controls preparation. The initial defaults
are a deterministic selection of at most 5,000 entries without finite/table keywords, with
LFS b-files, at most 20 GiB of b-file objects, eight development/eight final groups,
256 distinct training prompts and eight structural development examples. LFS
objects plus checkout can occupy twice their declared size. Both acquisition
and checkpointing preserve the 100 GiB free-disk floor. Source selection is a
bounded diagnostic subset; coverage/shortfalls remain in the import report.

Acquisition clones `https://github.com/oeis/oeisdata.git` shallowly, freezes its
commit and export timestamp, and fetches b-file objects in Git LFS batches at
that revision. There are no individual sequence HTTP requests and no historical
`git lfs fetch --all`. Repeating acquisition verifies/reuses completed objects.
Set `data.revision` to an exact 40-character commit to choose a particular
snapshot before the first fetch. Changing the snapshot/config needs a new
cache and input set; existing pins are never overwritten.

Import preserves original indices, signed terms, offset/keyword metadata and
raw entry hashes; unresolved LFS pointers, malformed records, incomplete
horizons and exclusions are reported. The SQLite index and term projections
bound VM memory use. Names/formulas/program text do not enter learner inputs.

OEIS provides evaluation targets and exclusion information. Training labels are
generated generic programs admitted through the existing native executor plus
independent reference evaluator. Cohorts and all-member reserved prefixes are
frozen before admission. Structural development/final programs cannot enter
the training view. Preparation has its own 1,800-second lifetime ledger;
shortfalls preserve decisions and remaining budget rather than receiving a
fresh allowance. Lowering a frozen target is not a resume operation.

The bulk source is available under CC BY-SA 4.0 from the OEIS Foundation. Keep
the source commit, attribution and license with derived data.

## Start, recover and evaluate

If preparation reports `too many open files`, retain the complete stderr output.
The CLI prints the open-file soft/hard limits and traceback for `EMFILE`/`ENFILE`.
SQLite import connections and failed worker-start sockets/process handles are
explicitly closed; increasing `ulimit` is not a substitute for cleanup.

An incomplete preparation freezes its runtime identity in `archive-plan.json`.
After updating execution code, use a new input directory if that frozen plan
rejects the changed runtime; retain the old journal and budget evidence. The
existing bulk download cache is reused:

```bash
python scripts/foundation_run.py prepare --output tmp-fresh-inputs-fd-fix
```

```bash
python scripts/foundation_run.py smoke \
  --inputs tmp-fresh-inputs --run runs/fresh-smoke

python scripts/foundation_run.py train \
  --inputs tmp-fresh-inputs --run runs/fresh-001
```

Smoke checkpoints random weights, scores the initial model on development,
performs one update, reloads complete state, performs updates two/three, scores
development, freezes checkpoint selection and evaluates final groups. Its
outer cap is 30 minutes; this is distinct from the original frozen smoke test.

The one-hour pilot targets two15-minute training segments, with initial and
post-segment development evaluation. Reserve time includes loading, checkpoints,
native scoring and cleanup; budget pressure can shorten training or fail a
stage. The initial model is the existing qualified-model shape: integer bytes,
d=384, four encoder/six decoder layers, six heads, feed-forward1,536,
context1,024, sinusoidal positions, eager attention, FP32, AdamW at0.0003,
batch four. This diagnostic does not assert qualification by using that shape.

```bash
python scripts/foundation_run.py status --run runs/fresh-001
python scripts/foundation_run.py cancel --run runs/fresh-001
python scripts/foundation_run.py resume --run runs/fresh-001
python scripts/foundation_run.py evaluate \
  --run runs/fresh-001 --checkpoint latest --split development
python scripts/foundation_run.py finalize \
  --run runs/fresh-001 --checkpoint best-development
python scripts/foundation_run.py evaluate --run runs/fresh-001 --split final
```

Commands are synchronous. Status/cancel can run in another VM terminal without
waiting for the active project/GPU lease. Ctrl-C also requests interruption.
Logs include periodic progress and bounded role deadlines. The host owns final
reclamation; broker busy is75, host timeout124. A stale lease requires the lab
administrator recovery procedure.

Resume restores model, AdamW, module modes, Python/NumPy/Torch/device RNGs,
permutation, cursor and committed update. Runtime, profile and input identity
drift rejects. Invalid checkpoint blobs are quarantined; generations are not
reused. Failed updates can repeat after restoring a committed checkpoint, but
their wall time remains charged. An interrupted gap is conservatively charged
until recovery; finalized runs cannot resume training. Keep the same run and
host state, including its budget journal.

The selection rule is development solved-group count, then selected top-one
count, then earliest update. It is locked before final truth is mounted. Final
evaluation reuses sealed candidates on rerun. A zero solved count is valid.
These are finite100-term results from20 visible terms, not theorem claims.
Once inspected to guide another experiment, this diagnostic final cohort is
consumed; freeze an independent final cohort for later assessment.

## Upload results for analysis

```bash
python scripts/foundation_run.py pack \
  --run runs/fresh-001 --output tmp-analysis/fresh-001.zip

# Optional separate complete checkpoint package, up to40GiB:
python scripts/foundation_run.py pack --checkpoint \
  --run runs/fresh-001 --output tmp-analysis/fresh-001-checkpoint.zip
```

Training also attempts to create the small ZIP after success or failure. Upload
`tmp-analysis/fresh-001.zip` here first. It includes a summary, frozen settings,
source/admission coverage, losses/gradient/entropy/resource/sensor metrics,
development candidate failures, final aggregate scores, budgets, runtime and
checkpoint metadata, logs, and a verified bundle inventory. Default report
payloads are capped at25MiB; long logs use their last1MiB and large metric streams
retain first/last100 rows. Every omission/sampling operation records original
size/hash. Bulk sources, private inputs, hidden values and weight tensors are
excluded. The optional checkpoint package adds only the selected complete blob
and matching manifests/settings; retain local recovery state as well.

Throughput uses explicitly measured training phases, not a promise of steady
state. Attempt records can include work rolled back during recovery. Unavailable
sensors are null with a reason; GPU and host memory on the APU overlap and must
not be summed. Use this report to set corpus, evaluation and checkpoint budgets
before increasing duration or changing one model/learning setting.

## Validation scope

CPU tests cover parsing and batched fetch/resume, real Python-Wasmtime/reference
admission against synthetic source fixtures, exact real-model update/resume
parity, corrupt-generation recovery, host budget accounting, private role view
construction, sealed finite scoring, interruption recovery and ZIP verification.
Broker tests cover named-operation restrictions, disabled-policy rejection,
rootless extraction and status/cancel dispatch. These fixtures do not establish
real OEIS/Rust acceptance, rootless Podman containment or HIP performance.

The wrapper prefers the copied `system/lab-gpu.py` client, so refreshing that
host-managed copy is sufficient for its commands even before replacing the
installed VM client. Direct `lab-gpu foundation` commands require the new client.

The separate actual-device regression uses the full configured model and a
workspace temporary directory that can satisfy the disk floor:

```bash
lab-gpu exec -- python -m pytest -q tests/integration/test_fresh_gpu.py \
  --basetemp=/workspace/tmp-fresh-gpu-tests
```

Run `check`, `smoke` and the first pilot on the actual host to obtain those
observations. Full T115 host kill/quota acceptance, T117 historical wrappers,
T118 overflow strata, fixed T119/T120 measurements and T121 release handoff stay
under their owning progression gates. The new workflow does not remove those
blockers or mark their tasks complete.
