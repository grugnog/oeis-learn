# Round028 workstation release acceptance

**T044, T046 and T047 passed on candidate `3ecac2fdca8613dc1601864d51f610254308a58a`.** The reader at that commit verified the returned `round028-release.zip` without the donor workspace. Its embedded pilot is byte-identical to the previously verified one-seed result. The pilot retained D and kept the final assessment sealed. [Pilot outcome](round028-pilot.md#verified-one-seed-result)

## Regression and recovery results

| Check | Workstation result |
| --- | --- |
| Source and installed wheel | All 253 registered tests passed in each installation across the same 27 test paths. Source took 197.865 s and the wheel took 198.440 s. The candidate native evaluator verified 200 terms. |
| Paired HIP training | Ordinary and first-divergence training each completed two updates of 2,060 and 1,836 tokens. Each arm restored its full 13-file, 1,257,820,683-byte checkpoint. |
| GPU measurements | The paired qualification held the GPU lease for 189.324 s. Across 19 readable samples, 4.467 s were busy in 18.000 s of coverage. Peak device memory was 4,852,388,352 bytes. |
| Interrupted learner | Killing the learner during its second backward pass discarded one update. After restore it completed two useful updates totaling 3,896 tokens and reproduced the uninterrupted final model. The interruption and recovery held the lease for 12.698 s. |
| Recovery and storage | All 60 recovery tests passed, covering interrupted native fits, committed batches, staged freeze publication and ownership across checkouts. The disposable allocation reached `ENOSPC` (errno 28) after exercising 16,683,008 bytes and recovered. |

The collector requires identical release receipts on the first invocation and two resumes. It then verifies the private pilot replay before copying the ZIP and comparing its archive-only receipt. Per-invocation logs stay on the workstation. [Pinned collector](collect_audit.py)

## Production-scale measurements

Both producer encodings expanded to 461,574,003 bytes across 352 files. Cold and warm journal reads each credited the same 145,000 committed operations.

| Fixture | Stored bytes | Cold | Warm |
| --- | ---: | ---: | ---: |
| Compressible producers | 489,082 | 1.155 s | 1.155 s |
| Varied producers | 461,750,962 | 7.379 s | 0.528 s |
| Journal with 290,000 events | 120,988,345 | 6.905 s | 0.029 s |

The campaign held 56,569,701 bytes and calibration held 49,001,445 bytes. Peak allocation was 1,614,614,528 bytes and 717 inodes. Admission required 2,360,290,764 bytes above the 5 GiB reserve and 736 inodes. It found 221,195,083,776 free bytes and 119,396,419 free inodes.

The revised planning forecast was 2 h 57 min 31 s for development and 3 h 34 min 38 s for a conditional replication, leaving 1 h 27 min 51 s in an eight-hour shift. The pilot's registered decision ended research after one seed.

## Verify the returned archive

Run this from the candidate checkout with the archive path supplied as shown.

```bash
python3 -S -B scripts/foundation_run.py --engine podman discovery round028 \
  --action verify --archive /path/to/round028-release.zip
```

| Identifier | Verified value |
| --- | --- |
| SHA-256 of the 147,747-byte ZIP | `45cab7c100e914a70ef5253b43c85ddf1ec80212f66326b090def75555fd5967` |
| Release report | `sha256:66c54ddf7deb88f8ecd9615b7ad13e86f5f9b03e78daef572e8dbde542545fdc` |
| Candidate source snapshot | `sha256:00df6db1262ca685825609f0f64635f13fb318de2a199f20c6206d3d04538190` |
