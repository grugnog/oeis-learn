# Reviewed round019 storage block

The uploaded 2026-11-18 01:47:21 ZIP preserves the original stopping reason:
canonical freeze exited 2 after 39,212.39 seconds with a sealed BLOCKED result,
FREEZE_DEFICIT and missing_evidence=[storage_bytes]. Its producer_error is null.
The T230 collector incorrectly called `.get` on that null, hiding the block behind
an AttributeError. Both conditions are covered by regression tests now.

Free storage was 291,429,883,904 bytes. The unchanged 352 task partitions reserve
375,541,145,822 bytes. Journal, shard and packaging bring the minimum to
378,252,121,337 bytes before input/checkpoint projection. At least 86.82 GB extra
is therefore necessary, and the final requirement is higher. Do not treat the
prepared aggregate disk cap as a substitute for the canonical projection estimate.
The diagnostic checks the minimum first; if it fits, it measures the full closure
using the same pinned storage functions and arithmetic as freeze. These checks
are diagnostic only; canonical freeze still validates all original authorities.

Keep b616b116dad59a05ad1911687583d8ab1626af13 checked out. Copy the new helper
`continue-round019-storage.py` into run-state/discovery-tools under that new name.
**Retain the original continue-round019.py unchanged:** it is hash-bound historical
evidence. No image build, source change, qualification or calibration rerun is
required. The helper checks that exact source and the active preparation owner.

Default invocation checks capacity only. `--inspect-storage` additionally collects
read-only engine system-df, graph-store information, qualified image histories,
and workspace file accounting. Each engine collector is bounded and independent;
a failed collector does not discard the other evidence. It neither prunes images
nor deletes data. Results, errors and original failures use the existing session
ZIP under share/prepare-research.zip, including on failure.

Once space is available, invoke the same helper with `--continue`. It requires
the exact reviewed result, stdout, acknowledged process termination and original
charged allowance. It archives the blocked publication and original journal,
then runs canonical freeze, acceptance, readiness and run/resume. A completed
freeze is reused; a new failed/blocked freeze is retained for review, not silently
retried. Existing parent and child allowance records remain untouched. Both prior
freeze costs are charged before allocating a fresh bounded attempt from unused
parent preparation time; there is no scientific budget increase or time refund.

The successful validations inside the failed freeze were not published as a
reusable authority, so that CPU validation must currently repeat. This attempt
took about 10h53m; 24 hours is a retry ceiling, not a promised runtime. Five-minute
safe stack dumps remain enabled. The final assessment route is still separate.

## Image disk accounting

The GPU build log shows cache hits throughout dependency installation and native
compilation. Final COPY and installation reran. That installation ends with
recursive chmod over the checkout/build-info and entire virtualenv. On overlay
storage, metadata writes can copy lower files into the new layer depending on
metadata-copy support. This is a plausible avoidable contributor, not a measured
per-layer attribution yet. Retained old source/install layers and build cache
can also accumulate even when large base layers remain shared.

Nearby 30-second telemetry samples show roughly 10.34 GB less free disk during
GPU image building, 38.66 GB during base qualification and 65.77 GB during model
qualification. These are temporal associations, not exclusive ownership accounting.
The long freeze retry changed disk use only minimally. The added read-only
measurements distinguish image unique/shared bytes from retained test outputs.
Do not prune the exact CPU/GPU image IDs required by current qualification.
