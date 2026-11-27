"""One resumable workstation acceptance, isolated from the completed scientific run."""

import io
import json
import subprocess
import traceback
import zipfile

from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.generalization.progressive.contracts import (
    _atomic_write,
    digest,
    load,
    read_bound_record,
    seal,
)
from oeis_learn.generalization.progressive.program_search_outcome import archive_bytes
from oeis_learn.generalization.progressive.program_search_pilot import verify_pilot
from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
    encode,
    verify_transport,
)
from oeis_learn.generalization.progressive.program_search_release import (
    PHASES,
    guard_inventory,
    release_forecast,
    verify_release,
)
from .identity import confined
from .preparation_recovery import retain, revise
from .progressive_panel_sources import require
from .progressive_research_inputs import checked_file
from .progressive_release_checks import scale_targets, release_cases
from .progressive_round028_pilot import PilotRole, SHARE as PILOT_SHARE
from .progressive_round028_rehearsal import failure_locations, original_scale, rehearse, space
from .research_limits import ACCEPTANCE_SECONDS


class ReleaseRole(PilotRole):
    repeatable_preparation = PilotRole.repeatable_preparation | {
        "release_suite",
        "release_scale",
        "release_interrupt",
    }


def release_admission(area, targets, free_floor, *, registration=None):
    # Both producer corpora and their transports coexist at peak. Include block rounding,
    # the event spool and atomic staging. Completed scale fixtures remove their scratch.
    block_overhead = (4 * targets["tasks"] + 32) * 4096
    required = (
        4 * targets["candidate_score_bytes"]
        + 2 * (targets["campaign_bytes"] + targets["calibration_bytes"])
        + targets["journal_events"] * 1024
        + block_overhead
    )
    inodes = 2 * targets["tasks"] + 32
    credited = False
    assignment = area / "assignments/release-scale-release_scale.json"
    if registration is not None and assignment.is_file():
        from .progressive_pilot_recovery import assigned_request

        key = assigned_request(area, assignment)
        stage = confined(area, "roles/" + key[7:], exists=False)
        if (stage / "boundary.json").is_file():
            req = load(confined(stage, "request.json"))
            boundary = load(confined(stage, "boundary.json"))
            producer = read_bound_record(
                stage, boundary["producer"], version=ReleaseRole.producer_version
            )
            packet = seal(dict(block="release-scale", targets=targets))
            require(
                req == seal(req)
                and boundary == seal(boundary)
                and key == req["identity"] == boundary["request"] == producer["request"]
                and req["registration"] == registration["identity"]
                and req["source"] == registration["source"]
                and req["role"] == "release_scale"
                and req["data"] == packet["identity"]
                and producer["result"]["targets"] == targets,
                "scale credit does not match this release",
            )
            credited = True
            required = inodes = 0
    available = space(area)
    return dict(
        **available,
        required_bytes=required,
        required_inodes=inodes,
        credited_scale=credited,
        deficit_bytes=max(0, free_floor + required - available["free_bytes"]),
        deficit_inodes=max(0, inodes - available["free_inodes"]),
    )


def release_command(args, source, root, folder, revision, store, host_factory):
    from .progressive_round028 import audit_archive, reopen
    from .progressive_round028_receipts import recover
    from oeis_learn.generalization.progressive.program_search_report import verify_archive

    pilot_path = confined(folder, PILOT_SHARE, exists=False)
    if not pilot_path.is_file():
        return dict(mechanical_status="BLOCKED", reason="COMPLETED_PILOT_RECEIPT_REQUIRED"), 2
    pilot = verify_pilot(pilot_path)
    audit = verify_archive(audit_archive(folder))
    require(
        pilot["next_action"] == "RETAIN_REFERENCE"
        and audit["report"] == store.state["publication"]
        and audit["decision"]["action"] == "MATCHED_GENERATOR",
        "original scientific decisions differ",
    )
    relative = "private/release-acceptance/" + revision
    area = confined(folder, relative, exists=False)
    area.mkdir(parents=True, exist_ok=True, mode=0o700)
    cut = reopen(root, folder, store.campaign, store.state["audit"], recheck_inputs=False)
    # The registered child ran 160 endpoint cases; the original graph is the larger comparator.
    # Reopen the original campaign and all selected producer references before drawing scale from them.
    from .progressive_round028_audit import ORIGINAL

    measurements = original_scale(cut)
    originals = [
        r for r in cut["inputs"] if r["path"] == ORIGINAL + "/prepare-research/frozen/campaign.json"
    ]
    require(len(originals) == 1, "original campaign is absent or duplicated in the audit")
    original = originals[0]
    measurements["calibration_bytes"] = max(
        (r["bytes"] for r in cut["inputs"] if "calibrat" in r["path"]), default=0
    )
    campaign = read_bound_record(root, original, identity=store.campaign["policy"]["campaign"])
    targets = scale_targets(measurements, len(campaign["payload"]["tasks"]))
    qualification = seal(
        dict(
            version="round028-release-registration/v1",
            candidate=revision,
            source=digest(collect_snapshot(source)),
            pilot=pilot["report"],
            targets=targets,
            suite=digest(release_cases()),
        )
    )
    retain(area / "registration.json", qualification)
    cpu = ReleaseRole(args, source, root, revision, area, qualification, host_factory)
    host = host_factory(args, source_root=source, workspace=root, revision=revision)
    running, failures, evidence = [], {}, {}
    locations = {}
    capacity = None

    def progress(phase):
        revise(
            area / "progress.json",
            seal(
                dict(
                    candidate=revision,
                    phase=phase,
                    completed=running[:],
                    failures=dict(failures),
                    total=len(PHASES),
                )
            ),
        )

    for phase in PHASES:
        progress(phase)
        try:
            if phase == "hardware":
                probe_dir = relative + "/training"
                observed, code = rehearse(
                    args,
                    root,
                    source,
                    folder,
                    store,
                    revision,
                    {"decision": audit["decision"], "receipt": audit},
                    host_factory=host_factory,
                    output_area=probe_dir,
                )
                if code:
                    failures[phase] = observed["reason"]
                    locations[phase] = observed.get("failure_locations", [])
                    continue
                training = recover(root, folder, source, revision, store, output_area=probe_dir)
                require(
                    set(training["checkpoint_closures"].values()) == {"VERIFIED"},
                    "full checkpoint missing",
                )
                hardware = area / "rehearsal.zip"
                if hardware.exists():
                    require(
                        verify_transport(hardware)["evidence"] == training["identity"],
                        "hardware receipt changed",
                    )
                else:
                    with zipfile.ZipFile(audit_archive(folder)) as archived:
                        audit_bytes = archive_bytes({"audit.json": archived.read("audit.json")})
                    packed = archive_bytes(
                        {"rehearsal.json": encode(training), "round028.zip": audit_bytes}
                    )
                    require(
                        verify_transport(io.BytesIO(packed))["evidence"] == training["identity"],
                        "release rehearsal parent differs",
                    )
                    _atomic_write(hardware, packed)
                hip_cut = observed["qualification_receipts"]["hip"]
                stage = confined(folder, probe_dir + "/stages/" + hip_cut["request"][7:])
                bound = load(confined(stage, "receipt.json"))
                require(bound["identity"] == hip_cut["boundary"], "original HIP boundary differs")
                proof = read_bound_record(stage, bound["producer"], identity=hip_cut["producer"])
                expected = proof["details"]["arms"]["ordinary"]["final"]
                sample = confined(folder, probe_dir + "/training.json")
                from oeis_learn.generalization.progressive.contracts import reference

                gpu = ReleaseRole(args, source, root, revision, area, qualification, host_factory)
                gpu.phase = "hip"
                packet = seal(
                    dict(
                        block="release-interrupt",
                        expected=expected,
                        sample=dict(reference(sample.parent, sample), path="inputs/training.json"),
                    )
                )
                fault = gpu(
                    "release_interrupt",
                    packet,
                    seconds=ACCEPTANCE_SECONDS,
                    inputs=[(sample, "/work/inputs/training.json")],
                )
                evidence["interrupt"] = dict(
                    image=gpu.image,
                    native=gpu.native,
                    producer=gpu.last_producer["identity"],
                    wall_us=gpu.last_wall_us,
                    lease_us=gpu.last_lease_us,
                    result=fault,
                )
            else:
                if phase == "scale":
                    capacity = release_admission(
                        area,
                        targets,
                        host.policy["free_disk_floor_bytes"],
                        registration=qualification,
                    )
                    revise(area / "scale-admission.json", seal(capacity))
                    if capacity["deficit_bytes"] or capacity["deficit_inodes"]:
                        failures[phase] = "RELEASE_SCALE_RESOURCE_DEFICIT"
                        continue
                    admission_path = area / "scale-original-admission.json"
                    if not admission_path.exists():
                        require(
                            not capacity["credited_scale"],
                            "completed scale lacks its original admission",
                        )
                        retain(
                            admission_path,
                            seal(
                                dict(**capacity, floor_bytes=host.policy["free_disk_floor_bytes"])
                            ),
                        )
                role, parameters = {
                    "source": ("release_suite", {"mode": "source"}),
                    "wheel": ("release_suite", {"mode": "wheel"}),
                    "scale": ("release_scale", {"targets": targets}),
                    "storage": ("storage", {}),
                    "recovery": ("recovery", {}),
                }[phase]
                result = dict(
                    cpu(
                        role,
                        seal(dict(block="release-" + phase, **parameters)),
                        seconds=ACCEPTANCE_SECONDS,
                    )
                )
                for key in ("junit", "report"):
                    if key in result:
                        checked_file(cpu.last_output, result[key])
                        result[key] = result[key]["sha256"]
                boundary = load(cpu.last_boundary)
                timing = read_bound_record(cpu.last_boundary.parent, boundary["timing"])
                evidence[phase] = dict(
                    image=cpu.image,
                    native=cpu.native,
                    producer=cpu.last_producer["identity"],
                    wall_us=timing["elapsed_us"],
                    result=result,
                )
            running.append(phase)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            (area / (phase + "-error.txt")).write_text(traceback.format_exc())
            failures[phase] = type(exc).__name__
            locations[phase] = failure_locations(source, area, role_folder="roles")
    progress("blocked" if failures else "verified")
    if failures:
        return (
            dict(
                mechanical_status="BLOCKED",
                reason="RELEASE_ACCEPTANCE_INCOMPLETE",
                completed=running,
                failures=failures,
                failure_locations=locations,
                **({"capacity": capacity} if capacity is not None else {}),
            ),
            2,
        )
    with zipfile.ZipFile(pilot_path) as child:
        forecast = release_forecast(training, json.loads(child.read("pilot.json")))
    report = seal(
        dict(
            version="round028-release/v1",
            candidate=revision,
            source=qualification["source"],
            registration=qualification["identity"],
            targets=targets,
            pilot=pilot["report"],
            rehearsal=training["identity"],
            stages=evidence,
            forecast=forecast,
            admission=load(confined(area, "scale-original-admission.json")),
            hardware=training["hip"]["image"],
            guards=guard_inventory(evidence["scale"]["result"], training["hip"]),
        )
    )
    packed = archive_bytes(
        {
            "release.json": encode(report),
            "rehearsal.zip": hardware.read_bytes(),
            "pilot.zip": pilot_path.read_bytes(),
        }
    )
    receipt = verify_release(io.BytesIO(packed))
    path = confined(
        folder, "share/release-acceptance/" + revision + "/round028-release.zip", exists=False
    )
    if path.exists():
        require(path.read_bytes() == packed, "completed release acceptance changed")
    else:
        _atomic_write(path, packed)
    return (
        dict(
            mechanical_status="COMPLETE",
            reason="RELEASE_ACCEPTANCE_VERIFIED",
            receipt=receipt,
            archive=str(path.relative_to(folder)),
            completed=running,
        ),
        0,
    )
