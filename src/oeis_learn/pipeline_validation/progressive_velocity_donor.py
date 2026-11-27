"""Read immutable 028 outputs in place. A public score alone cannot authorize a new probe."""

import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES,
    digest,
    load,
    read_bound_record,
    seal,
)
from .identity import confined, file_hash
from .progressive_research_inputs import checked_file
from .progressive_round028_audit import SESSION as PARENT
from .progressive_round028_pilot import AREA, SHARE, checkpoint_inputs

# Published, independently verified bytes in the 015 pilot and release handoffs.
ARCHIVES = {
    "pilot": ("pilot", "4ad17a6f9e252de72025603a2e5ae7322f77c51b98260196eaec557ab22314fd"),
    "qualification": (
        "pilot-qualification",
        "ef61b2e243c2bb95035baaa230b71093ef7ecb36433f577063c39a3d65350812",
    ),
    "release": ("release", "45cab7c100e914a70ef5253b43c85ddf1ec80212f66326b090def75555fd5967"),
}
PILOT_REPORT = "sha256:71c8c2f807cbc9101983a1c74194e04629f535666d878b06664274464b0198ab"


def reuse_layers(cut, receipt, request):
    """Compare authenticated operands from the donor and frozen requested measurement."""
    same_cut = bool(cut.get("weights") and cut.get("lineage") and
                    all(cut[k] == request.get(k) for k in ("weights", "lineage")))
    eligible = bool(same_cut and cut.get("full_state"))
    continuation = eligible and cut.get("continuation") == "verified_disposable_next_update"
    action, reason = "rerun_control", "weights or training lineage unavailable"
    if same_cut:
        action, reason = "new_inference", "search, selection or sealed choice differs"
        if (
            receipt
            and receipt.get("choice")
            and all(
                receipt.get(k) and receipt[k] == request.get(k)
                for k in ("weights", "lineage", "inference", "selection")
            )
        ):
            action, reason = "rescore", "private truth or scorer differs"
            if receipt.get("score") and all(
                receipt.get(k) and receipt[k] == request.get(k) for k in ("scorer", "truth")
            ):
                action, reason = "reuse_score", "choice, scorer and truth match"
    return dict(action=action, continuation=continuation, eligible_for_parity=eligible, reason=reason)


def _archives(root, policy):
    paths = {}
    result = {}
    tools = confined(root, "run-state/discovery-tools", exists=False)
    originals = {
        "pilot": f"{PARENT}/{SHARE}",
        "release": f"{PARENT}/share/release-acceptance/{policy['release_reader']}/round028-release.zip",
    }
    for kind, (suffix, sha) in ARCHIVES.items():
        name = "round028-" + suffix + ".zip"
        candidates = [root / name, *sorted(tools.glob("round028-*/" + name))]
        if kind in originals:
            candidates.insert(0, root / originals[kind])
        available = []
        for candidate in candidates:
            path = confined(root, candidate, exists=False)
            if path.is_file():
                actual = file_hash(path)
                available.append(dict(bytes=path.stat().st_size, sha256=actual))
                if actual == "sha256:" + sha:
                    paths[kind] = path
                    break
        result[kind] = dict(available=available, matched=kind in paths, reader=None)
    return paths, result


def _verify_with_pin(root, revision, archive, qualification=None):
    """Disjoint temporary reader. No ref checkout, cache writes or imports from the donor."""
    with TemporaryDirectory(prefix="oeis-028-reader-") as temp:
        with subprocess.Popen(
            ["git", "-C", str(root), "archive", revision],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ) as git:
            extraction = subprocess.run(
                ["tar", "-xf", "-", "-C", temp], stdin=git.stdout, capture_output=True
            )
            if git.wait() or extraction.returncode:
                raise ValueError("reader commit unavailable: " + revision)
        cmd = [
            sys.executable,
            "-I",
            "-B",
            "-S",
            "-c",
            "import sys,runpy;sys.path.insert(0,sys.argv[1]);sys.argv=sys.argv[2:];"
            "runpy.run_path(sys.argv[0],run_name='__main__')",
            str(Path(temp) / "scripts"),
            str(Path(temp) / "scripts/foundation_run.py"),
            "--engine",
            "podman",
            "discovery",
            "round028",
            "--action",
            "verify",
            "--archive",
            str(archive),
        ]
        if qualification:
            cmd.extend(["--qualification-archive", str(qualification)])
        proc = subprocess.run(cmd, cwd=temp, capture_output=True, text=True)
        if proc.returncode:
            raise ValueError("pinned historical verifier rejected " + archive.name)
        return json.loads(proc.stdout)


def _checkpoint(cut, row, roles):
    """The public milestone fixes the manifest hash even when optimizer blobs are missing."""
    for role, bound, producer, result in roles:
        if result.get("milestone") != cut:
            continue
        if result["checkpoint"]["sha256"] != cut["parent"]:
            continue
        output = confined(role, Path(bound["producer"]["path"]).parent)
        base = confined(output, "checkpoint")
        manifest = read_bound_record(base, result["checkpoint"])
        weights = manifest["states"]["weights"]
        checked_file(base, weights)
        row.update(
            weights=weights["sha256"],
            weight_bytes=weights["bytes"], permitted_roles=["generate"],
            inference=True,
            reason="verified weights; new measurement requires a frozen inference key",
        )
        try:
            mounts, _ = checkpoint_inputs(output, result)
        except (ValueError, FileNotFoundError):
            row["reason"] += "; full restore data missing or inconsistent"
        else:
            row.update(full_state=cut["parent"], full_state_bytes=sum(p.stat().st_size for p,_ in mounts),
                       continuation="requires_next_update_parity", permitted_roles=["generate", "disposable_restore"])
        return


def _training_boundaries(area):
    roles = []
    for path in sorted(area.glob("roles/*/boundary.json")):
        try:
            bound = load(confined(area, path))
            producer = read_bound_record(path.parent, bound["producer"])
            if bound == seal(bound) and producer["request"] == bound["request"]:
                if "milestone" in producer["result"]:
                    roles.append((path.parent, bound, producer, producer["result"]))
        except (ValueError, KeyError, FileNotFoundError):
            continue
    return roles


def donated_ordinary(root):
    """Open only the original ordinary 4M full cut authorized by the published pilot."""
    from .progressive_velocity_continuation import continuation_plan

    policy = velocity_policy()
    files, _ = _archives(root, policy)
    if not {"pilot", "qualification"} <= files.keys():
        raise ValueError("original pilot and independent qualification archives required for donation")
    verified = _verify_with_pin(root, policy["pilot_reader"], files["pilot"], files["qualification"])
    if verified.get("report") != PILOT_REPORT:
        raise ValueError("original pilot reader did not verify the registered report")
    with zipfile.ZipFile(files["pilot"]) as archive:
        pilot = json.loads(archive.read("pilot.json"))
    cut = next(v for v in pilot["looks"] if v["target"] == 4_000_000)["cuts"]["ordinary"]
    area = confined(root, f"{PARENT}/{AREA}")
    freeze = load(confined(area, "freeze.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
    if seal(freeze) != freeze or freeze["identity"] != pilot["design"]["private_freeze"]:
        raise ValueError("original private freeze differs")
    training = checked_file(area, freeze["training"])
    package = load(training, maximum=CAMPAIGN_DOCUMENT_BYTES)
    if package["stream"]["identity"] != freeze["stream"] or package["stream"]["seeds"] != freeze["seeds"]:
        raise ValueError("original training seed or stream differs")
    for role, bound, _, parent in _training_boundaries(area):
        if parent["milestone"] == cut and parent["checkpoint"]["sha256"] == cut["parent"]:
            output = confined(role, Path(bound["producer"]["path"]).parent)
            mounts, _ = checkpoint_inputs(output, parent)
            plan, _, _ = continuation_plan(package, parent)
            return dict(parent=parent, plan=plan, training=dict(freeze["training"], path="training.json")), [
                (training, "/work/inputs/training.json"), *mounts,
            ]
    raise ValueError("authenticated ordinary 4M full checkpoint missing")


def inventory(root):
    policy = velocity_policy()
    files, archives = _archives(root, policy)
    cuts = [
        dict(
            method=m,
            look=k,
            weights=None,
            full_state=None,
            continuation=False,
            inference=False,
            reason="original matched weights and provenance unavailable",
        )
        for m in ("00", "first_divergence")
        for k in (0, 1000000, 4000000)
    ]
    cuts.append(
        dict(
            method="D",
            look=None,
            weights=None,
            full_state=None,
            continuation=False,
            inference=False,
            reason="original qualified D weights unavailable",
        )
    )
    out = dict(
        archives=archives,
        cuts=cuts,
        original_clocks=[],
        prices={},
        stream=None,
        historical_scores=[],
        gpu_dispatches=0,
    )
    if "release" in files:
        try:
            _verify_with_pin(root, policy["release_reader"], files["release"])
            archives["release"]["reader"] = policy["release_reader"]
        except ValueError as exc:
            archives["release"]["reason"] = str(exc)
    if not {"pilot", "qualification"} <= files.keys():
        return out
    try:
        verified = _verify_with_pin(
            root, policy["pilot_reader"], files["pilot"], files["qualification"]
        )
        if verified.get("report") != PILOT_REPORT:
            raise ValueError("original pilot report identity differs")
        archives["pilot"]["reader"] = archives["qualification"]["reader"] = policy["pilot_reader"]
        from oeis_learn.generalization.progressive.program_search_costing import exposed_timing_rates

        with zipfile.ZipFile(files["pilot"]) as archive:
            pilot = json.loads(archive.read("pilot.json"))
            with (
                zipfile.ZipFile(io.BytesIO(archive.read("round028-parent.zip"))) as wrapper,
                zipfile.ZipFile(io.BytesIO(wrapper.read("round028.zip"))) as evidence,
            ):
                out["prices"]["exposed_028_timing"] = exposed_timing_rates(
                    json.loads(evidence.read("timing.json")), json.loads(evidence.read("panels.json"))
                )
        out["original_clocks"] = pilot["costs"]
        with zipfile.ZipFile(files["qualification"]) as archive:
            qualification = json.loads(archive.read("qualification.json"))
        balance = qualification.get("balance")
        if balance is not None:
            out["prices"]["original_028_balance"] = dict(
                receipt=balance["identity"],
                settled_us=sum(row[2] for row in balance["charges"]),
                outstanding_us={
                    state: sum(row[3] for row in balance["opened"] if row[2] == state)
                    for state in ("open", "ended", "unstarted")
                },
                stopped_after=balance["stopped_after"],
            )
        scale = qualification["scale"]
        out["prices"]["resident_028"] = dict(
            receipt=scale["identity"], candidate=qualification["registration"]["candidate"],
            cases=scale["cases"], empty=scale["no_selection"], slowest_case_us=scale["slowest_case_us"],
            search_supervisor_us=scale["search_supervisor_us"], score_supervisor_us=scale["score_supervisor_us"],
            nonempty_score_supervisor_us=(scale.get("score_control") or {}).get("supervisor_us"),
        )
        area = confined(root, f"{PARENT}/{AREA}")
        freeze = load(confined(area, "freeze.json"), maximum=CAMPAIGN_DOCUMENT_BYTES)
        if seal(freeze) != freeze or freeze["identity"] != pilot["design"]["private_freeze"]:
            raise ValueError("original private freeze differs")
    except (ValueError, FileNotFoundError) as exc:
        out["reason"] = str(exc)
        return out
    for look in pilot["looks"]:
        out["historical_scores"].append(
            dict(
                look=look["target"],
                attempts=8 if look["target"] == 4000000 else 1,
                denominator=len(
                    next(iter(freeze["graph"][str(look["target"])].values()))["members"]
                ),
                disposition="historical_only_until_ordered_inference_and_score_match",
            )
        )
    try:
        path = checked_file(area, freeze["training"])
        out["stream"] = dict(
            bytes=path.stat().st_size,
            artifact=freeze["training"]["sha256"],
            identity=freeze["stream"],
            seeds=freeze["seeds"],
        )
    except (ValueError, FileNotFoundError) as exc:
        out["stream_reason"] = str(exc)
    try:
        from collections import Counter

        prior = load(
            checked_file(root, freeze["roster_source"]), maximum=freeze["roster_source"]["bytes"]
        )
        if prior != seal(prior):
            raise ValueError("original reserved panel supply differs")
        counts = {}
        for role, members in prior["allocations"]["panels"].items():
            by_stratum = Counter(r["stratum"] for r in members)
            counts[role] = {
                s: dict(cases=n, families=len({r["group"] for r in members if r["stratum"] == s}))
                for s, n in by_stratum.items()
            }
        out["historical_reservations"] = dict(
            source=freeze["roster_source"]["sha256"], counts=counts
        )
    except (ValueError, FileNotFoundError, KeyError) as exc:
        out["historical_reservations"] = dict(counts=None, reason=str(exc))
    try:
        checked_file(root, freeze["reference"])
        cuts[-1].update(
            weights=freeze["reference"]["sha256"],
            weight_bytes=freeze["reference"]["bytes"], permitted_roles=["generate"],
            inference=True,
            reason="normal D recipe and authenticated weights",
            recipe=freeze["recipe"], source=pilot["registration"]["source"], images=freeze["images"],
            registration=pilot["registration"]["identity"], lineage=digest(["D", freeze["recipe"], freeze["reference"]["sha256"]]),
        )
    except (ValueError, FileNotFoundError) as exc:
        cuts[-1]["reason"] = str(exc)
    roles = _training_boundaries(area)
    for row in cuts[:-1]:
        look = next((v for v in pilot["looks"] if v["target"] == row["look"]), None)
        if look is None:
            continue
        arm = "ordinary" if row["method"] == "00" else row["method"]
        cut = look["cuts"][arm]
        row.update(
            lineage=cut["stream"],
            initial=cut["initial"],
            useful_tokens=cut["useful_tokens"], updates=cut["updates"], cursor=cut["cursor"], seeds=cut["seeds"],
            discarded_tokens=cut["discarded"], overshoot=cut["overshoot"], codec=cut["codec"],
            charged_tokens=cut["charged"],
            source=pilot["registration"]["source"], registration=pilot["registration"]["identity"],
            images=freeze["images"],
            image=freeze["images"]["hip"],
            policy=cut["common_policy"],
        )
        try:
            _checkpoint(cut, row, roles)
        except (ValueError, FileNotFoundError) as exc:
            row["reason"] = str(exc)
    return out
