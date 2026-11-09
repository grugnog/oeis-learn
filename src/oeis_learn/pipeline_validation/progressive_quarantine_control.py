"""Host publication of independently checked refutations; no history rewriting."""

import json

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    load,
    reference,
    seal,
    verify_refs,
)
from oeis_learn.generalization.progressive.quarantine import Journal
from oeis_learn.pipeline_validation.identity import confined


def require_clean_inputs(root, journal, artifacts):
    identities = []
    for group in artifacts.values():
        for descriptor in group:
            value = load(confined(root, descriptor.ref["path"]))
            if value["kind"] == "checkpoint":
                ref = value["data"].get("artifact")
                if ref:
                    identities.extend(Journal(journal.root).weight_checkpoints(ref["sha256"]))
            if value["kind"] == "library":
                data = json.loads(
                    verify_refs(root, [value["data"]["artifact"]], max_bytes=256 << 20)[0]
                )
                identities.append(data["library"]["identity"])
    Journal(journal.root).require_clean(identities)


def reconcile(root, output, result, journal):
    ref = result.get("metrics", {}).get("dossier_record")
    if ref is None:
        return None
    from oeis_learn.generalization.progressive.claims import validate_dossier

    dossier = json.loads(verify_refs(output, [ref], max_bytes=32 << 20)[0])
    # The installed reviewer has already verified the native transcript. Replay
    # independent source checks and all original hash bindings before publishing.
    claim = dossier["payload"]["claim"]
    if claim.get("review") is not None:
        # Novelty trust is unrelated to refutation authority. Replay the original
        # execution under a non-promoted copy of that claim.
        candidate = dict(
            dossier,
            payload=dict(
                dossier["payload"],
                claim={
                    "statement": "conjecture",
                    "novelty": "not_reviewed",
                    "proof": None,
                    "review": None,
                },
            ),
        )
        candidate["payload"].pop("proof_check", None)
        validate_dossier(output, seal(candidate))
    else:
        validate_dossier(output, dossier)
    q = Journal(journal.root)
    selection = json.loads(
        verify_refs(output, [dossier["payload"]["selection"]["artifact"]], max_bytes=256 << 20)[0]
    )
    checkpoint = selection.get("checkpoint")
    known = {n["id"] for n in q.read()[0]}
    parents = [checkpoint] if checkpoint in known else []
    if isinstance(checkpoint, str):
        parents.extend(q.weight_checkpoints(checkpoint))
    library = dossier["payload"].get("library")
    library_identity = None
    if library:
        lib = json.loads(verify_refs(output, [library], max_bytes=256 << 20)[0])
        library_identity = lib.get("library", lib).get("identity")
        if library_identity not in known:
            raise ValueError("review library lineage was not registered")
    if not dossier["payload"]["counterexamples"]:
        q.require_clean(parents + ([library_identity] if library_identity else []))
    q.dossier(dossier, checkpoints=sorted(set(parents)), library_identity=library_identity)
    overlay = q.refute_dossier(dossier, evidence=reference(root, confined(output, ref["path"])))
    if overlay:
        path = output / "quarantine.json"
        value = seal(
            dict(
                version="progressive-refutation-publication/v1",
                dossier=dossier["identity"],
                **overlay,
            )
        )
        if not path.exists():
            atomic_write(path, value)
        elif load(path) != value:
            raise ValueError("original refutation publication differs")
    return overlay
