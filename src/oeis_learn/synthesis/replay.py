"""Replay native scoped fitting evidence without loading retired algorithms."""

from .admission import VisiblePrompt, admit, source_identity
from .client import SynthesisClient
from .contracts import identity


def replay_native(record):
    try:
        prompt = VisiblePrompt(tuple(record["indices"]), tuple(record["values"]))
        profile = record["profile"]
        response = record["result"]
        with SynthesisClient(identity(dict(replay=record["identity_sha256"]))) as client:
            if response.get("outcome") == "verified_solution":
                rows = response.get("admitted", [])
                if not 1 <= len(rows) <= 8:
                    return False
                for row in rows:
                    if row.get("original_source_sha256") != source_identity(
                        record["original_source"]
                    ):
                        return False
                    if row.get("language_profile") != profile:
                        return False
                    raw = dict(row, outcome="assignment_found")
                    raw.pop("verification", None)
                    result = admit(raw, prompt, runtime_sha256=client.runtime_identity(profile))
                    if result.outcome != "verified_solution":
                        return False
                    actual = client.execute(row["source"], prompt.indices, profile=profile)
                    if actual.get("values") != list(prompt.values) or actual.get("outcome") != "ok":
                        return False
                return True
            if response.get("outcome") != "unsat_in_scope":
                return False
            native = response["native"]
            evidence = native["evidence"]
            result = client.request(
                "replay_unsat",
                [
                    dict(
                        candidate_id="replay",
                        source=record["original_source"],
                        evidence=evidence,
                        abstraction_policy=record["policy"],
                    )
                ],
                prompt.indices,
                prompt.values,
                profile=profile,
            )
            return (
                result["complete"]
                and len(result["results"]) == 1
                and result["results"][0].get("outcome") == "replayed_unsat"
            )
    except (ValueError, KeyError, TypeError, OSError):
        return False
