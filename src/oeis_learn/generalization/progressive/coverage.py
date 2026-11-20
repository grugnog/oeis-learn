"""Private constructive witnesses separate expression support from search failure."""

import json
import time
from .contracts import record, verify_refs
from oeis_learn.synthesis.contracts import ResourceProfile


def audit(
    root,
    task,
    witness,
    *,
    client=None,
    limits=ResourceProfile(),
    token_cap=1024,
    audit_limits=None,
    operator_class=(),
    missing_operations=(),
    created_at,
    purpose="acceptance",
    track="synthetic_strict",
):
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.reference_parser import parse
    from oeis_learn.decoder.bigint_codec import encode_body, decode_body, fitting_policy, CodecError
    from oeis_learn.synthesis.contracts import canonical_integer

    classification = "unresolved"
    reason = "no_constructive_witness"
    ref_out = "unavailable"
    native_out = "unavailable"
    fit = "unavailable"
    if witness is not None:
        raw = json.loads(verify_refs(root, [witness], max_bytes=1 << 20)[0])
        if (
            set(raw) != {"source", "indices", "values"}
            or not isinstance(raw["source"], str)
            or not 1 <= len(raw["indices"]) <= 100
            or len(raw["indices"]) != len(raw["values"])
        ):
            raise ValueError("closed independent witness required")
        for v in raw["indices"] + raw["values"]:
            canonical_integer(v, bits=limits.input_bits)
        selected = evaluate(raw["source"], raw["indices"], limits)
        ref_out = selected.reason or selected.outcome
        broader = audit_limits or limits
        verified = evaluate(raw["source"], raw["indices"], broader)
        if verified.outcome != "ok":
            reason = "witness_unverified:" + str(verified.reason)
        elif list(verified.values) != raw["values"]:
            reason = "witness_counterexample"
        else:
            parse(raw["source"], broader)
            try:
                tokens = encode_body(raw["source"])
                decode_body(tokens)
                token_ok = len(tokens) <= token_cap
                fit = "solver_support_unmeasured:" + fitting_policy(raw["source"])
            except CodecError as exc:
                token_ok = False
                fit = "unsupported"
                if str(exc) != "token cap":
                    reason = "grammar_unsupported"
            if client is None:
                reason = "native_evidence_unavailable"
            else:
                if getattr(client, "limits", broader) != broader:
                    raise ValueError("native/reference audit profile differs")
                deadline = time.monotonic() + broader.evaluation_seconds
                native = client.execute(raw["source"], raw["indices"], deadline=deadline)
                native_out = native.get("reason") or native.get("outcome", "unresolved")
                if time.monotonic() > deadline:
                    reason = "late_native_result"
                    native_out = "deadline"
                elif (
                    native.get("outcome") != "ok" or list(native.get("values", [])) != raw["values"]
                ):
                    reason = "native_reference_disagreement"
                elif selected.outcome == "ok" and token_ok:
                    classification = "supported"
                    reason = "independent_witness_within_declared_limits"
                elif (
                    ref_out
                    in {
                        "integer_limit",
                        "reference_limit",
                        "loop_limit",
                        "output_limit",
                        "deadline",
                    }
                    or not token_ok
                ):
                    classification = "over_budget"
                    reason = "verified_witness_exceeds_selected_limits_no_inexpressibility_claim"
                else:
                    reason = "selected_profile_unresolved"
    return record(
        "coverage",
        dict(
            task=task,
            operator_class=list(operator_class),
            witness=witness,
            language="bigint-body/v1",
            limits=limits.identity,
            classification=classification,
            reason=reason,
            missing_operations=list(missing_operations),
            reference_outcome=ref_out,
            native_outcome=native_out,
            fitting_support=fit,
            verification_scope="finite independent witness only; not a learner target or an impossibility proof",
        ),
        created_at=created_at,
        purpose=purpose,
        track=track,
    )
