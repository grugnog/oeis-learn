"""Immutable multi-program archive; admission precedes every index mutation."""

from copy import deepcopy
from pathlib import Path

from oeis_learn.data.structural_splits import StructuralHoldouts, structural_signature
from oeis_learn.decoder.program_codec import encode_body, decode_body, codec_digest
from oeis_learn.experiments.artifacts import (
    compute_canonical_digest as digest,
    canonical_bytes,
    load_json,
    atomic_write,
    ArtifactPath,
)
from oeis_learn.experiments.qualified_models import seal_record, validate_record
from oeis_learn.experiments.models import check_digest
from oeis_learn.experiments.profiles import I256_MIN, I256_MAX
from oeis_learn.sandbox.pipeline import digest_bytes


def eligibility(source, context_limit, *, codec="foundation"):
    if type(context_limit) is not int or not 2 <= context_limit <= 1025:
        raise ValueError("bounded context limit required")
    if codec == "foundation":
        tokens = encode_body(source)  # includes sampled EOS
        if decode_body(tokens) != " ".join(source.split()):
            raise ValueError("canonical complete source required")
        count = len(tokens) + 1  # BOS plus codec's EOS
    elif codec == "legacy":
        from oeis_learn.decoder.wat_grammar import encode_wat, UNK_ID

        tokens = encode_wat(source)
        if UNK_ID in tokens:
            raise ValueError("legacy source contains unrepresentable tokens")
        count = len(tokens) + 2
    else:
        raise ValueError("unknown codec")
    return count <= context_limit, len(tokens), count


class ProgramArchive:
    def __init__(self, root, *, holdouts=None, context_limit=1024, quota_bytes=200 << 30):
        self.root = Path(root)
        if not 0 < quota_bytes <= 200 << 30:
            raise ValueError("archive quota out of bounds")
        self.holdouts = holdouts or StructuralHoldouts()
        self.context_limit, self.quota = context_limit, quota_bytes
        self.root.mkdir(parents=True, exist_ok=True)
        self.index = self.root / "index.json"
        self._entries = []
        if self.index.exists():
            index = load_json(self.index.read_bytes())
            if (
                set(index) != {"version", "holdouts", "context_limit", "entries", "identity_sha256"}
                or index["version"] != "program-archive/v1"
                or index["holdouts"] != self.holdouts.identity()
                or index["context_limit"] != context_limit
                or index["identity_sha256"] != digest(index, "identity_sha256")
            ):
                raise ValueError("incompatible or corrupt archive index")
            self._entries = list(index["entries"])
            for key in self._entries:
                self.get(key)

    def get(self, key):
        check_digest(key, "archive object")
        path = self.root / (key[7:] + ".json")
        envelope = load_json(path.read_bytes())
        return self.validate_packet(envelope,key)

    def validate_packet(self,envelope,key):
        """Shared admission replay for an immutable transported archive object."""
        if (
            set(envelope) != {"entry", "source", "tokens", "execution", "provenance"}
            or digest(envelope) != key
        ):
            raise ValueError("archive object mismatch")
        record = envelope["entry"]
        validate_record(record)
        if (
            digest_bytes(envelope["source"].encode()) != record["source_sha256"]
            or encode_body(envelope["source"]) != envelope["tokens"]
            or digest(envelope["execution"]) != record["execution_sha256"]
        ):
            raise ValueError("archive source/execution mismatch")
        expected = envelope["execution"]
        if (
            record["codec_sha256"] != codec_digest()
            or record["ir_sha256"] != structural_signature(envelope["source"])
            or record["sample_sha256"] != digest(envelope["provenance"])
            or record["admission_sha256"] != self.holdouts.identity()
            or record["runtime_sha256"] != expected["identities"]["runtime_sha256"]
            or record["fit_indices"] != list(range(20))
            or record["validation_indices"] != list(range(20, len(record["outputs"])))
            or record["conditioning_values"] != record["outputs"][:20]
            or record["body_tokens"] != len(envelope["tokens"])
            or record["fuel"] != expected["fuel"]
            or record["elapsed_ns"] != expected["elapsed_ns"]
        ):
            raise ValueError("unbound archive admission fields")
        if (
            expected["outcome"] is not None
            or expected["outputs"] != record["outputs"]
            or expected["reference_outputs"] != record["outputs"]
        ):
            raise ValueError("unverified archive execution")
        self._provenance(
            record["origin"],
            envelope["provenance"],
            envelope["source"],
            record["conditioning_values"],
            record["model_sha256"],
        )
        allowed, reason = self.holdouts.decision(
            envelope["source"], [int(v) for v in record["conditioning_values"]]
        )
        context, _, count = eligibility(envelope["source"], self.context_limit)
        if (
            record["training_eligible"] != (allowed and context)
            or record["eligibility_reason"]
            != (reason if not allowed else "eligible" if context else "over_context")
            or record["context_tokens"] != count
        ):
            raise ValueError("archive eligibility mismatch")
        return deepcopy(envelope)

    def _provenance(self, origin, provenance, source, conditioning, model_sha):
        if origin == "generic_sample":
            from oeis_learn.data.sampler_factory import sampler as GenericSampler
            from oeis_learn.data.program_admission import require_sample

            if set(provenance) == {"config", "sample"}:
                require_sample(provenance["sample"], GenericSampler(provenance["config"]))
            elif set(provenance) == {"config", "sample", "structure"}:
                from oeis_learn.data.generic_programs import QualifiedGenericSampler

                qualified = QualifiedGenericSampler(provenance["config"]).sample_variant(
                    provenance["sample"]["sample_counter"],
                    provenance["structure"]["parameter_counter"],
                )
                if (
                    qualified != provenance["structure"]
                    or qualified["sample"] != provenance["sample"]
                ):
                    raise ValueError("qualified structural draw provenance mismatch")
            else:
                raise ValueError("generic provenance keys differ")
            if provenance["sample"]["canonical_source"] != source:
                raise ValueError("generic sample provenance mismatch")
        elif origin in ("generic_model", "training_prefix_hypothesis"):
            if set(provenance) == {"search"}:
                self._search_provenance(provenance["search"], source, conditioning, model_sha)
                return
            if set(provenance) != {"rollout"}:
                raise ValueError("model rollout provenance required")
            rollout = provenance["rollout"]
            validate_record(rollout)
            from oeis_learn.decoder.program_codec import BOS_ID

            if (
                rollout["kind"] != "rollout_record"
                or model_sha
                != digest([rollout["behavior_encoder_sha256"], rollout["behavior_decoder_sha256"]])
                or rollout["actions"][0] != BOS_ID
                or decode_body(rollout["actions"][1:]) != source
                or rollout["visible_prompt_sha256"] != digest(conditioning)
                or any(rollout["forced_mask"][1:])
                or any(rollout["pad_mask"])
            ):
                raise ValueError("unbound model trajectory")
        else:
            raise ValueError("teacher/imported origin cannot enter strict archive")

    def _search_provenance(self, packet, source, conditioning, model_sha, depth=0):
        from oeis_learn.evaluation.search import typed_repairs
        from oeis_learn.experiments.models import check_digest

        if (
            depth > 1
            or set(packet)
            != {
                "version",
                "method",
                "source_sha256",
                "body_tokens",
                "visible_prompt_sha256",
                "encoder_sha256",
                "decoder_sha256",
                "execution",
                "parent",
                "identity_sha256",
            }
            or packet["version"] != "search-hypothesis/v1"
            or packet["method"] not in ("beam", "stochastic", "greedy", "typed_repair")
            or digest(packet, "identity_sha256") != packet["identity_sha256"]
        ):
            raise ValueError("closed bounded search provenance required")
        check_digest(packet["encoder_sha256"], "search encoder")
        check_digest(packet["decoder_sha256"], "search decoder")
        if (
            packet["source_sha256"] != digest_bytes(source.encode())
            or decode_body(packet["body_tokens"]) != source
            or packet["visible_prompt_sha256"] != digest(conditioning)
            or model_sha != digest([packet["encoder_sha256"], packet["decoder_sha256"]])
        ):
            raise ValueError("search source/prompt/model binding mismatch")
        evidence = packet["execution"]
        if evidence["source_sha256"] != packet["source_sha256"] or evidence["indices"] != list(
            range(20)
        ):
            raise ValueError("search execution binding mismatch")
        if packet["method"] == "typed_repair":
            parent = packet["parent"]
            if not isinstance(parent, dict):
                raise ValueError("repair parent required")
            if parent.get("kind") == "model" and set(parent) == {"kind", "hypothesis"}:
                old = decode_body(parent["hypothesis"]["body_tokens"])
                self._search_provenance(
                    parent["hypothesis"], old, conditioning, model_sha, depth + 1
                )
            elif parent.get("kind") == "archive" and set(parent) == {"kind", "key", "snapshot"}:
                view = dict(self.training_view(parent["snapshot"]))
                if parent["key"] not in view:
                    raise ValueError("repair retrieval parent is excluded")
                old = view[parent["key"]]["source"]
            else:
                raise ValueError("unsupported search parent")
            if source not in typed_repairs(old):
                raise ValueError("unreplayed typed source repair")
        elif packet["parent"] is not None:
            raise ValueError("untransformed model hypothesis has a parent")

    def admit(
        self,
        source,
        conditioning,
        outputs,
        *,
        origin,
        provenance,
        runner,
        request_id,
        model_sha256=None,
        parent_sha256=None,
    ):
        if (
            not 20 < len(outputs) <= 100
            or len(conditioning) != 20
            or outputs[:20] != conditioning
            or any(type(v) is not int or not I256_MIN <= v <= I256_MAX for v in outputs)
        ):
            raise ValueError("exact visible values plus independent training continuation required")
        tokens = encode_body(source)
        source = decode_body(tokens)
        text = [str(v) for v in conditioning]
        self._provenance(origin, provenance, source, text, model_sha256)
        allowed, reason = self.holdouts.decision(source, conditioning)
        in_context, body_count, context_count = eligibility(source, self.context_limit)
        execution = runner.run_single(
            source, list(range(len(outputs))), request_id=request_id
        ).to_dict()
        if (
            execution["outcome"] is not None
            or execution["outputs"] != [str(v) for v in outputs]
            or execution["reference_outputs"] != execution["outputs"]
        ):
            raise ValueError("independent execution mismatch; admission is atomic")
        entry = seal_record(
            "archive_entry",
            source_sha256=digest_bytes(source.encode()),
            ir_sha256=structural_signature(source),
            codec_sha256=codec_digest(),
            runtime_sha256=execution["identities"]["runtime_sha256"],
            origin=origin,
            parent_sha256=parent_sha256,
            model_sha256=model_sha256,
            sample_sha256=digest(provenance),
            conditioning_values=text,
            outputs=execution["outputs"],
            fit_indices=list(range(20)),
            validation_indices=list(range(20, len(outputs))),
            execution_sha256=digest(execution),
            admission_sha256=self.holdouts.identity(),
            training_eligible=allowed and in_context,
            eligibility_reason=reason
            if not allowed
            else "eligible"
            if in_context
            else "over_context",
            body_tokens=body_count,
            context_tokens=context_count,
            elapsed_ns=execution["elapsed_ns"],
            fuel=execution["fuel"],
        )
        envelope = {
            "entry": entry,
            "source": source,
            "tokens": tokens,
            "execution": execution,
            "provenance": deepcopy(provenance),
        }
        key = digest(envelope)
        if key in self._entries:
            return key
        encoded = canonical_bytes(envelope)
        if (
            sum(p.stat().st_size for p in self.root.glob("*.json"))
            + len(encoded)
            + 2 * len(canonical_bytes(self._entries + [key]))
            > self.quota
        ):
            raise OSError("archive quota exhausted")
        atomic_write(ArtifactPath(self.root, key[7:] + ".json"), encoded)
        index = {
            "version": "program-archive/v1",
            "holdouts": self.holdouts.identity(),
            "context_limit": self.context_limit,
            "entries": self._entries + [key],
        }
        index["identity_sha256"] = digest(index)
        from oeis_learn.tracking.foundation_metrics import replace_json

        replace_json(self.index, index)
        self._entries.append(key)
        return key

    def freeze(self):
        entries = tuple(sorted(set(self._entries)))
        return {
            "version": "archive-snapshot/v1",
            "entries": list(entries),
            "root_sha256": digest(["archive-snapshot/v1", self.holdouts.identity(), entries]),
        }

    def training_view(self, snapshot=None):
        snapshot = self.freeze() if snapshot is None else deepcopy(snapshot)
        if snapshot["root_sha256"] != digest(
            ["archive-snapshot/v1", self.holdouts.identity(), tuple(snapshot["entries"])]
        ) or not set(snapshot["entries"]) <= set(self._entries):
            raise ValueError("unbound archive snapshot")
        groups = {}
        for key in snapshot["entries"]:
            envelope = self.get(key)
            if not envelope["entry"]["training_eligible"]:
                continue
            groups.setdefault(digest(envelope["entry"]["outputs"]), []).append((key, envelope))
        selected = []
        for items in groups.values():
            unique = {}
            for key, value in items:
                identity = (value["entry"]["source_sha256"], value["entry"]["runtime_sha256"])
                unique.setdefault(identity, (key, value))
            candidates = sorted(
                unique.values(),
                key=lambda row: (
                    row[1]["entry"]["context_tokens"],
                    row[1]["entry"]["fuel"],
                    row[0],
                ),
            )
            frontier = []
            for item in candidates:
                record = item[1]["entry"]
                if any(
                    other[1]["entry"]["context_tokens"] < record["context_tokens"]
                    and other[1]["entry"]["fuel"] <= record["fuel"]
                    and other[1]["entry"]["ir_sha256"] == record["ir_sha256"]
                    for other in frontier
                ):
                    continue
                frontier.append(item)
            selected.extend(frontier[:4])
        return tuple(sorted(selected, key=lambda row: row[0]))

    def reverify(self, runner):
        for key in tuple(self._entries):
            value = self.get(key)
            result = runner.run_single(
                value["source"],
                list(range(len(value["entry"]["outputs"]))),
                request_id="reverify-" + key[7:23],
            ).to_dict()
            if (
                result["outcome"] is not None
                or result["outputs"] != value["entry"]["outputs"]
                or result["reference_outputs"] != result["outputs"]
                or result["identities"]["runtime_sha256"] != value["entry"]["runtime_sha256"]
            ):
                raise ValueError(
                    "archive must be independently readmitted under changed semantics/runtime"
                )
        return self.freeze()
