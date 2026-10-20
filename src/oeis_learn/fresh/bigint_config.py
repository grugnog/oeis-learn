"""Closed image-side recipe resolution for the qualified Phase 2 algorithms."""

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

from oeis_learn.synthesis.contracts import ResourceProfile, identity
from oeis_learn.synthesis.service_policy import ServiceLimits
from .bootstrap_contract import PRIMES


def defaults():
    return dict(
        version="bigint-bootstrap/v1",
        data=dict(
            cache="tmp-oeis-bulk",
            revision=None,
            seed=20261016,
            max_records=5000,
            max_download_bytes=20 << 30,
            free_disk_floor_bytes=5 << 30,
            generator="bigint-connected/v1",
            variants=16,
            max_families=4096,
            training=512,
            parameter_development=128,
            structural_development=128,
            oeis_development=8,
            oeis_final=8,
        ),
        execution=dict(
            language="bigint-body/v1",
            codec="bigint-codec/v1",
            runtime="bigint-wasm/v1",
            resources=asdict(ResourceProfile()),
            artifacts=None,
        ),
        inputs=dict(
            profile="hybrid-chunks/v1",
            primes=list(PRIMES),
            max_valuation=32,
            use_film=True,
            summary_tokens=False,
            mask=dict(magnitude=True, modular=True, difference=True, exact_bytes=True),
            ablation_mode="post_training",
        ),
        model=dict(
            d_model=256,
            heads=4,
            encoder_layers=4,
            decoder_layers=4,
            ff=1024,
            precision="float32",
            attention="eager",
            token_scale=1,
            position_scale=16,
            dropout=0,
            seed=20261016,
        ),
        learning=dict(
            target_view="literal",
            learning_rate=1e-4,
            batch_size=8,
            max_updates=3000,
            validation_every=500,
            patience=8,
            min_improvement=1,
            curriculum=False,
            replay=False,
            representation_comparison=True,
            replay_capacity=4096,
            replay_fraction=0.25,
            earlier_fraction=0.25,
            advancement_threshold=0.8,
            advancement_coverage=0.95,
            advancement_validations=2,
            cost_preflight=None,
        ),
        fitting=dict(
            profile="native-fitting/v1",
            disabled=False,
            max_parameters=8,
            max_assignments=8,
            coefficient_bits=4096,
            bounds=[16, 64, 256, 1024, 4096],
            native_seconds=2,
            reference_seconds=2,
            service_limits=asdict(ServiceLimits()),
        ),
        search=dict(attempts=8, greedy=1, stochastic=7, tokens=1024, seconds=60, beam=None),
        evaluation=dict(
            visible=20,
            horizon=100,
            periodic_panel=32,
            primary_transfer_metric="continuation_accuracy_100",
            portfolio="exact_equal_budget",
            controls=["correct", "shuffled", "absent", "template"],
            final_access="sealed",
            hidden_seconds=30,
            coverage=0.95,
            family_resamples=2000,
            paired_seeds=[20261016, 20261017, 20261018],
        ),
        track=dict(kind="generic_only", foundation=None, assisted_manifest=None),
        archive=dict(max_alternatives=8, max_bytes=512 << 20),
        budgets=dict(
            preparation_seconds=7200,
            training_seconds=7200,
            cleanup_seconds=5,
            ledger="debit-before-work/v1",
        ),
        reporting=dict(
            bundle_bytes=64 << 20,
            core_reserve_bytes=8 << 20,
            max_examples=8,
            detail_chunk_bytes=1 << 20,
            max_core_bytes=8 << 20,
        ),
    )


def _merge(base, patch, path="recipe"):
    if not isinstance(patch, dict) or set(patch) - set(base):
        raise ValueError(
            f"unknown {path} settings: {set(patch) - set(base) if isinstance(patch, dict) else 'object required'}"
        )
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(base[key], dict):
            out[key] = _merge(base[key], value, path + "." + key)
        else:
            out[key] = deepcopy(value)
    return out


def resolve(value, *, seed=None, artifacts=None):
    if not isinstance(value, dict) or value.get("version") != "bigint-bootstrap/v1":
        raise ValueError("bigint-bootstrap/v1 required")
    value = deepcopy(value)
    existing = value.pop("identity", None)
    result = _merge(defaults(), value)
    if seed is not None:
        if "seed" in value.get("model", {}) and value["model"]["seed"] != seed:
            raise ValueError("conflicting explicit recipe/seed")
        result["model"]["seed"] = seed
    if artifacts is not None:
        if result["execution"]["artifacts"] not in (None, artifacts):
            raise ValueError("runtime artifact mismatch")
        result["execution"]["artifacts"] = deepcopy(artifacts)
    baseline = defaults()
    for section, keys in {
        "data": [
            "generator",
            "variants",
            "training",
            "parameter_development",
            "structural_development",
        ],
        "execution": ["language", "codec", "runtime"],
        "inputs": ["primes", "max_valuation", "use_film", "summary_tokens"],
        "model": [
            "d_model",
            "heads",
            "encoder_layers",
            "decoder_layers",
            "ff",
            "precision",
            "attention",
            "token_scale",
            "position_scale",
            "dropout",
        ],
        "fitting": [
            "profile",
            "max_parameters",
            "max_assignments",
            "coefficient_bits",
            "bounds",
            "native_seconds",
            "reference_seconds",
        ],
        "search": ["attempts", "greedy", "stochastic", "tokens", "seconds"],
        "evaluation": [
            "visible",
            "horizon",
            "periodic_panel",
            "primary_transfer_metric",
            "portfolio",
            "controls",
            "coverage",
            "family_resamples",
            "paired_seeds",
        ],
        "learning": [
            "patience",
            "min_improvement",
            "replay_fraction",
            "earlier_fraction",
            "advancement_threshold",
            "advancement_coverage",
            "advancement_validations",
        ],
        "budgets": ["cleanup_seconds", "ledger"],
    }.items():
        if any(result[section][k] != baseline[section][k] for k in keys):
            raise ValueError(f"unqualified {section} policy; return semantic changes to Phase 2")
    ResourceProfile(**result["execution"]["resources"])
    ServiceLimits(**result["fitting"]["service_limits"])
    for section, keys in {
        "data": [
            "seed",
            "max_records",
            "max_download_bytes",
            "free_disk_floor_bytes",
            "max_families",
            "oeis_development",
            "oeis_final",
        ],
        "learning": ["batch_size", "max_updates", "validation_every", "replay_capacity"],
        "model": ["seed"],
        "budgets": ["preparation_seconds", "training_seconds"],
        "reporting": [
            "bundle_bytes",
            "core_reserve_bytes",
            "max_examples",
            "detail_chunk_bytes",
            "max_core_bytes",
        ],
        "archive": ["max_alternatives", "max_bytes"],
    }.items():
        for key in keys:
            n = result[section][key]
            if type(n) is not int or n < (0 if key == "seed" else 1):
                raise ValueError(f"positive exact {section}.{key} required")
    if not 0 <= result["model"]["seed"] < 2**32 or not 0 <= result["data"]["seed"] < 2**32:
        raise ValueError("invalid seed")
    for section, key in [("learning", "learning_rate"), ("evaluation", "hidden_seconds")]:
        n = result[section][key]
        if type(n) not in (int, float) or not math.isfinite(n) or n <= 0:
            raise ValueError("finite positive cost/rate required")
    for section, keys in [
        ("learning", ["curriculum", "replay", "representation_comparison"]),
        ("fitting", ["disabled"]),
    ]:
        if any(type(result[section][key]) is not bool for key in keys):
            raise ValueError("explicit boolean required")
    if result["evaluation"]["final_access"] != "sealed":
        raise ValueError("ordinary recipes must keep final access sealed")
    if result["learning"]["target_view"] not in {"literal", "parameterized"}:
        raise ValueError("unknown target view")
    if result["learning"]["representation_comparison"] and result["learning"]["replay"]:
        raise ValueError("replay disabled in representation comparisons")
    if (
        not 1 <= result["learning"]["max_updates"] <= 3000
        or result["budgets"]["training_seconds"] > 7200
        or result["budgets"]["preparation_seconds"] > 7200
    ):
        raise ValueError("bounded registered screen required")
    if (
        result["reporting"]["bundle_bytes"] < 16 << 20
        or result["reporting"]["core_reserve_bytes"] != 8 << 20
        or result["reporting"]["max_core_bytes"] > 8 << 20
    ):
        raise ValueError("bundle needs minimum16MiB and core reserve8MiB")
    if result["archive"]["max_alternatives"] != 8 or result["learning"]["replay_capacity"] > 4096:
        raise ValueError("archive/replay cap exceeded")
    if (
        result["data"]["max_families"] > 4096
        or result["data"]["max_records"] > 10000
        or result["data"]["max_download_bytes"] > 100 << 30
        or any(result["data"][k] > 1000 for k in ("oeis_development", "oeis_final"))
    ):
        raise ValueError("bounded pinned data/cohort sizes required")
    from oeis_learn.encoder.ablation import InputMask

    InputMask(**result["inputs"]["mask"])
    if result["inputs"]["profile"] not in {"hybrid-chunks/v1", "tri-stream/v1"} or result["inputs"][
        "ablation_mode"
    ] not in {"post_training", "retrained"}:
        raise ValueError("unknown input profile/mask mode")
    if result["track"]["kind"] not in {"generic_only", "assisted"}:
        raise ValueError("unknown provenance track")
    if result["track"]["kind"] == "assisted":
        if not result["track"]["foundation"] or not result["track"]["assisted_manifest"]:
            raise ValueError(
                "assisted recipe requires compatible generic foundation and reviewed manifest"
            )
    elif result["track"]["assisted_manifest"] is not None:
        raise ValueError("strict recipient cannot use assisted data")
    for key in ("foundation", "assisted_manifest"):
        if result["track"][key] is not None and not isinstance(result["track"][key], str):
            raise ValueError("workspace manifest/checkpoint path required")
    for section, key in [
        ("data", "cache"),
        ("track", "foundation"),
        ("track", "assisted_manifest"),
    ]:
        p = result[section][key]
        if p is not None and (Path(p).is_absolute() or ".." in Path(p).parts):
            raise ValueError("workspace-relative path required")
    if result["search"]["beam"] is not None:
        raise ValueError("beam is not registered for the qualified eight-attempt portfolio")
    if result["inputs"]["profile"] == "tri-stream/v1" and result["inputs"]["mask"] != dict(
        magnitude=True, modular=True, difference=True, exact_bytes=False
    ):
        raise ValueError("tri-stream control requires mathematical streams without byte branch")
    if (
        result["reporting"]["max_examples"] > 8
        or result["reporting"]["detail_chunk_bytes"] > 1 << 20
        or result["archive"]["max_bytes"] > 512 << 20
    ):
        raise ValueError("bounded example/detail/archive caps required")
    if (
        result["learning"]["validation_every"] not in (100, 500)
        or result["learning"]["batch_size"] != 8
        or result["learning"]["learning_rate"] != 1e-4
    ):
        raise ValueError("registered LR1e-4, batch8 and100/500-update cadence required")
    cost = result["learning"]["cost_preflight"]
    if cost is not None:
        from oeis_learn.synthesis.state import require_hash
        from .cost_policy import screen_cost

        if (
            not isinstance(cost, dict)
            or set(cost) != {"image", "profile_sha256", "scoring_data", "costs", "fixture_sha256"}
            or cost["scoring_data"] is not False
        ):
            raise ValueError("closed non-scoring measured cost preflight required")
        for key in ("image", "profile_sha256", "fixture_sha256"):
            require_hash(cost[key])
        phases = {
            "training_p95",
            "teacher_forcing_p95",
            "generation_p95",
            "fit_p95",
            "reference_p95",
            "hidden_scoring_p95",
            "checkpoint_p95",
        }
        if set(cost["costs"]) not in (
            phases,
            phases | {"generation_batch_p95", "generation_batch_size"},
        ):
            raise ValueError("complete measured p95 phases required")
        screen_cost(**cost["costs"])
    result["identity"] = identity(result)
    if existing is not None and existing != result["identity"]:
        raise ValueError("resolved recipe identity mismatch")
    return result


def load(path, **kwargs):
    import yaml

    return resolve(yaml.safe_load(Path(path).read_text()), **kwargs)


def packaged_artifacts():
    from .common import read, file_hash

    build = read("/opt/oeis-build-info/bigint-build.json")
    return dict(
        build_sha256=file_hash("/opt/oeis-build-info/bigint-build.json"),
        worker_sha256=file_hash("/opt/oeis-build-info/synthesis-worker"),
        guest_sha256=build["runtime_sha256"],
        native=build["native"],
        locks=build["locks"],
        rustc=build["rustc"],
    )
