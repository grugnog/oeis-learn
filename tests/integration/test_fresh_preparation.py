"""Real Python Wasmtime/reference admission on a synthetic bulk fixture, never source/Rust acceptance."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from oeis_learn.experiments.profiles import qualified_model_profile
from oeis_learn.fresh.bulk import metadata
from oeis_learn.fresh.common import file_hash, read, write
from oeis_learn.fresh.inputs import prepare
from oeis_learn.tracking.budget_ledger import BudgetLedger


def test_real_generic_admission_bulk_source_and_small_fixture_rejection(tmp_path):
    cache = tmp_path / "cache"
    repo = cache / "oeisdata"
    records = []
    functions = [
        lambda n: 7,
        lambda n: n,
        lambda n: n * n,
        lambda n: n**3,
        lambda n: 2**n,
        lambda n: (-1) ** n,
    ]
    for number, function in enumerate(functions, 1):
        aid = f"A{number:06d}"
        seq = repo / "seq/A000" / (aid + ".seq")
        terms = repo / "files/A000" / f"b{number:06d}.txt"
        seq.parent.mkdir(parents=True, exist_ok=True)
        terms.parent.mkdir(parents=True, exist_ok=True)
        seq.write_text(f"%I {aid}\n%S {aid} 0,1\n%O {aid} 0,1\n%K {aid} sign,easy\n")
        terms.write_text("".join(f"{i} {function(i)}\n" for i in range(120)))
        records.append(
            {
                "aid": aid,
                "path": str(terms.relative_to(repo)),
                "sha256": file_hash(terms),
                "bytes": terms.stat().st_size,
                "entry_sha256": file_hash(seq),
                "metadata": metadata(seq.read_text(), aid, "synthetic-fixture"),
            }
        )
    write(cache / "fetch-plan.json", {"selected": records, "snapshot": {"commit": "a" * 40}})
    write(cache / "fetch-result.json", {"plan_sha256": file_hash(cache / "fetch-plan.json")})
    root = tmp_path / "inputs"
    config = {
        "cohort": {"seed": 20260913, "dev_count": 2, "final_count": 2},
        "sampler_config": str(Path("configs/foundation/generic_sampler.yaml").resolve()),
        "archive": {"training_prompts": 8, "development_structures": 1, "seconds": 120},
        "run": {
            "model": qualified_model_profile(attention_backend="eager"),
            "seed": 7,
            "training_seconds": 60,
            "evaluation": {"groups": 2, "attempts": 2, "tokens": 128, "per_group_seconds": 30},
        },
    }

    def fixture_clock():
        return {
            "monotonic_ns": time.monotonic_ns(),
            "utc_ns": time.time_ns(),
            "boot_id": "synthetic-clock",
        }

    def fixture_ledger(*args):
        return BudgetLedger(*args, clock=fixture_clock)

    with patch("oeis_learn.tracking.budget_ledger.BudgetLedger", fixture_ledger):
        manifest = prepare(config, cache, root, backend="python_wasmtime")
        assert not manifest["qualified"]
        assert manifest["files"]["archive-plan.json"]["sha256"] == file_hash(
            root / "archive-plan.json"
        )
        view = read(root / "examples.json")
        assert len(view["training"]) >= 8 and view["development"]
        assert not {r["structure"] for r in view["training"]} & {
            r["structure"] for r in view["development"]
        }
        assert read(root / "release-source-audit.json")["all_member_prefixes"] >= 4
        retained = file_hash(root / "prepared.json")
        # Reduced CPU fixtures cannot pass the public128-prompt readiness gate.
        with pytest.raises(ValueError, match="128"):
            prepare(config, cache, root, backend="python_wasmtime")
        assert file_hash(root / "prepared.json") == retained
