"""Read-only diagnosis of an existing checkpoint under explicitly recorded code."""

from pathlib import Path
from .common import digest, read, write


def diagnose(inputs, output):
    from .worker import models, runtime
    from .diagnostics import evaluate
    from .bootstrap_config import validate_learning
    from oeis_learn.tracking.role_coordinator import QueueRunner
    from oeis_learn.tracking.role_worker import validation_loss
    from oeis_learn.tracking.training_checkpoint import QualifiedCheckpointStore

    inputs, output = Path(inputs), Path(output)
    settings, examples = read(inputs / "settings.json"), read(inputs / "examples.json")
    if digest(examples) != settings["examples_sha256"]:
        raise ValueError("diagnostic inputs differ from source run")
    learning = validate_learning(settings["learning"])
    encoder, decoder, counts = models(settings)
    store = QualifiedCheckpointStore(inputs / "checkpoints", free_floor_bytes=0)
    # Verify the original checkpoint's contract/profile/runtime. This is an
    # explicit weights-only diagnostic, never a resume under changed code.
    _, base, state = store.load_learning(
        inputs / "checkpoints" / settings["checkpoint"],
        expected_contract=settings["source_contract"],
        expected_profile=settings["model"]["identity_sha256"],
        expected_runtime=settings["source_runtime"],
    )
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    result = evaluate(
        encoder, decoder, examples, learning, QueueRunner(output / "runtime-queue"),
        full_development=True, compare_generation=True,
    )
    result["checkpoint_sha256"] = base["blob_sha256"]
    result["source_runtime_sha256"] = settings["source_runtime"]
    result["diagnostic_runtime"] = runtime(settings["device"])
    result["parameter_count"] = counts
    result["validation_loss"] = validation_loss(
        encoder,
        decoder,
        [
            {"tokens": r["tokens"], "entry": {"conditioning_values": r["values"]}}
            for r in examples["development"]
        ],
        settings["device"],
        limit=None,
        batch_size=4,
    )
    result["validation_denominator"] = len(examples["development"])
    result["mode"] = "weights-only diagnostic; optimizer, source run and budgets unchanged"
    result["qualified"] = False
    result["identity"] = digest({k: v for k, v in result.items() if k != "identity"})
    write(output / "diagnostic.json", result)
    return result
