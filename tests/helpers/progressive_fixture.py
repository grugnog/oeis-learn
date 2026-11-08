"""Diagnostic-only records; expected arithmetic lives in hand-checked fixture JSON."""

import json
from pathlib import Path
from oeis_learn.pipeline_validation.identity import digest, seal

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/progressive"


def manifest():
    return json.loads((FIXTURES / "manifest.json").read_text())


def envelope(kind, payload, *, purpose="acceptance", track="synthetic_strict", parents=()):
    return seal(
        dict(
            version=f"discovery-{kind}/v1",
            created_at="2026-11-08T00:00:00Z",
            parents=list(parents),
            purpose=purpose,
            track=track,
            payload=payload,
        )
    )


def caps(**overrides):
    return (
        dict(
            tokens=100,
            draws=10,
            rejections=10,
            attempts=8,
            native_work=1000,
            elapsed_us=100_000_000,
            disk_bytes=1 << 20,
            archive_bytes=1 << 19,
            **{},
        )
        | overrides
    )


def graph():
    return dict(
        selected_stages=["capability", "diversity"],
        nodes=[
            dict(
                id="capability",
                stage="capability",
                kind="routine",
                dependencies=[],
                maximum_visits=1,
                caps=caps(),
            ),
            dict(
                id="diversity",
                stage="diversity",
                kind="routine",
                dependencies=["capability"],
                maximum_visits=1,
                caps=caps(),
            ),
        ],
        edges=[
            dict(
                source="capability",
                target="diversity",
                condition="COMPLETE",
                action="ADVANCE_REGISTERED",
            )
        ],
        diagnostic_pool=caps(
            tokens=0,
            draws=0,
            rejections=0,
            attempts=0,
            native_work=0,
            elapsed_us=0,
            disk_bytes=0,
            archive_bytes=0,
        ),
    )


def task(campaign=None, **changes):
    payload = dict(
        campaign=campaign or digest("campaign"),
        id="d-seed-1",
        stage="diversity",
        kind="training",
        arm="static",
        seed=1,
        initializer=dict(mode="random", artifact=None, compatibility=digest("learning")),
        token_target=100,
        caps=caps(),
        startup_us=1_000_000,
        finalization_us=2_000_000,
        policy=digest("policy"),
        slot="diversity",
        output="runs/d-seed-1",
    )
    payload.update(changes)
    return envelope("task", payload)


def blueprint():
    return envelope(
        "blueprint",
        dict(
            executable=False,
            selected_stages=["capability", "diversity"],
            references=[],
            unresolved=["retained_checkpoint", "qualification", "prepared_inputs"],
            proposed_caps=caps(),
            reference_campaign="run-state/prior",
        ),
    )
