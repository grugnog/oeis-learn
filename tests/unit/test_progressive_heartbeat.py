import time

import pytest

from oeis_learn.generalization.progressive.contracts import load
from oeis_learn.pipeline_validation.progressive_training import Heartbeat, Progress


def test_background_progress_continues_while_update_is_blocked(tmp_path):
    progress = Progress(tmp_path, interval=0.02)
    state = {
        "committed": 3,
        "charged": 6,
        "discarded": 1,
        "remaining": {"tokens": 10},
        "phase": "UPDATE_IN_PROGRESS",
    }
    with Heartbeat(progress, device="cpu", initial=state):
        time.sleep(0.11)
    records = [load(p) for p in sorted(tmp_path.glob("*.json"))]
    assert len(records) >= 3
    assert all(r["committed"] == 3 and r["charged"] == 6 and r["discarded"] == 1 for r in records)
    assert records[-1]["last_boundary_age_us"] > records[0]["last_boundary_age_us"]
    assert records[-1]["stalled_reason"] and records[-1]["generation_tail"] is None


def test_progress_preflight_protects_disk_and_finalization(tmp_path):
    progress = Progress(tmp_path)
    with pytest.raises(TimeoutError):
        progress.preflight(
            bytes_required=2**63,
            inodes_required=1,
            remaining_us=100,
            finalization_us=1,
            research=False,
        )
    with pytest.raises(ValueError):
        progress.preflight(
            bytes_required=0, inodes_required=1, remaining_us=100, finalization_us=1, research=True
        )


def test_generation_queue_fit_and_tail_clocks_remain_separate_from_wall_time():
    from oeis_learn.pipeline_validation.identity import seal
    from oeis_learn.pipeline_validation.progressive_evidence import generation_timing

    value = seal(
        {
            "version": "discovery-candidates/v1",
            "elapsed_us": 5000000,
            "records": {
                "prompt": {
                    "costs": {
                        "correct": {
                            "details": [
                                {
                                    "generation_seconds": 0.1,
                                    "complete": True,
                                    "source": "private",
                                    "fitting": {
                                        "clock": {"admitted": 1, "dispatched": 2, "completed": 4}
                                    },
                                },
                                {"generation_seconds": 0.2, "complete": False},
                            ]
                        }
                    }
                }
            },
        }
    )
    report = generation_timing(value)
    assert report["generation_tail"]["p95_us"] == 200000
    assert report["generation_tail"]["token_cap_or_incomplete_events"] == 1
    assert report["queue"]["total_us"] == 1000000 and report["fit"]["total_us"] == 2000000
    assert report["wall_elapsed_us"] == 5000000 and report["fit"]["unavailable"] == 1
    assert "private" not in str(report)
