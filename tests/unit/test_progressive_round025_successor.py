"""Boundary cases for the single post-capability learner replacement."""

import pytest

from oeis_learn.generalization.progressive.contracts import seal
from oeis_learn.pipeline_validation import progressive_round025_successor as recovery
from oeis_learn.pipeline_validation.progressive_workstation_state import Blocked


def test_preserve_full_capability_roster():
    assert len(recovery.CARRIED) == len(set(recovery.CARRIED)) == 35
    assert recovery.CARRIED[:18] == recovery.previous.CARRIED
    assert recovery.CARRIED[-1] == 'capability-readiness'
    assert recovery.TASK not in recovery.CARRIED


def test_unchanged_setup_envelope():
    evidence = seal(
        dict(version=recovery.VERSION, reviewed_audit=recovery.AUDIT, additional_prepare_seconds=21600))
    assert recovery.setup_limits({'qualify': 300}, evidence) == {
        'qualify': 300,
        'prepare': 21600,
        'freeze': 21600
    }
    with pytest.raises(Blocked):
        recovery.setup_limits({}, seal(dict(evidence, additional_prepare_seconds=21601)))


def test_unrelated_run_uses_its_normal_scope(tmp_path):
    campaign = dict(payload=dict(references=[]))
    original = tmp_path / 'original.json'
    original.touch()
    with recovery.research_scope(tmp_path, campaign, original, None, None):
        pass


def test_reject_round026_without_matching_proof(tmp_path):
    path = tmp_path / recovery.SUCCESSOR / 'prepare-research/frozen/campaign.json'
    path.parent.mkdir(parents=True)
    path.touch()
    (tmp_path / recovery.SUCCESSOR / 'preparation-owner-proof.json').write_text('{}')
    with pytest.raises(Blocked, match='single round026'):
        with recovery.research_scope(tmp_path, dict(payload=dict(references=[])), path, None, None):
            pass
