"""Learning CLI options reach the actual command handlers without silent defaults."""

import pytest
from oeis_learn.cli.main import cli
from oeis_learn.cli import foundation


@pytest.mark.parametrize("command", ["learning-train", "learning-search"])
def test_learning_cli_preserves_required_inputs_and_selected_options(monkeypatch, command):
    expected = {
        "config": "config.json",
        "gate_report": "g7.json",
        "learning_gate_report": "g8.json",
        "profiles": "profiles",
        "archive": "archive",
        "output": "output",
        "device": "cpu",
        "diagnostic": True,
        "as_json": True,
    }
    flags = ["foundation", command]
    for key, value in list(expected.items())[:-2]:
        flags += ["--" + key.replace("_", "-"), value]
    flags += ["--diagnostic", "--json"]
    if command == "learning-train":
        expected.update(updates=2, seed=7, resume="qualified-00000000.json")
        flags += ["--updates", "2", "--seed", "7", "--resume", "qualified-00000000.json"]
    else:
        expected.update(
            checkpoint="qualified-00000001.json", prompt="visible.json", portfolio="greedy"
        )
        flags += [
            "--checkpoint",
            "qualified-00000001.json",
            "--prompt",
            "visible.json",
            "--portfolio",
            "greedy",
        ]

    def handler(**kwargs):
        assert kwargs == expected
        return 0

    monkeypatch.setattr(foundation, "cmd_" + command.replace("-", "_"), handler)
    assert cli(flags) == 0


def test_learning_hardware_flag_is_not_silently_dropped(monkeypatch):
    def handler(output, hardware=False, as_json=False):
        assert (output, hardware, as_json) == ("evidence", True, True)
        return 0

    monkeypatch.setattr(foundation, "cmd_qualify_learning", handler)
    assert (
        cli(["foundation", "qualify-learning", "--output", "evidence", "--hardware", "--json"]) == 0
    )


def test_qualified_activation_receives_learning_evidence(monkeypatch):
    def handler(
        config, gate_report, profiles, as_json=False, learning_gate_report=None, diagnostic=False
    ):
        assert (config, gate_report, profiles, learning_gate_report, diagnostic, as_json) == (
            "config.json",
            "g7.json",
            "profiles",
            "g8.json",
            True,
            True,
        )
        return 0

    monkeypatch.setattr(foundation, "cmd_activate_qualified", handler)
    assert (
        cli(
            [
                "foundation",
                "activate-qualified",
                "--config",
                "config.json",
                "--gate-report",
                "g7.json",
                "--profiles",
                "profiles",
                "--learning-gate-report",
                "g8.json",
                "--diagnostic",
                "--json",
            ]
        )
        == 0
    )
