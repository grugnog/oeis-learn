"""Observed runtime metadata must preserve configurable fresh-worker threads."""

import torch

from oeis_learn.rl.foundation_sft import runtime_record
from oeis_learn.fresh.worker import runtime


def test_runtime_record_observes_actual_threads_without_reconfiguring_them():
    previous = torch.get_num_threads()
    try:
        for count in (1, 3):
            torch.set_num_threads(count)
            assert runtime_record("cpu")["cpu_threads"] == count
            assert runtime("cpu")["cpu_threads"] == count
            assert torch.get_num_threads() == count
    finally:
        torch.set_num_threads(previous)
