"""Hourly model snapshots over the existing irreversible progressive cost journal.

Like post-012 research persistence, a crash restores the last complete model and
policy, while newer charges survive. The cost ledger already includes discarded
work here; adding the 012 discarded counter a second time would double-charge it.
Only the leased learner holds unsaved state. No live progress pointer authorizes
an evaluator or successor.
"""
import copy
import time

from oeis_learn.generalization.budgets import Ledger, integer
from oeis_learn.pipeline_validation.generalization_persistence import snapshot_due
from .contracts import seal
from oeis_learn.fresh.common import write
from .recovery import Recovery


class PeriodicRecovery(Recovery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active = False
        self.dirty = False
        self.entries = set()
        self.last_full = time.monotonic()

    def activate(self):
        current = super().current()
        if current['pending'] is not None:
            raise ValueError('hourly snapshots must be complete update boundaries')
        self.recoverable_useful = self.state["useful_tokens"]
        self.live_ledger = current['ledger']
        self.live_cuts = current['cuts']
        self.active = True
        self.last_full = time.monotonic()

    def read(self):
        if self.active:
            if not self.owned:
                raise ValueError('exclusive task lease required')
            return copy.deepcopy(self.state)
        return super().read()

    def append(self, **changes):
        if self.active:
            if not self.owned:
                raise ValueError("exclusive task lease required")
            self.state = dict(self.state, **changes)
        else:
            super().append(**changes)

    def current(self):
        if not self.active:
            return super().current()
        if not self.owned:
            raise ValueError('exclusive task lease required')
        spent = self.base['tokens'] + self.p['caps']['tokens'] - self.journal.task_remaining(self.task)['tokens']
        if spent < self.live_ledger.consumed or spent < self.state['useful_tokens']:
            raise ValueError('cost journal rolled back behind live model')
        return dict(ledger=Ledger(self.token_cap, consumed=spent, updates=self.live_ledger.updates),
                    discarded_tokens=spent-self.state['useful_tokens'],
                    pending=copy.deepcopy(self.state['pending']), cuts=copy.deepcopy(self.live_cuts))

    def completed(self, update, batch):
        integer(update, 1)
        if not self.owned:
            raise ValueError("exclusive task lease required")
        if not self.active:
            return super().completed(update, batch)
        # The current owner may retry its last acknowledgement. Older requests
        # require restoring a published snapshot, never reconstructing tensors.
        last = self.state['last']
        if last is not None and last['update'] == update:
            if last['batch'] != batch:
                raise ValueError('conflicting committed update replay')
            return self.state['checkpoint']
        if type(update) is not int or update != self.live_ledger.updates + 1:
            raise ValueError('ordered logical update required')
        return None

    def commit_lazy(self, capture, ledger, cuts):
        if not self.active or not self.owned or self.state['status'] != 'RUNNING':
            raise ValueError('active leased hourly learner required')
        pending = self.state['pending']
        if (pending is None or ledger.pending is not None or
                ledger.updates != self.live_ledger.updates + 1 or
                ledger.consumed != self.current()['ledger'].consumed):
            raise ValueError('complete matching optimizer and cost cut required')
        from .milestones import MILESTONES
        crossed = any(self.live_ledger.consumed < n <= ledger.consumed for n in MILESTONES)
        self.entries.update(pending['batch']['entries'])
        self.state.update(useful_tokens=self.state['useful_tokens']+pending['tokens'], pending=None,
                          last=dict(update=ledger.updates, batch=pending['batch']))
        self.live_ledger, self.live_cuts, self.dirty = ledger, copy.deepcopy(cuts), True
        if snapshot_due('hourly/v1', self.last_full, time.monotonic(), force=crossed):
            self.flush(capture)
        self.progress()
        return self.state['checkpoint']

    def progress(self):
        # Informational only. Recovery authority is the immutable snapshot event
        # plus the independently durable debit-before-work journal.
        write(self.root/'progress.json', seal(dict(version='progressive-periodic-progress/v1',
            checkpoint=self.state['checkpoint']['identity'], ledger=self.live_ledger.snapshot(),
            useful_tokens=self.state['useful_tokens'], recoverable_useful_tokens=self.recoverable_useful,
            recovery_authority=False)))

    def flush(self, capture):
        if not self.owned or not self.active or self.state['pending'] is not None:
            raise ValueError('complete live update boundary required')
        if not self.dirty:
            return self.state['checkpoint']
        record = self.save(capture(), self.live_ledger, self.live_cuts, entries=sorted(self.entries))
        # Publication is the sole commit point. If this fails, the caller must
        # discard the live window, including any newer orphan tensor generation.
        self.active = False
        try:
            super().append(checkpoint=record)
        finally:
            self.active = True
        self.recoverable_useful = self.state["useful_tokens"]
        self.entries.clear()
        self.dirty = False
        self.last_full = time.monotonic()
        return record

    def finish_window(self, capture, *, failed=False):
        if not self.active:
            return
        try:
            if not failed:
                self.flush(capture)
        finally:
            self.active = False
            self.dirty = False
            self.entries.clear()
            super().read()  # on failure, forget all unsaved policy/model progress

    def restore(self):
        if self.active:
            raise ValueError('finish or discard live window before full restore')
        return super().restore()
