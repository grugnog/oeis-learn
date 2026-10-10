"""Frozen round replay; cursor and counters advance after optimizer commit."""

from copy import deepcopy


class FrozenReplay:
    def __init__(self, archive, snapshot=None, cursor=0):
        self.archive = archive
        self.snapshot = archive.freeze() if snapshot is None else deepcopy(snapshot)
        self.entries = archive.training_view(self.snapshot)
        if type(cursor) is not int or cursor < 0:
            raise ValueError("invalid replay cursor")
        self.cursor, self.committed_examples = cursor, 0

    def peek(self, size):
        if type(size) is not int or size < 0:
            raise ValueError("invalid replay batch size")
        if size and not self.entries:
            raise ValueError("configured replay has no eligible examples")
        return tuple(self.entries[(self.cursor + i) % len(self.entries)] for i in range(size))

    def commit(self, size):
        self.peek(size)
        self.cursor += size
        self.committed_examples += size

    def state_dict(self):
        return {
            "version": "frozen-replay/v1",
            "snapshot": deepcopy(self.snapshot),
            "cursor": self.cursor,
            "committed_examples": self.committed_examples,
        }

    def load_state_dict(self, state):
        if state["version"] != "frozen-replay/v1" or state["snapshot"] != self.snapshot:
            raise ValueError("replay round/root cannot change on resume")
        if any(type(state[k]) is not int or state[k] < 0 for k in ("cursor", "committed_examples")):
            raise ValueError("invalid replay continuation")
        self.cursor, self.committed_examples = state["cursor"], state["committed_examples"]
