"""Frozen SFT/search rounds with a controller-owned training truth boundary."""

from copy import deepcopy
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.experiments.profiles import I256_MIN, I256_MAX
from oeis_learn.rl.replay import FrozenReplay


class TrainingTruth:
    """Only an explicitly pinned training partition can supply continuations.

    Proposers receive visible() copies. This object stays in the controller and
    never accepts dev/final feedback, filenames or evaluator truth objects.
    """

    def __init__(self, partition, *, expected_sha256, reserved_prefixes):
        if (
            set(partition) != {"version", "partition", "records"}
            or partition["version"] != "training-truth/v1"
            or partition["partition"] != "train"
            or digest(partition) != expected_sha256
        ):
            raise ValueError("pinned training partition required")
        self._truth = {}
        self.identity = expected_sha256
        self.reserved_prefixes = frozenset(reserved_prefixes)
        for values in partition["records"]:
            if not 20 < len(values) <= 100 or any(
                type(v) is not int or not I256_MIN <= v <= I256_MAX for v in values
            ):
                raise ValueError("exact training continuation required")
            prefix = tuple(values[:20])
            key = digest([str(v) for v in prefix])
            if key in reserved_prefixes or prefix in self._truth:
                raise ValueError("reserved/ambiguous prefix cannot enter self-training")
            self._truth[prefix] = tuple(values)
        if not self._truth:
            raise ValueError("empty training partition")

    def visible(self):
        return tuple(self._truth)

    def continuation(self, visible):
        try:
            return list(self._truth[tuple(visible)])
        except KeyError as exc:
            raise ValueError("continuation is unavailable outside the training partition") from exc


class SelfTrainingRounds:
    def __init__(self, archive, truth):
        if not isinstance(truth, TrainingTruth):
            raise ValueError("training-only truth capability required")
        if truth.reserved_prefixes != archive.holdouts.reserved_prefixes:
            raise ValueError("truth and archive reserved-prefix manifests differ")
        self.archive, self.truth = archive, truth
        self.round = 0
        self.snapshot = archive.freeze()
        self.replay = FrozenReplay(archive, self.snapshot)
        self.admissions = []

    def training_prompts(self):
        return self.truth.visible()

    def search(self, visible, proposer, runner, **settings):
        from oeis_learn.evaluation.search import SearchSession

        # Membership only, before any model work; continuation stays here.
        self.truth.continuation(visible)
        session = SearchSession.with_archive(visible, self.archive, self.snapshot, **settings)
        return session, session.run(proposer, runner)

    def admit(self, source, visible, *, provenance, model_sha256, runner, request_id):
        outputs = self.truth.continuation(visible)
        key = self.archive.admit(
            source,
            list(visible),
            outputs,
            origin="training_prefix_hypothesis",
            provenance=provenance,
            model_sha256=model_sha256,
            runner=runner,
            request_id=request_id,
        )
        self.admissions.append(key)
        # The current round's replay/retrieval view remains immutable.
        return key

    def near_misses(self, visible, limit=8):
        self.truth.continuation(visible)
        if type(limit) is not int or not 0 <= limit <= 8:
            raise ValueError("bounded training-only lookup required")
        candidates = []
        for key, value in self.archive.training_view(self.snapshot):
            prefix = tuple(int(v) for v in value["entry"]["conditioning_values"])
            # Never join a training archive entry to evaluation metadata.
            if prefix not in self.truth._truth:
                continue
            mismatch = sum(a != b for a, b in zip(prefix, visible))
            candidates.append((mismatch, key, value["source"]))
        return tuple(sorted(candidates)[:limit])

    def next_round(self, runner):
        snapshot = self.archive.reverify(runner)
        replay = FrozenReplay(self.archive, snapshot)
        self.round += 1
        self.snapshot, self.replay = snapshot, replay
        self.admissions = []

    def run_round(
        self, encoder, decoder, optimizer, runner, *, updates=1, prompt_limit=4, seed=20260913
    ):
        """Run real frozen replay updates, then search training prefixes only.

        Stochastic discoveries preserve their behavior record. Beam and repair
        hypotheses carry separate replayable search provenance for SFT only.
        """
        from oeis_learn.rl.sft_trainer import replay_update
        from oeis_learn.evaluation.search import ModelProposer

        if (
            type(updates) is not int
            or not 0 <= updates <= 100
            or type(prompt_limit) is not int
            or not 1 <= prompt_limit <= 4
        ):
            raise ValueError("bounded round work required")
        generic = [
            key
            for key, value in self.archive.training_view(self.snapshot)
            if value["entry"]["origin"] == "generic_sample"
        ]
        if updates and not generic:
            raise ValueError("round has no generic training complement")
        training = []
        for _ in range(updates):
            start = self.replay.committed_examples * 3
            training.append(
                replay_update(
                    encoder,
                    decoder,
                    optimizer,
                    self.replay,
                    generic=[generic[(start + i) % len(generic)] for i in range(3)],
                    device=next(encoder.parameters()).device,
                )
            )
        reports = []
        proposer = ModelProposer(encoder, decoder, context_limit=self.archive.context_limit)
        for number, visible in enumerate(self.training_prompts()[:prompt_limit]):
            _, report = self.search(visible, proposer, runner, seed=seed + number)
            for candidate in report["results"]:
                if (
                    candidate.get("status") != "visible_fit"
                    or not {"rollout", "hypothesis"} & candidate.get("provenance", {}).keys()
                ):
                    continue
                if "rollout" in candidate["provenance"]:
                    rollout = candidate["provenance"]["rollout"]
                    provenance = {"rollout": rollout}
                    model = digest(
                        [rollout["behavior_encoder_sha256"], rollout["behavior_decoder_sha256"]]
                    )
                else:
                    packet = candidate["provenance"]["hypothesis"]
                    provenance = {"search": packet}
                    model = digest([packet["encoder_sha256"], packet["decoder_sha256"]])
                try:
                    self.admit(
                        candidate["source"],
                        visible,
                        provenance=provenance,
                        model_sha256=model,
                        runner=runner,
                        request_id=f"round-{self.round}-{number}-{len(self.admissions)}",
                    )
                except ValueError:
                    # A visible fit can fail the independent continuation; it
                    # stays in the search report and never enters replay.
                    candidate["admission"] = "continuation_rejected"
            reports.append(report)
        return {
            "round": self.round,
            "archive_root_sha256": self.snapshot["root_sha256"],
            "training": training,
            "search": reports,
            "admissions": list(self.admissions),
        }

    def state_dict(self):
        return {
            "version": "self-training-round/v1",
            "round": self.round,
            "truth_sha256": self.truth.identity,
            "snapshot": deepcopy(self.snapshot),
            "replay": self.replay.state_dict(),
            "admissions": list(self.admissions),
        }

    def load_state_dict(self, state):
        if (
            set(state) != {"version", "round", "truth_sha256", "snapshot", "replay", "admissions"}
            or state["version"] != "self-training-round/v1"
            or state["truth_sha256"] != self.truth.identity
            or type(state["round"]) is not int
            or state["round"] < 0
        ):
            raise ValueError("incompatible training round")
        replay = FrozenReplay(self.archive, state["snapshot"])
        replay.load_state_dict(state["replay"])
        for key in state["admissions"]:
            self.archive.get(key)
        self.round, self.snapshot, self.replay = state["round"], deepcopy(state["snapshot"]), replay
        self.admissions = list(state["admissions"])
