from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)
from oeis_learn.pipeline_validation import progressive_round028_selector as selector
from oeis_learn.pipeline_validation import progressive_round028_pilot as pilot
from oeis_learn.pipeline_validation.progressive_round028_replay import replay_decision
from oeis_learn.generalization.progressive.program_search_selector import fit_ranker
from tests.contract.test_round028_routing import sample


def test_replay_budget_and_one_decisive_prefix(tmp_path):
    votes = sample(["P", "NSP"], replay=True)
    charged = []

    def run(row):
        path = tmp_path / (row["task"] + ".json")
        if not path.exists():
            charged.append(row["task"])
            atomic_write(path, seal(dict(mask="P", elapsed=7)))
        old = load(path)
        return {"feasible": old["mask"]}, old["elapsed"]

    smaller = replay_decision(votes, True, 7, run)
    assert smaller["deficit_us"] == 1 and not charged
    for _ in range(2):
        result = replay_decision(votes, True, 8, run)
        assert result["decision"]["action"] == "MATCHED_GENERATOR"
        assert result["elapsed_us"] == 7 and result["original_service_us"] == 8
    assert len(charged) == 1


def test_remaining_endpoint_price_accounts_for_new_and_cached_roles():
    from oeis_learn.generalization.progressive.program_search_timing import remaining_price

    prices = {("arm-0", "train"): 13, ("arm-0", "score"): 20, ("arm-4000000", "search"): 80}
    assert remaining_price(prices, {("arm-0", "train")}, ("arm-0", "score"), 19) == 99
    assert 19 <= 50 < 99


def test_resumed_qualification_reconciles_the_last_microsecond(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation.progressive_pilot_funding import finish_role

    monkeypatch.setattr(pilot, "SHIFT_US", 100)
    monkeypatch.setattr(pilot, "accepted_scale", lambda *args: {})
    monkeypatch.setattr(pilot, "require_resident_measurements", lambda *args: None)
    registration, acceptance = seal({"candidate": "a"*40}), seal({"checks": "fixture"})
    prices = {("ordinary-0", "train"): 90, ("ordinary-0", "score"): 20}
    def qualify(area):
        return pilot.qualification_result(area, registration, acceptance, prices, "a"*40, {})[0]
    empty = tmp_path / "empty"
    (empty / "reservations").mkdir(parents=True)
    with pytest.raises(pilot.PilotDeficit) as exc:
        qualify(empty)
    assert exc.value.counts["deficit_us"] == 10
    for spent in (79, 81):
        area = tmp_path / str(spent)
        atomic_write(area / "reservations/ordinary-0-train.json", seal(dict(key="ordinary-0/train/0", us=100, estimate_us=90)))
        atomic_write(area / "costs/ordinary-0-train.json", seal(dict(block="ordinary-0", role="train",
                     charged_us=spent, wall_us=spent, lease_us=0)))
        finish_role(area, registration["identity"], "ordinary-0", "train")
        atomic_write(area / "reservations/ordinary-0-score.json", seal(dict(key="ordinary-0/score/0", us=100-spent, estimate_us=20)))
        if spent == 79:
            result = qualify(area)
            assert result["projected_us"] == 99 and result["scientific_status"] == "PILOT_INCOMPLETE"
            assert result["balance"]["opened"] == [["ordinary-0", "score", "unstarted", 20]]
        else:
            with pytest.raises(pilot.PilotDeficit) as exc:
                qualify(area)
            assert exc.value.counts["deficit_us"] == 1


def test_paid_producer_space_and_cached_role(tmp_path):
    from oeis_learn.pipeline_validation.progressive_pilot_capacity import capacity

    area = tmp_path / "private"
    role = area / "roles/key"
    out = role / "attempt/output"
    out.mkdir(parents=True)
    file = out / "candidates.json"
    file.write_bytes(b"z" * 4096)
    data = seal(dict(block="ordinary-0"))
    request = seal(dict(role="search", data=data["identity"]))
    atomic_write(role / "data.json", data)
    atomic_write(role / "request.json", request)
    req = request["identity"]
    atomic_write(
        out / "producer.json", seal(dict(request=req, result={"candidates": reference(out, file)}))
    )
    atomic_write(
        role / "boundary.json",
        seal(dict(request=req, producer=reference(role, out / "producer.json"))),
    )
    scale = dict(candidate_score_bytes=8000, campaign_bytes=2000)
    host = SimpleNamespace(policy={"free_disk_floor_bytes": 1000})

    def measure(p):
        return dict(free_bytes=7 * 100 + 2 * 10000 + 2000 + 1000 - 4096, free_inodes=10683)

    arg = dict(scale=scale, share=tmp_path, cached=False, measure=measure, verified={})
    (row,) = capacity(area, host, 100, **arg)
    assert row["required_bytes"] > measure(area)["free_bytes"]
    one = {("ordinary-0", "score")}
    many = one | {(arm + "-" + str(target), "score") for arm in ("ordinary", "first_divergence")
                  for target in (0, 1_000_000, 4_000_000)} | {("incumbent-4000000", "score")}
    measured = dict(scale, resident={"cases": 160, "candidates_bytes": 65536, "scored_bytes": 8192})
    simple = dict(arg, scale=measured, outstanding=one)
    limited = capacity(area, host, 100, **simple)[0]
    available = lambda p: dict(free_bytes=limited["required_bytes"], free_inodes=limited["required_inodes"])
    assert capacity(area, host, 100, **dict(simple, measure=available))[0]["deficit_bytes"] == 0
    deficit = capacity(area, host, 100, **dict(simple, measure=available, outstanding=many))[0]
    assert deficit["deficit_bytes"] == 6 * 8192 and deficit["deficit_inodes"] > 0
    control = dict(measured, resident=dict(measured["resident"], score_control={"scored_bytes": 9000}))
    priced = capacity(area, host, 100, **dict(simple, scale=control))[0]
    assert priced["required_bytes"] - limited["required_bytes"] == 2 * (9000 - 8192)
    cached_search = capacity(area, host, 100, **dict(simple, outstanding=one | {("ordinary-0", "search")}))[0]
    assert cached_search == limited
    assert not capacity(area, host, 100, **dict(arg, cached=True))
    (row,) = capacity(
        area, host, 100, **dict(arg, measure=lambda p: dict(free_bytes=0, free_inodes=0))
    )
    assert row["deficit_bytes"] > 0 and row["deficit_inodes"] > 0
    file.write_bytes(b"x" * 4096)
    with pytest.raises(ValueError, match="identity changed"):
        capacity(area, host, 100, **arg)


def test_storage_charges_the_roles_and_share_filesystems_separately(tmp_path, monkeypatch):
    import os
    from pathlib import Path
    from oeis_learn.pipeline_validation.progressive_pilot_capacity import capacity

    area, share = tmp_path / "private", tmp_path / "share"
    roles = area / "roles"
    roles.mkdir(parents=True)
    share.mkdir()
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        original = original_stat(path, *args, **kwargs)
        if path in (roles, share):
            fields = list(original)
            fields[2] = 123456 if path == roles else 123457
            return os.stat_result(fields)
        return original
    monkeypatch.setattr(Path, "stat", stat)
    records = capacity(area, SimpleNamespace(policy={"free_disk_floor_bytes": 10}), 100,
                       scale={"candidate_score_bytes": 90, "campaign_bytes": 40}, share=share,
                       cached=False, measure=lambda p: {"free_bytes": 100, "free_inodes": 1000},
                       outstanding={("ordinary-0", "train")})
    by_device = {row["device"]: row for row in records}
    assert by_device[123456]["required_bytes"] == 210
    assert by_device[123456]["deficit_bytes"] == 110
    assert by_device[123457]["required_bytes"] == 50


def test_nested_choice_and_missing_checkpoint_break_private_reexport(tmp_path):
    area = tmp_path
    for f in [
        "registration.json",
        "freeze.json",
        "public-freeze.json",
        "acceptance.json",
        "accounting/budget-binding.json",
        "accounting/budget.jsonl",
    ]:
        atomic_write(area / f, {})
    role = area / "roles/key"
    output = role / "attempt-0000/output"
    score = output / "scored/scores.json"
    atomic_write(score, seal(dict(exact=True)))
    atomic_write(output / "producer.json", seal(dict(result={"scores": reference(output, score)})))
    atomic_write(
        role / "boundary.json", seal(dict(producer=reference(role, output / "producer.json")))
    )
    prior = pilot.publication_inputs(area)
    nested = output / "checkpoint/state.bin"
    nested.parent.mkdir()
    nested.write_bytes(b"checkpoint")
    with_checkpoint = pilot.publication_inputs(area)
    nested.unlink()
    assert prior == pilot.publication_inputs(area) != with_checkpoint
    score.write_bytes(b"{}")
    with pytest.raises(ValueError):
        pilot.publication_inputs(area)


@pytest.mark.parametrize(("live", "action"), [("F", "RETAIN_REFERENCE"),
                                               ("U", "UNRESOLVED_SPENDING_CONTRAST"),
                                               ("S", "REPLICATION_UNFUNDED")])
def test_selector_budget_and_shuffled_control_govern_spending(live, action):
    from oeis_learn.generalization.progressive.program_search_selector import selector_spending

    def cells(stratum, outcomes, group):
        return dict(stratum=stratum, group=group, counts={outcomes: 2})

    # Incumbent, learned and shuffled on one list, followed by three independently timed arms.
    rows = [cells(s, "FSFF" + live + "F", i) for i, s in enumerate(("composition", "unseen_structure"))]
    rows.append(cells("retention", "SSSSSS", 2))
    decision = selector_spending(rows, comparable_cost=True)
    assert decision["action"] == action
    assert decision["contrasts"]["same_bank"]["gain"] == 1.0
    assert selector_spending(rows, comparable_cost=False)["action"] == "INCOMPARABLE"
    rows[0]["counts"] = rows[1]["counts"] = {"FSSF" + live + "F": 2}
    assert selector_spending(rows, comparable_cost=True)["action"] == "RETAIN_REFERENCE"


def test_selector_compares_same_bank_then_scores_sealed_choices(tmp_path, monkeypatch):
    from oeis_learn.fresh import bigint_diagnostics, synthesis_policy
    from oeis_learn.synthesis.admission import source_identity
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.reference import evaluate

    const = "(program (return (int 0)))"
    index = "(program (return $index))"
    training = [dict(case="zero", values=["0"] * 20, positive=const, negative=index)]
    model = fit_ranker(training)
    from oeis_learn.synthesis.admission import VisiblePrompt

    def envelope(body, prompt):
        candidate = dict(
            source=body,
            source_sha256=source_identity(body),
            verified=True,
            visible_count=20,
            cost=0.0,
        )
        return {
            "selected": synthesis_policy.select_visible([candidate]),
            "prompt_sha256": VisiblePrompt(
                tuple(prompt["indices"]), tuple(prompt["values"])
            ).sha256,
        }

    def portfolio(decoder, memory, prompt, recipe, client, **kwargs):
        bank = [envelope(const if prompt["values"][1] == "0" else index, prompt)]
        return {
            "selector_candidates": {"equal_budget": bank},
            "equal_budget": bank[0],
            "arm_evidence": {"equal_budget": {"complete": True}},
            "details": [{}] * 8,
        }

    monkeypatch.setattr(bigint_diagnostics, "portfolio", portfolio)
    monkeypatch.setattr(
        "oeis_learn.generalization.progressive.baselines.load_baseline",
        lambda *a, **kw: (None, None, {}),
    )
    monkeypatch.setattr(bigint_diagnostics, "model_snapshot", lambda *a: digest("weights"))

    class Client:
        limits = ResourceProfile()

        def execute(self, src, indices, **kw):
            return dict(outcome="ok", values=list(evaluate(src, indices).values))

    class Role:
        last_wall_us = 10

        def __call__(self, role, data, **opts):
            if role == "selector_score":
                assert (tmp_path / "selections.json").is_file()
                return selector.selector_score(data, Client())
            assert "truth" not in data
            return selector.selector_choice(data, Client())

    prefix = {"id": "fixture", "indices": list(map(str, range(20))), "values": ["0"] * 20}
    truth = dict(prefix, indices=list(map(str, range(100))), values=["0"] * 20 + ["1"] * 80)
    correct = "(program (return (select (lt $index (int 20)) (int 0) (int 1))))"
    names = (*selector.ARM_NAMES, *selector.LIVE_NAMES)
    for body, closed, opened in [(correct, "S", "S"), (const, "F", "U"), (None, "F", "U")]:
        for complete, expected in [(True, closed), (False, opened)]:
            row = dict(chosen=envelope(body, prefix) if body else None, complete=complete, comparable=True)
            scored = selector.selector_score(dict(truth=truth, selections=dict.fromkeys(names, row)), Client())
            assert scored["outcomes"] == dict.fromkeys(names, expected)

    keys = ["zero", "index"]
    roster = [dict(task=k, group=k, stratum="composition") for k in keys]
    idx = list(map(str, range(100)))
    witnesses = {
        k: dict(id=k, indices=idx, values=["0"] * 100 if k == "zero" else idx) for k in keys
    }
    recipe = {"model": {"seed": 1}}
    reg = seal(dict(seed=3, proposer="template_only", reference={"method": "retained_D"}))
    report = selector.compare_selector(
        tmp_path,
        reg,
        roster,
        witnesses,
        model,
        Role(),
        dict(template_only=recipe, retained_D=recipe),
        dict(template_only=(None, []), retained_D=(None, [])),
    )
    report_arms = report["arms"]
    assert all(x == {"S": 2} for x in report_arms.values())
    for item in load(tmp_path / "selections.json")["choices"].values():
        assert len({item[k]["bank"] for k in selector.ARM_NAMES}) == 1

    # Enter the same comparison through the installed round028 run dispatcher.
    from pathlib import Path
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.pipeline_validation import (
        progressive_round028 as command,
        progressive_round028_audit,
    )
    from oeis_learn.pipeline_validation import (
        progressive_panel_sources,
        progressive_round028_support,
    )
    from oeis_learn.pipeline_validation import (
        progressive_round028_panels,
        progressive_round028_replay,
    )
    from tests.integration.test_round028_audit import install_donor_fixture
    from tests.unit.test_round028_training import training_rows

    work = tmp_path / "work"
    work.mkdir()
    registration = install_donor_fixture(work, monkeypatch)
    owner = work / command.SESSION
    owner.mkdir(parents=True)
    with RunStore.owner(owner):
        stored = RunStore(owner, registration, diagnostic=True)
        command.audit_command(work, owner, stored, registration)
    report = dict(receipt={"report": digest("cut")}, decision={"action": "CPU_SELECTOR"})
    monkeypatch.setattr(command, "audit_command", lambda *a: report)
    monkeypatch.setattr(
        command,
        "metadata",
        lambda *a: dict(
            qualified_checkout=True,
            files={command.ORIGINAL + "/prepare-research/frozen/campaign.json": True},
        ),
    )
    monkeypatch.setattr(command, "candidate_revision", lambda *a: registration["candidate"])
    monkeypatch.setattr(command, "learned_search_policy", lambda: registration["policy"])
    atomic_write(work / "base.json", recipe)
    entries = [
        dict(name=name, artifact=reference(work, work / "base.json"))
        for name in ("recipe", "retained-D-evaluation-weights")
    ]
    donor = SimpleNamespace(
        policy={"campaign": digest("donor")},
        saved=lambda *a, **kw: {"payload": {"references": entries, "tasks": []}},
        replay_index=lambda: {},
    )
    monkeypatch.setattr(progressive_round028_audit, "Donor", lambda *a: donor)
    monkeypatch.setattr(
        progressive_round028_support, "training_package", lambda *a: {"shards": [training_rows()], "layout": {"probes": {}}}
    )
    monkeypatch.setattr(
        command,
        "reopen",
        lambda *a: {
            "producers": {},
            "cases": [dict(stratum="composition", group="g", observations={})],
        },
    )
    monkeypatch.setattr(progressive_panel_sources, "read_inputs", lambda *a, **kw: {})

    def prepare(root, area, inputs, reg, native):
        members = [dict(id=r["task"], group=r["group"], stratum=r["stratum"]) for r in roster]
        panels = dict(
            development=members, development_retention=[], replication=[], replication_retention=[]
        )
        atomic_write(
            area / "supply.json",
            seal(dict(allocations=dict(deficits={}, panels=panels), witnesses=witnesses)),
        )
        return {"calibrated": True}

    monkeypatch.setattr(progressive_round028_panels, "prepare", prepare)
    from oeis_learn.pipeline_validation import progressive_round028_pilot as pilot
    from oeis_learn.generalization.progressive.training_rows import matched_stream
    def prepare_training(root, area, reg, cpu, refs, supply, panel_area, *, pair=0):
        rows = training_rows()
        stream = matched_stream([rows], {"probes":{}}, reg["identity"], pair)
        return dict(rows=rows, stream=stream)
    monkeypatch.setattr(pilot, "prepare_training", prepare_training)
    monkeypatch.setattr(progressive_round028_replay, "DiagnosticRole", lambda *a: Role())
    args = SimpleNamespace(action="run", workspace=str(work), archive=None)
    outcome, code = command.execute(args, Path(__file__).resolve().parents[2])
    assert (
        code == 0
        and outcome["scientific_status"] == "SELECTOR_EXPLORATORY"
        and outcome["result"]["arms"] == report_arms
    )
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import verify_transport
    assert verify_transport(owner / outcome["archive"]) == outcome["receipt"]
    args.action = "verify"
    assert command.execute(args, Path(__file__).resolve().parents[2])[0]["private_replay"] == "VERIFIED"
    from oeis_learn.generalization.progressive.program_search_outcome import archive_bytes
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode
    tampered = seal(dict(outcome["result"], spending={"action":"RUN_RESERVED_REPLICATION"}))
    copied = tmp_path / "tampered.zip"
    copied.write_bytes(archive_bytes({"selector.json":encode(tampered)}))
    with pytest.raises(ValueError, match="spending"):
        verify_transport(copied)


def test_installed_audit_reserves_and_reopens_native_replay(tmp_path, monkeypatch):
    from pathlib import Path
    from oeis_learn.generalization.progressive.program_search_routing import project, route
    from oeis_learn.generalization.selector_fit_replay import original_attempt
    from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.pipeline_validation import progressive_round028 as commands
    from oeis_learn.pipeline_validation import (
        progressive_handoff,
        progressive_round028_panels as panels,
    )
    from oeis_learn.pipeline_validation import progressive_round028_rehearsal as rehearsal
    from oeis_learn.pipeline_validation.progressive_round028_replay import replay_attempts
    from tests.integration.test_round028_audit import install_donor_fixture
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.pipeline_validation.progressive_round028_timing import original_channels

    reg = install_donor_fixture(tmp_path, monkeypatch)
    folder = tmp_path / commands.SESSION
    folder.mkdir(parents=True)
    with RunStore.owner(folder):
        store = RunStore(folder, reg, diagnostic=True)
        store.append(dict(store.state, registration=reg["identity"]))
        cut = commands.audit(commands.Donor(tmp_path, folder, reg))
    primary = sorted(
        (r for r in cut["cases"] if r["stratum"] in ("composition", "unseen_structure")),
        key=lambda r: r["task"],
    )
    votes = sample(["P", "NSP"])
    for case, vote in zip(primary, votes, strict=True):
        case["diagnosis"] = dict(vote, task=case["task"], stratum=case["stratum"],
                                 search_closed=vote["feasible"] == "P", exact_witness=vote["feasible"] == "P")
    case = primary[1]
    prompts, _ = original_channels(tmp_path, [case])
    prompt = VisiblePrompt(tuple(prompts[0]["indices"]), tuple(prompts[0]["values"]))
    body = "(program (return (int 0)))"

    class Client:
        limits = ResourceProfile()

        def runtime_identity(self, profile):
            return digest("runtime")

        def fit(self, source, visible, **kwargs):
            assert source == body and visible == prompt and kwargs["seed"] == 39
            return dict(complete=True, admitted=[])

    draw = dict(
        position=0,
        seed=39,
        source=body,
        source_sha256=source_identity(body),
        generation=dict(complete=True),
        fitting=dict(outcome="resource_exhausted", clock=dict(admitted=0.0, deadline=1.0)),
        native=dict(
            evidence=dict(
                prompt_sha256=prompt.sha256,
                source_sha256=source_identity(body),
                resource_sha256=Client.limits.identity,
                runtime_sha256=digest("runtime"),
            )
        ),
    )
    expected = original_attempt(case["task"], draw, 0, "resource_exhausted", prompt)
    case["diagnosis"].update(
        replay=dict(ordinals=[0], cost_us=1_000_000, resolves="NP"),
        replay_inputs=[
            {
                k: expected[k]
                for k in (
                    "position",
                    "seed",
                    "original_clock",
                    "service_seconds",
                    "prompt_sha256",
                    "source_sha256",
                    "original_resource",
                )
            }
        ],
        eligible_verdicts={},
        missing_sources=[],
        exact_witness=True,
    )
    case["observations"]["trial-05-1M"]["companion"] = "failure"
    path = tmp_path / "offered.json"
    atomic_write(
        path,
        seal(
            dict(
                version="discovery-candidates/v1",
                records={case["task"]: dict(costs=dict(correct=dict(details=[draw], equal_budget=None,
                                                                  archive={"omitted": [], "retained": []})))},
            )
        ),
    )
    cut["producers"]["trial-05-1M"]["generation"] = reference(tmp_path, path)
    cut["support"].update(eligible_rows=1, status="measured", eligibility="executable-training/v1")
    initial = route(
        project([r["diagnosis"] for r in primary], training_support=True, replay_budget_us=None)
    )
    assert initial["blocker"] == "REPLAY_RESERVATION_UNMEASURED"
    cut["projection"] = project([r["diagnosis"] for r in primary], training_support=True, replay_budget_us=None)
    cut["decision"] = initial
    cut["methods"], cut["pairs"] = commands.summarize(cut["cases"], list(cut["methods"]))
    monkeypatch.setattr(commands, "audit", lambda *a, **kw: cut)
    monkeypatch.setattr(
        commands,
        "metadata",
        lambda *a: dict(
            qualified_checkout=True,
            files={commands.ORIGINAL + "/prepare-research/frozen/campaign.json": True},
        ),
    )
    monkeypatch.setattr(commands, "candidate_revision", lambda *a: reg["candidate"])
    monkeypatch.setattr(commands, "learned_search_policy", lambda: reg["policy"])
    image = digest("cpu")
    monkeypatch.setattr(panels, "candidate_image", lambda *a: (image, 0, True))
    monkeypatch.setattr(rehearsal, "physical_scope", lambda *a, **kw: digest("physical"))
    runs = []

    def stage(host, source, image, output, argv, *, gpu, seconds, readonly_inputs):
        assert not gpu
        mounts = {d: p for p, d in readonly_inputs}
        req = load(mounts["/work/inputs/request.json"])
        data = load(mounts["/work/inputs/data.json"])
        assert (
            set(data["prompt"]) == {"id", "indices", "values"}
            and len(data["prompt"]["values"]) == 20
        )
        result = replay_attempts(data, Client())
        runs.append(req["identity"])
        output.mkdir()
        atomic_write(
            output / "producer.json",
            seal(
                dict(
                    version=panels.NativeRoles.producer_version,
                    request=req["identity"],
                    native=digest("runtime"),
                    result=result,
                    elapsed_us=10,
                )
            ),
        )
        atomic_write(
            output / "container-inspect.json",
            [
                dict(
                    Image=image,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                    Mounts=[
                        dict(Type="bind", Source=str(p), Destination=d, RW=d == "/work/output")
                        for d, p in dict(mounts, **{"/work/output": output}).items()
                    ],
                )
            ],
        )

    monkeypatch.setattr(progressive_handoff, "stage", stage)
    host = SimpleNamespace(
        engine=SimpleNamespace(name="podman", image=lambda x: x),
        policy={"free_disk_floor_bytes": 0},
    )
    args = SimpleNamespace(
        action="audit", workspace=str(tmp_path), archive=None, replay_budget_seconds=1
    )
    source = Path(__file__).resolve().parents[2]
    first, code = commands.execute(args, source, host_factory=lambda *a, **kw: host)
    assert code == 0 and first["decision"]["action"] == "MATCHED_GENERATOR"
    args.replay_budget_seconds = None
    assert commands.execute(args, source, host_factory=lambda *a, **kw: host) == (first, 0)
    assert len(runs) == 1 and first["diagnostic_replay"]["original_service_us"] == 1_000_000
    args.action = "verify"
    private, code = commands.execute(args, source)
    assert code == 0 and private["public_recomputation"] == first["receipt"]
    copied = tmp_path / "copied.zip"
    copied.write_bytes((folder / first["archive"]).read_bytes())
    args.archive = str(copied)
    assert commands.execute(args, source) == (first["receipt"], 0)
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import audit_context
    assert audit_context(copied)[0] == first["receipt"]
    assert first["receipt"]["original_service_us"] == 1_000_000 and first["receipt"]["elapsed_us"] > 0
    assert len(runs) == 1
    import io
    import json
    from oeis_learn.generalization.progressive.program_search_outcome import archive_bytes, read_canonical
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode, verify_transport
    contents = read_canonical(copied, ("replay.json", "round028-audit.zip"))
    child = json.loads(contents["replay.json"])
    child["observations"][0]["after"] = "S"
    with pytest.raises(ValueError, match="replay"):
        verify_transport(io.BytesIO(archive_bytes(dict(contents, **{"replay.json": encode(seal(child))}))))


def test_historical_reference_weights_primary_strata_equally():
    cases = []
    for stratum in ("composition", "unseen_structure"):
        for i in range(32):
            group = i // 2 if stratum == "composition" else i
            observations = {name: {"frozen": "failure"} for name in
                            ("retained_D", *("retained_D/" + k for k in selector.CONVENTIONAL))}
            if stratum == "composition":
                observations["retained_D"]["frozen"] = "success"
            elif i < 24:
                observations["retained_D/template"]["frozen"] = "success"
            cases.append(dict(stratum=stratum, group=group, observations=observations))
    result = selector.historical_reference({"cases": cases}, trial_eligible=False)
    assert result["method"] == "retained_D" and not result["ambiguous"]
    assert result["bounds"]["retained_D"] == ["1/2", "1/2", 0]
    assert result["bounds"]["template"] == ["3/8", "3/8", 0]


def test_historical_reference_uses_family_macro_and_ambiguous_fallback():
    cases = [dict(stratum="composition", group="A", observations={}) for _ in range(3)]
    cases.extend(
        dict(stratum="unseen_structure", group=group, observations={}) for group in ("B", "C")
    )
    for row in cases:
        for name in ["retained_D", *("retained_D/" + k for k in selector.CONVENTIONAL)]:
            row["observations"][name] = {"frozen": "failure"}
        name = "retained_D" if row["group"] == "A" else "retained_D/polynomial"
        row["observations"][name] = {"frozen": "success"}
    assert (
        selector.historical_reference({"cases": cases}, trial_eligible=False)["method"]
        == "polynomial"
    )
    for row in cases:
        row["observations"].pop("retained_D/template")
    assert selector.historical_reference({"cases": cases}, trial_eligible=False)["ambiguous"]
