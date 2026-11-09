"""Finite campaign transitions; worker identity is owned by the existing supervisor."""

from contextlib import contextmanager
from pathlib import Path

from .campaign import validate_campaign, validate_task
from .contracts import digest, load, record, seal


class RunStore:
    def __init__(self, root, campaign):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.campaign = validate_campaign(campaign)
        binding = self.root / "campaign.json"
        if binding.exists():
            if load(binding,maximum=32<<20) != self.campaign:
                raise ValueError("run campaign changed")
        else:
            from .storage import journal_write

            journal_write(self.root, binding, self.campaign)
        self.read()

    @contextmanager
    def lease(self):
        import fcntl

        path = self.root / "run.lock"
        if path.is_symlink():
            raise ValueError("run lease alias")
        with path.open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("another invocation owns this campaign") from None
            self.read()
            yield

    def read(self):
        self.state = {"tasks": {}, "status": "FROZEN", "invocation": None, "decisions": {}}
        tip = self.campaign["identity"]
        paths = sorted(self.root.glob("transition-*.json"))
        if len(paths) > 4096:
            raise ValueError("finite campaign transition limit")
        for i, p in enumerate(paths):
            v = load(p)
            if p.name != f"transition-{i:04d}.json" or seal(v) != v or v["previous"] != tip:
                raise ValueError("invalid run transition")
            self.state = v["state"]
            tip = v["identity"]
        self.tip = tip
        self.ordinal = len(paths)
        return self.state

    def append(self, state):
        if self.ordinal >= 4096:
            raise ValueError("transition allowance exhausted")
        state = dict(state, choices=state.get('choices',self.state.get('choices',{})), outputs=state.get("outputs",self.state.get("outputs",{})), decisions=state.get("decisions", self.state.get("decisions", {})))
        from .storage import journal_write

        journal_write(
            self.root,
            self.root / f"transition-{self.ordinal:04d}.json",
            seal({"previous": self.tip, "state": state}),
        )
        return self.read()

    def next_task(self):
        if (self.root / "assessment-exposure.json").exists():
            raise ValueError("assessment exposed; adaptive continuation forbidden")
        nodes = {n["id"]: n for n in self.campaign["payload"]["graph"]["nodes"]}
        for original in self.campaign["payload"]["tasks"]:
            plan=self.resolved_plan(original)
            if plan is None:continue
            status = self.state["tasks"].get(plan["id"])
            if status in {"COMPLETE", "FAILED", "PARTIAL",'SKIPPED'}:
                continue
            node = nodes[plan["slot"]]
            deps = [
                p for p in self.campaign["payload"]["tasks"] if p["slot"] in node["dependencies"]
            ]
            if set(node["dependencies"]) - {p["slot"] for p in deps}:
                continue
            terminal = {"COMPLETE", "PARTIAL", "FAILED"}
            if any(self.state["tasks"].get(p["id"]) not in terminal for p in deps):
                continue
            if plan["initializer"].get("parent_task") and self.state["tasks"].get(
                plan["initializer"]["parent_task"]
            ) not in {"COMPLETE", "PARTIAL"}:
                continue
            edges = [
                e for e in self.campaign["payload"]["graph"]["edges"] if e["target"] == plan["slot"]
            ]
            if any(
                self.state["tasks"].get(p["id"]) != "COMPLETE"
                and not any(
                    e["source"] == p["slot"] and e["condition"] != "COMPLETE" for e in edges
                )
                for p in deps
            ):
                continue
            certificates = self.state.get("decisions", {})
            if any(e['action'] in {'RETAIN_BASELINE','DEFER_BRANCH','STOP_AFFECTED'} for e in edges):continue
            if any(
                (
                    any(
                        self.state["tasks"].get(p["id"]) != "COMPLETE"
                        for p in deps
                        if p["slot"] == e["source"]
                    )
                    if e["condition"] == "COMPLETE"
                    else not any(
                        v["source_slot"] == e["source"]
                        and v["target_slot"] == e["target"]
                        and v["verdict"] == e["condition"]
                        and v["action"] == e["action"]
                        for v in certificates.values()
                    )
                )
                for e in edges
            ):
                continue
            return record(
                "task",
                dict(plan, campaign=self.campaign["identity"]),
                created_at=self.campaign["created_at"],
                purpose=self.campaign["purpose"],
                track=self.campaign["track"],
            )
        return None

    def authorize(
        self, bundle, *, policy, remaining, panels=None, designs=None, evidence_root=None
    ):
        from .contracts import validate
        from .decisions import verify
        from .targets import validate_policy

        validate_policy(policy)
        if policy["identity"] != self.campaign["payload"]["target_policy"]:
            raise ValueError("decision target policy is not the frozen campaign policy")
        fields = {
            "version",
            "identity",
            "source_slot",
            "target_slot",
            "decision",
            "rows",
            "roster",
            "seeds",
            "design",
            "kwargs",
        }
        if (
            set(bundle) != fields
            or bundle["version"] != "progressive-decision-bundle/v1"
            or seal(bundle) != bundle
        ):
            raise ValueError("closed producer decision bundle required")
        key = digest([bundle["source_slot"], bundle["target_slot"]])
        prior = self.state.get("decisions", {}).get(key)
        if prior is not None:
            if prior.get("bundle") != bundle["identity"]:
                raise ValueError("conflicting decision edge")
            return prior
        training_seeds = sorted(
            {
                p["seed"]
                for p in self.campaign["payload"]["tasks"]
                if p["slot"] == bundle["source_slot"] and p["kind"] == "training"
            }
        )
        if (
            training_seeds
            and not bundle["kwargs"].get("confirmation")
            and sorted(bundle["seeds"]) != training_seeds
            and not (bundle['decision']['payload']['next_action'] in {'RETAIN_BASELINE','DEFER_BRANCH','STOP_AFFECTED'} and set(training_seeds)<=set(bundle['seeds']))
        ):
            raise ValueError("decision seed roster differs from frozen source tasks")
        decision = bundle["decision"]
        if (
            decision["purpose"] != self.campaign["purpose"]
            or decision["track"] != self.campaign["track"]
        ):
            raise ValueError("cross-purpose decision")
        edge = next(
            (
                e
                for e in self.campaign["payload"]["graph"]["edges"]
                if e["source"] == bundle["source_slot"]
                and e["target"] == bundle["target_slot"]
                and e["condition"] == decision["payload"]["evidence_verdict"]
                and e["action"] == decision["payload"]["next_action"]
            ),
            None,
        )
        if edge is None:
            no_work=decision['payload']['next_action'] in {'RETAIN_BASELINE','DEFER_BRANCH','STOP_AFFECTED'}
            nodes={n['id']:n for n in self.campaign['payload']['graph']['nodes']}
            if not no_work or bundle['target_slot'] not in nodes or bundle['source_slot'] not in nodes[bundle['target_slot']]['dependencies']:
                raise ValueError("decision lacks a frozen edge")
            edge=dict(action=decision['payload']['next_action'])
        if edge["action"] == "ADVANCE_REGISTERED" and any(
            self.state["tasks"].get(p["id"]) != "COMPLETE"
            for p in self.campaign["payload"]["tasks"]
            if p["slot"] == bundle["source_slot"] and p["kind"] == "training"
        ):
            raise ValueError("incomplete source learning cannot authorize confirmed expansion")
        kwargs = bundle["kwargs"]
        if set(kwargs) - {"target_id", "action", "trigger", "costs", "confirmation", "runtime"}:
            raise ValueError("closed decision arguments required")
        if edge["action"] != "EXPLORE_RESERVED" or self.campaign['purpose']=='research':
            from .decisions import verify_gate_producers

            verify_gate_producers(
                self.root if evidence_root is None else evidence_root, self.campaign, bundle,store=self
            )
        observation=kwargs.get('runtime')
        if observation is not None:
            from .contracts import verify_refs
            import json
            matches=[r['artifact'] for r in self.campaign['payload']['references'] if r['artifact']==observation.get('producer')]
            if len(matches)!=1:raise ValueError('original frozen runtime producer required')
            original=json.loads(verify_refs(self.root if evidence_root is None else evidence_root,matches)[0])
            if original.get('measurements')!=observation.get('measurements') or seal(original)!=original:
                raise ValueError('runtime measurements differ from original producer')
        visited = [
            v["failure_class"]
            for v in self.state.get("decisions", {}).values()
            if v.get("failure_class")
        ]
        verify(
            decision,
            bundle["rows"],
            bundle["roster"],
            bundle["seeds"],
            bundle["design"],
            policy=policy,
            remaining=remaining,
            visited=visited,
            **kwargs,
        )
        confirmation = kwargs.get("confirmation")
        if confirmation:
            original = confirmation["original"]["decision"]
            registration = confirmation["registration"]
            registered_branch = next(
                b for b in policy["payload"]["diagnostics"] if b["id"] == registration["branch"]
            )
            if not any(
                v["decision"] == original["identity"]
                and v.get("failure_class") == registered_branch["failure_class"]
                for v in self.state.get("decisions", {}).values()
            ):
                raise ValueError("original diagnostic decision must remain in the run chain")
            third = [
                p
                for p in self.campaign["payload"]["tasks"]
                if p["slot"] in registered_branch["task_slots"] and p["kind"] == "training"
            ]
            if (
                not third
                or {p["seed"] for p in third} != {registration["third_seed"]}
                or any(self.state["tasks"].get(p["id"]) != "COMPLETE" for p in third)
            ):
                raise ValueError("frozen third-seed diagnostic task must be complete")
        branch = None
        if edge["action"] == "EXPLORE_RESERVED":
            target = next(t for t in policy["payload"]["targets"] if t["id"] == kwargs["target_id"])
            branch = next(
                d for d in policy["payload"]["diagnostics"] if d["id"] == target["budget_ref"]
            )
            if branch["signal"] == "seed_disagreement":
                registration = (designs or {}).get(branch["design"], {})
                if (
                    registration.get("original_design") != digest(bundle["design"])
                    or registration.get("branch") != branch["id"]
                    or registration.get("original_seeds") != bundle["seeds"]
                ):
                    raise ValueError("original frozen third-seed design required")
            elif branch["design"] != digest(bundle["design"]):
                raise ValueError("diagnostic comparison design differs from frozen registration")
            if bundle["target_slot"] not in branch["task_slots"] or branch["signal"] not in {
                "heldout_yield",
                "seed_disagreement",
                "runtime_bottleneck",
                "support_deficit",
            }:
                raise ValueError("diagnostic lacks frozen held-out/runtime task reservation")
            if panels is None or branch["panel"] not in panels:
                raise ValueError("original reserved diagnostic panel required")
            panel = validate("cohort", panels[branch["panel"]])
            if (
                panel["payload"]["role"] in {"assessment", "training"}
                or panel["track"] != self.campaign["track"]
            ):
                raise ValueError("diagnostic cannot consume assessment/training truth")
            members = {r["id"]: r for r in panel["payload"]["rows"]}
            if any(
                r["task"] not in members
                or r["stratum"] != members[r["task"]]["stratum"]
                or r["group"] not in members[r["task"]]["groups"]
                for r in bundle["roster"]
            ) or {r["task"] for r in bundle["roster"]} != set(members):
                raise ValueError("diagnostic outcomes must cover the exact reserved panel roster")
            if any(
                v.get("panel") == panel["identity"]
                for v in self.state.get("decisions", {}).values()
            ):
                raise ValueError("diagnostic panel already consumed")
        certificates = dict(self.state.get("decisions", {}))
        cert = {
            "bundle": bundle["identity"],
            "source_slot": bundle["source_slot"],
            "target_slot": bundle["target_slot"],
            "decision": decision["identity"],
            "verdict": decision["payload"]["evidence_verdict"],
            "action": decision["payload"]["next_action"],
            "failure_class": branch["failure_class"] if branch else None,
            "panel": panel["identity"] if branch else None,
            "original_decision": confirmation["original"]["decision"]["identity"]
            if confirmation
            else None,
            "comparisons": bundle["design"]["comparisons"],
        }
        certificates[key] = cert
        self.append(dict(self.state, decisions=certificates))
        return cert

    def pause_path(self):
        return self.root / ("pause-" + digest(self.state["invocation"]).split(":")[1] + ".json")

    def pause(self):
        self.read()
        if not (self.pause_path()).exists():
            from .storage import journal_write

            journal_write(
                self.root,
                self.pause_path(),
                seal({"campaign": self.campaign["identity"], "requested": True}),
            )
        self.read()
        return (
            "PAUSED"
            if self.state["status"] in {"PAUSED", "COMPLETE", "FAILED", "PARTIAL"}
            else "PENDING_EXTERNAL_DEADLINE"
        )

    def boundary(self, task, status, invocation):
        validate_task(task, campaign=self.campaign)
        if status not in {"COMPLETE", "PARTIAL", "FAILED", "PAUSED"}:
            raise ValueError("closed task outcome required")
        tasks = dict(self.state["tasks"])
        tasks[task["payload"]["id"]] = status
        return self.append(
            {
                "tasks": tasks,
                "status": "PAUSED"
                if status != "FAILED" and (self.pause_path()).exists()
                else status,
                "invocation": invocation,
            }
        )

    def resolved_plan(self,plan):
        if 'trial_choices' not in plan:return plan
        index=self.state.get('choices',{}).get(plan['id'])
        if index is None:return None
        from .trial_scopes import resolve
        return resolve(plan,index)

    def choose_trial(self,proposal):
        choices=dict(self.state.get('choices',{}))
        selected_scopes=set()
        for plan in self.campaign['payload']['tasks']:
            trial=plan.get('trial')
            if 'trial_choices' not in plan or trial['phase'] in {'confirmation','selection'} or trial['study']!=proposal['study'] or trial['ordinal']!=proposal['ordinal']:continue
            if plan['seed']!=proposal['seed']:raise ValueError('frozen trial proposal seed differs')
            matching=[i for i,c in enumerate(plan['trial_choices']) if dict(c['parameters'],shape=c['parameters'].get('shape','reference'))==dict(proposal['parameters'],shape=proposal['parameters'].get('shape','reference'))]
            if len(matching)!=1:raise ValueError('proposal lacks a frozen prequalified trial alternative')
            if plan['id'] in choices and choices[plan['id']]!=matching[0]:raise ValueError('trial choice cannot change on resume')
            choices[plan['id']]=matching[0]
            if trial['phase']=='training':selected_scopes.add(plan['trial_choices'][matching[0]]['compatibility'])
        if not selected_scopes:raise ValueError('proposal lacks a frozen prequalified trial alternative')
        for plan in self.campaign['payload']['tasks']:
            trial=plan.get('trial')
            if trial and (trial['study'],trial['ordinal'],trial['phase'])==(proposal['study'],proposal['ordinal'],'generation'):
                scope=plan['trial_choices'][choices[plan['id']]]['compatibility'] if 'trial_choices' in plan else plan['initializer']['compatibility']
                if scope not in selected_scopes:raise ValueError('proposal lacks a frozen prequalified trial alternative')
        self.append(dict(self.state,choices=choices))
