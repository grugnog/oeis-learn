"""Show preparation, research stages, screening trials, and learner progress for a discovery round."""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path
import re
import sys
import time

ROUND_NAME = re.compile(r"013-research-round-(\d+)")


def read(path):
    with path.open() as f:
        return json.load(f)


def latest(folder, pattern):
    return max(folder.glob(pattern), default=None)


def millions(n):
    return f"{n / 1_000_000:.3f}M"


def preparation(lines, steps):
    rows = {k: r for k, r in steps.items() if k.startswith("prepare-research/")}
    for gate in ("freeze", "acceptance"):
        lines.append(f"{gate.capitalize():10}  {rows.get('prepare-research/' + gate, {}).get('status', 'PENDING')}")
    for name, row in sorted(rows.items(), key=lambda item: item[1].get("started_unix", 0)):
        status = row["status"]
        if status == "BLOCKED" and any(steps.get(dep, {}).get("status") != "COMPLETE"
                                      for dep in row.get("dependencies", [])):
            continue
        if status in {"RUNNING", "FAILED", "BLOCKED", "INTERRUPTED"}:
            reason = row.get("reason")
            lines.append(f"{name}    {status}" + (f"    {reason}" if reason else ""))


def research(lines, run, session, campaign, trial_ids, study, bindings):
    cid = campaign["identity"].split(":")[1]
    control = run / "discovery-control" / cid
    transition = latest(control, "transition-[0-9][0-9][0-9][0-9].json")
    if transition is None:
        lines.append("Waiting for the first recorded research transition.")
        return
    tasks = read(transition)["state"]["tasks"]
    plans = campaign["payload"]["tasks"]
    for stage in ("capability", "diversity"):
        ids = [p["id"] for p in plans if p["stage"] == stage]
        counts = Counter(tasks.get(key, "pending") for key in ids)
        lines.append(f"{stage.capitalize():10}  {counts['COMPLETE']}/{len(ids)} tasks complete")

    if trial_ids:
        folder = run / "discovery-studies" / cid / study
        states = {}
        for path in sorted(folder.glob("event-*.json")):
            event = read(path)
            states[int(event["ordinal"])] = event["state"]
        outcomes = Counter(s["result"]["status"] for s in (states.get(i, {}) for i in trial_ids) if s.get("result"))
        suffix = ", ".join(f"{n} {status.lower()}" for status, n in sorted(outcomes.items()))
        lines.append(f"Screening    {sum(outcomes.values())}/{len(trial_ids)} trials finalized"
                     + (f" ({suffix})" if suffix else ""))
        looks = {}
        for path in folder.glob("intermediate-*.json"):
            obs = read(path)
            ordinal = int(obs["ordinal"])
            looks[ordinal] = max(looks.get(ordinal, 0), obs["coordinate"])
        lines.append("Trial   Study status    Training milestone    Evaluated at")
        for i in trial_ids:
            prefix = f"trial-{i:02d}-"
            completed = [int(name[len(prefix):-6]) for name, value in tasks.items()
                         if value == "COMPLETE" and re.fullmatch(prefix + r"\d+-train", name)]
            status = states.get(i, {}).get("status", "PENDING")
            lines.append(f"{i:02d}      {status:14}  {millions(max(completed)) if completed else '—':18}"
                         f"  {millions(looks[i]) if i in looks else '—'}")

    # Task identities can change when a trial proposal is bound. Resolve the actual worker from its binding.
    for path in (run / "discovery-accounting" / cid).glob("task-binding-*.json"):
        if path not in bindings:
            binding = read(path)
            bindings[path] = (binding["identity"].split(":")[1], binding["payload"])
    execution = latest(session / "run-research", "execution-[0-9][0-9][0-9][0-9]")
    if execution:
        workers = (p for p in execution.glob("task-*") if re.fullmatch(r"task-[0-9a-f]{64}", p.name) and p.is_dir())
        worker = max(workers, key=lambda p: p.stat().st_mtime_ns, default=None)
        if worker:
            info = dict(bindings.values()).get(worker.name[5:])
            if info:
                lines.append(f"Latest worker directory    {info['id']}    "
                             f"(recorded task status: {tasks.get(info['id'], 'no final status yet')})")
                sample = latest(worker / "role/progress", "progress-*.json")
                if sample:
                    p = read(sample)
                    age = max(0, int(time.time() - sample.stat().st_mtime))
                    lines.append(f"Learner heartbeat    {p['phase']}    {p['committed']:,} useful tokens"
                                 f" / {info['token_target']:,} milestone target    sampled {age}s ago")
                    if "recoverable" in p:
                        lines.append(f"At recoverable checkpoint    {p['recoverable']:,} useful tokens")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repository root (default: current directory)")
    parser.add_argument("--round", type=int, help="round number (default: newest numbered session)")
    parser.add_argument("--interval", type=float, default=30, metavar="SECONDS", help="refresh interval (default: 30)")
    parser.add_argument("--once", action="store_true", help="print one snapshot and exit")
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error("--interval must be a positive finite number")
    run = args.root.expanduser().resolve() / "run-state"
    discovery = run / "discovery"
    if not discovery.is_dir():
        parser.error(f"discovery directory missing: {discovery}; set --root to the repository")
    if args.round is None:
        rounds = [(int(m[1]), p) for p in discovery.iterdir()
                  if (m := ROUND_NAME.fullmatch(p.name)) and p.is_dir()]
        if not rounds:
            parser.error(f"no discovery rounds in {discovery}")
        _, session = max(rounds)
    else:
        session = discovery / f"013-research-round-{args.round:03d}"
        if not session.is_dir():
            parser.error(f"round directory missing: {session}")

    campaign_file = session / "prepare-research/frozen/campaign.json"
    campaign = None
    trial_ids = []
    study = None
    bindings = {}
    while True:
        lines = [f"{session.name}    {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}"]
        try:
            journal = session / "journal.json"
            steps = read(journal)["steps"] if journal.is_file() else {}
            preparation(lines, steps)
            invocations = [k for k in steps if re.fullmatch(r"run-research/invocation-\d+", k)]
            if invocations:
                name = max(invocations, key=lambda k: int(k.rsplit("-", 1)[1]))
                row = steps[name]
                lines.append(f"{name}    {row['status']}" + (f"    {row['reason']}" if row.get('reason') else ""))
            if campaign is None and campaign_file.is_file():
                value = read(campaign_file)
                proposals = [p["trial"] for p in value["payload"]["tasks"]
                             if re.fullmatch(r"trial-\d+-propose", p["id"])]
                if proposals:
                    study, = {p["study"].split(":")[1] for p in proposals}
                    trial_ids = sorted(p["ordinal"] for p in proposals)
                campaign = value
            if campaign is None:
                lines.append("Waiting for the frozen campaign.")
            else:
                research(lines, run, session, campaign, trial_ids, study, bindings)
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            lines.append(f"Status record unavailable: {exc}")
            if args.once:
                print("\n".join(lines), file=sys.stderr)
                return 1
        display = "\n".join(lines)
        if sys.stdout.isatty() and not args.once:
            print("\033[2J\033[H", end="")
        print(display, flush=True)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
