"""Reopen the measured learner qualification twice on its original detached donor."""

import json
from pathlib import Path
import subprocess


CANDIDATE = "73009e595e589a1f6780cec06d5d16989ee8dcd5"
LAUNCHER = "specs/016-measure-learning-velocity/validation/launch-round029.sh"
GRANT = "44f3f04994007bc2463b4da006b85a27aa16c3e85995754648a9f814462d274c"
STATUS_ID = "sha256:0ea98dc480b85807daf1e9013573a71c385b432cd36b8718148c45f0828994b8"
RECEIPT_ID = "sha256:2e4f54628a8a6d9d1cfb4349f333e530acdedd944e7c7f76677d392c142a6e5b"


def replay(root, invoke):
    status = invoke("status")
    saved = status.get("progress", {}).copy()
    if (
        status.get("owner_active") is not False
        or status.get("last_committed_ref") != RECEIPT_ID
        or saved.pop("identity", None) != STATUS_ID
    ):
        raise SystemExit("The donor must be idle at the recorded learner qualification.")

    authority = root / "run-state/discovery-authorizations" / f"grant-{GRANT}.json"
    receipt = root / saved["receipt_path"]
    for _ in range(2):
        if not authority.is_file() or not receipt.is_file():
            raise SystemExit("The original learner grant and receipt must both be present.")
        if (
            json.loads(authority.read_bytes()).get("identity") != "sha256:" + GRANT
            or json.loads(receipt.read_bytes()) != saved["receipt"]
        ):
            raise SystemExit("The retained grant or receipt differs from the measured qualification.")
        if invoke("qualify", str(authority)) != saved:
            raise SystemExit("Replay differs from the original qualification or its charges.")

    return dict(
        candidate=CANDIDATE,
        receipt=RECEIPT_ID,
        verified_replays=2,
        elapsed_us=saved["elapsed_us"],
        charged_tokens=saved["charged_tokens"],
        research_dispatches=saved["research_dispatches"],
    )


def main():
    root = Path.cwd().resolve()
    launcher = subprocess.check_output(["git", "show", f"{CANDIDATE}:{LAUNCHER}"], cwd=root)

    def invoke(action, *args):
        run = subprocess.run(
            ["bash", "-s", "--", CANDIDATE, action, *args],
            input=launcher,
            cwd=root,
            stdout=subprocess.PIPE,
        )
        if run.returncode:
            raise SystemExit(run.stdout.decode().strip() or f"{action} exited {run.returncode}")
        return json.loads(run.stdout)

    print(json.dumps(replay(root, invoke)))


if __name__ == "__main__":
    main()
