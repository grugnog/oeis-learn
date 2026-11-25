"""Run inside the qualified image with a read-only source and separate child journal."""
from pathlib import Path
from oeis_learn.generalization.progressive.budgets import BudgetJournal
from oeis_learn.generalization.progressive.contracts import load, atomic_write, seal
from oeis_learn.generalization.progressive.recovery import Recovery


def import_cut(request, source_root, accounting_dir, state_dir):
    if seal(request) != request:
        raise ValueError("invalid checkpoint import request")
    task = request["task"]
    journal = BudgetJournal(accounting_dir,
                            task["payload"]["campaign"],
                            request["caps"],
                            journal_bytes=request["journal_bytes"])
    if task["payload"]["caps"]["tokens"] != 0:
        raise ValueError("imported donor must have zero new token authority")
    with Recovery(journal,
                  task,
                  task["payload"]["initializer"]["compatibility"],
                  state_root=state_dir,
                  checkpoint_bytes=request["checkpoint_bytes"]) as recovery:
        if recovery.state["checkpoint"] is None:
            recovery.initialize_import(source_root, request["source"])
        lineage = load(state_dir / "lineage.json")
        state = recovery.restore()
        if (lineage["parent"] != request["parent"] or recovery.state["pending"] is not None
                or state["ledger"].consumed != 1000564 or state["ledger"].updates != 1278
                or not state["states"]["optimizer"]["optimizer"]["state"]
                or not state["states"]["rng"]["device"]
                or state["states"]["scheduler"]["controller"]["cursor"] != 20448
                or recovery.base["ancestor_cost_us"] != 578661207
                or journal.task_remaining(task) != task["payload"]["caps"]):
            raise ValueError("imported state, inherited charges or zero-work boundary differs")
        if recovery.state["status"] != "COMPLETE":
            recovery.append(status="COMPLETE")
        result = seal(
            dict(version="trial00-import/v1",
                 task=task["identity"],
                 parent=lineage["parent"],
                 checkpoint=recovery.state["checkpoint"]["identity"],
                 base=recovery.base,
                 cursor=20448,
                 new_tokens=0,
                 new_updates=0))
        path = state_dir / "import.json"
        if path.exists() and load(path) != result:
            raise ValueError("conflicting import receipt")
        if not path.exists():
            atomic_write(path, result)


if __name__ == "__main__":
    import_cut(load("/inputs/import.json"), Path("/source"), Path("/accounting"),
               Path("/learning-state"))
