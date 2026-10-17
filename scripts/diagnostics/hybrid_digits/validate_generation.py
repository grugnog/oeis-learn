import json

import torch
from probe_common import ROOT, load

from oeis_learn.decoder.program_codec import decode_body
from oeis_learn.evaluation.search import ModelProposer, SearchExhausted, SearchLedger
from oeis_learn.sandbox.reference import execute_term
from oeis_learn.sandbox.wat_ast import parse

e, d, state, settings, rows, seqs, toks = load()
torch.set_num_threads(2)
results = []
for path in [None, *sorted(ROOT.glob("probe-drop*-gain*.pt"))]:
    data = torch.load(path, map_location="cpu", weights_only=True) if path else state
    label = path.name if path else "original-selected-update-900"
    e.load_state_dict(data["encoder"])
    d.load_state_dict(data["decoder"])
    e.eval()
    d.eval()
    proposer = ModelProposer(e, d, context_limit=65, diagnostics=True)
    result = {
        "checkpoint": label,
        "scope": "CPU production greedy sampler plus Python reference execution; not a GPU qualification",
        "attempts": [],
    }
    for index, (row, sequence) in enumerate(zip(rows, seqs)):
        ledger = SearchLedger(
            max_attempts=1, max_tokens=64, max_expansions=64, budget_ns=60_000_000_000
        )
        ledger.charge(attempts=1)
        try:
            source, evidence = proposer.stochastic(sequence, index, ledger, greedy=True)
            target = decode_body(row["tokens"])
            matched = False
            if source:
                ast = parse(source)
                expected = parse(target)
                values = [execute_term(ast, n) for n in range(100)]
                wanted = [execute_term(expected, n) for n in range(100)]
                matched = all(
                    a.outcome is None and b.outcome is None and a.value == b.value
                    for a, b in zip(values, wanted)
                )
            result["attempts"].append(
                {
                    "offset": row["values"][0],
                    "source": source,
                    "target": target,
                    "exact_source": source == target,
                    "execution_match": matched,
                }
            )
        except SearchExhausted as exc:
            result["attempts"].append(
                {"offset": row["values"][0], "failure": str(exc), "execution_match": False}
            )
    result["correct"] = sum(r["execution_match"] for r in result["attempts"])
    results.append(result)
    (ROOT / "generation.json").write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "checkpoint": label,
                "correct": result["correct"],
                "failed": [r for r in result["attempts"] if not r["execution_match"]],
            }
        ),
        flush=True,
    )
