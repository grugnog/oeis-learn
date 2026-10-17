import json
import math

import torch
from probe_common import ROOT, describe, load

from oeis_learn.decoder.program_codec import ID_TO_TOKEN
from oeis_learn.fresh.token_diagnostic import teacher_trace

e, d, state, settings, rows, seqs, toks = load()
report = {
    "torch": torch.__version__,
    "device": "cpu",
    "selected_update": 900,
    "fingerprints_verified": True,
}
with torch.no_grad():
    memory = e.forward_from_sequences(seqs, device="cpu")
    baseline = d(toks[:, :-1], memory, tgt_key_padding_mask=toks[:, :-1].eq(0))
    report["baseline"] = describe(baseline, toks[:, 1:], rows)
    report["digits"] = []
    for i, r in enumerate(rows):
        if r["values"][0] in ("11", "-11", "1", "-1"):
            probabilities = baseline[i].softmax(-1)
            report["digits"].append(
                {
                    "offset": r["values"][0],
                    "positions": [
                        {
                            "position": j + 1,
                            "target": ID_TO_TOKEN[int(toks[i, j + 1])],
                            "p_one": float(probabilities[j, 19]),
                            "p_end": float(probabilities[j, 17]),
                            "argmax": ID_TO_TOKEN[int(baseline[i, j].argmax())],
                        }
                        for j in range(17, min(21, len(r["tokens"])))
                    ],
                }
            )
    report["scales"] = {
        "embedding_weight_std": float(d.token_embedding.weight.std()),
        "scaled_one_embedding_norm": float(
            d.token_embedding.weight[19].norm() * math.sqrt(d.d_model)
        ),
        "pos18_norm": float(d.pos_encoder.pe[0, 18].norm()),
        "adjacent_position_distance": float(
            (d.pos_encoder.pe[0, 19] - d.pos_encoder.pe[0, 18]).norm()
        ),
        "byte_gate_mean": float(e.byte_gate.sigmoid().mean()),
    }
    original_pe = d.pos_encoder.pe.clone()
    report["position_gain_probe"] = []
    for gain in (0, 0.5, 1, 2, 4, 8, 16):
        d.pos_encoder.pe.copy_(original_pe * gain)
        logits = d(toks[:, :-1], memory, tgt_key_padding_mask=toks[:, :-1].eq(0))
        result = describe(logits, toks[:, 1:], rows)
        result.update(
            gain=gain,
            repeated=[
                {
                    "offset": r["values"][0],
                    "p_one_at_second": float(logits[i, 18].softmax(-1)[19]),
                    "p_end_at_second": float(logits[i, 18].softmax(-1)[17]),
                }
                for i, r in enumerate(rows)
                if r["values"][0] in ("11", "-11")
            ],
        )
        report["position_gain_probe"].append(result)
    d.pos_encoder.pe.copy_(original_pe)
report["cpu_teacher_traces"] = [
    {"offset": r["values"][0], **teacher_trace(e, d, r)}
    for r in rows
    if r["values"][0] in ("11", "-11")
]
(ROOT / "inspection.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "cpu_teacher_traces"}, indent=2))
