import json

import torch
from probe_common import ROOT, load

e, d, state, settings, rows, seqs, toks = load()
torch.set_num_threads(1)
saved = {}
hook = d.final_norm.register_forward_hook(lambda m, a, out: saved.update(hidden=out.detach()))
with torch.no_grad():
    memory = e.forward_from_sequences(seqs, device="cpu")
    logits = d(toks[:, :-1], memory, tgt_key_padding_mask=toks[:, :-1].eq(0))
    result = []
    for i, row in enumerate(rows):
        if row["values"][0] not in ("11", "-11"):
            continue
        hidden = saved["hidden"][i, 18:20]
        pre = d.lm_head(hidden)
        prob = d.cap_logits(pre).softmax(-1)
        grad = prob.clone()
        grad[0, 19] -= 1
        grad[1, 17] -= 1
        grad *= 1 - (pre / d.logit_cap_threshold).tanh().square()
        hcos = float(torch.nn.functional.cosine_similarity(hidden[0], hidden[1], dim=0))
        gcos = float(torch.nn.functional.cosine_similarity(grad[0], grad[1], dim=0))
        result.append(
            {
                "offset": row["values"][0],
                "hidden_cosine": hcos,
                "hidden_relative_distance": float(
                    (hidden[0] - hidden[1]).norm() / hidden[0].norm()
                ),
                "output_weight_gradient_cosine": hcos * gcos,
            }
        )
hook.remove()
(ROOT / "position_conflict.json").write_text(json.dumps(result, indent=2))
print(result)
