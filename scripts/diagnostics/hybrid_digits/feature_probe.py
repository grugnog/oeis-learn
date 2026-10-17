import json

import torch
from probe_common import ROOT, describe, load

from oeis_learn.decoder.program_codec import PAD_ID

e, d, state, settings, rows, seqs, toks = load()
torch.set_num_threads(1)
results = []
with torch.no_grad():
    for name, module in [
        ("baseline", None),
        ("magnitude", e.s1_magnitude),
        ("modular", e.s2_modulo),
        ("difference_padic", e.s3_diff_padic),
        ("bytes", e.byte_projection),
    ]:
        hook = (
            module.register_forward_hook(lambda mod, args, output: torch.zeros_like(output))
            if module
            else None
        )
        memory = e.forward_from_sequences(seqs, device="cpu")
        logits = d(toks[:, :-1], memory, tgt_key_padding_mask=toks[:, :-1].eq(PAD_ID))
        results.append(dict(ablated=name, **describe(logits, toks[:, 1:], rows)))
        if hook:
            hook.remove()
(ROOT / "features.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results, indent=2))
