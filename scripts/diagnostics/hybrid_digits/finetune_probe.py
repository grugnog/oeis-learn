import copy
import json
import random
import time

import torch
from probe_common import ROOT, describe, load

from oeis_learn.decoder.program_codec import PAD_ID
from oeis_learn.rl.policy import trajectory_mean
from oeis_learn.rl.sft_trainer import teacher_forced_logits
from oeis_learn.rl.token_loss import token_cross_entropy

e, d, state, settings, rows, seqs, toks = load()
results = []
for dropout, gain in ((0.1, 1), (0, 1), (0.1, 16), (0, 16)):
    e.load_state_dict(state["encoder"])
    d.load_state_dict(state["decoder"])
    d.pos_encoder.pe.mul_(gain)
    for model in (e, d):
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout
            if isinstance(module, torch.nn.MultiheadAttention):
                module.dropout = dropout
    optimizer = torch.optim.AdamW(
        [*e.parameters(), *d.parameters()], lr=0.0001, weight_decay=0.01, foreach=False
    )
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    torch.manual_seed(20261017)
    rng = random.Random(20261017)
    result = {
        "dropout": dropout,
        "position_gain": gain,
        "learning_rate": optimizer.param_groups[0]["lr"],
        "trace": [],
        "mode": "CPU intervention from selected weights and optimizer; new fixed sample order, no source run mutation",
    }
    start = time.monotonic()
    order = []
    for step in range(201):
        if step % 25 == 0:
            e.eval()
            d.eval()
            with torch.no_grad():
                memory = e.forward_from_sequences(seqs, device="cpu")
                logits = d(toks[:, :-1], memory, tgt_key_padding_mask=toks[:, :-1].eq(PAD_ID))
                observation = describe(logits, toks[:, 1:], rows)
                observation["step"] = step
                observation["repeated"] = [
                    {
                        "offset": r["values"][0],
                        "second_one": float(logits[i, 18].softmax(-1)[19]),
                        "end_after_second": float(logits[i, 19].softmax(-1)[17]),
                        "first_margin": float(logits[i, 18, 19] - logits[i, 18, 17]),
                        "second_margin": float(logits[i, 19, 19] - logits[i, 19, 17]),
                    }
                    for i, r in enumerate(rows)
                    if r["values"][0] in ("11", "-11")
                ]
                result["trace"].append(observation)
            print(json.dumps({"dropout": dropout, "gain": gain, **observation}), flush=True)
        if step == 200:
            break
        if not order:
            order = list(range(len(rows)))
            rng.shuffle(order)
        batch = order[:8]
        order = order[8:]
        e.train()
        d.train()
        optimizer.zero_grad(set_to_none=True)
        input_tokens = toks[batch]
        logits = teacher_forced_logits(
            e, d, [seqs[i] for i in batch], input_tokens, device="cpu", pad_id=PAD_ID
        )
        targets = input_tokens[:, 1:]
        loss = trajectory_mean(token_cross_entropy(logits, targets), targets.ne(PAD_ID))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*e.parameters(), *d.parameters()], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
    result["seconds"] = time.monotonic() - start
    torch.save(
        {"encoder": e.state_dict(), "decoder": d.state_dict()},
        ROOT / f"probe-drop{dropout}-gain{gain}.pt",
    )
    results.append(result)
    (ROOT / "finetune.json").write_text(json.dumps(results, indent=2))
