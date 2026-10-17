"""Bounded teacher-forced literal evidence; never changes model or target tokens."""


def teacher_trace(encoder, decoder, row):
    import torch
    from oeis_learn.rl.policy import scoring_mode
    from oeis_learn.decoder.environment_tracker import FoundationEnvironmentTracker
    from oeis_learn.decoder.program_codec import BOS_ID, ID_TO_TOKEN

    device = next(encoder.parameters()).device
    targets = row["tokens"]
    tokens = torch.tensor([[BOS_ID] + targets[:-1]], device=device)
    with scoring_mode(encoder, decoder), torch.no_grad():
        memory = encoder.forward_from_sequences([list(map(int, row["values"]))], device=device)
        logits = decoder(tokens, memory)
        cached, _ = decoder.prefill(tokens, memory, capacity=tokens.shape[1])
        error = (logits - cached).abs().max().item()
        tracker = FoundationEnvironmentTracker()
        trace = []
        for position, target in enumerate(targets):
            legal = sorted(tracker.get_valid_next_tokens())
            if target not in legal:
                raise ValueError("admitted target rejected by incremental grammar")
            current = logits[0, position]
            mask = torch.full_like(current, float("-inf"))
            mask[legal] = current[legal]
            raw = current.softmax(-1)
            grammar = mask.softmax(-1)
            trace.append(
                dict(
                    position=position + 1,
                    target=target,
                    spelling=ID_TO_TOKEN[target],
                    raw_probability=float(raw[target]),
                    grammar_probability=float(grammar[target]),
                    raw_argmax=int(current.argmax()),
                    grammar_argmax=int(mask.argmax()),
                )
            )
            tracker.update(ID_TO_TOKEN[target])
    return dict(
        tokens=trace,
        grammar_argmax_exact=all(t["grammar_argmax"] == t["target"] for t in trace),
        cache_parity=dict(
            max_absolute_error=error,
            passed=bool(torch.allclose(logits, cached, atol=2e-5, rtol=2e-4)),
        ),
    )


def target_comparison(targets, generated):
    """Locate digit/delimiter drift without assuming the output completed."""
    from oeis_learn.decoder.program_codec import BOS_ID, ID_TO_TOKEN

    generated = generated[1:] if generated[:1] == [BOS_ID] else generated
    for index in range(max(len(targets), len(generated))):
        expected = targets[index] if index < len(targets) else None
        actual = generated[index] if index < len(generated) else None
        if expected != actual:
            return dict(
                exact=False,
                position=index + 1,
                expected=ID_TO_TOKEN.get(expected),
                generated=ID_TO_TOKEN.get(actual),
                expected_id=expected,
                generated_id=actual,
            )
    return dict(exact=True, position=None)
