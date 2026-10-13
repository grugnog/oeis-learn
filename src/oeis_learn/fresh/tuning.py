"""Reproducible length batching and measured diagnostic batch selection."""


def permutation(rows, generator, batch_size, buckets=False):
    import torch

    order = torch.randperm(len(rows), generator=generator).tolist()
    if not buckets:
        return order
    batches = []
    for start in range(0, len(order), 256):
        block = sorted(order[start : start + 256], key=lambda i: len(rows[i]["tokens"]))
        batches.extend(block[i : i + batch_size] for i in range(0, len(block), batch_size))
    sequence = torch.randperm(len(batches), generator=generator).tolist()
    return [index for i in sequence for index in batches[i]]


def choose_trial(trials):
    eligible = [
        t
        for t in trials
        if t["status"] == "PASS" and t["steps"] >= 3 and t["peak_allocated_bytes"] <= 48 << 30
    ]
    if not eligible:
        raise ValueError("no benchmark batch passed with memory headroom")
    fastest = max(t["tokens_per_second"] for t in eligible)
    finalists = [t for t in eligible if t["tokens_per_second"] >= fastest * 0.95]
    return min(finalists, key=lambda t: (t["batch_size"], not t["length_buckets"]))
