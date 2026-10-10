"""Ancestor features from the observed token prefix, never the future AST."""

import torch


def ancestor_ids(tokens, vocabulary, max_depth=32):
    kinds = {"block": 1, "loop": 2, "if": 3, "then": 4, "else": 5, "result": 6}
    rows = []
    for row in tokens.detach().cpu().tolist():
        stack, result = [], []
        for token in row:
            word = vocabulary.get(token, "")
            if word == "(":
                if len(stack) >= max_depth:
                    raise ValueError("prefix ancestor depth cap")
                stack.append(0)
            elif word == ")":
                if stack:
                    stack.pop()
            elif stack and word in kinds:
                stack[-1] = kinds[word]
            result.append(stack + [0] * (max_depth - len(stack)))
        rows.append(result)
    return torch.tensor(rows, device=tokens.device, dtype=torch.long)
