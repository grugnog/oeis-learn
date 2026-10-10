"""One masked policy for rollout and differentiable scoring."""

from contextlib import contextmanager
from copy import deepcopy
import hashlib

import torch
import torch.nn.functional as F


@contextmanager
def scoring_mode(*models):
    modules = {
        module: module.training
        for model in models
        if model is not None
        for module in model.modules()
    }
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield
    finally:
        for module, training in modules.items():
            module.training = training


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(str((name, tuple(tensor.shape), str(tensor.dtype))).encode())
        digest.update(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    return "sha256:" + digest.hexdigest()


class FrozenReference:
    def __init__(self, encoder, decoder):
        self.encoder, self.decoder = deepcopy(encoder).eval(), deepcopy(decoder).eval()
        for model in (self.encoder, self.decoder):
            model.requires_grad_(False)
        self.encoder_sha256, self.decoder_sha256 = (
            model_digest(self.encoder),
            model_digest(self.decoder),
        )

    def logits(self, sequences, inputs, device, pad_id):
        with torch.no_grad():
            memory = self.encoder.forward_from_sequences(sequences, device=device)
            return self.decoder(inputs, memory, tgt_key_padding_mask=(inputs == pad_id))


def masked_distribution(logits, supports=None, temperatures=1.0):
    if supports is None:
        supports = torch.ones_like(logits, dtype=torch.bool)
    if supports.shape != logits.shape or supports.dtype != torch.bool or not supports.any(-1).all():
        raise ValueError("nonempty exact legal support required for each action")
    temperature = torch.as_tensor(temperatures, device=logits.device, dtype=logits.dtype)
    if not torch.isfinite(temperature).all() or (temperature <= 0).any():
        raise ValueError("positive finite scoring temperature required")
    if temperature.ndim:
        temperature = temperature[..., None]
    scaled = logits / temperature
    if not torch.isfinite(scaled[supports]).all():
        raise ValueError("nonfinite legal logits")
    log_probs = F.log_softmax(scaled.masked_fill(~supports, float("-inf")), -1)
    probabilities = log_probs.exp()
    safe_logs = torch.where(supports, log_probs, torch.zeros_like(log_probs))
    entropy = -(probabilities * safe_logs).sum(-1)
    return log_probs, probabilities, entropy


def trajectory_mean(values, mask):
    safe = torch.where(mask, values, torch.zeros_like(values))
    return (safe.sum(-1) / mask.sum(-1).clamp_min(1)).mean()


def group_advantages(rewards):
    if rewards.ndim != 1 or not len(rewards) or not torch.isfinite(rewards).all():
        raise ValueError("finite real sampled group rewards required")
    std = rewards.std(unbiased=False)
    if std == 0:
        return torch.zeros_like(rewards)
    return (rewards - rewards.mean()) / std


def policy_loss(
    logits,
    actions,
    behavior_logs,
    advantages,
    *,
    supports=None,
    temperatures=1.0,
    action_mask=None,
    objective="grpo",
    clip_eps=0.2,
    reference_logits=None,
    kl_coefficient=0.01,
    entropy_coefficient=0.0,
):
    if objective not in ("grpo", "reinforce") or not 0 < clip_eps < 1:
        raise ValueError("explicit supported objective and symmetric clipping required")
    logs, probs, entropy = masked_distribution(logits, supports, temperatures)
    if actions.shape != logits.shape[:-1] or behavior_logs.shape != actions.shape:
        raise ValueError("action and behavior shapes differ")
    if advantages.shape != (logits.shape[0],) or not torch.isfinite(advantages).all():
        raise ValueError("one finite advantage per trajectory required")
    if action_mask is not None and (
        action_mask.shape != actions.shape or action_mask.dtype != torch.bool
    ):
        raise ValueError("exact boolean action mask required")
    if actions.dtype != torch.long or (actions < 0).any() or (actions >= logits.shape[-1]).any():
        raise ValueError("valid categorical action IDs required")
    if not 0 <= kl_coefficient or not 0 <= entropy_coefficient:
        raise ValueError("nonnegative objective coefficients required")
    mask = torch.ones_like(actions, dtype=torch.bool) if action_mask is None else action_mask
    current = logs.gather(-1, actions[..., None]).squeeze(-1)
    if not torch.isfinite(current[mask]).all() or not torch.isfinite(behavior_logs[mask]).all():
        raise ValueError("sampled action outside support or missing finite behavior log")
    current = torch.where(mask, current, torch.zeros_like(current))
    old = torch.where(mask, behavior_logs.detach(), torch.zeros_like(behavior_logs))
    ratio = (current - old).exp()
    advantage = advantages.detach()[:, None]
    if objective == "reinforce":
        if not torch.allclose(current.detach()[mask], old[mask], atol=1e-5, rtol=1e-4):
            raise ValueError("REINFORCE requires a fresh rollout from the current policy")
        loss = -trajectory_mean(current * advantage, mask)
    else:
        loss = -trajectory_mean(
            torch.minimum(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage),
            mask,
        )
    if reference_logits is not None and kl_coefficient:
        reference, _, _ = masked_distribution(reference_logits.detach(), supports, temperatures)
        legal = torch.ones_like(logits, dtype=torch.bool) if supports is None else supports
        difference = torch.where(legal, logs - reference, torch.zeros_like(logs))
        loss = loss + kl_coefficient * trajectory_mean((probs * difference).sum(-1), mask)
    if entropy_coefficient:
        loss = loss - entropy_coefficient * trajectory_mean(entropy, mask)
    return loss
