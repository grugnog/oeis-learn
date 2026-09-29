"""Standalone Docker feasibility probe (T001) — Trustworthy Experiment Foundation.

This is the runnable entry point for the standalone pre-adoption GPU
feasibility probe. It is independent of the later foundation CLI/codec/pool
(FR-017, SC-005) and does not start a training experiment.

Behavior:
  * Loads a frozen preflight-only config (rejects unknown fields).
  * Inspects EXISTING host GPU/device permissions (ROCm/KFD/render nodes).
  * If the host cannot support the requested accelerator, it fails
    ``hardware_not_ready`` with specific diagnostics — never a CPU fallback,
    never a driver install, never a privileged container.
  * If the requested ROCm accelerator is available, it performs three finite
    FP32 forward/backward/AdamW parameter updates on real encoder/decoder
    parameters at 20 visible terms and 1,024 body tokens, including zero,
    alternating signed-256 extremes and adjacent large-integer prefixes,
    within the configured wall-time budget, and persists actual
    identity/memory/timing evidence.

Exit codes (match contracts/cli.md):
  0 success (including a successfully recorded rejection)
  2 invalid input/configuration
  3 hardware unavailable (hardware_not_ready)
  4 correctness/provenance/readiness gate failed
  5 resource or infrastructure interruption
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import yaml

# ── Exit codes ───────────────────────────────────────────────
EXIT_OK = 0
EXIT_INVALID = 2
EXIT_HARDWARE = 3
EXIT_GATE = 4
EXIT_RESOURCE = 5

# ── Allowed config schema (unknown fields are rejected) ───────
_ALLOWED_TOP = {
    "schema_version", "kind", "base_image", "base_image_digest", "dockerfile",
    "dependency_lock", "requested_device", "dtype", "no_cpu_fallback",
    "no_driver_install", "no_privileged_container", "updates", "optimizer",
    "visible_terms", "body_tokens", "timeout_minutes", "fixtures", "model",
}
_ALLOWED_FIXTURE = {"kind"}
_ALLOWED_MODEL = {
    "encoder", "decoder", "d_model", "n_heads", "n_encoder_layers",
    "n_decoder_layers", "d_ff", "dropout", "max_seq_len", "max_valuation",
    "moduli_count", "use_film", "enable_summary_tokens", "seed",
}
_FIXTURE_KINDS = {"zero", "alternating_signed256_extremes", "adjacent_large_integer_prefixes"}
_SIGNED_256_MAX = (1 << 255) - 1
_SIGNED_256_MIN = -(1 << 255)


class PreflightConfig:
    """Validated preflight-only configuration."""

    def __init__(self, data: Dict[str, Any], source: str) -> None:
        unknown = set(data) - _ALLOWED_TOP
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)} (source: {source})")
        if data.get("schema_version") != "preflight/v1":
            raise ValueError(f"unsupported schema_version: {data.get('schema_version')!r}")
        if data.get("kind") != "preflight_config":
            raise ValueError(f"unsupported kind: {data.get('kind')!r}")
        if not isinstance(data.get("base_image_digest"), str) or not data["base_image_digest"].startswith(
            "sha256:"
        ):
            raise ValueError("base_image_digest must be a sha256: digest")

        fixtures = data.get("fixtures", [])
        if not isinstance(fixtures, list) or len(fixtures) < 1:
            raise ValueError("fixtures must be a nonempty list")
        for fx in fixtures:
            if set(fx) - _ALLOWED_FIXTURE:
                raise ValueError(f"unknown fixture fields: {sorted(set(fx) - _ALLOWED_FIXTURE)}")
            if fx.get("kind") not in _FIXTURE_KINDS:
                raise ValueError(f"unknown fixture kind: {fx.get('kind')!r}")

        model = data.get("model", {})
        if set(model) - _ALLOWED_MODEL:
            raise ValueError(f"unknown model fields: {sorted(set(model) - _ALLOWED_MODEL)}")

        self.schema_version = data["schema_version"]
        self.kind = data["kind"]
        self.base_image = data["base_image"]
        self.base_image_digest = data["base_image_digest"]
        self.dockerfile = data.get("dockerfile", "")
        self.dependency_lock = data.get("dependency_lock", "")
        self.requested_device = data["requested_device"]
        self.dtype = data.get("dtype", "float32")
        self.no_cpu_fallback = bool(data.get("no_cpu_fallback", True))
        self.no_driver_install = bool(data.get("no_driver_install", True))
        self.no_privileged_container = bool(data.get("no_privileged_container", True))
        self.updates = int(data["updates"])
        self.optimizer = data.get("optimizer", "adamw")
        self.visible_terms = int(data["visible_terms"])
        self.body_tokens = int(data["body_tokens"])
        self.timeout_minutes = int(data.get("timeout_minutes", 10))
        self.fixtures = fixtures
        self.model = model
        self.source = source

        if self.dtype != "float32":
            raise ValueError("preflight dtype must be float32")
        if self.requested_device not in ("rocm", "hip"):
            raise ValueError(f"requested_device must be rocm/hip, got {self.requested_device!r}")
        if self.updates != 3:
            raise ValueError("preflight requires exactly three parameter updates")
        if self.visible_terms != 20:
            raise ValueError("preflight requires 20 visible terms")
        if self.body_tokens != 1024:
            raise ValueError("preflight requires 1,024 body tokens")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "base_image": self.base_image,
            "base_image_digest": self.base_image_digest,
            "dockerfile": self.dockerfile,
            "dependency_lock": self.dependency_lock,
            "requested_device": self.requested_device,
            "dtype": self.dtype,
            "updates": self.updates,
            "optimizer": self.optimizer,
            "visible_terms": self.visible_terms,
            "body_tokens": self.body_tokens,
            "timeout_minutes": self.timeout_minutes,
            "fixtures": self.fixtures,
            "model": self.model,
            "no_cpu_fallback": self.no_cpu_fallback,
            "no_driver_install": self.no_driver_install,
            "no_privileged_container": self.no_privileged_container,
        }


def load_config(path: str) -> PreflightConfig:
    """Loads and strictly validates the preflight config (YAML)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"preflight config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("preflight config must be a mapping")
    return PreflightConfig(data, path)


# ── Host GPU/device inspection (no torch required) ────────────
def _read_stat(path: str) -> Optional[Dict[str, Any]]:
    try:
        st = os.stat(path)
        return {
            "exists": True,
            "mode": oct(st.st_mode & 0o777),
            "uid": st.st_uid,
            "gid": st.st_gid,
            "type": "block" if (st.st_mode & 0o170000) == 0o10000 else "node",
        }
    except FileNotFoundError:
        return {"exists": False}


def inspect_host() -> Dict[str, Any]:
    """Inspects existing host GPU/device permissions (ROCm/KFD/render nodes)."""
    report: Dict[str, Any] = {}
    report["kfd"] = _read_stat("/dev/kfd")
    report["render_nodes"] = {}
    dri_dir = "/dev/dri"
    if os.path.isdir(dri_dir):
        for name in sorted(os.listdir(dri_dir)):
            report["render_nodes"][name] = _read_stat(os.path.join(dri_dir, name))
    report["any_render_node"] = any(
        n.get("exists") for n in report["render_nodes"].values()
    )

    # HIP/ROCm runtime identity (lazy torch import; missing torch is itself
    # a host environment diagnostic, never a silent CPU fallback).
    hip: Dict[str, Any] = {"available": False}
    try:
        import torch  # noqa: WPS433 (deliberate lazy import)

        hip["torch_version"] = torch.__version__
        hip["hip_version"] = getattr(torch.version, "hip", None)
        hip["cuda_available"] = bool(torch.cuda.is_available())
        hip["device_count"] = int(torch.cuda.device_count())
        hip["available"] = bool(
            hip["cuda_available"] and hip["device_count"] > 0 and hip["hip_version"]
        )
    except Exception as exc:  # pragma: no cover - host-dependent
        hip["error"] = f"{type(exc).__name__}: {exc}"
    report["hip"] = hip

    # Environment knobs relevant to device selection.
    report["env"] = {
        "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
        "HIP_DEVICE_AUTO_WAIT": os.environ.get("HIP_DEVICE_AUTO_WAIT"),
    }
    return report


def host_supports_rocm(env_report: Dict[str, Any]) -> bool:
    """True only if the host exposes a usable AMD/ROCm accelerator.

    No CPU fallback: an available CPU alone is NOT readiness.
    """
    hip = env_report.get("hip", {})
    if hip.get("available"):
        return True
    # Even without torch, existing AMD KFD + render node implies a usable
    # device for a containerized probe.
    if env_report.get("kfd", {}).get("exists") and env_report.get("any_render_node"):
        return True
    return False


# ── Deterministic generic tensor fixtures (signed-256) ────────
def make_fixture(kind: str, n: int) -> List[int]:
    """Builds a deterministic 20-term signed-256 fixture for the given kind."""
    if n < 20:
        raise ValueError("fixture length must be at least 20")
    terms: List[int] = []
    if kind == "zero":
        terms = [0] * n
    elif kind == "alternating_signed256_extremes":
        # index 0 is zero, then alternate -2^255 and (2^255 - 1).
        terms = [0]
        for i in range(1, n):
            terms.append(_SIGNED_256_MIN if (i % 2 == 1) else _SIGNED_256_MAX)
    elif kind == "adjacent_large_integer_prefixes":
        # Adjacent large-integer prefixes: pairs differing by 1 near the
        # signed-256 boundary, preceded by zero.
        terms = [0, _SIGNED_256_MAX, _SIGNED_256_MAX - 1, _SIGNED_256_MIN, _SIGNED_256_MIN + 1]
        while len(terms) < n:
            terms.append(0)
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"unknown fixture kind: {kind!r}")
    return terms[:n]


# ── Actual finite FP32 update probe (accelerator required) ────
def _build_model(cfg: PreflightConfig):
    """Builds the reused encoder/decoder architecture (imported lazily)."""
    import torch  # noqa: WPS433
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder  # noqa: WPS433
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder  # noqa: WPS433
    from oeis_learn.decoder.wat_grammar import VOCAB_SIZE  # noqa: WPS433

    m = cfg.model
    torch.manual_seed(int(m.get("seed", 20260913)))
    encoder = TriStreamEncoder(
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_encoder_layers=int(m["n_encoder_layers"]),
        d_ff=int(m["d_ff"]),
        dropout=float(m["dropout"]),
        max_seq_len=int(m.get("max_seq_len", 1024)),
        max_valuation=int(m.get("max_valuation", 16)),
        moduli_count=int(m.get("moduli_count", 16)),
        use_film=bool(m.get("use_film", True)),
        enable_summary_tokens=bool(m.get("enable_summary_tokens", False)),
    )
    decoder = WatTransformerDecoder(
        vocab_size=int(VOCAB_SIZE),
        d_model=int(m["d_model"]),
        n_heads=int(m["n_heads"]),
        n_decoder_layers=int(m["n_decoder_layers"]),
        d_ff=int(m["d_ff"]),
        dropout=float(m["dropout"]),
        max_seq_len=int(m.get("max_seq_len", 1024)),
        pad_idx=0,
    )
    return torch, encoder, decoder


def run_updates(cfg: PreflightConfig, env_report: Dict[str, Any]) -> Dict[str, Any]:
    """Runs three finite FP32 forward/backward/AdamW updates and persists
    actual identity/memory/timing evidence. Requires a real accelerator.
    """
    import torch  # noqa: WPS433

    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
    torch.manual_seed(int(cfg.model.get("seed", 20260913)))
    torch.cuda.manual_seed_all(int(cfg.model.get("seed", 20260913)))

    torch, encoder, decoder = _build_model(cfg)
    encoder = encoder.to(device=device, dtype=torch.float32)
    decoder = decoder.to(device=device, dtype=torch.float32)

    params = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    # Deterministic body tokens: 1,024 token IDs (no pad idx 0), < VOCAB_SIZE.
    vocab = decoder.vocab_size if hasattr(decoder, "vocab_size") else decoder.lm_head.weight.size(0)
    body = (torch.arange(cfg.body_tokens, device=device) % (vocab - 1)) + 1  # (body_tokens,)
    tgt = body.unsqueeze(0)  # (1, body_tokens)
    target = tgt[:, 1:]  # (1, body_tokens - 1)

    steps: List[Dict[str, Any]] = []
    elapsed_total = 0.0
    peak_before = torch.cuda.max_memory_allocated(device)
    for step in range(cfg.updates):
        fixture_kind = cfg.fixtures[step % len(cfg.fixtures)]["kind"]
        terms = make_fixture(fixture_kind, cfg.visible_terms)
        t0 = time.perf_counter()

        optimizer.zero_grad()
        memory = encoder.forward_from_sequences([terms], device=device)  # (1, 20, d)
        logits = decoder.forward(tgt_tokens=tgt, memory=memory)  # (1, body, vocab)
        # Shifted teacher-forced cross-entropy over body positions.
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, vocab),
            target.reshape(-1),
        )
        if torch.isnan(loss) or torch.isinf(loss):
            raise FloatingPointError(f"non-finite loss at update {step + 1}")

        loss.backward()
        grad_stats = {"finite": True, "max_abs": 0.0}
        for p in params:
            if p.grad is None:
                raise RuntimeError(f"missing gradient on {p.shape} at update {step + 1}")
            gmax = float(p.grad.abs().max())
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                grad_stats["finite"] = False
            grad_stats["max_abs"] = max(grad_stats["max_abs"], gmax)
        if not grad_stats["finite"]:
            raise FloatingPointError(f"non-finite gradient at update {step + 1}")

        before = [p.detach().clone() for p in params]
        optimizer.step()
        changed = 0
        max_change = 0.0
        for p, b in zip(params, before):
            d = float((p.detach() - b).abs().max())
            if d > 0.0:
                changed += 1
            max_change = max(max_change, d)

        t1 = time.perf_counter()
        elapsed = t1 - t0
        elapsed_total += elapsed
        steps.append(
            {
                "update": step + 1,
                "fixture_kind": fixture_kind,
                "visible_terms": cfg.visible_terms,
                "body_tokens": cfg.body_tokens,
                "device": str(device),
                "device_name": device_name,
                "dtype": "float32",
                "loss": float(loss.detach().cpu()),
                "loss_finite": bool(torch.isfinite(loss)),
                "grad_finite": grad_stats["finite"],
                "max_abs_grad": grad_stats["max_abs"],
                "params_changed": changed,
                "max_abs_param_change": max_change,
                "elapsed_s": elapsed,
            }
        )

        if elapsed_total > cfg.timeout_minutes * 60.0:
            raise TimeoutError(
                f"preflight exceeded {cfg.timeout_minutes}-minute wall-time budget"
            )

    peak_after = torch.cuda.max_memory_allocated(device)
    return {
        "device": str(device),
        "device_name": device_name,
        "hip_version": env_report.get("hip", {}).get("hip_version"),
        "steps": steps,
        "elapsed_total_s": elapsed_total,
        "within_timeout_minutes": cfg.timeout_minutes,
        "peak_memory_bytes_before": int(peak_before),
        "peak_memory_bytes_after": int(peak_after),
        "peak_memory_bytes_delta": int(max(peak_after - peak_before, 0)),
        "param_count": len(params),
        "status": "ready",
    }


# ── Evidence persistence ──────────────────────────────────────
def write_evidence(output_dir: str, report: Dict[str, Any]) -> str:
    """Atomically persists the preflight evidence record (JSON)."""
    os.makedirs(output_dir, exist_ok=True)
    canonical = json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report["evidence_sha256"] = digest
    path = os.path.join(output_dir, "preflight-report.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foundation-preflight",
        description="Standalone Docker GPU feasibility probe (T001). No CPU fallback.",
    )
    parser.add_argument("--config", required=True, help="preflight-only config (YAML)")
    parser.add_argument("--output", required=True, help="evidence output directory")
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="standalone pre-adoption feasibility; does not require cohort/pool",
    )
    args = parser.parse_args(argv)

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        cfg = load_config(args.config)
    except Exception as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return EXIT_INVALID

    env_report = inspect_host()
    base_report: Dict[str, Any] = {
        "command": "foundation preflight",
        "status": "pending",
        "schema_version": "preflight/v1",
        "created_at": now,
        "config_digest": cfg.base_image_digest,
        "requested_device": cfg.requested_device,
        "dtype": cfg.dtype,
        "updates": cfg.updates,
        "visible_terms": cfg.visible_terms,
        "body_tokens": cfg.body_tokens,
        "timeout_minutes": cfg.timeout_minutes,
        "no_cpu_fallback": cfg.no_cpu_fallback,
        "no_driver_install": cfg.no_driver_install,
        "no_privileged_container": cfg.no_privileged_container,
        "hardware_only": args.hardware_only,
        "environment_report": env_report,
    }

    if not host_supports_rocm(env_report):
        base_report["status"] = "hardware_not_ready"
        base_report["diagnostics"] = _hardware_diagnostics(env_report)
        base_report["gate"] = "pending"
        path = write_evidence(args.output, base_report)
        print(f"hardware_not_ready: host cannot support {cfg.requested_device} image; "
              "no CPU fallback, no driver install, no privileged container.", file=sys.stderr)
        print(json.dumps({"command": "foundation preflight", "status": "hardware_not_ready",
                          "evidence": path, "exit_code": EXIT_HARDWARE}))
        return EXIT_HARDWARE

    try:
        update_report = run_updates(cfg, env_report)
    except TimeoutError as exc:
        base_report["status"] = "timeout"
        base_report["diagnostics"] = [str(exc)]
        write_evidence(args.output, base_report)
        print(f"timeout: {exc}", file=sys.stderr)
        return EXIT_RESOURCE
    except Exception as exc:
        base_report["status"] = "gate_failed"
        base_report["diagnostics"] = [f"{type(exc).__name__}: {exc}"]
        write_evidence(args.output, base_report)
        print(f"gate failed: {exc}", file=sys.stderr)
        return EXIT_GATE

    base_report.update(update_report)
    base_report["status"] = "ready"
    base_report["gate"] = "passed"
    path = write_evidence(args.output, base_report)
    print(json.dumps({"command": "foundation preflight", "status": "ready",
                      "evidence": path, "exit_code": EXIT_OK}))
    return EXIT_OK


def _hardware_diagnostics(env_report: Dict[str, Any]) -> List[str]:
    diag: List[str] = []
    kfd = env_report.get("kfd", {})
    diag.append(f"/dev/kfd present={kfd.get('exists', False)}")
    render = env_report.get("render_nodes", {})
    present = [n for n, v in render.items() if v.get("exists")]
    diag.append(f"render nodes present={present or ['none']}")
    hip = env_report.get("hip", {})
    if hip.get("error"):
        diag.append(f"torch/hip: {hip['error']}")
    diag.append(f"hip_available={hip.get('available', False)}")
    diag.append("silent CPU fallback is prohibited; no driver install; no privileged container")
    return diag


if __name__ == "__main__":
    sys.exit(main())
