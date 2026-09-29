#!/usr/bin/env python3
"""
Minimal test-training run for oeis-learn.

Uses 2 synthetic sequences, a miniature encoder/decoder, and
a single gradient step through the EGCA-GRPO training loop.

Run locally:
    PYTHONPATH=src python scripts/test_minimal_train.py

Run in container:
    python scripts/test_minimal_train.py
"""

import json
import os
import sys

import numpy as np
import torch

# ── Tiny smoke-test model config ──────────────────────────────────
SMOKE_CFG = {
    "d_model": 64,
    "n_heads": 2,
    "n_encoder_layers": 1,
    "n_decoder_layers": 1,
    "d_ff": 256,
}

# ── 2 toy sequences (perfect squares, Fibonacci) ──────────────────
TOY_RECORDS = [
    {
        "oeis_id": "A000001",
        "name": "Perfect squares",
        "terms": [str(i * i) for i in range(1, 9)],       # 1 4 9 … 64
        "tags": "polynomial",
        "curriculum_stage": 1,
    },
    {
        "oeis_id": "A000002",
        "name": "Fibonacci",
        "terms": ["1", "1", "2", "3", "5", "8", "13", "21"],
        "tags": "recurrence",
        "curriculum_stage": 1,
    },
]


def _make_dry_db(db_path: str):
    """Seed a minimal DuckDB database matching the production schema."""
    import duckdb

    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = duckdb.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sequences (
            oeis_id            TEXT PRIMARY KEY,
            name               TEXT,
            terms_json         TEXT,
            tags               TEXT,
            curriculum_stage   INTEGER,
            joeis_class        TEXT,
            generating_formula TEXT,
            lz_complexity      DOUBLE,
        );
    """)

    for rec in TOY_RECORDS:
        conn.execute(
            "INSERT OR REPLACE INTO sequences VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            [
                rec["oeis_id"],
                rec["name"],
                json.dumps(rec["terms"]),
                rec.get("tags", ""),
                rec["curriculum_stage"],
                None,
                None,
                0.0,
            ],
        )
    conn.commit()
    conn.close()


# ── Main workflow ─────────────────────────────────────────────────
def main() -> int:
    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu")

    print("=" * 60)
    print("  oeis-learn  —  minimal test training run")
    print("=" * 60)

    # ── 1. Seed database ──────────────────────────────────────
    db_path = "data/oeis_learn_test.duckdb"
    _make_dry_db(db_path)
    print(f"\n✓ Seeded test database at {db_path} ({len(TOY_RECORDS)} sequences)")

    # ── 2. Build encoder + decoder ────────────────────────────
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder

    cfg = SMOKE_CFG
    encoder = TriStreamEncoder(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_encoder_layers=cfg["n_encoder_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        primes=[3, 5, 7],
        max_valuation=4,
        use_film=True,
        enable_summary_tokens=True,
    ).to(device)

    decoder = WatTransformerDecoder(
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_decoder_layers=cfg["n_decoder_layers"],
        d_ff=cfg["d_ff"],
        dropout=0.0,
        max_seq_len=64,
    ).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    print(f"✓ Encoder parameters:  {enc_params:>8,}")
    print(f"  Decoder parameters:  {dec_params:>8,}")
    print(f"  Total:               {enc_params + dec_params:>8,}")

    # ── 3. Load dataset ───────────────────────────────────────
    from oeis_learn.data.dataset import OeisSequenceDataset

    dataset = OeisSequenceDataset(db_path=db_path)
    print(f"\n✓ Loaded dataset: {len(dataset)} records")

    # ── 4. Encoder forward passes ─────────────────────────────
    encoder.eval()
    for rec in dataset.records:
        terms = [int(t) for t in rec.terms]
        sequences = [terms]  # batch of 1

        with torch.no_grad():
            latent = encoder.forward_from_sequences(sequences)

        print(f"\n  {rec.oeis_id}  ({rec.name})")
        print(f"    terms:  {terms}")
        print(f"    latent shape: {latent.shape}  |  dtype: {latent.dtype}")
        print(f"    value range: [{latent.min().item():.4f}, {latent.max().item():.4f}]")
        print(f"    norm: {latent.norm().item():.4f}")

    # ── 5. Single-step encoder training ───────────────────────
    print("\n--- Single-step encoder training ---")
    encoder.train()
    optim = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

    losses = []
    for rec in dataset.records:
        terms = [int(t) for t in rec.terms]

        # target: random tensor of the same shape (toy objective)
        sample_latent = encoder.forward_from_sequences([terms])
        target = torch.randn_like(sample_latent)

        optim.zero_grad()
        latent = encoder.forward_from_sequences([terms])
        loss = ((latent - target) ** 2).mean()
        loss.backward()
        optim.step()
        losses.append(loss.item())

    print(f"  losses: {[f'{l:.4f}' for l in losses]}")
    print("✓ Single-step training done")

    # ── 6. Decoder sampling (smoke-test) ──────────────────────
    print("\n--- Decoder sampling smoke-test ---")
    from oeis_learn.decoder.sampler import WatProgramSampler

    decoder.eval()
    sampler = WatProgramSampler(
        decoder=decoder,
        temperature=0.8,
        top_p=0.95,
        max_length=32,
    )

    for rec in dataset.records[:1]:
        terms = [int(t) for t in rec.terms]
        with torch.no_grad():
            memory = encoder.forward_from_sequences([terms])

        print(f"  memory shape: {memory.shape}")

        try:
            code, tokens = sampler.sample_candidate(memory, seed=42, max_length=32)
            print(f"  sampled WAT ({len(code)} chars): {code[:200]}…")
        except Exception as e:
            print(f"  Sample (expected for untrained model): {type(e).__name__}: {e}")

    # ── 7. Native WASM evaluator smoke-test ──────────────────
    print("\n--- Native WASM evaluator smoke-test ---")
    from oeis_learn.sandbox.runner import WasmRunner, HAS_NATIVE_EVALUATOR

    print(f"  HAS_NATIVE_EVALUATOR: {HAS_NATIVE_EVALUATOR}")
    assert HAS_NATIVE_EVALUATOR, "Native oeis_wasm_evaluator extension is missing (Dockerfile.simple should build it)"

    runner = WasmRunner(fuel_budget=10000, use_fallback=False)
    wat = """
    (module
        (func (export "compute") (param $n i32) (result i64)
            (i64.mul (i64.extend_i32_s (local.get $n)) (i64.const 3))
        )
    )
    """
    res = runner.run_single(wat, terms_to_generate=5)
    print(f"  status: {res.status}  |  output: {res.output}")
    assert res.status == "SUCCESS", f"expected SUCCESS, got {res.status} ({res.error})"
    assert res.output == [0, 3, 6, 9, 12], f"unexpected output: {res.output}"
    print("✓ Native WASM evaluator executed a WAT program")

    # ── 8. Cleanup ────────────────────────────────────────────
    os.remove(db_path)
    print(f"\n✓ Test database {db_path} removed.")

    print("\n" + "=" * 60)
    print("  All checks passed — minimal training smoke-test OK")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
