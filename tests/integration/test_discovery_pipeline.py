"""Integration test for complete discovery pipeline: Vector Search -> PSLQ -> SymPy Proof."""

import numpy as np
import pytest
from oeis_learn.discovery.manifold import cluster_latent_manifold, reduce_manifold_2d
from oeis_learn.discovery.pslq_solver import PslqRelationSolver
from oeis_learn.discovery.symbolic_prover import SymbolicProver
from oeis_learn.discovery.vector_search import VectorRelationSearcher


def test_discovery_pipeline_end_to_end():
    # 1. Mock latent embeddings satisfying vA + vB = vC
    va = np.array([1.0, 0.0, 0.5, 0.2], dtype=np.float32)
    vb = np.array([0.0, 1.0, 0.2, 0.3], dtype=np.float32)
    vc = va + vb  # exact sum

    embeddings = {
        "A000001": va,
        "A000002": vb,
        "A000003": vc,
    }

    # 2. Vector search finds relation triple
    searcher = VectorRelationSearcher(eps_distance=0.1)
    candidates = searcher.search_additive_triples(embeddings)
    assert len(candidates) >= 1
    cand = candidates[0]
    assert cand.sequences == ("A000001", "A000002", "A000003")

    # 3. PSLQ verifies integer relation
    terms_dict = {
        "A000001": [n for n in range(30)],
        "A000002": [2 * n for n in range(30)],
        "A000003": [3 * n for n in range(30)],
    }
    pslq_solver = PslqRelationSolver(precision_digits=100)
    verified = pslq_solver.verify_candidate(cand, terms_dict)
    assert verified.status == "PSLQ_VERIFIED"
    assert verified.pslq_vector is not None

    # 4. Symbolic prover proves identity
    prover = SymbolicProver()
    proven = prover.prove_relation(
        verified, formulas=["a(n) = n", "a(n) = 2*n", "a(n) = 3*n"]
    )
    assert proven.status == "PROVEN"
    assert "Q.E.D." in proven.symbolic_proof

    # 5. Manifold 2D reduction and clustering
    mat = np.stack([va, vb, vc] * 3, axis=0)
    reduced = reduce_manifold_2d(mat)
    assert reduced.shape == (9, 2)
    clusters = cluster_latent_manifold(mat)
    assert len(clusters) == 9


def test_discovery_pipeline_orchestration(tmp_path):
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
    from oeis_learn.discovery.pipeline import run_discovery_pipeline
    from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
    from oeis_learn.evaluation.checkpoint import save_checkpoint_v2
    import torch

    enc_cfg = {"d_model": 64, "n_heads": 2, "n_encoder_layers": 2, "d_ff": 128, "dropout": 0.0, "primes": [3, 5], "max_valuation": 16, "use_film": True}
    dec_cfg = {"d_model": 64, "n_heads": 2, "n_decoder_layers": 2, "d_ff": 128, "dropout": 0.0}
    enc = TriStreamEncoder(**enc_cfg)
    dec = WatTransformerDecoder(**dec_cfg)
    ckpt_file = str(tmp_path / "disc_test.v2.pt")
    save_checkpoint_v2(ckpt_file, enc, dec, enc_cfg, dec_cfg, epoch=1, producer_version="test")

    manifest_path = "data/benchmarks/trustworthy_synthesis_v1.json"
    proto_path = "configs/discovery_protocol_v1.json"
    defs_path = "data/benchmarks/symbolic_definitions_v1.json"

    res = run_discovery_pipeline(
        checkpoint_path=ckpt_file,
        manifest_path=manifest_path,
        protocol_path=proto_path,
        definitions_path=defs_path,
        seed=42,
    )

    assert "claims" in res
    assert "summary" in res
    assert "protocol" in res
    assert res["summary"]["latent_candidates"] >= 0
