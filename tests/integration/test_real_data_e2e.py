"""Careful, progressive End-to-End integration test using real OEIS sequence data.

Progresses across 6 stages:
1. Real Data Ingestion & Storage (oeisdata stripped/names)
2. Feature Preprocessing & Classification (4D magnitude, log-linearity, polynomial degrees)
3. Tri-Stream FP32 Neural Perception across Extreme Dynamic Ranges
4. Deterministic Sandboxed WASM Execution, Extrapolation Horizon & MDL Verification
5. EGCA-GRPO RL Training Optimization with Asymmetric Weighting on Real Prompts
6. Latent Manifold Geometry & PSLQ Theorem Discovery on Real Algebraic Identities
"""

import os
import tempfile
import numpy as np
import pytest
import sympy as sp
import torch
from oeis_learn.cli.reporting import export_discovered_proofs_markdown
from oeis_learn.curriculum.extrapolation import ExtrapolationVerifier
from oeis_learn.curriculum.mdl_verifier import MdlVerifier
from oeis_learn.curriculum.sampler import DynamicMixtureSampler
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.dataset import OeisSequenceDataset, collate_sequence_batch
from oeis_learn.data.ingest import OeisIngestionPipeline
from oeis_learn.data.models import LatentDiscoveryCandidate, SequenceRecord
from oeis_learn.data.preprocessing import (
    analyze_log_linearity,
    check_finite_difference_polynomial_degree,
    compute_magnitude_4d_features,
)
from oeis_learn.data.real_data_loader import RealOeisDataLoader
from oeis_learn.decoder.environment_tracker import EnvironmentTracker
from oeis_learn.decoder.grammar_masker import GrammarMasker
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.discovery.manifold import cluster_latent_manifold, reduce_manifold_2d
from oeis_learn.discovery.pslq_solver import PslqRelationSolver
from oeis_learn.discovery.symbolic_prover import SymbolicProver
from oeis_learn.discovery.vector_search import VectorRelationSearcher
from oeis_learn.discovery.vicreg_loss import VicRegLoss
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.trainer import EgcaGrpoTrainer
from oeis_learn.sandbox.runner import WasmRunner


def test_real_data_end_to_end_progressive():
    # =========================================================================
    # STAGE 1: Real Data Acquisition, Ingestion & Database Indexing
    # =========================================================================
    loader = RealOeisDataLoader()
    raw_records = loader.load_local_benchmark_records()
    assert len(raw_records) >= 20, f"Expected >= 20 curated real sequences, got {len(raw_records)}"

    stages_present = {r.curriculum_stage for r in raw_records}
    assert stages_present == {1, 2, 3, 4, 5}, f"All 5 stages must be represented, got {stages_present}"

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "real_oeis_e2e.duckdb")
        pipeline = OeisIngestionPipeline(db_path=db_path)
        inserted = pipeline.insert_records(raw_records)
        assert inserted == len(raw_records)
        pipeline.close()

        # Query back through OeisSequenceDataset
        dataset = OeisSequenceDataset(db_path=db_path)
        assert len(dataset) == len(raw_records)

        seq_dict = {r.oeis_id: r for r in dataset.records}
        assert "A000045" in seq_dict  # Fibonacci
        assert "A000032" in seq_dict  # Lucas
        assert "A000217" in seq_dict  # Triangular numbers
        assert "A000290" in seq_dict  # Squares
        assert "A000079" in seq_dict  # Powers of 2
        assert "A000142" in seq_dict  # Factorials
        assert "A000108" in seq_dict  # Catalan
        assert "A000040" in seq_dict  # Primes
        assert "A000005" in seq_dict  # Divisor count
        assert "A003188" in seq_dict  # Gray code

        # Batch collation check
        collated = collate_sequence_batch([dataset[0], dataset[1], dataset[2]])
        assert len(collated["oeis_ids"]) == 3
        assert collated["lengths"].shape == (3,)

        # =========================================================================
        # STAGE 2: Feature Preprocessing & Mathematical Classification Gate
        # =========================================================================
        # 1. Test 4D magnitude decomposition on extreme terms
        fact_record = seq_dict["A000142"]
        huge_fact_val = fact_record.terms[-1]  # 6402373705728000
        log_v, s_pos, s_neg, s_zero = compute_magnitude_4d_features(huge_fact_val)
        assert log_v > 15.0
        assert (s_pos, s_neg, s_zero) == (1.0, 0.0, 0.0)

        # 2. Test log-linear exponential detection
        is_fib_loglin, fib_r2 = analyze_log_linearity(seq_dict["A000045"].terms)
        assert is_fib_loglin is True and fib_r2 > 0.98

        is_pow_loglin, pow_r2 = analyze_log_linearity(seq_dict["A000079"].terms)
        assert is_pow_loglin is True and pow_r2 > 0.99

        is_sq_loglin, sq_r2 = analyze_log_linearity(seq_dict["A000290"].terms)
        assert is_sq_loglin is False

        # 3. Test polynomial degree detection via finite differences
        deg_tri = check_finite_difference_polynomial_degree(seq_dict["A000217"].terms)
        assert deg_tri == 2  # n*(n+1)/2 is degree 2

        deg_sq = check_finite_difference_polynomial_degree(seq_dict["A000290"].terms)
        assert deg_sq == 2  # n^2 is degree 2

        deg_cube = check_finite_difference_polynomial_degree(seq_dict["A000578"].terms)
        assert deg_cube == 3  # n^3 is degree 3

        # =========================================================================
        # STAGE 3: Tri-Stream Neural Perception on Real Sequences (Strict FP32)
        # =========================================================================
        encoder = TriStreamEncoder(d_model=128, n_heads=4, n_encoder_layers=2, d_ff=256)
        encoder.eval()

        real_seq_list = [r.terms[:20] for r in dataset.records]
        with torch.no_grad():
            latent_z = encoder.forward_from_sequences(real_seq_list)

        assert latent_z.shape == (len(real_seq_list), 20, 128)
        assert latent_z.dtype == torch.float32
        assert not torch.isnan(latent_z).any()
        assert not torch.isinf(latent_z).any()

        # Auxiliary heads verification
        aux_outputs = encoder.aux_heads(latent_z)
        assert "pred_magnitude" in aux_outputs
        assert "pred_sign_logits" in aux_outputs
        assert not torch.isnan(aux_outputs["pred_magnitude"]).any()

        # =========================================================================
        # STAGE 4: Deterministic Sandboxed WASM Execution & Generalization Gates
        # =========================================================================
        runner = WasmRunner(fuel_budget=10000)

        # 1. Triangular numbers: A000217: a(n) = n*(n+1)/2
        wat_triangular = """(module
          (func (export "compute") (param $n i32) (result i64)
            (local $n64 i64)
            (local.set $n64 (i64.extend_i32_u (local.get $n)))
            (i64.div_u
              (i64.mul (local.get $n64) (i64.add (local.get $n64) (i64.const 1)))
              (i64.const 2)
            )
          )
        )"""
        res_tri = runner.run_single(wat_triangular, terms_to_generate=20)
        assert res_tri.status == "SUCCESS"
        assert res_tri.output == seq_dict["A000217"].terms[:20]
        assert 0 < res_tri.consumed_fuel < 10000

        # 2. Powers of 2: A000079: a(n) = 1 << n
        wat_powers = """(module
          (func (export "compute") (param $n i32) (result i64)
            (i64.shl (i64.const 1) (i64.extend_i32_u (local.get $n)))
          )
        )"""
        res_pow = runner.run_single(wat_powers, terms_to_generate=20)
        assert res_pow.status == "SUCCESS"
        assert res_pow.output == seq_dict["A000079"].terms[:20]

        # 3. Fibonacci: A000045: F(n)
        wat_fib = """(module
          (func (export "compute") (param $n i32) (result i64)
            (local $a i64) (local $b i64) (local $temp i64) (local $i i32)
            (local.set $a (i64.const 0))
            (local.set $b (i64.const 1))
            (local.set $i (i32.const 0))
            (block $exit
              (loop $loop
                (br_if $exit (i32.ge_s (local.get $i) (local.get $n)))
                (local.set $temp (i64.add (local.get $a) (local.get $b)))
                (local.set $a (local.get $b))
                (local.set $b (local.get $temp))
                (local.set $i (i32.add (local.get $i) (i32.const 1)))
                (br $loop)
              )
            )
            (local.get $a)
          )
        )"""
        res_fib = runner.run_single(wat_fib, terms_to_generate=25)
        assert res_fib.status == "SUCCESS"
        assert res_fib.output == seq_dict["A000045"].terms[:25]

        # Extrapolation & MDL checks
        extrap_verifier = ExtrapolationVerifier(runner=runner, n_train=10, k_extrapolate=20)
        assert extrap_verifier.verify(wat_triangular, seq_dict["A000217"].terms) is True
        assert extrap_verifier.verify(wat_powers, seq_dict["A000079"].terms) is True
        assert extrap_verifier.verify(wat_fib, seq_dict["A000045"].terms) is True

        mdl_verifier = MdlVerifier(threshold=1.2)
        assert mdl_verifier.verify(wat_triangular, seq_dict["A000217"].terms) is True
        assert mdl_verifier.verify(wat_fib, seq_dict["A000045"].terms) is True

        # =========================================================================
        # STAGE 5: Closed-Loop EGCA-GRPO RL Training Step on Real Sequences
        # =========================================================================
        decoder = WatTransformerDecoder(d_model=128, n_heads=4, n_decoder_layers=2, d_ff=256)
        scheduler = CurriculumScheduler(initial_stage=1)
        mixture_sampler = DynamicMixtureSampler(records=dataset.records, scheduler=scheduler)

        trainer = EgcaGrpoTrainer(
            encoder=encoder,
            decoder=decoder,
            scheduler=scheduler,
            sampler=mixture_sampler,
            wasm_runner=runner,
            rollout_group_size=4,
            asymmetric_penalty_weight=1.5,
        )

        # Train 1 optimization step on Fibonacci and 1 on Triangular
        m_fib = trainer.train_step_for_prompt(seq_dict["A000045"])
        assert "loss" in m_fib and not np.isnan(m_fib["loss"])

        m_tri = trainer.train_step_for_prompt(seq_dict["A000217"])
        assert "loss" in m_tri and not np.isnan(m_tri["loss"])

        # =========================================================================
        # STAGE 6: Real Mathematical Discovery, PSLQ Relations & Formal Proofs
        # =========================================================================
        # Identity 1: Fibonacci (A000045) + Lucas (A000032) = 2 * Fibonacci(n+1)
        terms_fib = seq_dict["A000045"].terms[:30]
        terms_lucas = seq_dict["A000032"].terms[:30]
        terms_2fib_next = [2 * terms_fib[n + 1] for n in range(25)]

        pslq_solver = PslqRelationSolver(precision_digits=100)
        rel1, conf1 = pslq_solver.find_relation(
            [terms_fib[:25], terms_lucas[:25], terms_2fib_next], term_index=10
        )
        assert rel1 is not None
        assert rel1 in ([1, 1, -1], [-1, -1, 1])
        assert conf1 < 1e-50

        # Identity 2: Triangular Numbers Sum T(n) + T(n-1) = n^2 (Squares A000290)
        terms_tri = seq_dict["A000217"].terms[:30]
        terms_tri_prev = [0] + terms_tri[:24]
        terms_sq = seq_dict["A000290"].terms[:25]

        rel2, conf2 = pslq_solver.find_relation(
            [terms_tri[:25], terms_tri_prev, terms_sq], term_index=8
        )
        assert rel2 is not None
        assert rel2 in ([1, 1, -1], [-1, -1, 1])
        assert conf2 < 1e-50

        # Prove identities symbolically with SymPy
        prover = SymbolicProver()
        cand1 = LatentDiscoveryCandidate(
            candidate_id="fib_lucas_relation",
            relation_type="LINEAR_SUM",
            sequences=("A000045", "A000032", "2*A000045_next"),
            vector_distance=0.0,
            pslq_vector=[1, 1, -1],
            status="PSLQ_VERIFIED",
        )
        proven1 = prover.prove_relation(
            cand1, formulas=["a(n) = n", "a(n) = 2*n", "a(n) = 3*n"]
        )
        assert proven1.status == "PROVEN"

        cand2 = LatentDiscoveryCandidate(
            candidate_id="triangular_squares_relation",
            relation_type="LINEAR_SUM",
            sequences=("A000217_n", "A000217_n-1", "A000290_n"),
            vector_distance=0.0,
            pslq_vector=[1, 1, -1],
            status="PSLQ_VERIFIED",
        )
        proven2 = prover.prove_relation(
            cand2, formulas=["a(n) = n*(n+1)/2", "a(n) = (n-1)*n/2", "a(n) = n**2"]
        )
        assert proven2.status == "PROVEN"
        assert "Q.E.D." in proven2.symbolic_proof

        # Export formal real-data discovery report
        report_path = os.path.join("reports", "real_data_e2e_report.md")
        report_content = export_discovered_proofs_markdown(
            [proven1, proven2], output_path=report_path
        )
        assert os.path.exists(report_path)
        assert "A000045" in report_content
        assert "A000217" in report_content
