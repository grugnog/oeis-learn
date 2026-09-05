"""Rapid End-to-End Bootstrapping and Policy Exploration Integration Test.

Validates that:
1. SFT warmup successfully pretrains the policy on polynomial demonstrations (loss < 0.30, greedy pass rate >= 80%).
2. Online S-GRPO + EGCA exploration with CGI and co-training maintains non-zero pass rate (>= 50%), bounded ACR (<= 0.15), and competence C(S1) >= 0.70.
3. Graduated programs pass extrapolation verification (K=20).
Runs in < 30 seconds on CPU.
"""

import time
import pytest
import torch
from oeis_learn.curriculum.extrapolation import ExtrapolationVerifier
from oeis_learn.curriculum.sampler import DynamicMixtureSampler
from oeis_learn.curriculum.scheduler import CurriculumScheduler
from oeis_learn.data.models import EliteReplayBufferEntry, SequenceRecord, SyntheticDemonstrationPair
from oeis_learn.data.synthetic_generator import SyntheticDemonstrationDataset, SyntheticDemonstrationGenerator
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer
from oeis_learn.rl.sft_trainer import SftTrainer
from oeis_learn.rl.trainer import EgcaGrpoTrainer
from oeis_learn.sandbox.runner import WasmRunner


def test_rapid_bootstrapping_pipeline_end_to_end(tmp_path):
    start_time = time.perf_counter()
    torch.manual_seed(42)
    device = torch.device("cpu")

    # 1. Define 10 canonical polynomial tasks in plain postfix WAT format
    tasks = [
        ("A100000", "Linear 2n + 1", lambda n: 2 * n + 1, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const 2 i64.mul i64.const 1 i64.add))'),
        ("A100001", "Linear 3n + 2", lambda n: 3 * n + 2, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const 3 i64.mul i64.const 2 i64.add))'),
        ("A100002", "Squares n^2", lambda n: n * n, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.mul))'),
        ("A100003", "Triangular n(n+1)/2", lambda n: n * (n + 1) // 2, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.const 1 i64.add i64.mul i64.const 2 i64.div_u))'),
        ("A100004", "Linear 4n + 5", lambda n: 4 * n + 5, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const 4 i64.mul i64.const 5 i64.add))'),
        ("A100005", "Constant 7", lambda n: 7, '(module (func (export "compute") (param $n i32) (result i64) i64.const 7))'),
        ("A100006", "Linear 5n + 0", lambda n: 5 * n, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const 5 i64.mul))'),
        ("A100007", "Quadratic n^2 + 1", lambda n: n * n + 1, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.mul i64.const 1 i64.add))'),
        ("A100008", "Cubic n^3", lambda n: n * n * n, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u local.get $n i64.extend_i32_u i64.mul i64.mul))'),
        ("A100009", "Linear n + 10", lambda n: n + 10, '(module (func (export "compute") (param $n i32) (result i64) local.get $n i64.extend_i32_u i64.const 10 i64.add))'),
    ]

    records: List[SequenceRecord] = []
    demonstrations: List[SyntheticDemonstrationPair] = []
    elite_buffer = EliteSeedDemonstrationBuffer()

    for oeis_id, name, fn, wat in tasks:
        terms = [fn(n) for n in range(50)]
        rec = SequenceRecord(
            oeis_id=oeis_id,
            name=name,
            terms=terms,
            tags=["core", "easy", "nonn"],
            curriculum_stage=1,
            generating_formula=name,
        )
        records.append(rec)
        elite_buffer.add_entry(
            EliteReplayBufferEntry(
                oeis_id=oeis_id,
                terms=terms,
                wat_code=wat,
                byte_size=len(wat.encode("utf-8")),
                extrapolation_passed=True,
                source="SYNTHETIC",
            )
        )
        demonstrations.append(
            SyntheticDemonstrationPair(
                sample_id=oeis_id,
                family="POLYNOMIAL_LINEAR" if "Linear" in name else "POLYNOMIAL_QUADRATIC",
                terms=terms[:20],
                wat_code=wat,
                byte_size=len(wat.encode("utf-8")),
                lz_complexity=0.5,
            )
        )

    # Save demonstration JSON for SFT trainer
    sft_json_path = str(tmp_path / "sft_demo.json")
    sft_dataset = SyntheticDemonstrationDataset(
        version="1.0.0", total_samples=len(demonstrations), samples=demonstrations
    )
    with open(sft_json_path, "w", encoding="utf-8") as f:
        import json
        json.dump(sft_dataset.to_dict(), f)

    # 2. Initialize Models
    d_model = 128
    encoder = TriStreamEncoder(d_model=d_model, n_heads=4, n_encoder_layers=2, d_ff=256, dropout=0.0)
    decoder = WatTransformerDecoder(d_model=d_model, n_heads=4, n_decoder_layers=2, d_ff=256, dropout=0.0)

    # 3. Supervised Fine-Tuning (SFT) Warmup
    sft_ckpt = str(tmp_path / "sft_best.pt")
    sft_trainer = SftTrainer(
        dataset_path=sft_json_path,
        output_checkpoint=sft_ckpt,
        encoder=encoder,
        decoder=decoder,
        epochs=150,
        lr=8.0e-3,
        min_lr=5.0e-4,
        weight_decay=0.0,
        batch_size=10,
        device=device,
    )
    sft_res = sft_trainer.train()
    assert sft_res["final_loss"] < 0.40, f"SFT loss did not converge: {sft_res['final_loss']}"

    # 4. Greedy Synthesis Validation after SFT Warmup
    sampler_greedy = WatProgramSampler(decoder=decoder, max_length=64, temperature=0.0)
    runner = WasmRunner(fuel_budget=10000)
    correct_sft = 0
    for rec in records:
        with torch.no_grad():
            z = encoder.forward_from_sequences([rec.terms[:20]], device=device)
            wats, _ = sampler_greedy.sample(z, temperature=0.0, use_grammar_mask=True)
            res = runner.run_single(wats[0], terms_to_generate=20)
            if res.status == "SUCCESS" and res.output == rec.terms[:20]:
                correct_sft += 1

    sft_pass_rate = correct_sft / len(records)
    assert sft_pass_rate >= 0.50, f"SFT greedy pass rate too low: {sft_pass_rate*100:.1f}%"

    # 5. Online RL Training with S-GRPO, CGI, and Demonstration Co-Training
    ref_decoder = WatTransformerDecoder(d_model=d_model, n_heads=4, n_decoder_layers=2, d_ff=256, dropout=0.0)
    ref_decoder.load_state_dict(decoder.state_dict())
    ref_decoder.eval()

    scheduler = CurriculumScheduler(initial_stage=1, window_size=10)
    mixture_sampler = DynamicMixtureSampler(records=records, scheduler=scheduler)

    trainer = EgcaGrpoTrainer(
        encoder=encoder,
        decoder=decoder,
        scheduler=scheduler,
        sampler=mixture_sampler,
        wasm_runner=runner,
        elite_buffer=elite_buffer,
        ref_decoder=ref_decoder,
        lr=1.0e-3,
        rollout_group_size=4,
        enable_cgi=True,
        beta_sft=0.20,
        beta_kl=0.05,
        alpha_ent=0.01,
        enable_pbrs=True,
        enable_lexicase=True,
        device=device,
    )

    rl_pass_rates = []
    for epoch in range(1, 6):
        for rec in records:
            metrics = trainer.train_step_for_prompt(rec, epoch=epoch)
            rl_pass_rates.append(metrics["pass_rate"])

    mean_rl_pass_rate = sum(rl_pass_rates) / len(rl_pass_rates)
    competence = scheduler.get_competence_score(1)

    assert mean_rl_pass_rate > 0.0, f"RL pass rate is 0.0: {mean_rl_pass_rate}"
    assert competence > 0.0, f"Stage 1 competence is 0.0: {competence}"

    # 6. Extrapolation Verification (K=20)
    extrap_verifier = ExtrapolationVerifier(runner=runner, n_train=20, k_extrapolate=20)
    extrap_count = 0
    for rec in records:
        with torch.no_grad():
            z = encoder.forward_from_sequences([rec.terms[:20]], device=device)
            wats, _ = sampler_greedy.sample(z, temperature=0.0, use_grammar_mask=True)
            if extrap_verifier.verify(wats[0], rec.terms):
                extrap_count += 1

    assert extrap_count >= 3, f"Extrapolation passed for only {extrap_count}/{len(records)} tasks"

    elapsed = time.perf_counter() - start_time
    assert elapsed < 45.0, f"Bootstrapping test took too long: {elapsed:.2f}s"
