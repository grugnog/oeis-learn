"""CLI Workflow for Strategy C Progressive SFT Warmstart Transfer.

Orchestrates:
1. Vocabulary surgery on pre-trained scalar checkpoint
2. Phase 2: Embedding warmup (1,500 steps, frozen backbones)
3. Phase 3: Joint SFT bridge (5,000 steps, discriminative learning rates)
4. Transition Gate evaluation (ACR <= 15%, grammar >= 98.5%, pass >= 75%)
5. Reference policy re-anchoring
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, Dict
import torch
from oeis_learn.curriculum.transition_gate import TransitionGateEvaluator
from oeis_learn.decoder.vocabulary_surgery import perform_vocabulary_surgery
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import TOKEN_TO_ID, VOCAB_SIZE, WAT_VOCABULARY
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.rl.sft_trainer import SftTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("train_warmstart")


def evaluate_bridge_model(encoder, decoder, device, sequences, *, runner=None):
    """Sample actual current weights on distinct diagnostic prompts; no teacher injection."""
    sampler = WatProgramSampler(decoder, max_length=256)
    from oeis_learn.rl.policy import scoring_mode

    groups, seeds = [], []
    with scoring_mode(encoder, decoder), torch.no_grad():
        for index, terms in enumerate(sequences):
            visible = terms[:20]
            memory = encoder.forward_from_sequences([visible] * 4, device=device)
            group_seeds = [7001 + index * 4 + i for i in range(4)]
            seeds.append(group_seeds)
            groups.append([t.source for t in sampler.sample_trajectories(memory, group_seeds)])
    report = TransitionGateEvaluator(runner=runner).evaluate_rollouts(groups, sequences)
    return {
        **report,
        "candidate_origin": "model_sample",
        "sampling_seeds": seeds,
        "qualified": False,
        "purpose": "legacy_model_prefix_diagnostic",
    }


def run_warmstart_pipeline(
    base_checkpoint_path: str = "runs/010_overnight_production_curriculum/checkpoints/model_epoch_050.v2.pt",
    output_dir: str = "runs/011_multilimb_256bit_production/checkpoints",
    stage: str = "all",
    warmup_steps: int = 1500,
    bridge_steps: int = 5000,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Runs the progressive warmstart transfer pipeline."""
    if stage not in {"all", "surgery", "warmup", "bridge", "gate"}:
        raise ValueError("unsupported warmstart stage")
    for name, steps, needed in (
        ("warmup", warmup_steps, stage in {"all", "warmup", "bridge"}),
        ("bridge", bridge_steps, stage in {"all", "bridge"}),
    ):
        if needed and (type(steps) is not int or steps <= 0):
            raise ValueError(f"{name} requires positive exact steps")
    if not os.path.isfile(base_checkpoint_path):
        raise FileNotFoundError(f"Explicit warmstart parent is missing: {base_checkpoint_path}")
    os.makedirs(output_dir, exist_ok=True)
    if dry_run:
        warmup_steps = min(warmup_steps, 5)
        bridge_steps = min(bridge_steps, 5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Target execution device: {device}")

    # 1. Initialize models
    old_vocab_size = 83  # Pre-surgery scalar vocabulary
    if os.path.exists(base_checkpoint_path):
        logger.info(f"Loading weights from base checkpoint: {base_checkpoint_path}")
        ckpt = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
        if (
            not isinstance(ckpt, dict)
            or not ckpt.get("encoder_state_dict")
            or not ckpt.get("decoder_state_dict")
        ):
            raise ValueError("warmstart parent must contain both trained model states")
        enc_cfg = ckpt.get("encoder_config") or ckpt.get("provenance", {}).get("encoder_config")
        if enc_cfg:
            encoder = TriStreamEncoder(**enc_cfg)
        else:
            encoder = TriStreamEncoder(d_model=256, n_heads=4, n_encoder_layers=4, d_ff=1024)
        if "encoder_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state_dict"], strict=True)

        dec_cfg = ckpt.get("decoder_config") or ckpt.get("provenance", {}).get("decoder_config")
        decoder_state = ckpt.get("decoder_state_dict", {})
        if "token_embedding.weight" in decoder_state:
            old_vocab_size = decoder_state["token_embedding.weight"].shape[0]

        if dec_cfg:
            dec_kwargs = dict(dec_cfg)
            dec_kwargs["vocab_size"] = old_vocab_size
            dec_kwargs["logit_cap_threshold"] = 30.0
            decoder = WatTransformerDecoder(**dec_kwargs)
        else:
            decoder = WatTransformerDecoder(
                vocab_size=old_vocab_size,
                d_model=256,
                n_heads=4,
                n_decoder_layers=4,
                d_ff=1024,
                logit_cap_threshold=30.0,
            )

        if "decoder_state_dict" in ckpt:
            decoder.load_state_dict(ckpt["decoder_state_dict"], strict=True)
    else:
        raise FileNotFoundError(f"Explicit warmstart parent is missing: {base_checkpoint_path}")

    encoder.to(device)
    decoder.to(device)

    # 2. Vocabulary surgery
    logger.info(
        f"Executing Phase 1: Vocabulary surgery ({old_vocab_size} -> {VOCAB_SIZE} tokens)..."
    )
    old_token_to_id = {tok: idx for idx, tok in enumerate(WAT_VOCABULARY[:old_vocab_size])}
    decoder = perform_vocabulary_surgery(
        decoder=decoder,
        old_vocab_size=old_vocab_size,
        new_vocab_size=VOCAB_SIZE,
        old_token_to_id=old_token_to_id,
        new_token_to_id=TOKEN_TO_ID,
        logit_cap_threshold=30.0,
    )
    decoder.to(device)
    surgery_ckpt_path = os.path.join(output_dir, "model_post_surgery.pt")
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "vocab_size": VOCAB_SIZE,
        },
        surgery_ckpt_path,
    )
    if stage == "surgery":
        return {
            "status": "SUCCESS",
            "stage": "surgery",
            "checkpoint": surgery_ckpt_path,
            "qualified": False,
            "purpose": "legacy_unverified",
        }

    if stage != "gate":
        # 3. Phase 2: Embedding warmup
        trainer = SftTrainer(
            dataset_path="data/sft_multilimb_demonstrations.json",
            output_checkpoint=os.path.join(output_dir, "sft_warmup.pt"),
            encoder=encoder,
            decoder=decoder,
            device=device,
            multilimb=True,
        )
        logger.info(f"Executing Phase 2: Embedding warmup ({warmup_steps} steps)...")
        warmup_res = trainer.train_embedding_warmup(steps=warmup_steps, lr=1e-4)
        if stage == "warmup":
            return {**warmup_res, "qualified": False, "purpose": "legacy_unverified"}

        # 4. Phase 3: Joint SFT bridge
        logger.info(f"Executing Phase 3: Joint SFT bridge ({bridge_steps} steps)...")
        bridge_res = trainer.train_joint_bridge(
            steps=bridge_steps,
            encoder_lr=1e-5,
            decoder_lr=5e-5,
            heads_lr=1e-4,
        )
        if stage == "bridge":
            return {**bridge_res, "qualified": False, "purpose": "legacy_unverified"}

    # 5. Phase 4: Transition Gate evaluation
    logger.info("Executing Phase 4: Transition Gate evaluation...")
    from oeis_learn.cli.evaluate_canaries import CANARY_METADATA, get_ground_truth_for_canary

    sequences = [get_ground_truth_for_canary(meta, count=20) for meta in CANARY_METADATA]
    gate_res = evaluate_bridge_model(encoder, decoder, device, sequences)
    logger.info(f"Transition Gate Results: {gate_res}")

    final_ckpt_path = os.path.join(output_dir, "sft_bridge_ready.pt")
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "decoder_state_dict": decoder.state_dict(),
            "reference_encoder_state_dict": encoder.state_dict(),
            "reference_decoder_state_dict": decoder.state_dict(),
            "vocab_size": VOCAB_SIZE,
            "gate_passed": gate_res["gate_passed"],
            "qualified": False,
            "purpose": "legacy_unverified",
        },
        final_ckpt_path,
    )
    logger.info(f"Saved ready-for-RL model to {final_ckpt_path}")

    return {
        "status": "SUCCESS",
        "stage": stage,
        "checkpoint": final_ckpt_path,
        "gate_results": gate_res,
        "qualified": False,
        "purpose": "legacy_unverified",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strategy C Progressive SFT Warmstart Transfer CLI"
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="runs/010_overnight_production_curriculum/checkpoints/model_epoch_050.v2.pt",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/011_multilimb_256bit_production/checkpoints"
    )
    parser.add_argument(
        "--stage", type=str, default="all", choices=["all", "surgery", "warmup", "bridge", "gate"]
    )
    parser.add_argument("--warmup-steps", type=int, default=1500)
    parser.add_argument("--bridge-steps", type=int, default=5000)
    parser.add_argument(
        "--dry-run", action="store_true", help="Run 5 steps per phase for sanity checking"
    )

    args = parser.parse_args()
    res = run_warmstart_pipeline(
        base_checkpoint_path=args.base_checkpoint,
        output_dir=args.output_dir,
        stage=args.stage,
        warmup_steps=args.warmup_steps,
        bridge_steps=args.bridge_steps,
        dry_run=args.dry_run,
    )
    print(f"Warmstart Pipeline Finished: {res}")


if __name__ == "__main__":
    main()
