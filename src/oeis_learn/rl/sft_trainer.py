"""Supervised Fine-Tuning (SFT) Teacher-Forcing Trainer for Policy Warmup."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from oeis_learn.data.models import SyntheticDemonstrationPair
from oeis_learn.data.synthetic_generator import SyntheticDemonstrationDataset, SyntheticDemonstrationGenerator
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import BOS_ID, EOS_ID, PAD_ID, encode_wat
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder

logger = logging.getLogger("oeis_learn.sft_trainer")


class SftTrainer:
    """Trains Transformer Encoder and Decoder via Teacher-Forced Cross-Entropy on synthetic demonstrations."""

    def __init__(
        self,
        dataset_path: str = "data/sft_demonstrations.json",
        output_checkpoint: str = "checkpoints/sft_warmup_best.pt",
        encoder: Optional[TriStreamEncoder] = None,
        decoder: Optional[WatTransformerDecoder] = None,
        epochs: int = 5,
        lr: float = 5.0e-4,
        min_lr: float = 5.0e-5,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        device: Optional[torch.device] = None,
    ):
        self.dataset_path = dataset_path
        self.output_checkpoint = output_checkpoint
        self.epochs = epochs
        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        # Initialize models (FP32 strict)
        self.encoder = encoder or TriStreamEncoder(d_model=256, n_heads=4, n_encoder_layers=4, d_ff=1024)
        self.decoder = decoder or WatTransformerDecoder(d_model=256, n_heads=4, n_decoder_layers=4, d_ff=1024)
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=min_lr)

    def load_or_generate_dataset(self) -> List[SyntheticDemonstrationPair]:
        """Loads synthetic demonstration pairs from disk or generates a fresh set."""
        if os.path.exists(self.dataset_path):
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                import json
                data = json.load(f)
            dataset = SyntheticDemonstrationDataset.from_dict(data)
            logger.info(f"Loaded {len(dataset.samples)} SFT demonstration pairs from {self.dataset_path}")
            return dataset.samples

        logger.info(f"Dataset not found at {self.dataset_path}. Generating 1,000 synthetic demonstrations...")
        gen = SyntheticDemonstrationGenerator()
        dataset = gen.generate_dataset(num_samples=1000)
        gen.save_dataset(dataset, self.dataset_path)
        return dataset.samples

    def train_epoch(self, samples: List[SyntheticDemonstrationPair]) -> float:
        """Runs a single epoch of teacher-forced supervised fine-tuning."""
        self.encoder.train()
        self.decoder.train()

        total_loss = 0.0
        num_batches = 0

        # Shuffle samples
        import random
        shuffled = list(samples)
        random.shuffle(shuffled)

        for i in range(0, len(shuffled), self.batch_size):
            batch_samples = shuffled[i : i + self.batch_size]
            if not batch_samples:
                continue

            # Prepare encoder input sequences: list of lists of integers
            seq_list = [s.terms[:20] for s in batch_samples]

            # Prepare target tokens with BOS and EOS
            tgt_token_lists = []
            for s in batch_samples:
                encoded = [BOS_ID] + encode_wat(s.wat_code) + [EOS_ID]
                tgt_token_lists.append(torch.tensor(encoded, dtype=torch.long))

            # Pad targets to max length in batch
            max_len = max(len(t) for t in tgt_token_lists)
            tgt_batch = torch.full((len(batch_samples), max_len), PAD_ID, dtype=torch.long, device=self.device)
            for b_idx, t_tensor in enumerate(tgt_token_lists):
                tgt_batch[b_idx, : len(t_tensor)] = t_tensor.to(self.device)

            # Teacher forcing: input is tgt[:, :-1], target is tgt[:, 1:]
            dec_input = tgt_batch[:, :-1]
            dec_target = tgt_batch[:, 1:]

            self.optimizer.zero_grad()

            # Encoder forward
            memory = self.encoder.forward_from_sequences(seq_list, device=self.device)  # (batch, seq_len, d_model)

            # Decoder forward with explicit padding attention mask
            pad_mask = (dec_input == PAD_ID)
            logits = self.decoder(dec_input, memory, tgt_key_padding_mask=pad_mask)  # (batch, dec_len, vocab_size)

            loss = self.criterion(logits.reshape(-1, logits.size(-1)), dec_target.reshape(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self.scheduler.step()
        return total_loss / max(1, num_batches)

    def train(self) -> Dict[str, Any]:
        """Executes full SFT pretraining loop and saves model checkpoint."""
        samples = self.load_or_generate_dataset()
        logger.info(f"Starting SFT Warmup Training ({self.epochs} epochs, {len(samples)} samples)...")

        best_loss = float("inf")
        loss_history = []

        for epoch in range(1, self.epochs + 1):
            avg_loss = self.train_epoch(samples)
            loss_history.append(avg_loss)
            logger.info(f"SFT Epoch {epoch:02d}/{self.epochs:02d} | Cross-Entropy Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint(self.output_checkpoint, epoch=epoch, loss=best_loss)

        return {
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "best_loss": best_loss,
            "epochs_trained": self.epochs,
            "checkpoint": self.output_checkpoint,
        }

    def save_checkpoint(self, path: str, epoch: int, loss: float) -> None:
        """Saves encoder & decoder weights to checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "loss": loss,
                "encoder_state_dict": self.encoder.state_dict(),
                "decoder_state_dict": self.decoder.state_dict(),
            },
            path,
        )
        logger.debug(f"Saved SFT checkpoint to {path}")
