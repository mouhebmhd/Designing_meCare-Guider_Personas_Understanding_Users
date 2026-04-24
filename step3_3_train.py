"""
Step 3.3 – Training Loop for HyLitS

Implements:
  - HyLiTSDataset      : PyTorch Dataset wrapping model_ready/*.jsonl
  - WeightedSeq2SeqLoss: cross-entropy scaled by suitability weight
  - aux_classifier_loss: predicts tgt_profile from decoder hidden states
  - HyLiTSTrainer      : training + validation loop with checkpointing

Run:
  C:\\Python310\\python.exe step3_3_train.py

Key hyper-parameters are in the CONFIG block at the top.
"""

import json
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

# ── Re-use components from model setup ───────────────────────────────────────
from step3_2_model_setup import (
    build_tokenizer, build_model,
    ProfileClassifierHead,
    PROFILE_TO_TOKEN, N_PROFILES,
    APPROACH, LORA_R, LORA_ALPHA, LORA_DROPOUT, ADAPTER_DIM,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Paths
    "data_dir":        Path("data/model_ready"),
    "tokenizer_dir":   Path("checkpoints/hylits_init"),
    "checkpoint_dir":  Path("checkpoints/hylits_run"),

    # Model
    "max_input_len":   256,
    "max_target_len":  128,

    # Training
    "epochs":          30,
    "batch_size":      16,
    "grad_accum":      2,           # effective batch = batch_size × grad_accum
    "lr":              5e-4,
    "warmup_ratio":    0.10,
    "max_grad_norm":   1.0,
    "weight_decay":    0.01,

    # Loss
    "use_weighted_loss":  True,     # scale CE by suitability weight
    "aux_loss_weight":    0.2,      # λ for classifier head loss
    "label_smoothing":    0.1,

    # Misc
    "seed":            42,
    "log_every":       25,          # steps
    "eval_every":      200,         # steps
    "save_every":      500,        # steps
    "device":          "cuda" if torch.cuda.is_available() else "cpu",
}


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════

PROFILE_TO_IDX = {"Balanced": 0, "Transitional": 1, "Specialized": 2}


class HyLiTSDataset(Dataset):
    """
    Reads model_ready/{split}.jsonl and tokenises on-the-fly.
    Prepends an [HL:X] control token to the input so the model knows
    which readability level to target.
    """

    def __init__(self, jsonl_path: Path, tokenizer: AutoTokenizer,
                 max_input_len: int = 256, max_target_len: int = 128):
        self.tokenizer      = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len

        with open(jsonl_path, encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Prepend HL control token to tell the model the target readability
        hl_token   = PROFILE_TO_TOKEN.get(rec["tgt_profile"], "[HL:T]")
        input_text = f"{hl_token} {rec['input_text']}"

        enc = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tokenizer(
            rec["target_text"],
            max_length=self.max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = dec["input_ids"].squeeze()
        # Replace padding token id with -100 so it's ignored in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":       enc["input_ids"].squeeze(),
            "attention_mask":  enc["attention_mask"].squeeze(),
            "labels":          labels,
            "weight":          torch.tensor(rec["weight"], dtype=torch.float),
            "tgt_profile_idx": torch.tensor(rec["tgt_profile_idx"], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

class WeightedSeq2SeqLoss(nn.Module):
    """
    Cross-entropy with optional per-example suitability weighting
    and label smoothing.

    If use_weighted: loss_i = CE_i × weight_i
    Weights are normalised within each batch so the mean stays ~ 1.
    """

    def __init__(self, pad_id: int, label_smoothing: float = 0.1,
                 use_weighted: bool = True):
        super().__init__()
        self.pad_id          = pad_id
        self.label_smoothing = label_smoothing
        self.use_weighted    = use_weighted

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        logits  : [B, seq_len, vocab_size]
        labels  : [B, seq_len]   (-100 for padding)
        weights : [B]            suitability weights
        """
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
            reduction="none",
        ).reshape(B, T)                        # [B, T]

        # Mask padding tokens
        valid = (labels != -100).float()       # [B, T]
        loss  = (loss * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # [B]

        # Per-example weighting
        if self.use_weighted and weights is not None:
            # Normalise weights so batch mean ≈ 1
            w    = weights / weights.mean().clamp(min=1e-9)
            loss = loss * w

        return loss.mean()


def aux_classifier_loss(
    clf_head: ProfileClassifierHead,
    encoder_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    tgt_profile_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cross-entropy between predicted profiles and true target profiles.
    This is the auxiliary multi-task loss that encourages the model to
    produce output at the intended readability level.
    """
    logits = clf_head(encoder_hidden, attention_mask)   # [B, N_PROFILES]
    return F.cross_entropy(logits, tgt_profile_ids)


# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class HyLiTSTrainer:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg["device"])
        torch.manual_seed(cfg["seed"])

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])

        # ── Model components ──────────────────────────────────────────────────
        components         = build_model(self.tokenizer)
        self.model         = components["model"].to(self.device)
        self.clf_head      = components["classifier_head"].to(self.device)
        self.controllers   = components.get("controllers")
        if self.controllers is not None:
            self.controllers = self.controllers.to(self.device)

        # ── Datasets & loaders ────────────────────────────────────────────────
        def make_loader(split, shuffle):
            ds = HyLiTSDataset(
                cfg["data_dir"] / f"{split}.jsonl",
                self.tokenizer,
                cfg["max_input_len"],
                cfg["max_target_len"],
            )
            return DataLoader(ds, batch_size=cfg["batch_size"],
                              shuffle=shuffle, num_workers=0, pin_memory=True)

        self.train_loader = make_loader("train", shuffle=True)
        self.val_loader   = make_loader("val",   shuffle=False)

        # ── Optimiser: only trainable params ──────────────────────────────────
        trainable = (
            list(filter(lambda p: p.requires_grad, self.model.parameters()))
            + list(self.clf_head.parameters())
            + (list(self.controllers.parameters()) if self.controllers else [])
        )
        self.optimiser = torch.optim.AdamW(
            trainable, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )

        total_steps  = len(self.train_loader) * cfg["epochs"] // cfg["grad_accum"]
        warmup_steps = int(total_steps * cfg["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser, warmup_steps, total_steps
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        self.seq_loss_fn = WeightedSeq2SeqLoss(
            pad_id          = self.tokenizer.pad_token_id,
            label_smoothing = cfg["label_smoothing"],
            use_weighted    = cfg["use_weighted_loss"],
        )

        self.aux_weight = cfg["aux_loss_weight"]
        self.global_step = 0
        self.best_val_loss = float("inf")
        cfg["checkpoint_dir"].mkdir(parents=True, exist_ok=True)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one batch; return (total_loss, seq_loss)."""
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels         = batch["labels"].to(self.device)
        weights        = batch["weight"].to(self.device)
        profile_ids    = batch["tgt_profile_idx"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # ── Seq2seq loss (weighted CE) ────────────────────────────────────────
        seq_loss = self.seq_loss_fn(outputs.logits, labels, weights)

        # ── Auxiliary classifier loss ─────────────────────────────────────────
        enc_hidden = outputs.encoder_last_hidden_state   # [B, src_len, H]
        clf_loss   = aux_classifier_loss(
            self.clf_head, enc_hidden, attention_mask, profile_ids
        )

        total_loss = seq_loss + self.aux_weight * clf_loss
        return total_loss, seq_loss

    # ── Train one epoch ───────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int):
        self.model.train()
        self.clf_head.train()

        accum_loss  = 0.0
        accum_steps = 0
        t0 = time.time()

        self.optimiser.zero_grad()

        for step, batch in enumerate(self.train_loader):
            total_loss, seq_loss = self._forward(batch)
            (total_loss / self.cfg["grad_accum"]).backward()
            accum_loss  += seq_loss.item()
            accum_steps += 1

            if (step + 1) % self.cfg["grad_accum"] == 0:
                nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad]
                    + list(self.clf_head.parameters()),
                    self.cfg["max_grad_norm"]
                )
                self.optimiser.step()
                self.scheduler.step()
                self.optimiser.zero_grad()
                self.global_step += 1

                if self.global_step % self.cfg["log_every"] == 0:
                    avg = accum_loss / accum_steps
                    lr  = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    print(f"  Epoch {epoch:02d} | step {self.global_step:>6} | "
                          f"loss {avg:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                    accum_loss, accum_steps = 0.0, 0
                    t0 = time.time()

                if self.global_step % self.cfg["eval_every"] == 0:
                    val_loss = self._evaluate()
                    self.model.train()
                    self.clf_head.train()
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save("best")
                        print(f"  ✓ New best val_loss: {val_loss:.4f}")

                if self.global_step % self.cfg["save_every"] == 0:
                    self._save(f"step_{self.global_step}")

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> float:
        self.model.eval()
        self.clf_head.eval()
        total, count = 0.0, 0

        for batch in self.val_loader:
            loss, _ = self._forward(batch)
            total  += loss.item()
            count  += 1

        avg = total / max(count, 1)
        print(f"  [VAL] step {self.global_step} | val_loss {avg:.4f}")
        return avg

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(self, tag: str):
        out = self.cfg["checkpoint_dir"] / tag
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(),     out / "model.pt")
        torch.save(self.clf_head.state_dict(),  out / "clf_head.pt")
        if self.controllers is not None:
            torch.save(self.controllers.state_dict(), out / "adapters.pt")
        self.tokenizer.save_pretrained(out)
        with open(out / "training_state.json", "w") as f:
            json.dump({"global_step": self.global_step,
                       "best_val_loss": self.best_val_loss}, f)

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(self):
        print(f"\n{'='*60}")
        print(f"  HyLitS Training")
        print(f"  Device  : {self.device}")
        print(f"  Approach: {APPROACH.upper()}")
        print(f"  Epochs  : {self.cfg['epochs']}")
        print(f"  Eff batch size: {self.cfg['batch_size'] * self.cfg['grad_accum']}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.cfg["epochs"] + 1):
            print(f"\n── Epoch {epoch}/{self.cfg['epochs']} ──")
            self._train_epoch(epoch)

        # Final evaluation + save
        val_loss = self._evaluate()
        self._save("final")
        print(f"\n✅  Training complete.  Final val_loss: {val_loss:.4f}")
        print(f"   Best checkpoint → {self.cfg['checkpoint_dir'] / 'best'}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    trainer = HyLiTSTrainer(CONFIG)
    trainer.train()


if __name__ == "__main__":
    main()
