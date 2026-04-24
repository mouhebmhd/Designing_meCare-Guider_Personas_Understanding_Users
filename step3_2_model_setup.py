"""
Step 3.2 – Extend Tokenizer and Set Up Model Architecture

Two approaches implemented (select via APPROACH constant):
  A. LoRA  – rank-decomposition adapters on attention layers (PEFT library)
  B. Profile-Dependent Adapters – one small MLP adapter per HL profile,
     activated by the example's tgt_profile_idx at runtime.

Special tokens added:
  [REPLACE:X→Y]  [EXPLAIN:X]  [SPLIT]   (as atomic single tokens)

Also builds:
  - A profile classifier head (for auxiliary multi-task loss)
  - Saves tokenizer + model config to checkpoints/hylits_init/
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Config

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL  = "google/flan-t5-base"   # swap to "facebook/bart-base" if preferred
APPROACH    = "lora"                  # "lora" | "adapter"
SAVE_DIR    = Path("checkpoints/hylits_init")
N_PROFILES  = 3                       # Balanced / Transitional / Specialized

# LoRA hyper-parameters
LORA_R      = 16    # rank
LORA_ALPHA  = 32    # scaling  (effective lr scale = alpha / r)
LORA_DROPOUT = 0.05

# Profile adapter hyper-parameters
ADAPTER_DIM = 64    # bottleneck hidden size


# ── Special tokens ────────────────────────────────────────────────────────────
# Generic marker tokens — the model learns their semantics from training data.
# Individual jargon-specific markers (e.g. [REPLACE:myocardial→heart attack])
# are NOT added as specials because there would be thousands; instead, the
# tokenizer sees them as the three atomic parts: [REPLACE:, →, ].
SPECIAL_TOKENS = [
    "[REPLACE]",    # generic REPLACE operator
    "[EXPLAIN]",    # generic EXPLAIN operator
    "[SPLIT]",      # sentence-split instruction
    "[HL:B]",       # health literacy target = Balanced
    "[HL:T]",       # health literacy target = Transitional
    "[HL:S]",       # health literacy target = Specialized
]

PROFILE_TO_TOKEN = {
    "Balanced":     "[HL:B]",
    "Transitional": "[HL:T]",
    "Specialized":  "[HL:S]",
}


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def build_tokenizer(save_dir: Path) -> AutoTokenizer:
    print(f"  Loading base tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    n_before = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    n_after  = len(tokenizer)
    print(f"  Vocabulary: {n_before} → {n_after} tokens  "
          f"({n_after - n_before} new special tokens added)")

    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(save_dir)
    print(f"  Tokenizer saved → {save_dir}")
    return tokenizer


# ── LoRA layer ────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adapters.
    Only A and B matrices are trained; the original weight W is frozen.
    Output = W·x + (B·A·x) * (alpha/r)
    """
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.r       = r
        self.scaling = alpha / r
        d_out, d_in = linear.weight.shape

        # Freeze original weight
        self.weight = linear.weight
        self.bias   = linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Trainable LoRA matrices
        self.lora_A   = nn.Parameter(torch.randn(r, d_in)  * 0.01)
        self.lora_B   = nn.Parameter(torch.zeros(d_out, r))
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base   = torch.nn.functional.linear(x, self.weight, self.bias)
        lora   = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + lora * self.scaling


def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float) -> nn.Module:
    """
    Replace q_proj / v_proj (or q / v for T5) in every attention layer
    with LoRALinear. All other weights stay frozen.
    """
    frozen = 0
    adapted = 0

    for name, module in model.named_modules():
        # T5 uses .q, .v;  BART uses .q_proj, .v_proj
        for attr in ("q", "v", "q_proj", "v_proj", "k", "k_proj"):
            child = getattr(module, attr, None)
            if isinstance(child, nn.Linear):
                setattr(module, attr, LoRALinear(child, r, alpha, dropout))
                adapted += 1

    for p in model.parameters():
        if not p.requires_grad:
            frozen += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  LoRA injected: {adapted} layers adapted")
    print(f"  Trainable params: {trainable:,} / {total:,}  "
          f"({100*trainable/total:.2f}%)")
    return model


# ── Profile-Dependent Adapter ─────────────────────────────────────────────────

class ProfileAdapter(nn.Module):
    """
    A lightweight bottleneck adapter: down-project → ReLU → up-project.
    One instance per HL profile; activated by tgt_profile_idx at runtime.
    """
    def __init__(self, hidden_size: int, bottleneck: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck)
        self.up   = nn.Linear(bottleneck, hidden_size)
        self.act  = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden + self.up(self.act(self.down(hidden))))


class AdapterController(nn.Module):
    """
    Holds N_PROFILES adapters and routes each batch element to the
    correct adapter based on profile_ids (LongTensor of shape [B]).
    """
    def __init__(self, n_profiles: int, hidden_size: int, bottleneck: int):
        super().__init__()
        self.adapters = nn.ModuleList(
            [ProfileAdapter(hidden_size, bottleneck) for _ in range(n_profiles)]
        )

    def forward(self, hidden: torch.Tensor, profile_ids: torch.Tensor) -> torch.Tensor:
        """
        hidden     : [B, seq_len, hidden_size]
        profile_ids: [B]
        """
        out = torch.zeros_like(hidden)
        for pid in profile_ids.unique():
            mask = (profile_ids == pid)             # [B]
            out[mask] = self.adapters[pid](hidden[mask])
        return out


def inject_adapters(model: nn.Module, n_profiles: int,
                    bottleneck: int) -> tuple[nn.Module, nn.ModuleList]:
    """
    Freeze the entire base model, then attach one AdapterController
    per encoder layer and one per decoder layer.
    Returns (model, list_of_controllers).
    """
    for p in model.parameters():
        p.requires_grad_(False)

    hidden_size  = model.config.d_model
    controllers  = nn.ModuleList()

    # Attach to encoder layers
    for layer in model.encoder.block:
        ctrl = AdapterController(n_profiles, hidden_size, bottleneck)
        layer.adapter = ctrl
        controllers.append(ctrl)

    # Attach to decoder layers
    for layer in model.decoder.block:
        ctrl = AdapterController(n_profiles, hidden_size, bottleneck)
        layer.adapter = ctrl
        controllers.append(ctrl)

    trainable = sum(p.numel() for c in controllers for p in c.parameters())
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Adapters injected: {len(controllers)} controllers "
          f"({n_profiles} profiles × {len(controllers)//2} enc+dec layers)")
    print(f"  Trainable params: {trainable:,} / {total:,}  "
          f"({100*trainable/total:.2f}%)")

    return model, controllers


# ── Profile Classifier Head ───────────────────────────────────────────────────

class ProfileClassifierHead(nn.Module):
    """
    Auxiliary classification head: predicts HL profile from the mean-pooled
    encoder output of the GENERATED (or target) sequence.
    Encourages the model to produce text at the intended readability level.
    """
    def __init__(self, hidden_size: int, n_classes: int = N_PROFILES, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(hidden_size, n_classes)

    def forward(self, encoder_hidden: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        encoder_hidden : [B, seq_len, H]
        attention_mask : [B, seq_len]
        returns logits : [B, n_classes]
        """
        mask   = attention_mask.unsqueeze(-1).float()
        pooled = (encoder_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.linear(self.dropout(pooled))


# ── Full model builder ────────────────────────────────────────────────────────

def build_model(tokenizer: AutoTokenizer) -> dict:
    """
    Loads the base model, resizes embeddings for new tokens,
    injects LoRA or adapters, adds the classifier head.

    Returns a dict with all components:
      {model, classifier_head, controllers (adapter only), tokenizer}
    """
    print(f"\n  Loading base model: {BASE_MODEL}")
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # Resize embedding table to cover new special tokens
    old_vocab = model.config.vocab_size
    model.resize_token_embeddings(len(tokenizer))
    print(f"  Embedding resized: {old_vocab} → {len(tokenizer)}")

    hidden_size = model.config.d_model
    components  = {"tokenizer": tokenizer}

    if APPROACH == "lora":
        print(f"\n  Injecting LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
        model = inject_lora(model, LORA_R, LORA_ALPHA, LORA_DROPOUT)
        components["controllers"] = None

    elif APPROACH == "adapter":
        print(f"\n  Injecting Profile-Dependent Adapters (bottleneck={ADAPTER_DIM})")
        model, controllers = inject_adapters(model, N_PROFILES, ADAPTER_DIM)
        components["controllers"] = controllers

    else:
        raise ValueError(f"Unknown APPROACH: {APPROACH!r} — use 'lora' or 'adapter'")

    # Classifier head (always added regardless of approach)
    clf_head = ProfileClassifierHead(hidden_size, N_PROFILES)
    print(f"  Classifier head: {hidden_size}→{N_PROFILES} logits")

    components["model"]            = model
    components["classifier_head"]  = clf_head
    return components


# ── Save ──────────────────────────────────────────────────────────────────────

def save_components(components: dict, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model (state dict only — config is preserved in save_pretrained)
    torch.save(components["model"].state_dict(), save_dir / "model_init.pt")

    # Save classifier head
    torch.save(components["classifier_head"].state_dict(),
               save_dir / "classifier_head_init.pt")

    # Save adapter controllers if present
    if components.get("controllers") is not None:
        torch.save(components["controllers"].state_dict(),
                   save_dir / "adapters_init.pt")

    # Save config metadata
    meta = {
        "base_model":   BASE_MODEL,
        "approach":     APPROACH,
        "special_tokens": SPECIAL_TOKENS,
        "profile_to_token": PROFILE_TO_TOKEN,
        "profile_to_idx": {"Balanced": 0, "Transitional": 1, "Specialized": 2},
        "lora_r":       LORA_R,
        "lora_alpha":   LORA_ALPHA,
        "adapter_dim":  ADAPTER_DIM,
    }
    with open(save_dir / "hylits_config.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Components saved → {save_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  HyLitS – Tokenizer & Model Setup")
    print("=" * 60)

    tokenizer  = build_tokenizer(SAVE_DIR)
    components = build_model(tokenizer)
    save_components(components, SAVE_DIR)

    print("\n✅  Model setup complete.")
    print(f"   Approach : {APPROACH.upper()}")
    print(f"   Saved to : {SAVE_DIR}\n")


if __name__ == "__main__":
    main()
