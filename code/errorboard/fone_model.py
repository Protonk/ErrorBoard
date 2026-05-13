"""FoNE-augmented GPT for the FP8-add task.

Reuses the GPT building blocks (RMSNorm, Block) and adds:
  - input-side: FoNE features added at NUM-token positions
                (Zhou Def 3.4 step 5)
  - output-side: a per-digit cosine-similarity head reading the first
                 FONE_DIM dims of the result-position hidden state.
                 The standard lm_head is kept for sign prediction at
                 SIGN-token positions (and for any other vocab tokens).

Loss is a sum of (a) standard cross-entropy at the SIGN_c position and
(b) per-digit cross-entropy across the 6 digit positions at NUM_c.
For NaN samples the digit-loss is masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fone_encoder import FoneDigitHead, FoneEmbeddingAdd, fone_features
from .fone_tokenizer import (
    FONE_DIM,
    N_DIGITS,
    NUM_ID,
    POS_NUM_C,
    POS_SIGN_C,
    SEQ_LEN,
    SIGN_NAN_ID,
    SIGN_NEG_ID,
    SIGN_POS_ID,
    VOCAB_SIZE,
)
from .model import Block, GPTConfig, RMSNorm


@dataclass
class FoneGPTConfig:
    block_size: int = SEQ_LEN - 1
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 48
    d_mlp: int = 192
    pos_encoding: Literal["learned", "rope"] = "learned"
    init_std: float = 0.02

    def as_gpt_config(self) -> GPTConfig:
        return GPTConfig(
            block_size=self.block_size, vocab_size=self.vocab_size,
            n_layer=self.n_layer, n_head=self.n_head, n_embd=self.n_embd,
            d_mlp=self.d_mlp, pos_encoding=self.pos_encoding, init_std=self.init_std,
        )


class FoneGPT(nn.Module):
    """GPT with FoNE input augmentation + dual output head (vocab + digits)."""

    def __init__(self, config: FoneGPTConfig):
        super().__init__()
        if config.pos_encoding not in ("learned", "rope"):
            raise ValueError(
                f"pos_encoding must be 'learned' or 'rope', got {config.pos_encoding!r}"
            )
        self.config = config
        gpt_cfg = config.as_gpt_config()

        # Standard token embedding.
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_encoding == "learned":
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.wpe = None

        # FoNE input-side augmentation (zero-padded into first FONE_DIM dims).
        self.fone_add = FoneEmbeddingAdd(d_model=config.n_embd)

        # Transformer body (reuses GPT's Block / RMSNorm).
        self.blocks = nn.ModuleList([Block(gpt_cfg) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)

        # Two output heads.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.fone_digit_head = FoneDigitHead()

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,    # (B, T) token ids (T = SEQ_LEN - 1)
        num_values: torch.Tensor,   # (B, T) numeric values at NUM positions
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        x = self.wte(input_ids)                                  # (B, T, n_embd)
        if self.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            x = x + self.wpe(pos)

        # Add FoNE features at every NUM-token position.
        is_num_position = (input_ids == NUM_ID)
        x = self.fone_add(x, num_values, is_num_position)

        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        vocab_logits = self.lm_head(x)                           # (B, T, V)
        digit_logits = self.fone_digit_head(x)                   # (B, T, D, 10)

        return {"vocab_logits": vocab_logits, "digit_logits": digit_logits, "hidden": x}

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
        return n


def fone_loss(
    outputs: dict[str, torch.Tensor],
    target_tokens: torch.Tensor,      # (B, T) standard vocab targets
    sign_target: torch.Tensor,        # (B,) sign-class id (== target_tokens[:, sign_pos])
    digit_target: torch.Tensor,       # (B, N_DIGITS) digits 0..9
    is_nan: torch.Tensor,             # (B,) bool
    sign_target_pos: int = POS_SIGN_C - 1,   # target-tensor index of SIGN_c
    digit_target_pos: int = POS_NUM_C - 1,   # target-tensor index of NUM_c
) -> dict[str, torch.Tensor]:
    """Mixed CE loss:
        vocab_loss = full-sequence CE on `target_tokens` (covers structural
                     tokens + the SIGN_c prediction at sign_target_pos).
        digit_loss = mean per-digit CE at the NUM_c position (NaN samples masked).
    Returns dict with 'vocab_loss', 'sign_loss' (vocab CE restricted to sign
    position, for monitoring), 'digit_loss', 'loss' (vocab + digit sum).
    """
    vocab_logits = outputs["vocab_logits"]   # (B, T, V)
    digit_logits = outputs["digit_logits"]   # (B, T, D, 10)

    B, T, V = vocab_logits.shape

    # Full-sequence vocab CE. Mean over all valid positions (no ignore_index).
    vocab_loss = F.cross_entropy(
        vocab_logits.reshape(-1, V),
        target_tokens.reshape(-1),
    )

    # Monitoring: sign-position-only vocab CE (unused for gradients).
    sign_slice = vocab_logits[:, sign_target_pos, :]
    sign_loss = F.cross_entropy(sign_slice, sign_target)

    # Digit loss: per-digit CE at digit_target_pos, NaN-masked.
    digit_slice = digit_logits[:, digit_target_pos, :, :]  # (B, D, 10)
    D = digit_slice.shape[1]
    digit_logits_flat = digit_slice.reshape(B * D, 10)
    digit_target_flat = digit_target.reshape(B * D)
    non_nan_mask = (~is_nan).unsqueeze(1).expand(B, D).reshape(B * D).float()
    per_token_loss = F.cross_entropy(
        digit_logits_flat, digit_target_flat, reduction="none"
    )
    masked = per_token_loss * non_nan_mask
    denom = non_nan_mask.sum().clamp(min=1.0)
    digit_loss = masked.sum() / denom

    return {
        "vocab_loss": vocab_loss,
        "sign_loss": sign_loss,
        "digit_loss": digit_loss,
        "loss": vocab_loss + digit_loss,
    }


def fone_correct(
    outputs: dict[str, torch.Tensor],
    sign_target: torch.Tensor,        # (B,) sign class id
    digit_target: torch.Tensor,       # (B, N_DIGITS)
    is_nan: torch.Tensor,             # (B,) bool
    sign_target_pos: int = POS_SIGN_C - 1,
    digit_target_pos: int = POS_NUM_C - 1,
) -> torch.Tensor:
    """Per-pair correctness: predicted sign matches AND (sample is NaN OR
    predicted digits all match). Returns (B,) bool tensor."""
    sign_pred = outputs["vocab_logits"][:, sign_target_pos, :].argmax(dim=-1)
    sign_ok = (sign_pred == sign_target)
    digit_pred = outputs["digit_logits"][:, digit_target_pos, :, :].argmax(dim=-1)
    digits_ok = (digit_pred == digit_target).all(dim=-1)
    return sign_ok & (is_nan | digits_ok)


def load_fone(checkpoint_path, device: str = "cpu") -> "FoneGPT":
    """Load a FoNE checkpoint from disk."""
    from pathlib import Path
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    fone_cfg = FoneGPTConfig(**ckpt["fone_gpt_config"])
    model = FoneGPT(fone_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


@torch.no_grad()
def fone_predict_on_holdout(checkpoint_path, device: str = "cpu") -> dict:
    """Forward-pass a FoNE checkpoint on the holdout split.

    Returns:
        rows : the holdout table rows
        correct : (n,) bool, per-pair correctness (sign + digits, NaN-aware)
        sign_pred : (n,) int64 — predicted sign class id at SIGN_c
        digit_pred : (n, N_DIGITS) int64 — predicted digits at NUM_c
        sign_target : (n,) int64
        digit_target : (n, N_DIGITS) int64
        is_nan : (n,) bool
    """
    import numpy as np
    from .preprocess import build_table, split_train_holdout
    from .fone_tokenizer import encode_batch, encode_targets

    model = load_fone(checkpoint_path, device=device)
    table = build_table()
    _, holdout_idx = split_train_holdout(table, seed=0)
    rows = table[holdout_idx]
    triples = np.stack([rows["a_bits"], rows["b_bits"], rows["result_bits"]], axis=1)
    tokens, num_values = encode_batch(triples)
    sign_target, digit_target, is_nan = encode_targets(triples)

    n = len(rows)
    bsz = 512
    correct = np.zeros(n, dtype=bool)
    sign_pred = np.zeros(n, dtype=np.int64)
    digit_pred = np.zeros((n, digit_target.shape[1]), dtype=np.int64)
    sign_pos = POS_SIGN_C - 1
    digit_pos = POS_NUM_C - 1
    for i in range(0, n, bsz):
        input_ids = torch.from_numpy(tokens[i:i+bsz, :-1]).long().to(device)
        nv = torch.from_numpy(num_values[i:i+bsz, :-1]).float().to(device)
        st = torch.from_numpy(sign_target[i:i+bsz]).long().to(device)
        dt = torch.from_numpy(digit_target[i:i+bsz]).long().to(device)
        nan_t = torch.from_numpy(is_nan[i:i+bsz]).to(device)
        out = model(input_ids, nv)
        sp = out["vocab_logits"][:, sign_pos, :].argmax(dim=-1)
        dp = out["digit_logits"][:, digit_pos, :, :].argmax(dim=-1)
        c = fone_correct(out, st, dt, nan_t, sign_pos, digit_pos)
        sign_pred[i:i+bsz] = sp.cpu().numpy()
        digit_pred[i:i+bsz] = dp.cpu().numpy()
        correct[i:i+bsz] = c.cpu().numpy()
    return {
        "rows": rows,
        "correct": correct,
        "sign_pred": sign_pred,
        "digit_pred": digit_pred,
        "sign_target": sign_target,
        "digit_target": digit_target,
        "is_nan": is_nan,
    }


def _spot_checks() -> None:
    torch.manual_seed(0)
    cfg = FoneGPTConfig(n_embd=48, n_head=4, d_mlp=192, n_layer=4,
                        block_size=SEQ_LEN - 1, vocab_size=VOCAB_SIZE)
    model = FoneGPT(cfg)
    # No biases.
    for name, p in model.named_parameters():
        assert "bias" not in name, f"unexpected bias: {name}"

    B = 4
    T = SEQ_LEN - 1
    # Build a deterministic input mimicking the encoder layout.
    from .fone_tokenizer import encode_batch, encode_targets
    import numpy as np
    triples = np.array([
        [0x38, 0xB8, 0x00],   # 1 + -1 = 0
        [0x7E, 0x7E, 0x7E],   # 448 + 448 → saturate to 448
        [0x18, 0x18, 0x20],   # 0.0625 + 0.0625 = 0.125
        [0x7F, 0x00, 0x7F],   # NaN + 0 = NaN
    ], dtype=np.uint8)
    tokens, num_values = encode_batch(triples)
    sign_target, digit_target, is_nan = encode_targets(triples)

    input_ids = torch.from_numpy(tokens[:, :-1]).long()
    nv = torch.from_numpy(num_values[:, :-1]).float()
    sign_target_t = torch.from_numpy(sign_target).long()
    digit_target_t = torch.from_numpy(digit_target).long()
    is_nan_t = torch.from_numpy(is_nan)

    target_tokens_t = torch.from_numpy(tokens[:, 1:]).long()
    out = model(input_ids, nv)
    assert out["vocab_logits"].shape == (B, T, cfg.vocab_size)
    assert out["digit_logits"].shape == (B, T, N_DIGITS, 10)

    losses = fone_loss(
        out, target_tokens_t, sign_target_t, digit_target_t, is_nan_t,
    )
    for k in ("vocab_loss", "sign_loss", "digit_loss", "loss"):
        assert torch.isfinite(losses[k]), f"{k} = {losses[k]}"

    correct = fone_correct(out, sign_target_t, digit_target_t, is_nan_t)
    assert correct.shape == (B,)
    assert correct.dtype == torch.bool

    losses["loss"].backward()
    n_params = model.num_parameters()
    print(f"all FoNE model spot checks passed ({n_params:,} parameters)")


if __name__ == "__main__":
    _spot_checks()
