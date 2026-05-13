"""F2 (binary FoNE) GPT model + loss + correctness.

Same architecture as `fone_model.py` (input-side feature-add + dual head),
but with F2 constants pulled from `fone_f2_tokenizer` / `fone_f2_encoder`:
FONE_DIM=36, N_DIGITS=18, BASE=2.

Requires d_model >= 36; L4-E048 (d_model=48) is the minimum tractable
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fone_f2_encoder import FoneDigitHead, FoneEmbeddingAdd, fone_features
from .fone_f2_tokenizer import (
    BASE,
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
class FoneF2GPTConfig:
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


class FoneF2GPT(nn.Module):
    def __init__(self, config: FoneF2GPTConfig):
        super().__init__()
        if config.pos_encoding not in ("learned", "rope"):
            raise ValueError(
                f"pos_encoding must be 'learned' or 'rope', got {config.pos_encoding!r}"
            )
        self.config = config
        gpt_cfg = config.as_gpt_config()

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_encoding == "learned":
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.wpe = None

        self.fone_add = FoneEmbeddingAdd(d_model=config.n_embd)
        self.blocks = nn.ModuleList([Block(gpt_cfg) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
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
        input_ids: torch.Tensor,
        num_values: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape
        if T > self.config.block_size:
            raise ValueError(
                f"sequence length {T} exceeds block_size {self.config.block_size}"
            )

        x = self.wte(input_ids)
        if self.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long,
                               device=input_ids.device).unsqueeze(0)
            x = x + self.wpe(pos)

        is_num_position = (input_ids == NUM_ID)
        x = self.fone_add(x, num_values, is_num_position)

        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)

        vocab_logits = self.lm_head(x)                  # (B, T, V)
        digit_logits = self.fone_digit_head(x)          # (B, T, N_DIGITS, BASE)

        return {"vocab_logits": vocab_logits, "digit_logits": digit_logits, "hidden": x}

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
        return n


def fone_f2_loss(
    outputs: dict[str, torch.Tensor],
    target_tokens: torch.Tensor,
    sign_target: torch.Tensor,
    digit_target: torch.Tensor,
    is_nan: torch.Tensor,
    sign_target_pos: int = POS_SIGN_C - 1,
    digit_target_pos: int = POS_NUM_C - 1,
) -> dict[str, torch.Tensor]:
    """Mixed CE: full-sequence vocab + per-binary-digit at NUM_c (NaN-masked).

    Returns: vocab_loss, sign_loss (monitoring), digit_loss, loss.
    """
    vocab_logits = outputs["vocab_logits"]
    digit_logits = outputs["digit_logits"]

    B, T, V = vocab_logits.shape

    vocab_loss = F.cross_entropy(
        vocab_logits.reshape(-1, V),
        target_tokens.reshape(-1),
    )

    sign_slice = vocab_logits[:, sign_target_pos, :]
    sign_loss = F.cross_entropy(sign_slice, sign_target)

    digit_slice = digit_logits[:, digit_target_pos, :, :]  # (B, N_DIGITS, BASE)
    D = digit_slice.shape[1]
    digit_logits_flat = digit_slice.reshape(B * D, BASE)
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


def fone_f2_correct(
    outputs: dict[str, torch.Tensor],
    sign_target: torch.Tensor,
    digit_target: torch.Tensor,
    is_nan: torch.Tensor,
    sign_target_pos: int = POS_SIGN_C - 1,
    digit_target_pos: int = POS_NUM_C - 1,
) -> torch.Tensor:
    sign_pred = outputs["vocab_logits"][:, sign_target_pos, :].argmax(dim=-1)
    sign_ok = (sign_pred == sign_target)
    digit_pred = outputs["digit_logits"][:, digit_target_pos, :, :].argmax(dim=-1)
    digits_ok = (digit_pred == digit_target).all(dim=-1)
    return sign_ok & (is_nan | digits_ok)


def load_fone_f2(checkpoint_path, device: str = "cpu") -> "FoneF2GPT":
    from pathlib import Path
    ckpt = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    cfg = FoneF2GPTConfig(**ckpt["fone_f2_gpt_config"])
    model = FoneF2GPT(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


@torch.no_grad()
def fone_f2_predict_on_holdout(checkpoint_path, device: str = "cpu") -> dict:
    import numpy as np
    from .preprocess import build_table, split_train_holdout
    from .fone_f2_tokenizer import encode_batch, encode_targets

    model = load_fone_f2(checkpoint_path, device=device)
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
        c = fone_f2_correct(out, st, dt, nan_t, sign_pos, digit_pos)
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
    cfg = FoneF2GPTConfig(n_embd=48, n_head=4, d_mlp=192, n_layer=4,
                          block_size=SEQ_LEN - 1, vocab_size=VOCAB_SIZE)
    model = FoneF2GPT(cfg)
    for name, p in model.named_parameters():
        assert "bias" not in name, f"unexpected bias: {name}"

    import numpy as np
    from .fone_f2_tokenizer import encode_batch, encode_targets
    triples = np.array([
        [0x38, 0xB8, 0x00],
        [0x7E, 0x7E, 0x7E],
        [0x18, 0x18, 0x20],
        [0x7F, 0x00, 0x7F],
    ], dtype=np.uint8)
    tokens, num_values = encode_batch(triples)
    sign_target, digit_target, is_nan = encode_targets(triples)

    input_ids = torch.from_numpy(tokens[:, :-1]).long()
    nv = torch.from_numpy(num_values[:, :-1]).float()
    sign_t = torch.from_numpy(sign_target).long()
    digit_t = torch.from_numpy(digit_target).long()
    is_nan_t = torch.from_numpy(is_nan)
    target_tokens_t = torch.from_numpy(tokens[:, 1:]).long()

    out = model(input_ids, nv)
    assert out["vocab_logits"].shape == (4, SEQ_LEN - 1, cfg.vocab_size)
    assert out["digit_logits"].shape == (4, SEQ_LEN - 1, N_DIGITS, BASE)

    losses = fone_f2_loss(out, target_tokens_t, sign_t, digit_t, is_nan_t)
    for k in ("vocab_loss", "sign_loss", "digit_loss", "loss"):
        assert torch.isfinite(losses[k]), f"{k} = {losses[k]}"

    correct = fone_f2_correct(out, sign_t, digit_t, is_nan_t)
    assert correct.shape == (4,)
    losses["loss"].backward()
    n_params = model.num_parameters()
    print(f"all F2 model spot checks passed ({n_params:,} parameters)")


if __name__ == "__main__":
    _spot_checks()
