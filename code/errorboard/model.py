"""Decoder-only transformer for the ErrorBoard FP8-add task.

Forked from Karpathy's nanoGPT (code/nanoGPT/model.py) with the methodology.md
knobs baked in:

    - No biases anywhere (Llama-style). One linear object per role.
    - RMSNorm, pre-norm. Residual stream is the inspection target.
    - SiLU activation in the MLP. (GELU is an equivalent alternative; we picked
      SiLU for consistency with the Llama-style no-bias choice.)
    - No dropout. Synthetic task with infinite training data.
    - Untied embeddings: W_E (wte) and W_U (lm_head.weight) are separate objects,
      to be read independently as "input-direction" and "output-direction" for a
      given token.
    - Configurable positional encoding via `pos_encoding`:
        - "learned": standard learned-absolute embeddings (the default arm).
        - "rope":    Rotary Position Embedding applied to Q and K inside attention
                     (the comparison arm; methodology.md "Comparison arms").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- normalization ----

class RMSNorm(nn.Module):
    """Llama-style RMSNorm: x * rsqrt(mean(x**2) + eps) * gain. No bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.gain


# ---- rotary position embedding ----

def _precompute_rope_cache(
    d_head: int, max_seq_len: int, base: float = 10000.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute RoPE cos/sin. Returns (cos, sin), each shape (max_seq_len, d_head/2)."""
    if d_head % 2 != 0:
        raise ValueError(f"RoPE requires even d_head, got {d_head}")
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    pos = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(pos, inv_freq)  # (max_seq_len, half)
    return angles.cos(), angles.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x. x: (..., T, d_head). cos/sin: (T, d_head/2). Returns same shape as x.

    Uses the GPT-NeoX / Llama half-rotation convention: split the last dim into two halves,
    treat them as real and imaginary components, apply 2-D rotation per position.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    # Broadcast cos/sin: (T, half) -> (1, ..., 1, T, half) over leading dims of x.
    while cos.dim() < x.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


# ---- attention ----

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention. No biases. Optional RoPE on Q and K."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )
        self.n_head = config.n_head
        self.d_head = config.n_embd // config.n_head
        self.n_embd = config.n_embd
        self.use_rope = config.pos_encoding == "rope"

        # Fused QKV projection. No bias.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection. No bias.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if self.use_rope:
            cos, sin = _precompute_rope_cache(self.d_head, config.block_size)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)                          # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)       # each (B, T, C)
        # Reshape to (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        if self.use_rope:
            cos = self.rope_cos[:T]
            sin = self.rope_sin[:T]
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)

        # Scaled-dot-product attention with causal mask (uses FlashAttention if available).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # Re-assemble: (B, n_head, T, d_head) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ---- MLP ----

class MLP(nn.Module):
    """Two-layer feed-forward block. No biases. SiLU activation."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.d_mlp, bias=False)
        self.c_proj = nn.Linear(config.d_mlp, config.n_embd, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.act(self.c_fc(x)))


# ---- transformer block ----

class Block(nn.Module):
    """Pre-norm block: x + attn(RMSNorm(x)); x + mlp(RMSNorm(x))."""

    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---- config ----

@dataclass
class GPTConfig:
    block_size: int = 27        # input sequence length (= SEQ_LEN - 1 for the ErrorBoard task)
    vocab_size: int = 12        # ErrorBoard task vocab
    n_layer: int = 4            # methodology.md starting box
    n_head: int = 4
    n_embd: int = 128
    d_mlp: int = 512            # 4 * n_embd (standard ratio)
    pos_encoding: Literal["learned", "rope"] = "learned"
    init_std: float = 0.02      # standard nanoGPT/GPT-2 init scale


# ---- model ----

class GPT(nn.Module):
    """Small decoder-only transformer for FP8-add interpretability.

    Untied W_E and W_U: `self.wte.weight` and `self.lm_head.weight` are independent
    parameters and may be read separately as input- and output-side directions.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.pos_encoding not in ("learned", "rope"):
            raise ValueError(f"pos_encoding must be 'learned' or 'rope', got {config.pos_encoding!r}")
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        if config.pos_encoding == "learned":
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.wpe = None  # RoPE injects positional info inside attention; no token-side PE

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """idx: (B, T) token ids. targets: optional (B, T) target ids (use -1 to mask)."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.config.block_size}")

        x = self.wte(idx)                                       # (B, T, n_embd)
        if self.wpe is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
            x = x + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)                                # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Total parameter count. If non_embedding=True, exclude wte and wpe."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
        return n


def _spot_checks() -> None:
    torch.manual_seed(0)

    # ---- learned-absolute PE arm ----
    cfg_learned = GPTConfig(pos_encoding="learned")
    model = GPT(cfg_learned)

    # Verify no biases anywhere.
    for name, p in model.named_parameters():
        assert "bias" not in name, f"unexpected bias parameter: {name}"

    # Verify embeddings are untied (different objects).
    assert model.wte.weight.data_ptr() != model.lm_head.weight.data_ptr(), "embeddings should be untied"

    # Run a forward pass with random tokens.
    B, T = 4, cfg_learned.block_size
    idx = torch.randint(0, cfg_learned.vocab_size, (B, T))
    targets = torch.randint(0, cfg_learned.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg_learned.vocab_size), f"bad logits shape {logits.shape}"
    assert loss is not None and torch.isfinite(loss), f"loss should be finite, got {loss}"

    # Parameter count for the default config (methodology.md starting box).
    n_params = model.num_parameters()
    assert 700_000 < n_params < 900_000, f"expected ~0.8M params for default config, got {n_params}"

    # ---- RoPE arm ----
    cfg_rope = GPTConfig(pos_encoding="rope")
    model_rope = GPT(cfg_rope)
    assert model_rope.wpe is None, "RoPE arm should not have wpe"
    logits_rope, loss_rope = model_rope(idx, targets)
    assert logits_rope.shape == (B, T, cfg_rope.vocab_size)
    assert loss_rope is not None and torch.isfinite(loss_rope)

    # ---- loss masking via ignore_index ----
    masked_targets = targets.clone()
    masked_targets[:, : T - 8] = -1  # mask all but last 8 positions (mimics result-only eval)
    _, masked_loss = model(idx, masked_targets)
    assert torch.isfinite(masked_loss)

    # ---- end-to-end with the dataset ----
    from .dataset import StratifiedSampler
    from .preprocess import build_table, split_train_holdout

    table = build_table()
    train_idx, _ = split_train_holdout(table, seed=0)
    sampler = StratifiedSampler(table, train_idx, seed=0)
    batch = sampler.sample_batch(8)
    inp = torch.from_numpy(batch["input"]).long()
    tgt = torch.from_numpy(batch["target"]).long()
    logits_e, loss_e = model(inp, tgt)
    assert logits_e.shape == (8, cfg_learned.block_size, cfg_learned.vocab_size)
    assert torch.isfinite(loss_e), f"loss should be finite on real batch, got {loss_e}"

    print(f"all model spot checks passed (default config: {n_params:,} parameters)")


if __name__ == "__main__":
    _spot_checks()
