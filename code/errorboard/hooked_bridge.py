"""Convert ErrorBoard nanoGPT-style checkpoints into TransformerLens HookedTransformer.

Why this exists: HookedTransformer gives us run_with_cache + attention/MLP hooks
for free, but only if its state-dict layout matches. Our model.py has:

  - Fused QKV (c_attn.weight: (3*n_embd, n_embd))   → TL has separate W_Q,W_K,W_V
  - Output proj as (n_embd, n_embd)                  → TL splits into (n_head, d_head, d_model)
  - MLP weights as (d_mlp, n_embd) / (n_embd, d_mlp) → TL transposes both
  - lm_head.weight as (vocab, n_embd)                → TL transposes to (n_embd, vocab)
  - RMSNorm with .gain                               → TL names it .w
  - No biases anywhere                               → TL has biases; we zero them out
  - SiLU                                             → TL supports via act_fn="silu"
  - Pre-norm RMSNorm                                 → TL's normalization_type="RMS" is pre-norm

We bridge by writing a state dict in TL's expected layout, then loading. The
equivalence check (verify_equivalence) does a four-step residual-stream comparison
to catch any wrong-axis / wrong-transpose error early.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from .model import GPT, GPTConfig


def make_hooked_config(cfg: GPTConfig) -> HookedTransformerConfig:
    """Build the TransformerLens config matching a GPTConfig.

    Supports pos_encoding in {'learned', 'rope'}.
    """
    if cfg.pos_encoding not in ("learned", "rope"):
        raise NotImplementedError(
            f"hooked_bridge currently supports pos_encoding ∈ {{'learned', 'rope'}}, "
            f"got {cfg.pos_encoding!r}."
        )
    d_head = cfg.n_embd // cfg.n_head
    base = dict(
        n_layers=cfg.n_layer,
        d_model=cfg.n_embd,
        n_heads=cfg.n_head,
        d_head=d_head,
        d_mlp=cfg.d_mlp,
        n_ctx=cfg.block_size,
        d_vocab=cfg.vocab_size,
        d_vocab_out=cfg.vocab_size,
        act_fn="silu",
        normalization_type="RMS",
        # TL defaults eps=1e-5, ours is 1e-6. The bigger eps pulls RMS scale ~1% off
        # for typical residual magnitudes, which compounds across layers (saw 2.6e-2
        # at ln1.normalized during debugging).
        eps=1e-6,
        attention_dir="causal",
        tokenizer_name=None,
        default_prepend_bos=False,
        init_weights=False,
    )
    if cfg.pos_encoding == "learned":
        base["positional_embedding_type"] = "standard"
    else:  # rope
        base["positional_embedding_type"] = "rotary"
        base["rotary_dim"] = d_head
        # Our model uses GPT-NeoX / Llama convention with base=10000, applied to
        # full d_head (split into two halves and rotated). Match TL.
        base["rotary_base"] = 10000
    return HookedTransformerConfig(**base)


def convert_state_dict(gpt: GPT) -> dict[str, torch.Tensor]:
    """Build a TL-compatible state dict from a trained GPT model.

    Shape conversion notes (all derived by matching forward computations):

    Attention QKV (fused → split):
      Our c_attn.weight is (3*n_embd, n_embd) with rows ordered [Q | K | V].
      For projection P ∈ {Q,K,V}, after our `qkv.split` + `view(B,T,n_head,d_head)`,
      head h's component j is at output-row `h*d_head + j`.
      TL's W_P[h, d, m] computes: out[h, m] = sum_d x[d] * W_P[h, d, m].
      Therefore W_P[h, d, m] = our_P_weight[h*d_head + m, d]
      = reshape(P_weight, (n_head, d_head, n_embd)).transpose(1, 2).

    Attention output projection:
      Our c_proj.weight is (n_embd, n_embd). After concatenating heads,
      out[m] = sum_k y_flat[k] * c_proj.weight[m, k] where k = h*d_head + j.
      TL's W_O[h, j, m] computes: out[m] = sum_h sum_j y[h, j] * W_O[h, j, m].
      Therefore W_O = c_proj.weight.reshape(d_model, n_head, d_head).permute(1, 2, 0).

    MLP:
      Our c_fc.weight is (d_mlp, n_embd); forward is x @ c_fc.weight.T.
      TL's W_in is (d_model, d_mlp); forward is x @ W_in.
      Therefore W_in = c_fc.weight.T.
      Likewise W_out = c_proj.weight.T (MLP's c_proj).

    Unembed:
      Our lm_head.weight is (vocab, n_embd); logits = x @ lm_head.weight.T.
      TL's W_U is (d_model, d_vocab); logits = x @ W_U.
      Therefore W_U = lm_head.weight.T.

    Embeddings (wte/wpe) match TL's (W_E/W_pos) shape directly.
    RMSNorm gain copies directly to ln.w.
    All biases (b_Q/K/V/O, b_in/out, b_U) are zeroed since our model has none.
    """
    cfg = gpt.config
    sd_ours = gpt.state_dict()
    d_model = cfg.n_embd
    d_head = cfg.n_embd // cfg.n_head
    n_head = cfg.n_head
    d_mlp = cfg.d_mlp

    sd: dict[str, torch.Tensor] = {}

    # Embeddings — same shape, direct copy.
    sd["embed.W_E"] = sd_ours["wte.weight"].clone()
    # Positional embedding: only learned-PE has wpe; RoPE applies position inside
    # attention and has no pos_embed weights to copy.
    if cfg.pos_encoding == "learned":
        sd["pos_embed.W_pos"] = sd_ours["wpe.weight"].clone()

    # Final norm + unembed.
    sd["ln_final.w"] = sd_ours["norm_f.gain"].clone()
    sd["unembed.W_U"] = sd_ours["lm_head.weight"].T.contiguous()
    sd["unembed.b_U"] = torch.zeros(cfg.vocab_size)

    for i in range(cfg.n_layer):
        prefix = f"blocks.{i}"

        # Pre-norm gains.
        sd[f"{prefix}.ln1.w"] = sd_ours[f"blocks.{i}.norm1.gain"].clone()
        sd[f"{prefix}.ln2.w"] = sd_ours[f"blocks.{i}.norm2.gain"].clone()

        # Attention: split fused QKV and reshape per the docstring.
        c_attn = sd_ours[f"blocks.{i}.attn.c_attn.weight"]  # (3*n_embd, n_embd)
        Q_w = c_attn[0:d_model, :]                            # (n_embd, n_embd)
        K_w = c_attn[d_model:2*d_model, :]                    # (n_embd, n_embd)
        V_w = c_attn[2*d_model:3*d_model, :]                  # (n_embd, n_embd)
        # Reshape (n_embd, n_embd) → (n_head, d_head, n_embd) → (n_head, n_embd, d_head)
        sd[f"{prefix}.attn.W_Q"] = Q_w.reshape(n_head, d_head, d_model).transpose(1, 2).contiguous()
        sd[f"{prefix}.attn.W_K"] = K_w.reshape(n_head, d_head, d_model).transpose(1, 2).contiguous()
        sd[f"{prefix}.attn.W_V"] = V_w.reshape(n_head, d_head, d_model).transpose(1, 2).contiguous()
        sd[f"{prefix}.attn.b_Q"] = torch.zeros(n_head, d_head)
        sd[f"{prefix}.attn.b_K"] = torch.zeros(n_head, d_head)
        sd[f"{prefix}.attn.b_V"] = torch.zeros(n_head, d_head)

        # Attention output projection. c_proj.weight: (n_embd, n_embd).
        # W_O[h, j, m] = c_proj.weight[m, h*d_head + j]
        c_proj = sd_ours[f"blocks.{i}.attn.c_proj.weight"]   # (n_embd, n_embd)
        sd[f"{prefix}.attn.W_O"] = c_proj.reshape(d_model, n_head, d_head).permute(1, 2, 0).contiguous()
        sd[f"{prefix}.attn.b_O"] = torch.zeros(d_model)

        # MLP: simple transposes.
        sd[f"{prefix}.mlp.W_in"] = sd_ours[f"blocks.{i}.mlp.c_fc.weight"].T.contiguous()
        sd[f"{prefix}.mlp.W_out"] = sd_ours[f"blocks.{i}.mlp.c_proj.weight"].T.contiguous()
        sd[f"{prefix}.mlp.b_in"] = torch.zeros(d_mlp)
        sd[f"{prefix}.mlp.b_out"] = torch.zeros(d_model)

    return sd


def load_hooked(
    checkpoint_path: Path,
    device: str = "cpu",
) -> tuple[HookedTransformer, GPT]:
    """Load a checkpoint as both our GPT (the source) and a HookedTransformer
    (the bridged target). Returns (hooked, gpt) for downstream comparison.

    Expects training.py's checkpoint layout: {iter, model_state, optimizer_state,
    config, gpt_config} where gpt_config is a dict.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    gpt = GPT(gpt_cfg)
    gpt.load_state_dict(ckpt["model_state"])
    gpt.eval().to(device)

    hooked = HookedTransformer(make_hooked_config(gpt_cfg))
    # strict=False because TL has non-trainable buffers (attn.mask, attn.IGNORE)
    # that we don't supply — TL keeps its defaults.
    hooked.load_state_dict(convert_state_dict(gpt), strict=False)
    hooked.eval().to(device)
    return hooked, gpt


@torch.no_grad()
def verify_equivalence(
    gpt: GPT,
    hooked: HookedTransformer,
    n_samples: int = 4,
    device: str = "cpu",
    tol: float = 1e-4,
    verbose: bool = True,
) -> dict[str, float]:
    """Four-step layer-by-layer comparison. Returns max absolute differences at
    each stage. Raises AssertionError if any step exceeds `tol`."""
    gpt.eval(); hooked.eval()
    cfg = gpt.config
    # Move both to the same device. Prefer the caller-supplied device; if either model
    # lives elsewhere already (TL moves to CUDA automatically when available), follow it.
    target_device = next(hooked.parameters()).device
    gpt.to(target_device)
    torch.manual_seed(0)
    idx = torch.randint(0, cfg.vocab_size, (n_samples, cfg.block_size), device=target_device)

    # --- 1. Embedding ---
    with torch.no_grad():
        emb_ours = gpt.wte(idx)
        if gpt.wpe is not None:
            pos = torch.arange(cfg.block_size, device=target_device).unsqueeze(0)
            emb_ours = emb_ours + gpt.wpe(pos)
        # For RoPE, no positional addition at the embedding stage — rotation lives
        # inside attention.

    # TL: use run_with_cache and pull intermediate residuals.
    logits_hooked, cache = hooked.run_with_cache(idx, return_type="logits")
    # TL's `blocks.0.hook_resid_pre` is the post-embedding residual stream.
    emb_hooked = cache["blocks.0.hook_resid_pre"]
    diff_emb = (emb_ours - emb_hooked).abs().max().item()

    # --- 2. After block 0 ---
    with torch.no_grad():
        x = emb_ours
        x = gpt.blocks[0](x)
        block0_ours = x.clone()
    block0_hooked = cache["blocks.0.hook_resid_post"]
    diff_block0 = (block0_ours - block0_hooked).abs().max().item()

    # --- 3. After final norm ---
    with torch.no_grad():
        x = emb_ours
        for blk in gpt.blocks:
            x = blk(x)
        x_final_ours = gpt.norm_f(x)
    # TL's hook_normalized captures (x / scale) *before* the gain multiplication.
    # To compare apples-to-apples with our norm_f output (which includes gain),
    # multiply by ln_final.w.
    x_final_hooked = cache["ln_final.hook_normalized"] * hooked.ln_final.w
    diff_final = (x_final_ours - x_final_hooked).abs().max().item()

    # --- 4. Logits ---
    with torch.no_grad():
        logits_ours, _ = gpt(idx, targets=None)
    diff_logits = (logits_ours - logits_hooked).abs().max().item()

    result = {
        "embedding": diff_emb,
        "block0_post": diff_block0,
        "final_norm": diff_final,
        "logits": diff_logits,
    }
    if verbose:
        print("Equivalence check (max |Δ|):")
        for k, v in result.items():
            ok = "✓" if v < tol else "✗"
            print(f"  {k:<14s}  {v:.2e}  {ok}")
    for stage, d in result.items():
        if d >= tol:
            raise AssertionError(
                f"Bridge mismatch at stage {stage!r}: max |Δ| = {d:.3e} ≥ tol {tol:.0e}"
            )
    return result


def _spot_checks() -> None:
    """Test the bridge against fresh small models in both PE modes."""
    for pe in ["learned", "rope"]:
        print(f"\n--- pos_encoding={pe} ---")
        torch.manual_seed(0)
        # E=48 so d_head=12 (even), required for RoPE.
        cfg = GPTConfig(n_layer=4, n_head=4, n_embd=48, d_mlp=192,
                        block_size=27, vocab_size=12, pos_encoding=pe)
        gpt = GPT(cfg).eval()
        hooked = HookedTransformer(make_hooked_config(cfg))
        hooked.load_state_dict(convert_state_dict(gpt), strict=False)
        hooked.eval()
        verify_equivalence(gpt, hooked, n_samples=4, tol=1e-4, verbose=True)
    print(f"\nAll four stages within tolerance for both PE modes.")


if __name__ == "__main__":
    _spot_checks()
