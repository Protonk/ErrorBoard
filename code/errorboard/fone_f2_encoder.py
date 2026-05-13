"""F2 (binary) FoNE Fourier features + per-digit decoder.

Same shape as `fone_encoder.py` but with the F2 constants:
  FONE_DIM = 36, N_DIGITS = 18, BASE = 2.

Output-side prototypes are the 2 base-2 unit-circle points:
  φ(0, 2) = (1, 0), φ(1, 2) = (-1, 0).
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .fone_f2_tokenizer import BASE, FONE_DIM, N_DIGITS, PERIODS


def _digit_prototypes() -> torch.Tensor:
    """B unit-circle prototypes for base-B digit classification.
    Shape (B, 2). Row j = (cos(2πj/B), sin(2πj/B))."""
    angles = torch.arange(BASE, dtype=torch.float32) * (2 * math.pi / BASE)
    return torch.stack([angles.cos(), angles.sin()], dim=1)


def _periods_tensor() -> torch.Tensor:
    return torch.tensor(PERIODS, dtype=torch.float32)


def fone_features(values: torch.Tensor) -> torch.Tensor:
    """Compute FoNE features for a tensor of real values.

    Same math as F1; only the period set differs.
    """
    periods = _periods_tensor().to(values.device).to(values.dtype)
    angles = values.unsqueeze(-1) * (2 * math.pi) / periods
    cos = angles.cos()
    sin = angles.sin()
    feat = torch.stack([cos, sin], dim=-1)
    feat = feat.reshape(*values.shape, FONE_DIM)
    return feat


class FoneEmbeddingAdd(nn.Module):
    """Adds F2 FoNE features into the first FONE_DIM dims of the embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        if d_model < FONE_DIM:
            raise ValueError(
                f"d_model={d_model} < FONE_DIM={FONE_DIM}; cannot zero-pad."
            )
        self.d_model = d_model

    def forward(
        self,
        embed: torch.Tensor,
        num_values: torch.Tensor,
        is_num_position: torch.Tensor,
    ) -> torch.Tensor:
        feats = fone_features(num_values)
        pad = embed.new_zeros(*feats.shape[:-1], self.d_model - FONE_DIM)
        feats_padded = torch.cat([feats, pad], dim=-1)
        mask = is_num_position.to(embed.dtype).unsqueeze(-1)
        return embed + feats_padded * mask


class FoneDigitHead(nn.Module):
    """Per-binary-digit cosine-similarity decoder. Returns (..., N_DIGITS, BASE)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("prototypes", _digit_prototypes())  # (B, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        sliced = hidden[..., :FONE_DIM]
        pairs = sliced.reshape(*sliced.shape[:-1], N_DIGITS, 2)
        logits = pairs @ self.prototypes.T   # (..., N_DIGITS, BASE)
        return logits


def _spot_checks() -> None:
    # FoNE(0) features = all (1, 0).
    f0 = fone_features(torch.tensor(0.0))
    assert torch.allclose(f0, torch.tensor([1.0, 0.0] * N_DIGITS))

    # The 2 base-2 prototypes: (1, 0) and (-1, 0).
    head = FoneDigitHead()
    proto = head.prototypes
    assert torch.allclose(proto[0], torch.tensor([1.0, 0.0]), atol=1e-6)
    assert torch.allclose(proto[1], torch.tensor([-1.0, 0.0]), atol=1e-6)

    # Decoder roundtrip: bits at known FP8 values decode correctly.
    from .fone_f2_tokenizer import fp8_to_sign_and_digits, digits_to_magnitude
    for bits in [0x00, 0x38, 0x7E, 0x40, 0x18]:  # +0, +1, +448, +2, +0.0625
        _, true_digits = fp8_to_sign_and_digits(bits)
        hidden = torch.zeros(64)
        for i, d in enumerate(true_digits):
            hidden[2*i:2*i+2] = proto[d]
        logits = head(hidden)
        preds = logits.argmax(dim=-1).tolist()
        assert preds == true_digits, (
            f"per-digit decoder failed at bits={bits:#04x}: "
            f"true={true_digits} pred={preds}"
        )

    # Embedding-add: only NUM positions changed.
    add = FoneEmbeddingAdd(d_model=128)   # need at least 36
    B, T, d = 2, 10, 128
    embed = torch.randn(B, T, d)
    nv = torch.zeros(B, T)
    nv[0, 2] = 4.17
    nv[1, 5] = 123.0
    is_num = torch.zeros(B, T, dtype=torch.bool)
    is_num[0, 2] = True
    is_num[1, 5] = True
    out = add(embed, nv, is_num)
    assert torch.allclose(out[0, 0], embed[0, 0])
    diff_0_2 = out[0, 2] - embed[0, 2]
    expected_0_2 = torch.cat([fone_features(torch.tensor(4.17)),
                              torch.zeros(d - FONE_DIM)])
    assert torch.allclose(diff_0_2, expected_0_2, atol=1e-6)

    print("all F2 encoder spot checks passed")


if __name__ == "__main__":
    _spot_checks()
