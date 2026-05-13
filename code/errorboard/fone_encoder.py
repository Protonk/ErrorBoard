"""FoNE Fourier features and per-digit decoder.

Implements Zhou et al. Def 3.1/3.2 (Fourier embedding) and Def 3.6/3.7
(per-digit decoder via cosine similarity against unit-circle prototypes).

Notation:
    φ(x, T) = (cos(2πx/T), sin(2πx/T))
    FoNE(x) = [φ(x, T_0); φ(x, T_1); …; φ(x, T_{D-1})]   ∈ R^{2D}

For our (m=3, n=3) configuration, D = 6 and FoNE dim = 12.

Each digit at position i is decoded by:
    logits[j] = h[2i:2i+2] · φ(j, 10)   for j ∈ {0..9}

The prototype matrix `prototypes` ∈ R^{10, 2} carries `φ(j, 10)` for each
digit value. It is fixed (not learned), buffered into modules.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .fone_tokenizer import FONE_DIM, N_DIGITS, PERIODS


def _digit_prototypes() -> torch.Tensor:
    """Return the 10 unit-circle prototypes for base-10 digit classification.

    Shape (10, 2). Row j = (cos(2πj/10), sin(2πj/10)).
    """
    angles = torch.arange(10, dtype=torch.float32) * (2 * math.pi / 10.0)
    proto = torch.stack([angles.cos(), angles.sin()], dim=1)
    return proto


def _periods_tensor() -> torch.Tensor:
    return torch.tensor(PERIODS, dtype=torch.float32)


def fone_features(values: torch.Tensor) -> torch.Tensor:
    """Compute FoNE features for a tensor of real values.

    Args:
        values: any-shape float tensor of real numbers.

    Returns:
        features: shape values.shape + (FONE_DIM,). For each value x, the
            last-axis is [cos(2πx/T_0), sin(2πx/T_0), …, cos(2πx/T_{D-1}),
            sin(2πx/T_{D-1})].
    """
    periods = _periods_tensor().to(values.device).to(values.dtype)  # (D,)
    # values: (..., ). Broadcast against periods.
    angles = values.unsqueeze(-1) * (2 * math.pi) / periods  # (..., D)
    cos = angles.cos()
    sin = angles.sin()
    # Interleave cos/sin into the last axis: feature[2i] = cos_i, feature[2i+1] = sin_i.
    feat = torch.stack([cos, sin], dim=-1)        # (..., D, 2)
    feat = feat.reshape(*values.shape, FONE_DIM)  # (..., 2D)
    return feat


class FoneEmbeddingAdd(nn.Module):
    """Adds FoNE features to NUM-token embeddings.

    The FONE_DIM-wide feature vector occupies the first FONE_DIM dimensions of
    the residual stream — Zhou's "zero-pad to d_model" choice (Table 2). At
    non-NUM positions, the feature is suppressed by `is_num_position`.
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model < FONE_DIM:
            raise ValueError(
                f"d_model={d_model} < FONE_DIM={FONE_DIM}; cannot zero-pad."
            )
        self.d_model = d_model

    def forward(
        self,
        embed: torch.Tensor,        # (B, T, d_model) standard token embedding
        num_values: torch.Tensor,   # (B, T) numeric values; 0 at non-NUM positions
        is_num_position: torch.Tensor,  # (B, T) bool
    ) -> torch.Tensor:
        feats = fone_features(num_values)          # (B, T, FONE_DIM)
        pad = embed.new_zeros(*feats.shape[:-1], self.d_model - FONE_DIM)
        feats_padded = torch.cat([feats, pad], dim=-1)   # (B, T, d_model)
        mask = is_num_position.to(embed.dtype).unsqueeze(-1)  # (B, T, 1)
        return embed + feats_padded * mask


class FoneDigitHead(nn.Module):
    """Per-digit cosine-similarity decoder (Def 3.6 / 3.7).

    Reads the first 2*N_DIGITS dims of the hidden state at a position, splits
    them into N_DIGITS pairs, and dots each pair against the 10 digit prototypes.

    Forward returns logits of shape (..., N_DIGITS, 10).
    """

    def __init__(self):
        super().__init__()
        # Buffer (not parameter) — Zhou treats prototypes as fixed.
        self.register_buffer("prototypes", _digit_prototypes())  # (10, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: (..., d_model)
        # Slice off the FoNE channel.
        sliced = hidden[..., :FONE_DIM]                     # (..., 2*D)
        pairs = sliced.reshape(*sliced.shape[:-1], N_DIGITS, 2)  # (..., D, 2)
        # pairs @ prototypes.T → (..., D, 10)
        logits = pairs @ self.prototypes.T
        return logits


def _spot_checks() -> None:
    # Single value: input-side FoNE(0) should be all (1, 0).
    f0 = fone_features(torch.tensor(0.0))
    expected = torch.tensor([1.0, 0.0] * N_DIGITS)
    assert torch.allclose(f0, expected), f0

    # Output-side decoder roundtrip: the decoder reads per-digit Fourier
    # encodings `φ(d, 10) = (cos(2πd/10), sin(2πd/10))`, NOT the input-side
    # FoNE value features. Test with hand-crafted hidden states that carry the
    # per-digit encoding directly — this is what the model has to learn to
    # produce during training (Def 3.6).
    head = FoneDigitHead()
    head.eval()
    from .fone_tokenizer import fp8_to_sign_and_digits, digits_to_magnitude
    proto = head.prototypes  # (10, 2)
    for bits in [0x00, 0x38, 0x7E, 0x40, 0x18]:  # +0, +1, +448, +2, +0.0625
        _, true_digits = fp8_to_sign_and_digits(bits)
        # Build per-digit hidden by stacking φ(d_i, 10) into the first FONE_DIM dims.
        hidden = torch.zeros(64)
        for i, d in enumerate(true_digits):
            hidden[2*i:2*i+2] = proto[d]
        logits = head(hidden)                                    # (D, 10)
        preds = logits.argmax(dim=-1).tolist()
        assert preds == true_digits, (
            f"per-digit decoder failed at bits={bits:#04x}: "
            f"true={true_digits} pred={preds}"
        )

    # Embedding-add module: should leave non-NUM positions unchanged.
    add = FoneEmbeddingAdd(d_model=48)
    B, T, d = 2, 10, 48
    embed = torch.randn(B, T, d)
    nv = torch.zeros(B, T)
    nv[0, 2] = 4.17
    nv[1, 5] = 123.0
    is_num = torch.zeros(B, T, dtype=torch.bool)
    is_num[0, 2] = True
    is_num[1, 5] = True
    out = add(embed, nv, is_num)
    # Unchanged positions:
    assert torch.allclose(out[0, 0], embed[0, 0])
    assert torch.allclose(out[0, 3], embed[0, 3])
    assert torch.allclose(out[1, 0], embed[1, 0])
    # NUM positions: differ by FoNE features in first FONE_DIM dims.
    diff_0_2 = out[0, 2] - embed[0, 2]
    expected_0_2 = torch.cat([fone_features(torch.tensor(4.17)),
                              torch.zeros(d - FONE_DIM)])
    assert torch.allclose(diff_0_2, expected_0_2, atol=1e-6), (diff_0_2, expected_0_2)

    print("all FoNE encoder spot checks passed")


if __name__ == "__main__":
    _spot_checks()
