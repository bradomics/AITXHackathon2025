from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    n_features: int
    n_cells: int
    d_model: int
    arch: str


def build_model(*, n_features: int, n_cells: int, d_model: int = 128, arch: str = "gru"):
    import torch.nn as nn

    if arch == "mamba":
        try:
            from mamba_ssm import Mamba  # type: ignore[import-not-found]
        except Exception:
            arch = "gru"

    if arch == "mamba":
        class _MambaEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_proj = nn.Linear(n_features, d_model)
                self.block = Mamba(d_model)  # minimal defaults

            def forward(self, x):  # (B, T, D)
                x = self.in_proj(x)
                x = self.block(x)
                return x[:, -1, :]

        encoder: nn.Module = _MambaEncoder()
    else:
        encoder = nn.GRU(input_size=n_features, hidden_size=d_model, num_layers=1, batch_first=True)

    class HotspotSeqModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.arch = arch
            self.encoder = encoder
            self.head_coll = nn.Linear(d_model, n_cells)
            self.head_inc = nn.Linear(d_model, n_cells)

        def forward(self, x):
            if self.arch == "gru":
                out, h = self.encoder(x)
                h_last = h[-1]
            else:
                h_last = self.encoder(x)
            coll = self.head_coll(h_last)
            inc = self.head_inc(h_last)
            return coll, inc

    return HotspotSeqModel()

