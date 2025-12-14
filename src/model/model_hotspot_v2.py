from __future__ import annotations

import torch.nn as nn


class HotspotV2MLP(nn.Module):
    def __init__(self, *, n_features: int, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        layers: list[nn.Module] = [
            nn.Linear(self.n_features, self.hidden_dim),
            nn.ReLU(),
        ]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.extend(
            [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            ]
        )
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(self.hidden_dim, 2)

    def forward(self, x):
        h = self.backbone(x)
        out = self.head(h)
        logits_coll = out[:, 0]
        logits_inc = out[:, 1]
        return logits_coll, logits_inc


def build_hotspot_v2_model(*, n_features: int, hidden_dim: int = 128, dropout: float = 0.0) -> HotspotV2MLP:
    return HotspotV2MLP(n_features=n_features, hidden_dim=hidden_dim, dropout=dropout)

