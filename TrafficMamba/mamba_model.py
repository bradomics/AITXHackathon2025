import torch
import torch.nn as nn

from mamba_ssm import Mamba


class MambaIncidentModel(nn.Module):
    """
    Mamba-based temporal model for traffic incident prediction.

    Input:  x [B, T, F]
    Output: logits [B]
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                )
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        """
        x: [B, T, F]
        """
        x = self.input_proj(x)  # [B, T, d_model]

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Use last timestep representation
        last = x[:, -1, :]       # [B, d_model]
        logits = self.head(last)

        return logits.squeeze(-1)
