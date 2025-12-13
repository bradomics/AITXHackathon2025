from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DemandGRUConfig:
    n_controls: int
    context_steps: int
    exog_dim: int = 2  # how_sin, how_cos
    hidden_dim: int = 128
    num_layers: int = 1
    log_multiplier_min: float = -0.69  # ~0.5x
    log_multiplier_max: float = 0.69   # ~2.0x


def build_model(cfg: DemandGRUConfig):
    import torch.nn as nn

    in_dim = cfg.n_controls + cfg.exog_dim
    gru = nn.GRU(input_size=in_dim, hidden_size=cfg.hidden_dim, num_layers=cfg.num_layers, batch_first=True)
    head = nn.Linear(cfg.hidden_dim, cfg.n_controls)

    class DemandGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cfg = cfg
            self.gru = gru
            self.head = head

        def forward(self, x):
            # x: (B, T, n_controls + exog_dim)
            _out, h = self.gru(x)
            h_last = h[-1]
            return self.head(h_last)  # (B, n_controls) log-multipliers

        def predict_multipliers(self, x):
            import torch

            log_m = self.forward(x)
            log_m = torch.clamp(log_m, self.cfg.log_multiplier_min, self.cfg.log_multiplier_max)
            return torch.exp(log_m)

    return DemandGRU()


def save_checkpoint(path, *, model) -> None:
    import torch

    ckpt = {
        "config": asdict(model.cfg),
        "state_dict": model.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def load_checkpoint(path, *, map_location: str | None = None):
    import torch

    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = DemandGRUConfig(**ckpt["config"])
    model = build_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model

