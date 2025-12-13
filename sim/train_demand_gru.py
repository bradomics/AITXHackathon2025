from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from sim.common import how_sin_cos
from sim.model_gru import DemandGRUConfig, build_model, save_checkpoint


def _pick_device(s: str | None) -> torch.device:
    if not s or s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def _make_batch(
    *,
    batch_size: int,
    context_steps: int,
    n_controls: int,
    control_interval_s: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      x: (B, context_steps, n_controls + 2)
      y: (B, n_controls)  target log-multipliers for next step
    """
    # Baseline flow per control (veh/hour)
    base = torch.empty((batch_size, n_controls), device=device).uniform_(200.0, 1400.0)

    # Start time (random hour-of-week)
    start = datetime(2025, 1, 6)  # Monday
    start_h = torch.randint(0, 168, (batch_size,), device=device).tolist()
    start_dt = [start + timedelta(hours=int(h)) for h in start_h]

    # Simple latent log-multiplier AR(1) with shared weekly rhythm
    alpha = 0.85
    sigma = 0.08
    cycle_amp = 0.25

    log_m = torch.zeros((batch_size, context_steps + 1, n_controls), device=device)
    log_m[:, 0, :] = torch.randn((batch_size, n_controls), device=device) * 0.05

    flows = torch.zeros((batch_size, context_steps + 1, n_controls), device=device)

    for t in range(context_steps + 1):
        if t > 0:
            noise = torch.randn((batch_size, n_controls), device=device) * sigma

            how = []
            for b in range(batch_size):
                dt = start_dt[b] + timedelta(seconds=control_interval_s * t)
                s, c = how_sin_cos(dt)
                how.append((s, c))
            how_sc = torch.tensor(how, device=device)
            cyc = cycle_amp * how_sc[:, 0:1]  # use sin only

            log_m[:, t, :] = alpha * log_m[:, t - 1, :] + (1.0 - alpha) * cyc + noise

        mult = torch.exp(log_m[:, t, :])
        obs_noise = 1.0 + 0.05 * torch.randn((batch_size, n_controls), device=device)
        flows[:, t, :] = torch.clamp(base * mult * obs_noise, min=0.0)

    ratio = flows[:, :context_steps, :] / base.unsqueeze(1)

    exog = torch.zeros((batch_size, context_steps, 2), device=device)
    for t in range(context_steps):
        how = []
        for b in range(batch_size):
            dt = start_dt[b] + timedelta(seconds=control_interval_s * t)
            how.append(how_sin_cos(dt))
        exog[:, t, :] = torch.tensor(how, device=device)

    x = torch.cat([ratio, exog], dim=-1)
    y = log_m[:, context_steps, :]
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a tiny GRU demand-multiplier forecaster (synthetic data).")
    ap.add_argument("--out", default="sim/artifacts/demand_gru.pt")
    ap.add_argument("--n-controls", type=int, default=64)
    ap.add_argument("--context-steps", type=int, default=12)
    ap.add_argument("--control-interval-s", type=float, default=60.0)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batches-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _pick_device(args.device)
    cfg = DemandGRUConfig(
        n_controls=int(args.n_controls),
        context_steps=int(args.context_steps),
        hidden_dim=int(args.hidden_dim),
    )
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total = 0.0
        for _ in range(int(args.batches_per_epoch)):
            x, y = _make_batch(
                batch_size=int(args.batch_size),
                context_steps=cfg.context_steps,
                n_controls=cfg.n_controls,
                control_interval_s=float(args.control_interval_s),
                device=device,
            )
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu().item())

        avg = total / max(int(args.batches_per_epoch), 1)
        print(f"[train] epoch={epoch} mse={avg:.6f}")

    out_path = Path(args.out)
    save_checkpoint(out_path, model=model.cpu())
    print(f"[train] wrote {out_path}")


if __name__ == "__main__":
    main()
