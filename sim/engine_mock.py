from __future__ import annotations

import math
import random
from dataclasses import dataclass

from sim.common import ControlPoint


@dataclass
class EdgeState:
    edge_id: str
    veh_count: int
    mean_speed_mps: float


class MockSimEngine:
    """
    Minimal, local engine for wiring/debugging without SUMO.
    It converts demand multipliers into synthetic counts + speeds.
    """

    def __init__(self, *, controls: list[ControlPoint], control_interval_s: float, seed: int = 13) -> None:
        self.controls = controls
        self.control_interval_s = float(control_interval_s)
        self.sim_time_s = 0.0
        self._rng = random.Random(seed)
        self._last_counts = [0 for _ in controls]

    def apply_multipliers(self, multipliers: list[float]) -> None:
        counts: list[int] = []
        for c, m in zip(self.controls, multipliers, strict=True):
            vph = max(0.0, float(c.baseline_veh_per_hour) * float(m))
            expected = vph * (self.control_interval_s / 3600.0)
            # deterministic-ish rounding with small jitter
            jitter = self._rng.uniform(-0.25, 0.25)
            n = max(0, int(expected + jitter))
            counts.append(n)
        self._last_counts = counts

    def step_interval(self) -> None:
        self.sim_time_s += self.control_interval_s

    def observe_flows_ratio(self) -> list[float]:
        ratios: list[float] = []
        for c, n in zip(self.controls, self._last_counts, strict=True):
            base = float(c.baseline_veh_per_hour)
            obs_vph = (float(n) * 3600.0) / self.control_interval_s if self.control_interval_s > 0 else 0.0
            ratios.append((obs_vph / base) if base > 0 else 0.0)
        return ratios

    def edge_states(self) -> list[EdgeState]:
        states: list[EdgeState] = []
        for c, n in zip(self.controls, self._last_counts, strict=True):
            # crude congestion: higher flow -> slower
            base = max(float(c.baseline_veh_per_hour), 1.0)
            load = float(n) / max((base * self.control_interval_s / 3600.0), 1.0)
            free = 20.0  # m/s
            speed = free / (1.0 + 1.2 * load)
            states.append(EdgeState(edge_id=c.edge_id, veh_count=int(n), mean_speed_mps=float(speed)))
        return states

