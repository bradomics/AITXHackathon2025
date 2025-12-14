from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from sim.common import ControlPoint


@dataclass
class EdgeState:
    edge_id: str
    veh_count: int
    mean_speed_mps: float


class SumoSimEngine:
    """
    SUMO engine controlled via TraCI.

    Requirements:
    - SUMO installed
    - Python can import `traci` (often via SUMO_HOME/tools)
    - A valid `.sumocfg` and edge IDs/routes matching that network
    """

    def __init__(
        self,
        *,
        sumo_cfg: Path,
        controls: list[ControlPoint],
        control_interval_s: float,
        sumo_binary: str = "sumo",
        step_length_s: float = 1.0,
        close_speed_mps: float = 0.1,
    ) -> None:
        self.sumo_cfg = sumo_cfg
        self.controls = controls
        self.control_interval_s = float(control_interval_s)
        self.step_length_s = float(step_length_s)
        self.close_speed_mps = float(close_speed_mps)

        try:
            import traci  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("SUMO engine requires `traci` (install SUMO and set SUMO_HOME).") from e

        self._traci = traci
        self._veh_seq = 0
        self._carry: dict[str, float] = {c.control_id: 0.0 for c in controls}
        self._last_inserted: dict[str, int] = {c.control_id: 0 for c in controls}

        cmd = [
            sumo_binary,
            "-c",
            str(sumo_cfg),
            "--step-length",
            str(self.step_length_s),
            "--no-step-log",
            "true",
            "--quit-on-end",
            "true",
        ]
        traci.start(cmd)

        # Ensure routes exist for controls that define them.
        for c in self.controls:
            if not c.route_edges:
                continue
            rid = f"route_{c.control_id}"
            try:
                self._traci.route.add(rid, c.route_edges)
            except Exception:
                # If already exists, that's fine.
                pass

    def close_edges(self, edge_ids: list[str]) -> None:
        for e in edge_ids:
            try:
                self._traci.edge.setMaxSpeed(e, self.close_speed_mps)
            except Exception:
                continue

    def apply_multipliers(self, multipliers: list[float]) -> None:
        """
        Inject vehicles for the next interval using baseline_veh_per_hour * multiplier.
        Controls without `route_edges` are ignored.
        """
        self._last_inserted = {c.control_id: 0 for c in self.controls}
        for c, m in zip(self.controls, multipliers, strict=True):
            if not c.route_edges:
                continue

            base = max(0.0, float(c.baseline_veh_per_hour))
            expected = base * float(m) * (self.control_interval_s / 3600.0) + self._carry[c.control_id]
            n_add = int(expected)
            self._carry[c.control_id] = expected - n_add

            rid = f"route_{c.control_id}"
            for _ in range(n_add):
                vid = f"v{self._veh_seq}"
                self._veh_seq += 1
                try:
                    self._traci.vehicle.add(vid, rid, depart="now")
                    self._last_inserted[c.control_id] += 1
                except Exception:
                    # If insertion fails (jammed edge, invalid route), skip.
                    continue

    def step_interval(self) -> None:
        n_steps = int(self.control_interval_s / self.step_length_s) if self.step_length_s > 0 else 0
        for _ in range(max(1, n_steps)):
            self._traci.simulationStep()

    async def step_interval_async(self, *, yield_every_steps: int = 200) -> None:
        """
        Async-friendly version of step_interval() that yields to the event loop periodically.
        This keeps the WebSocket server responsive during long SUMO intervals.
        """
        n_steps = int(self.control_interval_s / self.step_length_s) if self.step_length_s > 0 else 0
        n_steps = max(1, n_steps)
        yield_every_steps = max(1, int(yield_every_steps))
        for i in range(n_steps):
            self._traci.simulationStep()
            if (i + 1) % yield_every_steps == 0:
                await asyncio.sleep(0)

    @property
    def sim_time_s(self) -> float:
        try:
            return float(self._traci.simulation.getTime())
        except Exception:  # pragma: no cover
            return 0.0

    def observe_flows_ratio(self) -> list[float]:
        ratios: list[float] = []
        for c in self.controls:
            base = float(c.baseline_veh_per_hour)
            inserted = float(self._last_inserted.get(c.control_id, 0))
            obs_vph = (inserted * 3600.0) / self.control_interval_s if self.control_interval_s > 0 else 0.0
            ratios.append((obs_vph / base) if base > 0 else 0.0)
        return ratios

    def edge_states(self) -> list[EdgeState]:
        out: list[EdgeState] = []
        for c in self.controls:
            try:
                n = int(self._traci.edge.getLastStepVehicleNumber(c.edge_id))
                s = float(self._traci.edge.getLastStepMeanSpeed(c.edge_id))
            except Exception:
                n = 0
                s = 0.0
            out.append(EdgeState(edge_id=c.edge_id, veh_count=n, mean_speed_mps=s))
        return out

    def close(self) -> None:
        try:
            self._traci.close()
        except Exception:
            pass
