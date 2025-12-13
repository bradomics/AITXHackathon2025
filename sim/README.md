## Digital Twin (SUMO + Torch “driver”)

This folder is a separate, KISS “digital twin” backend:

- **SUMO** simulates traffic dynamics (car-following, lane-changing, routing).
- A small **Torch** model runs on GPU (DGX) to infer a compact control vector (e.g., inflow multipliers) that **drives SUMO**.
- The backend broadcasts state over WebSocket as JSON.

This is intentionally isolated from the main ETL code under `src/data_pipeline/`.

### What the AI model does

At a fixed control interval (e.g., every 60s), the model predicts one multiplier per “control point”.
Each control point is typically a **source / on-ramp** you can inject vehicles from:

- `multiplier = 1.0` → baseline flow
- `multiplier > 1.0` → more demand
- `multiplier < 1.0` → less demand

SUMO then runs forward for the next interval with those injected vehicles.

### Files

- `digital_twin_server.py`: orchestrator (loads model, runs engine, serves WS)
- `model_gru.py`: tiny GRU forecaster (Torch)
- `train_demand_gru.py`: trains the GRU on synthetic data (fast smoke training)
- `engine_mock.py`: local mock engine (no SUMO required)
- `engine_sumo.py`: SUMO engine via TraCI (requires SUMO + `traci`)
- `controls_example.json`: example control-point config

### Install

The sim server uses `torch` (already in repo `requirements.txt`) plus WebSockets:

```bash
.venv/bin/python -m pip install -r sim/requirements.txt
```

For the real SUMO engine you must have SUMO installed and `traci` importable (usually via `SUMO_HOME`).

### Train a tiny model (synthetic)

This produces a small checkpoint you can run on DGX:

```bash
.venv/bin/python sim/train_demand_gru.py --out sim/artifacts/demand_gru.pt
```

### Run (mock engine)

Useful to validate wiring without SUMO:

```bash
.venv/bin/python sim/digital_twin_server.py --engine mock --model sim/artifacts/demand_gru.pt
```

WebSocket default: `ws://127.0.0.1:8765`

### Run (SUMO engine)

Provide your own Austin `.sumocfg` and a controls JSON with valid edge routes:

```bash
.venv/bin/python sim/digital_twin_server.py \
  --engine sumo \
  --sumo-cfg /path/to/austin.sumocfg \
  --controls sim/controls_example.json \
  --model sim/artifacts/demand_gru.pt
```

### WebSocket payload (one message per control interval)

```json
{
  "ts": "2025-12-13T10:00:00",
  "sim_time_s": 3600.0,
  "controls": [
    {"control_id": "ramp_1", "edge_id": "E1", "baseline_veh_per_hour": 600.0, "multiplier": 1.12}
  ],
  "edges": [
    {"edge_id": "E1", "veh_count": 7, "mean_speed_mps": 12.3}
  ]
}
```
