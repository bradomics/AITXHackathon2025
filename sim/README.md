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
.venv/bin/python eta_sim_run.py --controls sim/controls_example.json --out sim/artifacts/demand_gru.pt
# or: .venv/bin/python sim/train_demand_gru.py --out sim/artifacts/demand_gru.pt
```

### Train on real Austin traffic counts (radar/camera)

If you’ve downloaded counts under `data/bronze/austin_traffic_counts/` (via `scripts/fetch_austin_traffic_counts.py`),
you can train the same GRU on real per-bin volumes.

Controls config additions (per control):

- `radar_detids`: detector IDs for `i626-g7ub` radar counts (`detid` column)
- `camera_device_ids`: device IDs for `sh59-i6y9` camera counts (`atd_device_id` column)

The counts are typically **15-minute bins**, so set `control_interval_s` to `900` in the controls JSON.

Example:

```bash
.venv/bin/python eta_sim_run.py \
  --mode counts \
  --controls sim/controls_counts_example.json \
  --write-controls-out sim/controls_counts_example_filled.json \
  --out sim/artifacts/demand_gru_counts.pt \
  --epochs 1 --batches-per-epoch 50 --batch-size 32 --device cpu
```

Then run the server using the filled baselines file:

```bash
.venv/bin/python eta_sim_go.py \
  --engine mock \
  --controls sim/controls_counts_example_filled.json \
  --model sim/artifacts/demand_gru_counts.pt
```

### Run the server

```bash
.venv/bin/python eta_sim_go.py --engine mock --controls sim/controls_example.json --model sim/artifacts/demand_gru.pt --realtime
```

WebSocket default: `ws://127.0.0.1:8765`

### Run (mock engine, direct)

Useful to validate wiring without SUMO:

```bash
.venv/bin/python sim/train_demand_gru.py --out sim/artifacts/demand_gru.pt
.venv/bin/python sim/digital_twin_server.py --engine mock --model sim/artifacts/demand_gru.pt
```

### Run (SUMO engine)

Provide your own Austin `.sumocfg` and a controls JSON with valid edge routes:

```bash
# Generate a controls file from the existing SUMO routes file (optional helper)
.venv/bin/python sim/gen_controls_from_routes.py \
  --routes-xml sumo/austin/routes.rou.xml \
  --out sim/controls_austin_from_routes.json \
  --top-k 50

# Train a model sized to that controls file
.venv/bin/python eta_sim_run.py --controls sim/controls_austin_from_routes.json --out sim/artifacts/demand_gru_austin.pt

# Run headless SUMO + inference (requires SUMO + traci)
.venv/bin/python eta_sim_go.py \
  --engine sumo \
  --sumo-cfg sumo/austin/sim.sumocfg \
  --controls sim/controls_austin_from_routes.json \
  --model sim/artifacts/demand_gru_austin.pt \
  --sumo-step-s 0.1 \
  --realtime
```

Direct server form:

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
