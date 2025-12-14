# Traffic Incident Insights (Backend / Data Pipeline)

This repo contains a simple, local-first data pipeline to prepare Austin traffic-incident data (plus weather + TxDOT AADT exposure) for model training and later inference.

## Quickstart

### 1) (Optional) Create a virtualenv for the weather fetchers

The Open-Meteo fetch scripts and the tokenizer/training scripts use third-party packages. On macOS/Homebrew Python (PEP 668), install them in a venv:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

### 2) Run ETL only (bronze → silver → gold)

This pipeline code itself uses only the Python standard library.

```bash
python3 eta_mvp_run.py --config configs/pipeline.toml --only-etl
```

### 3) Run everything end-to-end (silver + gold + tokenize + train)

This requires the venv (tokenizer + training use `h3` + `torch`):

```bash
.venv/bin/python eta_mvp_run.py --config configs/pipeline.toml
```

To run only a single stage:

```bash
.venv/bin/python eta_mvp_run.py --only-silver
.venv/bin/python eta_mvp_run.py --only-etl
.venv/bin/python eta_mvp_run.py --only-gold
.venv/bin/python eta_mvp_run.py --only-tok
.venv/bin/python eta_mvp_run.py --only-train
.venv/bin/python eta_mvp_run.py --only-hotspot-v2-data
.venv/bin/python eta_mvp_run.py --only-hotspot-v2-train
```

## Data layout

- `data/bronze/`
  - Unified incident snapshot (preferred): `Real-Time_Traffic_Incident_Reports_YYYYMMDD.csv`
  - Legacy incident category files (CSV): still supported, but ignored when the unified snapshot exists
  - Weather history (CSV): `austin_weather_training_data.csv` (the silverizer handles both “normal CSV” and the current “1-column embedded CSV” format)
  - TxDOT AADT stations (CSV): `TxDOT_AADT_Annuals_ALL.csv`
- `data/silver/`
  - `incidents.csv`: normalized incident rows with `event_class`
  - `weather_hourly.csv`: bucketed weather keyed by `bucket_start` (bucket size from config; default 2h)
  - `aadt_stations.csv`: one row per station (latest year), with `aadt_log1p`
- `data/gold/`
  - `features/hotspot_features.csv`: per-cell, per-bucket features + labels split into collisions vs other incidents

## Config

`configs/pipeline.toml` controls:
- Input/output paths (bronze/silver/gold)
- Time bucket size (default 2 hours)
- Spatial resolution (grid rounding decimals)
- Label horizon and lookback windows
- AADT nearest-station max distance
- EMA half-life (hours)
- Tokenizer (H3 resolution, Austin radius, output dir). Default `h3_resolution=6` is a coarse “~3-mile-ish” hex grid.
- Training (context window length, in buckets)

## Gold feature table (high level)

`data/gold/features/hotspot_features.csv` contains one row per `(cell_id, bucket_start)` where at least one incident occurred in that cell/hour.

Column groups:
- **Time context:** `hour_of_day`, `day_of_week`, `hour_of_week`, `month`, `season` + cyclic `sin/cos` features
- **Recent history:** lookback counts, EMA counts, and “hours since last incident”
- **Type mix:** per-`issue_reported` counts (`n_issue_*`)
- **Weather:** temperature/precip/visibility/wind/weather_code
- **Traffic exposure:** nearest-station TxDOT AADT (`aadt`, `aadt_log1p`, `aadt_dist_km`)

## Entry points

- `eta_mvp_run.py`: end-to-end orchestrator (silver → gold → tokenize/train). Hotspot v2 is the default: `--hotspot v1|v2`
- Model CLIs (live under `src/model/`):
  - `tokenize_h3.py`: builds an H3 (hex) dataset under `tokenizer.output_dir`
  - `train_hotspot_cli.py`: trains a sequence model to predict per-hex probabilities
  - `infer_hotspot.py`: emits HeatPoint arrays to `output/phase1_output.json`
  - (v2 risk map) `build_h3_hotspot_v2_dataset.py`: builds an enriched per-`(h3_cell, bucket)` dataset under `hotspot_v2.output_dir`
  - (v2 risk map) `train_hotspot_v2_cli.py`: trains a simple per-cell classifier (train-only scaling, AMP/grad-clip/logit-clamp guards)
  - (v2 risk map) `infer_hotspot_v2.py`: scores every cell for a future bucket (and applies a base-rate logit shift for calibration)

Notes:
- Training uses **Mamba** if `mamba_ssm` is installed; otherwise it falls back to **GRU** for local development.
- On DGX (CUDA), install `mamba-ssm` and run training with `--arch mamba`.

### Weather (data acquisition)

Run these with the venv Python:

```bash
.venv/bin/python scripts/gather_weather.py
.venv/bin/python scripts/fetch_forecast.py
```

Outputs:
- `scripts/gather_weather.py` writes `data/bronze/austin_weather_training_data.csv`
- `scripts/fetch_forecast.py` writes `data/bronze/austin_forecast_live.csv`

## Key assumptions / gotchas

- **Collision vs traffic incident:** `event_class=collision` if the source filename contains `collision` *or* `Issue Reported` contains `crash`/`collis` (case-insensitive); otherwise `traffic_incident`.
- **Weather join:** joined by `bucket_start` (bucket size from config). The weather silverizer now outputs a dense timeline with simple interpolation (ffill/bfill + linear).
- **AADT exposure:** joined by nearest TxDOT station within `aadt_max_distance_km` (default 5 km). Cells without a nearby station get blank AADT fields.
- **Gold rows are event-driven:** `hotspot_features.csv` only emits rows for buckets where at least one incident occurred in that cell (not a dense grid). Hotspot v2 training compensates via negative sampling.

## Code layout

- Pipeline code lives under `src/`:
  - `src/data_pipeline/`: silver + gold feature factory (stdlib-first)
  - `src/model/`: tokenizer + training + inference (uses `h3` + `torch`)
  - Shared: `src/config.py` and `src/util.py`
- Helper scripts live under `scripts/` (weather fetchers).

## Phase 2 commands (tokenize/train/infer)

Tokenizer and training are typically run via `eta_mvp_run.py`, but CLIs exist under `src/model/`:

```bash
.venv/bin/python eta_mvp_run.py --only-tok
.venv/bin/python eta_mvp_run.py --only-train
.venv/bin/python eta_mvp_run.py --only-hotspot-v2-data
.venv/bin/python eta_mvp_run.py --only-hotspot-v2-train
.venv/bin/python src/model/infer_hotspot.py --config configs/pipeline.toml --forecast-csv data/bronze/austin_forecast_live.csv
```

### Live inference loop

`eta_mvp_go.py` fetches the live Open-Meteo forecast and re-runs hotspot inference on a loop (writes `output/phase1_output.json`).
For v2, it also appends new incidents from the City of Austin Socrata feed into `.runtime/hotspot_v2/incidents.ndjson` (gitignored) unless you pass `--skip-live-incidents`.
Set `SODA_APP_TOKEN` via `.env.local` (gitignored) to avoid committing credentials.

Configure loop interval in `.env`:

```bash
mvp_infer_loop_min=5
```

Run one cycle:

```bash
python3 eta_mvp_go.py --once              # v2 default
python3 eta_mvp_go.py --variant v1 --once
```

Run continuously:

```bash
python3 eta_mvp_go.py
```
