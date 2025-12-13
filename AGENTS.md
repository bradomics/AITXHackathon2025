# Agent Notes (Repo Working Agreements)

## What this repo is

Backend/data-pipeline proof-of-concept for “Traffic Incident Insights”.

Pipeline stages:
- **Bronze**: raw CSVs in `data/bronze/`
- **Silver**: normalized CSVs in `data/silver/`
- **Gold**: feature table in `data/gold/`

Entry point:
- ETL only (stdlib): `python3 eta_mvp_run.py --config configs/pipeline.toml --only-etl`
- End-to-end (needs venv): `.venv/bin/python eta_mvp_run.py --config configs/pipeline.toml`

Phase 2 (tokenize/train/infer):
- `src/model/tokenize_h3.py` → `data/gold/tokens/h3_v1/`
- `src/model/train_hotspot_cli.py` → `artifacts/h3_hotspot_model.pt`
- `src/model/infer_hotspot.py` → `output/phase1_output.json`

End-to-end orchestrator:
- `.venv/bin/python eta_mvp_run.py` (supports `--only-silver/--only-etl/--only-gold/--only-tok/--only-train`)

## Local environment

The ETL modules in `src/data_pipeline/` use only the Python standard library.

Weather acquisition scripts and the Phase 2 tokenizer/training scripts require packages listed in `requirements.txt`. On macOS/Homebrew Python you should install them in a venv:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Key files

- Config: `configs/pipeline.toml`
- Pipeline modules: `src/`
- Helper scripts: `scripts/`

## Simplicity-first rules (KISS/YAGNI)

- Prefer straightforward scripts over frameworks.
- Keep `src/data_pipeline/` dependency-light (stdlib-first).
- Avoid speculative features and heavy abstractions.
- Keep each script doing one job (silverize incidents, silverize weather, etc.).
- Only add validation/error-handling when there’s a known failure mode.

## Data conventions

- `data/silver/incidents.csv` adds `event_class`:
  - `collision` if the source filename contains `collision`
  - otherwise `traffic_incident`
- Weather is hourly and keyed by `bucket_start` in `data/silver/weather_hourly.csv`.
- TxDOT AADT is reduced to one row per station (latest year) in `data/silver/aadt_stations.csv`.
- Gold features in `data/gold/features/hotspot_features.csv` include:
  - per-hour counts split into `n_collisions` / `n_traffic_incidents`
  - lookback windows for both
  - time context (`hour_of_week`, cyclic `sin/cos`, `month`, `season`)
  - EMA counts (half-life from `configs/pipeline.toml`) and time-since-last-event features
  - per-`issue_reported` type mix columns (`n_issue_*`)
  - horizon labels (`y_collisions_next_*h`, `y_traffic_incidents_next_*h`)
  - weather columns + nearest-station AADT exposure

## Training notes

- The training/inference model predicts per-hex probabilities for collisions vs other incidents from time + weather inputs.
- `HotspotSeqModel` uses **Mamba** if `mamba_ssm` is installed; otherwise it falls back to **GRU**.
