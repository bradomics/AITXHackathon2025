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
- (v2 risk map) `src/model/build_h3_hotspot_v2_dataset.py` → `data/gold/tokens/h3_hotspot_v2/`
- (v2 risk map) `src/model/train_hotspot_v2_cli.py` → `artifacts/h3_hotspot_v2_model.pt`
- (v2 risk map) `src/model/infer_hotspot_v2.py` → `output/phase1_output.json`
- (v2 live buffer) `scripts/fetch_live_incidents.py` → `.runtime/hotspot_v2/incidents.ndjson` (gitignored; set `SODA_APP_TOKEN`, ideally via `.env.local`)

End-to-end orchestrator:
- `.venv/bin/python eta_mvp_run.py` (Hotspot v2 default; supports `--hotspot v1|v2` plus `--only-silver/--only-etl/--only-gold/--only-tok/--only-train/--only-hotspot-v2-data/--only-hotspot-v2-train`)

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
- Weather is bucketed (see `configs/pipeline.toml` `features.bucket_minutes`) and keyed by `bucket_start` in `data/silver/weather_hourly.csv` (dense timeline via simple interpolation).
- TxDOT AADT is reduced to one row per station (latest year) in `data/silver/aadt_stations.csv`.
- Gold features in `data/gold/features/hotspot_features.csv` include:
  - per-bucket counts split into `n_collisions` / `n_traffic_incidents`
  - lookback windows for both
  - time context (`hour_of_week`, cyclic `sin/cos`, `month`, `season`)
  - EMA counts (half-life from `configs/pipeline.toml`) and time-since-last-event features
  - per-`issue_reported` type mix columns (`n_issue_*`)
  - radar volume/speed/occupancy (if traffic radar counts are available)
  - horizon labels (`y_collisions_next_*h`, `y_traffic_incidents_next_*h`)
  - weather columns + nearest-station AADT exposure

Built to scale (not a bug):
- `src/data_pipeline/feature_factory.py` intentionally produces a wide gold feature table (lookbacks/EMA/time-since/issue mix + AADT + radar) as a proof of concept so those columns are ready to ingest once live AADT/radar streams are wired in.
- The v1 tokenizer (`src/model/tokenizer_h3.py`) currently trains on a smaller input: 5 time features (`hour_of_week`, `how_sin`, `how_cos`, `month`, `season`) + weather columns, so the extra gold columns are not used yet.

## Training notes

- The training/inference model predicts per-hex probabilities for collisions vs other incidents from time + weather inputs.
- `HotspotSeqModel` uses **Mamba** if `mamba_ssm` is installed; otherwise it falls back to **GRU**.
- Hotspot v2 trains a simple MLP on an enriched per-`(h3_cell, bucket)` dataset (all positives + sampled negatives per bucket), and v2 inference applies a base-rate logit shift so weights behave like probabilities on the full grid.
