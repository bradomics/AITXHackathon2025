from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    bronze_dir: Path
    silver_dir: Path
    gold_dir: Path
    weather_bronze_path: Path
    aadt_bronze_path: Path


@dataclass(frozen=True)
class SilverizeConfig:
    incidents_output_name: str
    weather_output_name: str
    aadt_output_name: str
    datetime_format: str


@dataclass(frozen=True)
class FeaturesConfig:
    bucket_minutes: int
    cell_round_decimals: int
    lookback_hours: list[int]
    label_horizon_hours: list[int]
    aadt_max_distance_km: float
    ema_half_life_hours: float


@dataclass(frozen=True)
class TokenizerConfig:
    h3_resolution: int
    austin_center_lat: float
    austin_center_lon: float
    austin_radius_km: float
    output_dir: Path


@dataclass(frozen=True)
class TrainConfig:
    context_steps: int


@dataclass(frozen=True)
class HotspotV2Config:
    output_dir: Path
    runtime_dir: Path
    neg_per_hour: int
    seed_hours: int


@dataclass(frozen=True)
class PipelineConfig:
    paths: PathsConfig
    silverize: SilverizeConfig
    features: FeaturesConfig
    tokenizer: TokenizerConfig
    train: TrainConfig
    hotspot_v2: HotspotV2Config


def load_config(config_path: str) -> PipelineConfig:
    cfg_path = Path(config_path)
    repo_root = cfg_path.resolve().parent.parent

    def to_path(v: str) -> Path:
        p = Path(v)
        return p if p.is_absolute() else repo_root / p

    with cfg_path.open("rb") as f:
        raw = tomllib.load(f)

    paths = raw["paths"]
    silverize = raw["silverize"]
    features = raw["features"]
    tokenizer = raw["tokenizer"]
    train = raw["train"]
    hotspot_v2 = raw.get("hotspot_v2") or {}

    return PipelineConfig(
        paths=PathsConfig(
            bronze_dir=to_path(paths["bronze_dir"]),
            silver_dir=to_path(paths["silver_dir"]),
            gold_dir=to_path(paths["gold_dir"]),
            weather_bronze_path=to_path(paths["weather_bronze_path"]),
            aadt_bronze_path=to_path(paths["aadt_bronze_path"]),
        ),
        silverize=SilverizeConfig(
            incidents_output_name=silverize["incidents_output_name"],
            weather_output_name=silverize["weather_output_name"],
            aadt_output_name=silverize["aadt_output_name"],
            datetime_format=silverize["datetime_format"],
        ),
        features=FeaturesConfig(
            bucket_minutes=int(features["bucket_minutes"]),
            cell_round_decimals=int(features["cell_round_decimals"]),
            lookback_hours=[int(x) for x in features["lookback_hours"]],
            label_horizon_hours=[int(x) for x in features["label_horizon_hours"]],
            aadt_max_distance_km=float(features["aadt_max_distance_km"]),
            ema_half_life_hours=float(features["ema_half_life_hours"]),
        ),
        tokenizer=TokenizerConfig(
            h3_resolution=int(tokenizer["h3_resolution"]),
            austin_center_lat=float(tokenizer["austin_center_lat"]),
            austin_center_lon=float(tokenizer["austin_center_lon"]),
            austin_radius_km=float(tokenizer["austin_radius_km"]),
            output_dir=to_path(tokenizer["output_dir"]),
        ),
        train=TrainConfig(context_steps=int(train["context_steps"])),
        hotspot_v2=HotspotV2Config(
            output_dir=to_path(str(hotspot_v2.get("output_dir") or "data/gold/tokens/h3_hotspot_v2")),
            runtime_dir=to_path(str(hotspot_v2.get("runtime_dir") or ".runtime/hotspot_v2")),
            neg_per_hour=int(hotspot_v2.get("neg_per_hour") or 50),
            seed_hours=int(hotspot_v2.get("seed_hours") or 168),
        ),
    )
