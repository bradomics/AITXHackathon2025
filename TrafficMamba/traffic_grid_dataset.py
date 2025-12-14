import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import holidays

# =========================
# SPLIT CONFIG
# =========================
@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self):
        if not np.isclose(self.train + self.val + self.test, 1.0):
            raise ValueError("Split ratios must sum to 1.0")


# =========================
# HELPERS
# =========================
def cyclic_encode(x: np.ndarray, period: float):
    ang = 2 * np.pi * (x / period)
    return np.sin(ang), np.cos(ang)


# =========================
# DATASET
# =========================
class TrafficGridDataset(Dataset):
    """
    One sample = one temporal window from ONE grid.

    Returns:
      x_seq : [seq_len, num_features]
      t_seq : [seq_len, num_time_features]   (optional)
      y     : scalar (binary) or vector (counts)
    """

    def __init__(
        self,
        csv_path: str,
        split: str = "train",                     # train | val | test
        seq_len: int = 16,                        # 16 * 3h = 48h history
        pred_horizon: int = 0,                    # predict at last step
        split_cfg: SplitConfig = SplitConfig(),
        scale: bool = True,
        time_encoding: str = "cyclic",            # none | raw | cyclic
        target: str = "any_incident",             # any_incident | counts
        interpolate_weather: bool = True,
        add_holiday: bool = True,
    ):
        super().__init__()

        assert split in {"train", "val", "test"}
        assert time_encoding in {"none", "raw", "cyclic"}
        assert target in {"any_incident", "counts"}

        self.csv_path = csv_path
        self.split = split
        self.seq_len = seq_len
        self.pred_horizon = pred_horizon
        self.split_cfg = split_cfg
        self.scale = scale
        self.time_encoding = time_encoding
        self.target = target
        self.interpolate_weather = interpolate_weather
        self.add_holiday = add_holiday

        # Column definitions
        self.GRID_LAT = "grid_center_lat"
        self.GRID_LON = "grid_center_lon"
        self.TIME_COL = "time_bucket"
        self.BUCKET_COL = "bucket_3h"

        self.COUNT_COLS = [
            "count_group_0",
            "count_group_1",
            "count_group_2",
            "count_group_3",
            "count_group_4",
            "count_group_5",
        ]
        self.WEATHER_COLS = [
            "temperature_2m",
            "windspeed_10m",
            "precipitation",
        ]

        self._load()
        self._build_index()

    # =========================
    # LOAD + PREP
    # =========================
    def _load(self):
        df = pd.read_csv(self.csv_path)
        df[self.TIME_COL] = pd.to_datetime(df[self.TIME_COL], errors="coerce")
        df = df.dropna(subset=[self.TIME_COL, self.GRID_LAT, self.GRID_LON])
        df = df.sort_values([self.GRID_LAT, self.GRID_LON, self.TIME_COL])

        # stable grid key (internal only)
        df["_grid"] = (
            df[self.GRID_LAT].round(6).astype(str)
            + "_"
            + df[self.GRID_LON].round(6).astype(str)
        )

        # -------------------------
        # Weather interpolation
        # -------------------------
        if self.interpolate_weather:
            df[self.WEATHER_COLS] = (
                df.groupby("_grid")[self.WEATHER_COLS]
                .apply(
                    lambda g: g.interpolate("linear", limit_direction="both")
                              .ffill()
                              .bfill()
                )
                .reset_index(level=0, drop=True)
            )

        # -------------------------
        # Global time split
        # -------------------------
        all_times = np.sort(df[self.TIME_COL].unique())
        n = len(all_times)
        n_train = int(n * self.split_cfg.train)
        n_val = int(n * self.split_cfg.val)

        if self.split == "train":
            keep_times = all_times[:n_train]
        elif self.split == "val":
            keep_times = all_times[n_train : n_train + n_val]
        else:
            keep_times = all_times[n_train + n_val :]

        df = df[df[self.TIME_COL].isin(keep_times)].reset_index(drop=True)

        # -------------------------
        # Feature matrix
        # -------------------------
        feat_cols = [
            self.GRID_LAT,
            self.GRID_LON,
            *self.WEATHER_COLS,
        ]

        X = df[feat_cols].astype(np.float32).values

        # -------------------------
        # Scaling (train only)
        # -------------------------
        self.scaler = None
        if self.scale:
            df_all = pd.read_csv(self.csv_path)
            df_all[self.TIME_COL] = pd.to_datetime(df_all[self.TIME_COL])
            df_all = df_all.sort_values(self.TIME_COL)
            train_times = df_all[self.TIME_COL].unique()[:n_train]
            df_train = df_all[df_all[self.TIME_COL].isin(train_times)]

            self.scaler = StandardScaler()
            self.scaler.fit(df_train[feat_cols].astype(np.float32).values)
            X = self.scaler.transform(X)

        self.X = X.astype(np.float32)

        # -------------------------
        # Targets
        # -------------------------
        if self.target == "any_incident":
            self.y = df["any_incident"].astype(np.float32).values
        else:
            self.y = df[self.COUNT_COLS].astype(np.float32).values

        # -------------------------
        # Time features
        # -------------------------
        self.T = self._build_time_features(df)

        # -------------------------
        # Grid grouping
        # -------------------------
        self.groups = {
            g: idx.values
            for g, idx in df.groupby("_grid").groups.items()
        }

    # =========================
    # TIME FEATURES
    # =========================
    def _build_time_features(self, df):
        if self.time_encoding == "none" and not self.add_holiday:
            return None

        t = df[self.TIME_COL]
        hour = t.dt.hour.values.astype(np.float32)
        dow = t.dt.dayofweek.values.astype(np.float32)
        month = (t.dt.month.values - 1).astype(np.float32)
        bucket = df[self.BUCKET_COL].values.astype(np.float32)

        feats = []

        if self.time_encoding == "raw":
            feats += [hour, dow, month, bucket]

        elif self.time_encoding == "cyclic":
            feats += cyclic_encode(hour, 24)
            feats += cyclic_encode(dow, 7)
            feats += cyclic_encode(month, 12)
            feats += cyclic_encode(bucket, 8)

        if self.add_holiday:
            us_holidays = holidays.US()
            is_holiday = np.array(
                [1.0 if d.date() in us_holidays else 0.0 for d in t],
                dtype=np.float32,
            )
            feats.append(is_holiday)

        return np.stack(feats, axis=1).astype(np.float32)

    # =========================
    # INDEX
    # =========================
    def _build_index(self):
        self.index = []
        needed = self.seq_len + self.pred_horizon

        for g, idxs in self.groups.items():
            L = len(idxs)
            for s in range(0, L - needed + 1):
                self.index.append((g, s))

    # =========================
    # PYTORCH API
    # =========================
    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        g, s = self.index[i]
        idxs = self.groups[g]

        seq_idx = idxs[s : s + self.seq_len]
        y_idx = idxs[s + self.seq_len - 1 + self.pred_horizon]

        x = torch.tensor(self.X[seq_idx], dtype=torch.float32)
        y = torch.tensor(self.y[y_idx], dtype=torch.float32)

        if self.T is None:
            return x, y
        else:
            t = torch.tensor(self.T[seq_idx], dtype=torch.float32)
            return x, t, y
