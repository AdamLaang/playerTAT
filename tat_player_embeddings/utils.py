from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yaml


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_feature_data_and_scaler(cfg: Dict) -> Tuple[pd.DataFrame, Dict, Dict]:
    split_csv = Path(cfg["data"]["split_csv"])
    feature_cfg_yaml = Path(cfg["data"]["feature_config_yaml"])
    scaler_joblib = Path(cfg["data"]["scaler_joblib"])

    df = pd.read_csv(split_csv, low_memory=False)
    feature_cfg = load_yaml(feature_cfg_yaml)
    scaler_payload = joblib.load(scaler_joblib)
    return df, feature_cfg, scaler_payload


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
