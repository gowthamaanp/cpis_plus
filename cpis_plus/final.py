"""The locked CPIS+ configuration shared by the evaluation scripts: the
24-hour horizon, the 339-feature set (Tier-A clinical signals + SOFA) and the
LightGBM hyperparameters from scripts/tune_hyperparameters.py."""
import json

import pandas as pd

from cpis_plus import config as C

H = 24


def params():
    """LightGBM hyperparameters for the 24-hour horizon."""
    for e in json.loads((C.COHORT / "best_config.json").read_text()):
        if int(e["horizon"]) == H:
            q = dict(e["params"])
            for k in ("num_leaves", "max_depth", "min_child_samples",
                      "n_estimators", "subsample_freq", "n_jobs"):
                if k in q:
                    q[k] = int(float(q[k]))
            q.pop("n_jobs", None)
            return q


def build_final_dataset():
    """Landmark table plus the locked list of 339 raw feature columns."""
    ds = pd.read_parquet(C.COHORT / f"landmarks_{H}h_final.parquet").reset_index(drop=True)
    with open(C.COHORT / f"final_features_{H}h.json") as f:
        cols = json.load(f)["features"]
    return ds, cols
