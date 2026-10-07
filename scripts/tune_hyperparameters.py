"""Random search (16 draws) over LightGBM hyperparameters, scored by 5-fold
patient-grouped cross-validated AUROC on all landmarks.

Output: data/cohort/best_config.json
"""
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C
from cpis_plus import data as D

warnings.filterwarnings("ignore")
SEED = 0
N_ITER = 16


def log(msg):
    print(f"[tune] {msg}", flush=True)


def cv(P, X, y, df, oof_out=False):
    oof = np.full(len(y), np.nan)
    for tr, te in D.folds(df, 5):
        g = df.subject_id.values[tr]
        itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                        .split(np.zeros(len(tr)), groups=g))
        m = lgb.LGBMClassifier(**P)
        m.fit(X.iloc[tr[itr]], y.iloc[tr[itr]],
              eval_set=[(X.iloc[tr[iva]], y.iloc[tr[iva]])], eval_metric="auc",
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    a = roc_auc_score(y, oof)
    return (a, oof) if oof_out else a


def sample_params(rng, y):
    return {
        "n_estimators": 3000,
        "learning_rate": float(rng.choice([0.01, 0.02, 0.05])),
        "num_leaves": int(rng.choice([15, 31, 63])),
        "max_depth": int(rng.choice([3, 4, 5, 6, -1])),
        "min_child_samples": int(rng.choice([50, 100, 200, 400])),
        "subsample": float(rng.choice([0.6, 0.8, 1.0])), "subsample_freq": 1,
        "colsample_bytree": float(rng.choice([0.2, 0.3, 0.5])),
        "reg_alpha": float(rng.choice([0, 0.1, 1.0])),
        "reg_lambda": float(rng.choice([0, 1.0, 10.0])),
        "scale_pos_weight": float(rng.choice([1.0, (y == 0).sum() / max(y.sum(), 1)])),
        "n_jobs": 8, "verbose": -1, "random_state": SEED,
    }


def run(horizon):
    ds = pd.read_parquet(C.COHORT / f"landmarks_{horizon}h_extended.parquet")
    y = ds.y
    X = D.encode(ds, D.feature_columns(ds, include_culture=False))
    log(f"{horizon}h: {len(ds):,} landmarks, {y.sum():,} positive, {X.shape[1]} features")

    rng = np.random.default_rng(SEED)
    best = (0, None)
    for i in range(N_ITER):
        P = sample_params(rng, y)
        a = cv(P, X, y, ds)
        if a > best[0]:
            best = (a, P)
            log(f"  [{i+1}/{N_ITER}] best cross-validated AUROC {a:.4f}")
    return {"horizon": horizon, "auroc_p1": best[0], "params": best[1]}


if __name__ == "__main__":
    out = [run(h) for h in C.PREDICTION_HORIZONS]
    (C.COHORT / "best_config.json").write_text(json.dumps(out, indent=2, default=str))
    log("done")
