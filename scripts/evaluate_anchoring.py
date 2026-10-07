"""Table 3 row "asymmetric temporal anchoring": one prediction per ventilation
episode, with controls taken at their first landmark (fixed 48 h after
intubation) and cases at their last pre-onset landmark (closest to clinical
suspicion). Same model and features; different unit of analysis.

Output: data/cohort/index_day_anchoring_24h.json
"""
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cpis_plus import config as C
from cpis_plus import data as D
from cpis_plus import final as FIN
from cpis_plus import metrics as M
from cpis_plus import split as SPL

warnings.filterwarnings("ignore")
SEED = 0
H = 24


def log(m):
    print(f"[anchoring] {m}", flush=True)


def build_anchored(df):
    """One row per patient (stay_id): cases at their last pre-onset landmark
    (closest to, i.e. anchored at, clinical suspicion), controls at their
    first landmark (fixed 48h-post-intubation reference point)."""
    has_onset = df.vap_onset.notna()
    cases = (df[has_onset].sort_values("t")
             .groupby("stay_id", as_index=False).tail(1))
    controls = (df[~has_onset].sort_values("t")
                .groupby("stay_id", as_index=False).head(1))
    cases = cases.copy(); cases["y"] = 1
    controls = controls.copy(); controls["y"] = 0
    out = pd.concat([cases, controls], ignore_index=True)
    return out


def main():
    ds, cols = FIN.build_final_dataset()
    P = FIN.params()

    tagged = SPL.apply_split(ds, SPL.load_canonical_split())
    train_lm = tagged[tagged.split == "train"].reset_index(drop=True)
    test_lm = tagged[tagged.split == "test"].reset_index(drop=True)

    train = build_anchored(train_lm)
    test = build_anchored(test_lm)
    log(f"anchored cohort: {len(train)} train patients ({train.y.sum()} cases), "
        f"{len(test)} test patients ({test.y.sum()} cases)")

    vocab = D.fit_categorical_vocab(train, cols)
    Xtr = D.encode(train, cols, vocab=vocab)
    Xte = D.align_columns(Xtr, D.encode(test, cols, vocab=vocab))

    g = train.subject_id.values
    itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                     .split(np.zeros(len(train)), groups=g))
    m = lgb.LGBMClassifier(**P)
    m.fit(Xtr.iloc[itr], train.y.values[itr],
          eval_set=[(Xtr.iloc[iva], train.y.values[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    p = m.predict_proba(Xte)[:, 1]

    r = M.evaluate(test.y.values, p, name="index_day_anchoring", groups=test.subject_id.values)
    log(f"index-day anchoring: AUROC {r['auroc']:.4f} (95% CI {r['auroc_lo']:.4f}-{r['auroc_hi']:.4f}), "
        f"n={len(test)}, n_pos={int(test.y.sum())}")

    result = {
        "description": "One prediction per ventilation episode. Controls anchored at "
                        "their first landmark (48h post-intubation); cases at their last "
                        "pre-onset landmark.",
        "n_train": int(len(train)), "n_pos_train": int(train.y.sum()),
        "n_test": int(len(test)), "n_pos_test": int(test.y.sum()),
        "auroc": float(r["auroc"]), "auroc_lo": float(r["auroc_lo"]), "auroc_hi": float(r["auroc_hi"]),
    }
    (C.COHORT / f"index_day_anchoring_{H}h.json").write_text(json.dumps(result, indent=2))
    log(f"saved data/cohort/index_day_anchoring_{H}h.json")


if __name__ == "__main__":
    main()
