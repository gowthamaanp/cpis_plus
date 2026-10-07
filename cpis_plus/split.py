"""Patient-level 60/20/20 train/validation/test split, stratified by whether a
patient ever has a positive landmark and grouped by subject_id, so that no
patient crosses a boundary. Built once by scripts/patient_split.py and then
loaded from disk everywhere else."""
import json

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 0
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.6, 0.2, 0.2


def _path():
    from cpis_plus import config as C
    return C.COHORT / "canonical_split.json"


def make_canonical_split(df: pd.DataFrame, seed: int = SEED) -> dict:
    """Build the patient -> 'train' | 'val' | 'test' map and write it to disk."""
    pat = (df.groupby("subject_id")
             .agg(ever_pos=("y", "max"), n_episodes=("stay_id", "nunique"),
                  n_landmarks=("y", "size"))
             .reset_index())
    ids = pat.subject_id.values
    strata = pat.ever_pos.values

    rest_idx, test_idx = next(StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRAC, random_state=seed).split(ids, strata))
    val_of_rest = VAL_FRAC / (TRAIN_FRAC + VAL_FRAC)
    train_rel, val_rel = next(StratifiedShuffleSplit(
        n_splits=1, test_size=val_of_rest, random_state=seed)
        .split(ids[rest_idx], strata[rest_idx]))
    train_idx, val_idx = rest_idx[train_rel], rest_idx[val_rel]

    assignment, summary = {}, {}
    for idx, name in ((train_idx, "train"), (val_idx, "val"), (test_idx, "test")):
        for sid in ids[idx]:
            assignment[int(sid)] = name
        sub = pat.iloc[idx]
        summary[name] = {"n_patients": int(len(sub)),
                         "n_episodes": int(sub.n_episodes.sum()),
                         "n_landmarks": int(sub.n_landmarks.sum()),
                         "patient_prevalence": float(sub.ever_pos.mean())}

    payload = {"seed": seed, "summary": summary, "assignment": assignment}
    _path().write_text(json.dumps(payload, indent=2))
    return payload


def load_canonical_split() -> dict:
    """subject_id (int) -> 'train' | 'val' | 'test'."""
    p = _path()
    if not p.exists():
        raise FileNotFoundError(f"{p} does not exist; run scripts/patient_split.py first.")
    return {int(k): v for k, v in json.loads(p.read_text())["assignment"].items()}


def apply_split(df: pd.DataFrame, assignment=None) -> pd.DataFrame:
    """Tag each row with its patient's split; raise if any patient is unmapped."""
    assignment = assignment or load_canonical_split()
    missing = set(df.subject_id.unique()) - set(assignment)
    if missing:
        raise AssertionError(f"{len(missing)} patients are not in the split "
                             f"(e.g. {sorted(missing)[:5]}).")
    out = df.copy()
    out["split"] = out.subject_id.map(assignment)
    return out


def check_split_integrity(df_with_split: pd.DataFrame):
    """Every patient in exactly one split, and all three splits present."""
    per_patient = df_with_split.groupby("subject_id")["split"].nunique()
    bad = per_patient[per_patient > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} patient(s) appear in more than one split")
    missing = {"train", "val", "test"} - set(df_with_split["split"].unique())
    if missing:
        raise AssertionError(f"split(s) absent from this table: {missing}")
    return True
