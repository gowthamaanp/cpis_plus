"""Feature selection and patient-level splitting for the landmark dataset.

Two rules are enforced here rather than left to each experiment, because
violating either silently inflates every metric downstream:

1. Splits are grouped by `subject_id`. A patient contributes many landmarks and
   often several episodes; splitting on rows would put the same patient's
   near-duplicate landmarks in both train and test.
2. `leaky_*` columns are never returned as model features. See README.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# columns that identify/administer a row rather than describe the patient state
# Any column that is (or encodes) an outcome. `vap_proxy`/`vap_icd`/`vap_any`
# are alternative labels for the same event, so letting one in as a feature
# while predicting another is perfect leakage - it produced AUROC 1.0000 once.
ID_COLS = {
    "stay_id", "subject_id", "hadm_id", "vent_start", "t", "y", "vap_any",
    "vap_proxy", "vap_icd", "vap_onset", "hrs_vent_to_onset", "vap_window_start",
    "vent_end", "cxr_dicom_id", "cxr_study_id",
}
CATEGORICAL = ["gender", "first_careunit", "admission_type",
               "sputum_color", "sputum_consistency", "sputum_amount",
               "o2_device", "vent_mode"]


# Culture-derived features encode the clinician's *suspicion* of VAP - the very
# thing the model must anticipate independently - so they are excluded from the
# model feature set by default even though `cpis_culture` is a legitimate
# component of the published CPIS score. It ranked as the #1 feature when left
# in, which is the signature of suspicion leakage rather than physiology.
# `cpis_total` sums the culture component, so it carries the same contamination;
# `cpis_total_nc` (added by load_landmarks) is the culture-free equivalent.
SUSPICION_COLS = {"cpis_culture", "cpis_total",
                  "leaky_culture_sent_72h", "leaky_culture_positive"}


def load_landmarks(horizon, cohort_dir=None):
    """Load a landmark table and add the culture-free CPIS total."""
    from cpis_plus import config as C
    from cpis_plus import scoring
    d = pd.read_parquet((cohort_dir or C.COHORT) / f"landmarks_{horizon}h.parquet")
    nc = [c for c in scoring.COMPONENTS if c != "cpis_culture"]
    obs = d[nc].notna().sum(axis=1)
    d["cpis_total_nc"] = d[nc].fillna(0).sum(axis=1).where(obs >= 3)
    return d


def feature_columns(df: pd.DataFrame, include_cpis=True, include_leaky=False,
                    include_cxr_labels=False, include_culture=False,
                    include_image=False):
    """Model feature columns.

    include_cpis        the five non-culture CPIS components + total (legitimate,
                        clinically-grounded - CPIS+ *augments* CPIS)
    include_culture     culture-derived features. Off by default; see above.
                        Turn on only to reproduce the classical CPIS baseline.
    include_leaky       the explicitly-named leaky_* columns. Off by default.
    include_cxr_labels  CheXpert-derived findings on the linked film.
    include_image       img_* / zs_* columns (not produced in this repository).
    """
    cols = []
    for c in df.columns:
        if c in ID_COLS:
            continue
        if c in SUSPICION_COLS and not include_culture:
            continue
        if c.startswith("leaky_") and not include_leaky:
            continue
        if c.startswith("cpis_") and not include_cpis:
            continue
        if c.startswith(("img_", "zs_")) and not include_image:
            continue
        if c.startswith("cxr_") and not include_cxr_labels:
            # cxr_age_h is timing metadata, not a finding; keep it out too so the
            # tabular model cannot infer "a film was ordered"
            continue
        cols.append(c)
    return cols


def assert_no_single_feature_leak(X, y, thresh=0.95, max_report=5):
    """Tripwire: no single feature should separate the outcome almost perfectly.

    A univariate AUROC above `thresh` is essentially never real physiology - it
    means a label, an identifier, or a direct consequence of the outcome has
    entered the feature set. This check exists because three separate ad-hoc
    label columns (`vap_icd`, `vap_proxy`, `y_diag`) each slipped through a
    name-based blocklist and produced AUROC 0.97-1.00.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return
    hits = []
    for c in X.columns:
        v = X[c].values
        m = np.isfinite(v)
        if m.sum() < 200 or len(np.unique(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], v[m])
        a = max(a, 1 - a)
        if a > thresh:
            hits.append((c, round(float(a), 4)))
    if hits:
        hits.sort(key=lambda t: -t[1])
        raise AssertionError(
            f"{len(hits)} feature(s) separate the outcome almost perfectly "
            f"(univariate AUROC > {thresh}); this is leakage, not signal: "
            f"{hits[:max_report]}")


def fit_categorical_vocab(df: pd.DataFrame, cols, top_k=12):
    """Top-k category vocabulary per categorical column, fit on the rows passed
    in. Call this on TRAIN rows only, then pass the result to every later
    `encode()` call (train, val and test alike) so the set of categories that
    survive to a dummy column - and therefore the dummy columns that exist at
    all - is decided without ever looking at val or test. Fitting vocabulary on
    the full dataset (train+val+test together) is a real, if usually small,
    leakage vector: a category that is common in the held-out set but rare
    overall can be kept or dropped depending on data the model is not supposed
    to have seen. `first_careunit` (14 distinct values here) is the column in
    this cohort where "top 12" actually drops anything.
    """
    cats = [c for c in CATEGORICAL if c in cols and c in df.columns]
    vocab = {}
    for c in cats:
        v = df[c].astype(str).str.upper().str.strip()
        vocab[c] = v.value_counts().head(top_k).index.tolist()
    return vocab


def encode(df: pd.DataFrame, cols, vocab=None):
    """Numeric matrix: one-hot the low-cardinality categoricals, pass the rest.

    `vocab`, if given, is the output of `fit_categorical_vocab` on TRAIN rows;
    pass the SAME vocab to every call (train/val/test) so dummy columns line up
    without needing a separate alignment step for categoricals. If `vocab` is
    omitted, the top-k categories are computed from whatever rows are passed to
    this call - fine for exploratory/CV-only work where the same fold structure
    already prevents the categories seen from ever including the current test
    fold's rows in a way that matters, but NOT fine for the canonical
    train/val/test evaluation, where `vocab` must be supplied.
    """
    bad = set(cols) & ID_COLS
    if bad:
        raise AssertionError(f"outcome/identifier columns in feature set: {sorted(bad)}")
    # any column literally named like a label
    lab = [c for c in cols if c == "y" or c.startswith("y_")
           or c.startswith("label") or c.startswith("target")]
    if lab:
        raise AssertionError(f"label-like columns in feature set: {sorted(lab)}")
    X = df[cols].copy()
    cats = [c for c in CATEGORICAL if c in X.columns]
    for c in cats:
        X[c] = X[c].astype(str).str.upper().str.strip()
        top = vocab[c] if vocab and c in vocab else X[c].value_counts().head(12).index
        X[c] = X[c].where(X[c].isin(top), "OTHER")
    X = pd.get_dummies(X, columns=cats, dummy_na=True)
    return X.astype("float32")


def align_columns(X_ref: pd.DataFrame, X_other: pd.DataFrame) -> pd.DataFrame:
    """Reindex `X_other` onto `X_ref`'s columns (fills any column X_other lacks
    with 0, drops any column X_other has that X_ref doesn't). Needed whenever
    two splits are encoded separately and a rare dummy category happens to
    appear in one but not the other - without this, a model fit on X_ref and
    scored on a differently-shaped X_other silently misaligns columns rather
    than erroring, which is worse than either leaking or crashing."""
    return X_other.reindex(columns=X_ref.columns, fill_value=0.0)


def fit_numeric_medians(X: pd.DataFrame) -> pd.Series:
    """Column medians, meant to be computed on TRAIN rows only and then applied
    (via `impute`) to train, val and test alike. LightGBM handles missing
    values natively and needs none of this, but every non-tree baseline in this
    project (logistic regression, GRU, Transformer, feedforward) requires dense
    input, and imputing with a statistic computed across train+val+test
    together lets held-out rows influence the value used to fill training rows'
    gaps - a real leak, not a theoretical one: an earlier sequence-model script
    did exactly this (global `X.median()` before any split) until this pass."""
    return X.median()


def impute(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """Fill missing values using medians computed elsewhere (train-only)."""
    return X.fillna(medians)


def split(df: pd.DataFrame, test_size=0.2, seed=0):
    """Grouped train/test split on subject_id."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df.subject_id))
    return df.iloc[tr].copy(), df.iloc[te].copy()


def folds(df: pd.DataFrame, n_splits=5):
    """Grouped k-fold, as the proposal's k-fold cross-validation.

    As of the canonical train/val/test split (`cpis.split`), `df` here should
    be the TRAIN subset only. Every configuration decision - features,
    architecture, protocol variant, hyperparameters - belongs inside this CV
    loop on train; none of it may be informed by a number computed on val or
    test. Calling this on the full cohort (train+val+test together), as every
    script before the canonical split did, is not itself leakage - CV still
    holds out each fold - but it means the "held-out" fold was, in every one of
    those earlier runs, also used to help choose the configuration being
    evaluated, across the dozens of other times CV was rerun with a different
    configuration. That nested-selection optimism is exactly what the
    canonical test set (evaluated once, at the end) is meant to bound.
    """
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, df.y, groups=df.subject_id))


def check_no_leakage(train, test):
    """Fail loudly if any patient appears on both sides."""
    overlap = set(train.subject_id) & set(test.subject_id)
    if overlap:
        raise AssertionError(f"{len(overlap)} subjects in both train and test")
    return True
