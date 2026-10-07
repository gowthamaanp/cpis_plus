"""Table 3 rows that change data partitioning and resampling.

All rows use the locked 339-feature configuration, the locked LightGBM
hyperparameters and the same fitting procedure as evaluate_constructions.py (fit on an
inner 80% of the training rows, early-stop against a patient-grouped inner
20%), so the reference row is reproduced exactly.

  A_honest         reference protocol, patient-level split (canonical split)
  A_honest_imputed same fit with median imputation; reported only to price the
                   imputation that SMOTE forces on configurations C and D
  B_row_wise       random landmark-level split that ignores patient identity,
                   same train:test size ratio (patient overlap; invalid)
  C_smote_inside   SMOTE applied within the training partition only (correct)
  D_smote_before   SMOTE applied to the whole dataset before a random split
                   (leakage; invalid)

Confidence intervals: patient-clustered bootstrap for A, A_imputed, B and C.
For D, most test rows are synthetic points that belong to no patient, so the
bootstrap is necessarily row-level and is not comparable in precision.

Output: data/cohort/partition_resampling_24h.json
"""
import json
import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score, roc_auc_score
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
NBOOT = 1000


def log(m):
    print(f"[partition] {m}", flush=True)


def load_locked():
    ds, cols = FIN.build_final_dataset()
    return ds, cols, FIN.params()


def fit_predict(P, Xtr, ytr, Xte, groups=None):
    """Fit exactly as evaluate_constructions.fit_score does: inner
    80/20 split of the training rows, early stopping on AUC against the inner
    20%. `groups` makes that inner split patient-grouped, which is possible
    wherever the training rows still have patient identities; it is None only
    for the SMOTE-before-split configuration, whose training rows are largely
    synthetic. The locked hyperparameters are used unmodified throughout, so
    the model is held fixed and not merely the feature set."""
    ytr = np.asarray(ytr)
    if groups is not None:
        itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                        .split(np.zeros(len(ytr)), groups=groups))
    else:
        rs = np.random.default_rng(SEED).permutation(len(ytr))
        cut = int(0.8 * len(ytr))
        itr, iva = rs[:cut], rs[cut:]
    m = lgb.LGBMClassifier(**P)
    m.fit(Xtr.iloc[itr], ytr[itr],
          eval_set=[(Xtr.iloc[iva], ytr[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    return m.predict_proba(Xte)[:, 1]


def score(name, yte, p, groups, results):
    """Point estimate plus a bootstrap CI whose resampling unit is `groups`
    (patient-grouped) or None (row-level, only where patient identity does not
    exist - see the module docstring)."""
    a = roc_auc_score(yte, p)
    pr = average_precision_score(yte, p)
    lo, hi = M.bootstrap_ci(yte, p, roc_auc_score, n=NBOOT, seed=SEED, groups=groups)
    unit = "patient-grouped" if groups is not None else "row-level"
    results[name] = {"auroc": a, "auprc": pr, "auroc_lo": lo, "auroc_hi": hi,
                     "boot_unit": unit}
    log(f"{name}: AUROC {a:.4f} (95% CI {lo:.3f}-{hi:.3f}, {unit}) AUPRC {pr:.4f}")


def main():
    ds, cols, P = load_locked()
    assignment = SPL.load_canonical_split()
    tagged = SPL.apply_split(ds, assignment)
    train = tagged[tagged.split == "train"].reset_index(drop=True)
    test = tagged[tagged.split == "test"].reset_index(drop=True)
    log(f"locked feature set: {len(cols)} raw columns")
    log(f"train {len(train):,} / test {len(test):,} landmarks (canonical partition)")

    vocab = D.fit_categorical_vocab(train, cols)
    Xtr_raw = D.encode(train, cols, vocab=vocab)
    Xte_raw = D.align_columns(Xtr_raw, D.encode(test, cols, vocab=vocab))
    ytr, yte = train.y.values, test.y.values
    g_test = test.subject_id.values

    results = {}

    # ---- A: honest reference. Native missing handling, exactly as the locked
    # model does it, so this row IS the reference protocol row and the table
    # has a single baseline -------------------------------------------------
    p = fit_predict(P, Xtr_raw, ytr, Xte_raw, groups=train.subject_id.values)
    score("A_honest", yte, p, g_test, results)

    # ---- A': the same fit with median imputation. Not a protocol variant -
    # it exists only to price the imputation that configurations C and D are
    # forced into by SMOTE, so those rows are read against the right number --
    med = Xtr_raw.median()
    p = fit_predict(P, Xtr_raw.fillna(med), ytr, Xte_raw.fillna(med),
                    groups=train.subject_id.values)
    score("A_honest_imputed", yte, p, g_test, results)

    # ---- B: row-wise split - pool all rows, single random split ignoring
    # patient identity, same train:test size ratio as the canonical split.
    # Native missing handling, so it is directly comparable to A ------------
    Xall = D.encode(tagged, cols, vocab=vocab)
    yall = tagged.y.values
    g_all = tagged.subject_id.values
    rng = np.random.default_rng(SEED)
    n = len(tagged)
    perm = rng.permutation(n)
    n_test = len(test)
    te_rw, tr_rw = perm[:n_test], perm[n_test:]
    p = fit_predict(P, Xall.iloc[tr_rw], yall[tr_rw], Xall.iloc[te_rw],
                    groups=g_all[tr_rw])
    score("B_row_wise", yall[te_rw], p, g_all[te_rw], results)

    # ---- C: SMOTE inside training partition only (correct usage). Test set
    # is the canonical held-out set, so patient grouping still applies -------
    # The inner early-stopping split is taken BEFORE resampling and grouped by
    # patient, so SMOTE touches only the inner-training rows and the rows the
    # model early-stops against stay real and patient-disjoint. Resampling
    # first would put synthetic neighbours of the early-stopping rows into the
    # fit - a smaller version of the very violation configuration D isolates.
    Xtr_imp = Xtr_raw.fillna(med)
    itr_c, iva_c = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                        .split(np.zeros(len(ytr)), groups=train.subject_id.values))
    xs, ys = SMOTE(random_state=SEED).fit_resample(Xtr_imp.iloc[itr_c], ytr[itr_c])
    m = lgb.LGBMClassifier(**P)
    m.fit(xs, ys, eval_set=[(Xtr_imp.iloc[iva_c], ytr[iva_c])], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    p = m.predict_proba(Xte_raw.fillna(med))[:, 1]
    score("C_smote_inside", yte, p, g_test, results)

    # ---- D: SMOTE before the split. Most test rows are now synthetic points
    # belonging to no patient, so the bootstrap must be row-level: patient
    # identity has been destroyed by the violation itself --------------------
    med_all = Xall.median()
    Xs_all, ys_all = SMOTE(random_state=SEED).fit_resample(Xall.fillna(med_all), yall)
    n_synth = len(Xs_all) - len(Xall)
    perm2 = rng.permutation(len(Xs_all))
    te_d, tr_d = perm2[:n_test], perm2[n_test:]
    frac_synth = (te_d >= len(Xall)).mean()
    log(f"D: {n_synth:,} synthetic rows created; {frac_synth:.1%} of its test "
        f"rows are synthetic and have no patient identity")
    p = fit_predict(P, Xs_all.iloc[tr_d], ys_all[tr_d], Xs_all.iloc[te_d],
                    groups=None)
    score("D_smote_before", ys_all[te_d], p, None, results)
    results["D_smote_before"]["frac_test_synthetic"] = float(frac_synth)

    (C.COHORT / f"partition_resampling_{H}h.json").write_text(
        json.dumps(results, indent=2, default=float))
    log(f"saved data/cohort/partition_resampling_{H}h.json")


if __name__ == "__main__":
    main()
