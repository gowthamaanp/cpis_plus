"""Evaluation metrics for VAP early prediction.

Reports AUROC and F1 as the proposal specifies, plus two the reviewers will
expect for a rare, time-sensitive outcome:

- **AUPRC**, because at ~3-8% landmark prevalence AUROC flatters every model.
- **Episode-level** performance, because the clinically meaningful question is
  "was this patient flagged before onset", not "was each 12-hourly landmark
  classified correctly".
- **Sensitivity/specificity at the CPIS>6 operating point**, so the comparison
  against the published 61%/43% is like-for-like.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)


def bootstrap_ci(y, p, fn, n=1000, seed=0, alpha=0.05, groups=None):
    """Percentile CI, resampling with replacement.

    `groups`, if given, makes this a PATIENT-grouped bootstrap: each resample
    draws whole patients (with replacement) and keeps all of that patient's
    rows together, rather than drawing individual rows independently. This
    matters here specifically: a patient contributes many landmarks, and
    row-level resampling treats those landmarks as independent draws, which
    understates the true resampling variance and produces a confidence
    interval narrower than the data support - a real evaluation fault in a
    project whose central claim rests on interval widths being trustworthy.
    Every CI in this project computed before this fix used plain row-level
    resampling (groups=None); comparisons against confidence intervals
    reported by that earlier code are not on the same footing as one computed
    with `groups` supplied.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    stats = []
    if groups is None:
        for _ in range(n):
            i = rng.integers(0, len(y), len(y))
            if len(np.unique(y[i])) < 2:
                continue
            stats.append(fn(y[i], p[i]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
        for _ in range(n):
            sampled = rng.choice(uniq, size=len(uniq), replace=True)
            i = np.concatenate([idx_by_group[g] for g in sampled])
            if len(np.unique(y[i])) < 2:
                continue
            stats.append(fn(y[i], p[i]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def at_threshold(y, p, thr):
    yh = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {
        "threshold": thr,
        "sensitivity": tp / (tp + fn_) if tp + fn_ else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ppv": tp / (tp + fp) if tp + fp else np.nan,
        "npv": tn / (tn + fn_) if tn + fn_ else np.nan,
        "f1": f1_score(y, yh, zero_division=0),
    }


def sens_at_spec(y, p, target_spec=0.43):
    """Sensitivity at the specificity CPIS achieves - the honest comparison."""
    fpr, tpr, thr = roc_curve(y, p)
    spec = 1 - fpr
    i = np.argmin(np.abs(spec - target_spec))
    return {"sensitivity": tpr[i], "specificity": spec[i], "threshold": thr[i]}


def evaluate(y, p, ci=True, name="", groups=None):
    """`groups` (e.g. subject_id per row), if given, makes the reported CIs
    patient-grouped rather than row-level - see `bootstrap_ci`. Pass it
    whenever y/p are landmark-level and a patient contributes more than one
    row, which is true of every primary analysis in this project."""
    y, p = np.asarray(y), np.asarray(p)
    out = {
        "name": name,
        "n": len(y),
        "n_pos": int(y.sum()),
        "prevalence": float(y.mean()),
        "auroc": roc_auc_score(y, p),
        "auprc": average_precision_score(y, p),
    }
    if ci:
        out["auroc_lo"], out["auroc_hi"] = bootstrap_ci(y, p, roc_auc_score, groups=groups)
        out["auprc_lo"], out["auprc_hi"] = bootstrap_ci(y, p, average_precision_score, groups=groups)

    # best-F1 operating point
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    out.update({f"f1opt_{k}": v for k, v in
                at_threshold(y, p, thr[best] if len(thr) else 0.5).items()})
    out.update({f"cpisop_{k}": v for k, v in sens_at_spec(y, p).items()})
    return out


def episode_level(df, prob_col="p", how="max", label_col="vap_onset"):
    """Collapse landmarks to one score per ventilation episode.

    `max` = the patient is flagged if the model ever fired before onset, which
    is how an early-warning system is actually used.

    `label_col` defaults to `vap_onset` (truthy iff notna), NOT the `vap_any`
    column this function used until this fix. `vap_any` is defined upstream
    (scripts/build_cohort.py) as `vap_proxy | vap_icd` - the union of the
    proxy label this study's landmark-level `y` is built from AND a separate
    ICD-code-based flag. Every landmark-level result in this study is scored
    against the pure proxy label; episode-level results computed through this
    function were therefore, until this fix, silently scored against a
    broader and materially different outcome (25.6% episode prevalence via
    vap_any vs. the 16.2% vap_onset-based prevalence stated everywhere else in
    this study's Results) - not a different aggregation of the same label, a
    different label. `vap_onset.notna()` matches what landmark-level `y` is
    derived from and is the correct default for a metric meant to be compared
    against this study's own landmark-level figures. Pass `label_col="vap_any"`
    explicitly if a result genuinely wants the broader ICD-inclusive outcome -
    but that is a distinct sensitivity analysis, not the primary quantity.
    """
    g = df.groupby("stay_id")
    if label_col == "vap_onset":
        y = g["vap_onset"].apply(lambda s: s.notna().any()).astype(int)
    else:
        y = g[label_col].first().astype(int)
    return pd.DataFrame({
        "y": y,
        "p": g[prob_col].max() if how == "max" else g[prob_col].mean(),
    })


def report(results):
    """Tidy comparison table across models."""
    d = pd.DataFrame(results)
    cols = ["name", "n", "n_pos", "auroc", "auroc_lo", "auroc_hi", "auprc",
            "f1opt_f1", "f1opt_sensitivity", "f1opt_specificity",
            "cpisop_sensitivity"]
    return d[[c for c in cols if c in d.columns]].round(3)
