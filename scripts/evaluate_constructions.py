"""Table 3 rows scored on the held-out test partition. The reference model is fitted
once on the training partition and scored once on the test partition
(cpis_plus.split), then:

  P0_reference  landmark-level scoring of the reference model
  P_episode     the same predictions reduced to one score per episode (maximum)
  P_cc          1:1 matched case-control (VAP-free controls; 10-year age band, sex,
                ventilation-duration band), built separately in train and test
  P_cc_leak     P_cc plus culture-ordering and culture-result features
                (cpis_plus.data.SUSPICION_COLS)

Intervals: 95% patient-clustered bootstrap (cpis_plus.metrics).

Output: data/cohort/constructions_24h.json
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
from cpis_plus import final as FIN
from cpis_plus import metrics as M
from cpis_plus import split as SPL

warnings.filterwarnings("ignore")
SEED = 0
H = 24
MIN_N = 30


def log(m):
    print(f"[constructions] {m}", flush=True)


def load_locked():
    ds, cols = FIN.build_final_dataset()
    return ds, cols, FIN.params()


def fit_score(train, test, cols, P, vocab=None, min_n=MIN_N, name=""):
    """Fit once on `train`, score once on `test`. Returns None (logged, not
    raised) if either side is too small to trust rather than silently
    reporting a number built on a handful of events."""
    if len(train) < min_n or len(test) < min_n or train.y.sum() < 5 or test.y.sum() < 5:
        log(f"  {name}: SKIPPED (train n={len(train)} pos={int(train.y.sum())}, "
            f"test n={len(test)} pos={int(test.y.sum())} - below the {min_n}-row / "
            f"5-event minimum)")
        return None
    vocab = vocab or D.fit_categorical_vocab(train, cols)
    Xtr = D.encode(train, cols, vocab=vocab)
    Xte = D.align_columns(Xtr, D.encode(test, cols, vocab=vocab))
    g = train.subject_id.values
    itr, iva = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED)
                    .split(np.zeros(len(train)), groups=g))
    m = lgb.LGBMClassifier(**P)
    m.fit(Xtr.iloc[itr], train.y.iloc[itr],
          eval_set=[(Xtr.iloc[iva], train.y.iloc[iva])], eval_metric="auc",
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    p_test = m.predict_proba(Xte)[:, 1]
    r = M.evaluate(test.y, p_test, name=name, groups=test.subject_id.values)
    log(f"  {name}: AUROC {r['auroc']:.4f} (95% CI {r['auroc_lo']:.4f}-{r['auroc_hi']:.4f}), "
        f"n_train={len(train):,} n_test={len(test):,}")
    return {"auroc": r["auroc"], "auroc_lo": r["auroc_lo"], "auroc_hi": r["auroc_hi"],
            "n_train": int(len(train)), "n_test": int(len(test)),
            "n_pos_train": int(train.y.sum()), "n_pos_test": int(test.y.sum())}, p_test


def build_matched(df, rng):
    """1:1 age-band/sex/ventilation-duration-band matched case-control,
    VAP-free-only negatives, applied WITHIN whatever frame is passed in (call
    separately on train and on test so the matching never crosses the
    partition boundary)."""
    co = pd.read_parquet(C.COHORT / "cohort.parquet")
    lab = pd.read_parquet(C.COHORT / "labels.parquet")
    info = co.merge(lab[["stay_id", "vap_proxy"]], on="stay_id")
    info = info[info.subject_id.isin(df.subject_id.unique())]
    cases = info[info.vap_proxy].copy()
    ctrls = info[~info.vap_proxy].copy()
    for d in (cases, ctrls):
        d["ageb"] = (d.age // 10).astype(int)
        d["ventb"] = pd.cut(d.vent_hours, [48, 96, 168, 336, 1e9], labels=False)
    used, pairs = set(), []
    for c in cases.itertuples():
        pool = ctrls[(ctrls.ageb == c.ageb) & (ctrls.gender == c.gender)
                     & (ctrls.ventb == c.ventb) & (~ctrls.stay_id.isin(used))]
        if pool.empty:
            continue
        pick = pool.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        used.add(pick.stay_id); pairs.append((c.stay_id, pick.stay_id))
    case_ids = {a for a, _ in pairs}; ctrl_ids = {b for _, b in pairs}
    pos = df[df.stay_id.isin(case_ids) & (df.y == 1)].copy()
    neg = df[df.stay_id.isin(ctrl_ids)].copy(); neg["y"] = 0
    out = pd.concat([pos, neg], ignore_index=True)
    return out, len(pairs)


def main():
    ds, cols, P = load_locked()
    assignment = SPL.load_canonical_split()
    tagged = SPL.apply_split(ds, assignment)
    SPL.check_split_integrity(tagged)
    train = tagged[tagged.split == "train"].reset_index(drop=True)
    test = tagged[tagged.split == "test"].reset_index(drop=True)
    log(f"train {len(train):,} landmarks / {train.subject_id.nunique():,} patients | "
        f"test {len(test):,} landmarks / {test.subject_id.nunique():,} patients")

    results = {}
    vocab = D.fit_categorical_vocab(train, cols)

    # ---------------- P0 reference (train once, score test once) -----------
    r0, p0_test = fit_score(train, test, cols, P, vocab=vocab, name="P0 reference")
    results["P0_reference"] = r0

    # ---------------- P_episode (aggregate P0's test predictions) ----------
    s = test.copy(); s["p"] = p0_test
    ep = M.episode_level(s)
    ep_groups = test.drop_duplicates("stay_id").set_index("stay_id").loc[ep.index, "subject_id"].values
    r_ep = M.evaluate(ep.y, ep.p, name="P_episode", groups=ep_groups)
    log(f"  P_episode: AUROC {r_ep['auroc']:.4f} (95% CI {r_ep['auroc_lo']:.4f}-"
        f"{r_ep['auroc_hi']:.4f}), n_episodes={len(ep):,}, n_pos={int(ep.y.sum())}")
    results["P_episode"] = {"auroc": r_ep["auroc"], "auroc_lo": r_ep["auroc_lo"],
                            "auroc_hi": r_ep["auroc_hi"], "n_episodes": int(len(ep)),
                            "n_pos": int(ep.y.sum())}

    # ---------------- P_cc / P_cc_leak (matched within train, within test) -
    rng = np.random.default_rng(SEED)
    cc_train, n_pairs_train = build_matched(train, rng)
    cc_test, n_pairs_test = build_matched(test, rng)
    log(f"matched cohort: {n_pairs_train} pairs in train, {n_pairs_test} pairs in test")
    out = fit_score(cc_train, cc_test, cols, P, name="P_cc matched case-control")
    if out:
        r_cc, oof_cc = out
        results["P_cc"] = r_cc

    ext = pd.read_parquet(C.COHORT / f"landmarks_{H}h_extended.parquet")
    susp_cols = sorted(c for c in D.SUSPICION_COLS if c in ext.columns)
    susp = ext[["stay_id", "t"] + susp_cols].drop_duplicates(["stay_id", "t"])
    cc_train_leak = cc_train.merge(susp, on=["stay_id", "t"], how="left", validate="one_to_one")
    cc_test_leak = cc_test.merge(susp, on=["stay_id", "t"], how="left", validate="one_to_one")
    n_matched = cc_train_leak[susp_cols[0]].notna().sum() if susp_cols else 0
    if not susp_cols or n_matched == 0:
        raise AssertionError("P_cc_leak: suspicion-column merge found nothing")
    out = fit_score(cc_train_leak, cc_test_leak, cols + susp_cols, P,
                    name="P_cc_leak matched case-control + suspicion")
    if out:
        r_leak, _ = out
        results["P_cc_leak"] = r_leak

    (C.COHORT / f"constructions_{H}h.json").write_text(
        json.dumps(results, indent=2, default=float))
    log(f"saved data/cohort/constructions_{H}h.json")


if __name__ == "__main__":
    main()
