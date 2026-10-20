"""Revision analyses (analysis plan, Amendment 4). CPU only; reuses saved predictions.

1. Extra UQ baselines on the gene protocol: CQR (LightGBM quantile models on the same inputs), weighted split
   conformal under covariate shift (Tibshirani et al. 2019), Mondrian conformal by predicted-effect bins.
2. Coverage conditional on the *predicted* effect (|mu| > tau) next to outcome-conditional coverage.
3. Predictor-agnostic comparison: the same UQ methods on Pangolin-ridge and LightGBM predictors, with a
   TAC-style difficulty model trained on their own out-of-gene residuals.
4. Few-shot target calibration for the unseen-study protocol.
Outputs: results/rev_*.csv
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

import experiment as ex
import uq
from analyze import load, p_large
from config import PROCESSED, RESULTS, SEED

ALPHAS = (0.2, 0.1, 0.05)
TAU = 0.2
RNG = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------- data without GPU tensors
def load_df():
    df = pd.read_parquet(PROCESSED / "variants_v1.parquet")
    tab = pd.read_parquet(PROCESSED / "tabular_v1.parquet").drop(columns="sample_id")
    pg = pd.read_parquet(PROCESSED / "pangolin_features_v1.parquet").drop(columns="sample_id")
    pgsum = [c for c in pg.columns if c.startswith("pg_")]
    df = pd.concat([df.reset_index(drop=True), tab, pg], axis=1)
    df["pg_absmax"] = df[[c for c in pgsum if "_d_" in c or "gain" in c or "loss" in c]].abs().max(1)
    df["abs_dist_ss"] = np.log1p(np.minimum(np.abs(df.dist_acc), np.abs(df.dist_don)))
    ctxemb = df[[f"pe_{k}" for k in range(128)]].to_numpy(np.float32)
    return df, pgsum, ctxemb


def gene_folds(df, seed=SEED):
    pool = np.where(df.study_id.isin(["MFASS", "VexSeq"]) & df.delta_psi.notna())[0]
    folds = ex.gene_split(df, pool, [0.2] * 5, seed)
    out = []
    for k in range(5):
        dev = np.concatenate([folds[j] for j in range(5) if j != k])
        tr, va, ca = ex.gene_split(df, dev, [0.7, 0.1, 0.2], seed + k)
        out.append((tr, va, ca, folds[k]))
    return out


# ----------------------------------------------------------------------------- interval constructions
def weighted_quantile_per_test(scores, w_cal, w_test, alpha):
    """Tibshirani et al. (2019) weighted split-conformal quantile, one per test point."""
    o = np.argsort(scores)
    s, wc = scores[o], w_cal[o]
    cw = np.cumsum(wc)
    W = cw[-1]
    q = np.full(len(w_test), np.inf)
    for i, wt in enumerate(w_test):
        level = (1 - alpha) * (W + wt)
        j = np.searchsorted(cw, level)
        if j < len(s):
            q[i] = s[j]
    return q


def density_ratio(F_cal, F_test):
    X = np.vstack([F_cal, F_test])
    z = np.r_[np.zeros(len(F_cal)), np.ones(len(F_test))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), z)
    p = lambda F: np.clip(clf.predict_proba(sc.transform(F))[:, 1], 1e-3, 1 - 1e-3)
    odds = lambda F: p(F) / (1 - p(F)) * len(F_cal) / len(F_test)
    return odds(F_cal), odds(F_test)


def mondrian_bins(mu_c, mu_t, res, alpha, nbins=3):
    edges = np.quantile(np.abs(mu_c), np.linspace(0, 1, nbins + 1)[1:-1])
    bc, bt = np.digitize(np.abs(mu_c), edges), np.digitize(np.abs(mu_t), edges)
    q = {b: uq.split_quantile(res[bc == b], alpha) for b in np.unique(bc)}
    return np.array([q.get(b, np.inf) for b in bt])


def cqr_models(X, y, tr, va, alpha):
    ms = []
    for q in (alpha / 2, 1 - alpha / 2):
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=600, learning_rate=0.03,
                              num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, verbose=-1, random_state=SEED)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
        ms.append(m)
    return ms


def metrics(y, lo, hi, mu, genes, regions, alpha):
    m = uq.interval_metrics(y, lo, hi, genes, alpha, regions=regions)
    cov = (y >= lo) & (y <= hi)
    big_pred = np.abs(mu) > TAU
    m["cov_pred_large"] = float(cov[big_pred].mean()) if big_pred.any() else np.nan
    m["cov_pred_small"] = float(cov[~big_pred].mean()) if (~big_pred).any() else np.nan
    m["n_pred_large"] = int(big_pred.sum())
    return m


def clip(mu, h):
    return np.maximum(mu - h, -1), np.minimum(mu + h, 1)


# ----------------------------------------------------------------------------- 1+2: extra baselines (gene)
def extra_baselines(df, X):
    p = load("gene")
    y_all = df.delta_psi.to_numpy()
    sid = df.sample_id.to_numpy()
    rows, pooled = [], {}
    feats = ["f_" + f for f in uq.TRANSFER_FEATS if f != "assay_unknown"]
    for k, (tr, va, ca, te) in enumerate(gene_folds(df)):
        f = p[p.fold == f"gene{k}"]
        c = f[f.split == "cal"].set_index("sample_id").loc[sid[ca]]
        t = f[f.split == "test"].set_index("sample_id").loc[sid[te]]
        assert len(c) == len(ca) and len(t) == len(te), "split reconstruction mismatch"
        yc, yt = c.delta_psi.to_numpy(), t.delta_psi.to_numpy()
        res = np.abs(yc - c.mu.to_numpy())
        uc = np.sqrt(c.ens_std ** 2 + c.ale_sig ** 2).to_numpy()
        ut = np.sqrt(t.ens_std ** 2 + t.ale_sig ** 2).to_numpy()
        wc, wt = density_ratio(c[feats].to_numpy(), t[feats].to_numpy())
        for alpha in ALPHAS:
            out = {}
            # weighted scaled conformal (covariate-shift correction)
            qw = weighted_quantile_per_test(res / uc, wc, wt, alpha)
            out["weighted_scaled_conformal"] = clip(t.mu.to_numpy(), qw * ut)
            # Mondrian by predicted-effect bins on scaled scores
            out["mondrian_pred_effect"] = clip(t.mu.to_numpy(),
                                               mondrian_bins(c.mu.to_numpy(), t.mu.to_numpy(), res / uc, alpha) * ut)
            # CQR on the same inputs (own quantile predictor)
            lo_m, hi_m = cqr_models(X, y_all, tr, va, alpha)
            lc, hc = lo_m.predict(X[ca]), hi_m.predict(X[ca])
            lt, ht = lo_m.predict(X[te]), hi_m.predict(X[te])
            sc = np.maximum(lc - yc, yc - hc)
            qs = uq.split_quantile(sc, alpha)
            qg = uq.group_quantile(sc, c.gene_name.to_numpy(), alpha)
            out["CQR"] = (np.maximum(lt - qs, -1), np.minimum(ht + qs, 1))
            out["CQR_gene_weighted"] = (np.maximum(lt - qg, -1), np.minimum(ht + qg, 1))
            # reference methods recomputed for the same table
            for mname in ("scaled_conformal_ens", "group_conformal", "TAC"):
                lo, hi = ex.intervals(mname, c, t, yc, c.gene_name, t.gene_name, c.a_TAC, t.a_TAC,
                                      c.region, t.region, alpha)
                out[mname] = (np.asarray(lo), np.asarray(hi))
            for mname, (lo, hi) in out.items():
                mu_ref = (lt + ht) / 2 if mname.startswith("CQR") else t.mu.to_numpy()
                pooled.setdefault((alpha, mname), []).append(
                    pd.DataFrame({"y": yt, "lo": lo, "hi": hi, "mu": mu_ref, "gene": t.gene_name.to_numpy(),
                                  "region": t.region.to_numpy()}))
        print("fold", k, "done", flush=True)
    for (alpha, mname), parts in pooled.items():
        d = pd.concat(parts)
        r = metrics(d.y.to_numpy(), d.lo.to_numpy(), d.hi.to_numpy(), d.mu.to_numpy(), d.gene.to_numpy(),
                    d.region.to_numpy(), alpha)
        rows.append({"alpha": alpha, "method": mname, **r})
    pd.DataFrame(rows).to_csv(RESULTS / "rev_extra_baselines_gene.csv", index=False)


# ----------------------------------------------------------------------------- 3: predictor-agnostic
def difficulty_features(df, idx, mu, ref_exons_emb, ctxemb):
    F = pd.DataFrame(index=idx)
    F["abs_mu"] = np.abs(mu)
    F["knn_ctx"] = uq.knn_distance(ref_exons_emb, ctxemb[idx])
    for c in ("abs_dist_ss", "exon_len_l", "pg_absmax"):
        F[c] = df.loc[idx, c].to_numpy()
    F["is_intronic"] = df.loc[idx, "is_intronic"].astype(float).to_numpy()
    return F


def fit_predictor(kind, df, X, pgsum, tr, va):
    y = df.delta_psi.to_numpy()
    if kind == "pangolin_ridge":
        m = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(df.loc[tr, pgsum], y[tr])
        return lambda idx: m.predict(df.loc[idx, pgsum])
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, objective="huber", alpha=0.1,
                          verbose=-1, random_state=SEED)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    return lambda idx: m.predict(X[idx])


def predictor_agnostic(df, X, pgsum, ctxemb):
    y = df.delta_psi.to_numpy()
    rows, tri = [], []
    DFEATS = ["abs_mu", "knn_ctx", "abs_dist_ss", "exon_len_l", "pg_absmax", "is_intronic"]
    for kind in ("pangolin_ridge", "lightgbm"):
        pooled = {}
        for k, (tr, va, ca, te) in enumerate(gene_folds(df)):
            pred = fit_predictor(kind, df, X, pgsum, tr, va)
            mu_c, mu_t = pred(ca), pred(te)
            # out-of-gene residuals inside the training genes
            inner = ex.gene_split(df, tr, [0.2] * 5, SEED + k + 991)
            Fo, Ro = [], []
            for j in range(5):
                itr = np.concatenate([inner[i] for i in range(5) if i != j])
                pj = fit_predictor(kind, df, X, pgsum, itr, va)
                mj = pj(inner[j])
                Fo.append(difficulty_features(df, inner[j], mj, ex.exon_ref_embeddings(ctxemb, df, itr), ctxemb))
                Ro.append(np.abs(y[inner[j]] - mj))
            dm = uq.fit_difficulty(pd.concat(Fo), np.concatenate(Ro), DFEATS)
            ref = ex.exon_ref_embeddings(ctxemb, df, tr)
            a_c = uq.predict_difficulty(dm, difficulty_features(df, ca, mu_c, ref, ctxemb))
            a_t = uq.predict_difficulty(dm, difficulty_features(df, te, mu_t, ref, ctxemb))
            yc, yt = y[ca], y[te]
            gc, gt = df.gene_name.to_numpy()[ca], df.gene_name.to_numpy()[te]
            res = np.abs(yc - mu_c)
            for alpha in ALPHAS:
                out = {"split_conformal": clip(mu_t, uq.split_quantile(res, alpha)),
                       "group_conformal": clip(mu_t, uq.group_quantile(res, gc, alpha)),
                       "mondrian_pred_effect": clip(mu_t, mondrian_bins(mu_c, mu_t, res, alpha)),
                       "TAC": clip(mu_t, uq.group_quantile(res / a_c, gc, alpha) * a_t)}
                for mname, (lo, hi) in out.items():
                    pooled.setdefault((alpha, mname), []).append(pd.DataFrame(
                        {"y": yt, "lo": lo, "hi": hi, "mu": mu_t, "gene": gt, "region": df.region.to_numpy()[te]}))
            # class-conditional triage with sigma ~ a(x) * sqrt(pi/2)
            for gw in (False, True):
                sig_c, sig_t = a_c * np.sqrt(np.pi / 2), a_t * np.sqrt(np.pi / 2)
                s_c = norm.sf((TAU - mu_c) / sig_c) + norm.cdf((-TAU - mu_c) / sig_c)
                s_t = norm.sf((TAU - mu_t) / sig_t) + norm.cdf((-TAU - mu_t) / sig_t)
                zc = np.abs(yc) > TAU
                q = {}
                for cls, nonc in ((1, 1 - s_c), (0, s_c)):
                    m = zc == cls
                    q[cls] = uq.group_quantile(nonc[m], gc[m], 0.1) if gw else uq.split_quantile(nonc[m], 0.1)
                L, S = ((1 - s_t) <= q[1]), (s_t <= q[0])
                call = np.where(L & ~S, "large", np.where(S & ~L, "small", "abstain"))
                tri.append(pd.DataFrame({"predictor": kind, "gw": gw, "call": call, "big": np.abs(yt) > TAU}))
            print(kind, "fold", k, "done", flush=True)
        for (alpha, mname), parts in pooled.items():
            d = pd.concat(parts)
            r = metrics(d.y.to_numpy(), d.lo.to_numpy(), d.hi.to_numpy(), d.mu.to_numpy(), d.gene.to_numpy(),
                        d.region.to_numpy(), alpha)
            rows.append({"predictor": kind, "alpha": alpha, "method": mname, **r,
                         "MAE": float(np.mean(np.abs(d.y - d.mu)))})
    pd.DataFrame(rows).to_csv(RESULTS / "rev_predictor_agnostic.csv", index=False)
    t = pd.concat(tri)
    tr_rows = []
    for (kind, gw), g in t.groupby(["predictor", "gw"]):
        b = g.big
        tr_rows.append({"predictor": kind, "method": "Mondrian-" + ("gene" if gw else "split"),
                        "frac_decided": (g.call != "abstain").mean(),
                        "recall_large_confident": ((g.call == "large") & b).sum() / b.sum(),
                        "frac_large_called_small": ((g.call == "small") & b).sum() / b.sum(),
                        "precision_large_confident": ((g.call == "large") & b).sum() / max(1, (g.call == "large").sum())})
    pd.DataFrame(tr_rows).to_csv(RESULTS / "rev_predictor_agnostic_triage.csv", index=False)


# ----------------------------------------------------------------------------- 4: few-shot target calibration
def fewshot_study():
    p = load("study")
    rows = []
    for fold, f in p.groupby("fold"):
        t = f[f.split == "test"].reset_index(drop=True)
        genes = t.gene_name.unique()
        for rep in range(20):
            rng = np.random.default_rng(SEED + rep)
            cal_genes = set(rng.choice(genes, max(1, int(0.2 * len(genes))), replace=False))
            m = t.gene_name.isin(cal_genes).to_numpy()
            c, te = t[m], t[~m]
            for alpha in (0.1,):
                for mname in ("split_conformal", "scaled_conformal_ens", "group_conformal", "TAC"):
                    lo, hi = ex.intervals(mname, c, te, c.delta_psi, c.gene_name, te.gene_name, c.a_TAC,
                                          te.a_TAC, c.region, te.region, alpha)
                    r = metrics(te.delta_psi.to_numpy(), np.asarray(lo), np.asarray(hi), te.mu.to_numpy(),
                                te.gene_name.to_numpy(), te.region.to_numpy(), alpha)
                    rows.append({"direction": fold, "rep": rep, "method": mname, "alpha": alpha, **r})
    d = pd.DataFrame(rows)
    agg = d.groupby(["direction", "method"]).agg(["mean", "std"]).drop(columns=["rep", "alpha"], level=0)
    agg.columns = ["_".join(c) for c in agg.columns]
    agg.reset_index().to_csv(RESULTS / "rev_fewshot_study.csv", index=False)


# ----------------------------------------------------------------------------- predicted-effect coverage for main methods
def predicted_effect_coverage():
    rows = []
    for prot in ("gene", "study", "random"):
        p = load(prot)
        for alpha in ALPHAS:
            for mname in ex.UQ_METHODS:
                parts = []
                for fold, f in p.groupby("fold"):
                    c, t = f[f.split == "cal"], f[f.split == "test"]
                    lo, hi = ex.intervals(mname, c, t, c.delta_psi, c.gene_name, t.gene_name, c.a_TAC, t.a_TAC,
                                          c.region, t.region, alpha)
                    parts.append(t.assign(lo=np.asarray(lo), hi=np.asarray(hi)))
                d = pd.concat(parts)
                for study, g in [("all", d)] + list(d.groupby("study_id")):
                    r = metrics(g.delta_psi.to_numpy(), g.lo.to_numpy(), g.hi.to_numpy(), g.mu.to_numpy(),
                                g.gene_name.to_numpy(), g.region.to_numpy(), alpha)
                    rows.append({"protocol": prot, "study": study, "alpha": alpha, "method": mname,
                                 "cov_pred_large": r["cov_pred_large"], "cov_pred_small": r["cov_pred_small"],
                                 "n_pred_large": r["n_pred_large"], "cov_large_effect": r["cov_large_effect"]})
    pd.DataFrame(rows).to_csv(RESULTS / "rev_predicted_effect_coverage.csv", index=False)


# ----------------------------------------------------------------------------- 5: robustness (second seed, MSE loss)
def robustness():
    from analyze import mondrian_triage
    rows = []
    for tag, label in (("gene", "main (seed 20261019, Huber)"), ("gene_seed2", "second partition (seed 20261020)"),
                       ("gene_mse", "squared-error loss")):
        if not (RESULTS / f"pred_{tag}.parquet").exists():
            continue
        p = load(tag)
        t = p[p.split == "test"]
        point = uq.reg_metrics(t.delta_psi.to_numpy(), t.mu.to_numpy())
        has_tac = "a_TAC" in p.columns
        for mname in ["split_conformal", "scaled_conformal_ens", "group_conformal"] + (["TAC"] if has_tac else []):
            parts = []
            for fold, f in p.groupby("fold"):
                c, tt = f[f.split == "cal"], f[f.split == "test"]
                lo, hi = ex.intervals(mname, c, tt, c.delta_psi, c.gene_name, tt.gene_name,
                                      c.a_TAC if has_tac else None, tt.a_TAC if has_tac else None,
                                      c.region, tt.region, 0.1)
                parts.append(tt.assign(lo=np.asarray(lo), hi=np.asarray(hi)))
            d = pd.concat(parts)
            r = metrics(d.delta_psi.to_numpy(), d.lo.to_numpy(), d.hi.to_numpy(), d.mu.to_numpy(),
                        d.gene_name.to_numpy(), d.region.to_numpy(), 0.1)
            rows.append({"run": label, "method": mname, "MAE": point["MAE"], "Spearman": point["Spearman"],
                         "mean_abs_mu": float(np.abs(t.mu).mean()), **r})
        tri = mondrian_triage(p, TAU, gene_weighted=True)
        b = tri.delta_psi.abs() > TAU
        rows.append({"run": label, "method": "class-conditional triage (gene-weighted)",
                     "MAE": point["MAE"], "Spearman": point["Spearman"],
                     "frac_large_called_small": float(((tri.call == "small") & b).sum() / b.sum()),
                     "recall_large_confident": float(((tri.call == "large") & b).sum() / b.sum()),
                     "frac_decided": float((tri.call != "abstain").mean())})
    pd.DataFrame(rows).to_csv(RESULTS / "rev_robustness.csv", index=False)


def main():
    import sys
    if "--robustness" in sys.argv:
        robustness(); print("robustness done"); return
    df, pgsum, ctxemb = load_df()
    X = ex.tabular_matrix(df, pgsum)
    predicted_effect_coverage(); print("predicted-effect coverage done", flush=True)
    fewshot_study(); print("few-shot done", flush=True)
    extra_baselines(df, X); print("extra baselines done", flush=True)
    predictor_agnostic(df, X, pgsum, ctxemb); print("predictor-agnostic done", flush=True)


if __name__ == "__main__":
    main()
