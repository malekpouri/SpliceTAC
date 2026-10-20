"""Conformal calibration, transferability scoring, and evaluation metrics."""
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, matthews_corrcoef,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors

ALPHAS = (0.2, 0.1, 0.05)


# ----------------------------------------------------------------------------- conformal
def split_quantile(scores, alpha):
    """Standard split-conformal quantile (assumes exchangeable variants)."""
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def group_quantile(scores, groups, alpha):
    """Gene-weighted (two-layer hierarchical) conformal quantile.

    Each calibration gene receives total weight 1/m regardless of its number of variants
    (pooled-CDF construction of Dunn, Wasserman & Ramdas, 2023), targeting coverage of a random
    variant from a new, exchangeable gene rather than a random variant from the pooled data.
    """
    scores = np.asarray(scores, dtype=float)
    g = pd.Series(np.asarray(groups))
    m = g.nunique()
    w = 1.0 / g.map(g.value_counts()).to_numpy() / m
    o = np.argsort(scores)
    cw = np.cumsum(w[o])
    level = min(1.0, (1 - alpha) * (m + 1) / m)
    if level >= 1.0:
        return np.inf
    return scores[o][np.searchsorted(cw, level - 1e-12)]


def mondrian_quantiles(scores, cats, alpha):
    return {c: split_quantile(scores[cats == c], alpha) for c in np.unique(cats)}


# ----------------------------------------------------------------------------- transferability
TRANSFER_FEATS = ["ens_std", "ale_sig", "abs_mu", "knn_ctx", "knn_z", "abs_dist_ss", "exon_len_l",
                  "is_intronic", "pg_absmax", "assay_unknown"]


def knn_distance(train_X, query_X, k=10, exclude_self=False):
    nn_ = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(train_X)
    d, _ = nn_.kneighbors(query_X)
    return d[:, int(exclude_self):].mean(1)


def fit_difficulty(F, abs_res, feats=TRANSFER_FEATS):
    """Learn a(x) ~ E[|y - mu| | transferability features] from out-of-gene residuals."""
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, objective="l1",
                              verbose=-1, random_state=0)
    model.fit(F[list(feats)], abs_res)
    model.feats_ = list(feats)
    return model


def predict_difficulty(model, F, floor=0.005):
    return np.maximum(model.predict(F[model.feats_]), floor)


# ----------------------------------------------------------------------------- metrics
def reg_metrics(y, mu):
    return {"MAE": float(np.mean(np.abs(y - mu))), "RMSE": float(np.sqrt(np.mean((y - mu) ** 2))),
            "Spearman": float(spearmanr(y, mu).statistic), "Pearson": float(np.corrcoef(y, mu)[0, 1])}


def direction_acc(y, mu, thr=0.1):
    m = np.abs(y) > thr
    return float(np.mean(np.sign(y[m]) == np.sign(mu[m]))) if m.any() else np.nan


def interval_metrics(y, lo, hi, groups, alpha, min_group=10, regions=None, effect_thr=0.2):
    cov = (y >= lo) & (y <= hi)
    extra = {}
    if regions is not None:
        rc = pd.Series(cov).groupby(np.asarray(regions)).mean()
        extra = {"region_cov_min": float(rc.min()), **{f"cov_{k}": float(v) for k, v in rc.items()}}
    big = np.abs(y) > effect_thr
    extra["cov_large_effect"] = float(cov[big].mean()) if big.any() else np.nan
    extra["width_large_effect"] = float((hi - lo)[big].mean()) if big.any() else np.nan
    width = hi - lo
    iscore = width + 2 / alpha * ((lo - y) * (y < lo) + (y - hi) * (y > hi))
    gdf = pd.DataFrame({"g": groups, "c": cov})
    gc = gdf.groupby("g").c.agg(["mean", "size"])
    gc = gc[gc["size"] >= min_group]["mean"]
    return {"coverage": float(cov.mean()), "width_mean": float(width.mean()),
            "width_median": float(np.median(width)), "interval_score": float(iscore.mean()),
            "gene_cov_mean": float(gc.mean()) if len(gc) else np.nan,
            "gene_cov_p10": float(gc.quantile(0.1)) if len(gc) else np.nan,
            "gene_cov_min": float(gc.min()) if len(gc) else np.nan,
            "frac_genes_undercovered": float((gc < (1 - alpha) - 0.1).mean()) if len(gc) else np.nan,
            "n_genes_eval": int(len(gc)), **extra}


def risk_coverage(y, mu, conf, effect_thr=0.2, grid=np.linspace(0.05, 1, 96)):
    """Selective regression: accept most-confident fraction c; risk = MAE of accepted."""
    o = np.argsort(-conf)
    err = np.abs(y - mu)[o]
    eff = (np.abs(y) > effect_thr)[o]
    n = len(y)
    risks, effret = [], []
    for c in grid:
        k = max(1, int(round(c * n)))
        risks.append(err[:k].mean())
        effret.append(eff[:k].sum() / max(1, eff.sum()))
    risks, effret = np.array(risks), np.array(effret)
    full = err.mean()
    out = {"AURC": float(np.trapezoid(risks, grid) / (grid[-1] - grid[0])), "risk_full": float(full)}
    for c in (0.5, 0.7, 0.9):
        i = np.argmin(np.abs(grid - c))
        out[f"risk@{int(c * 100)}"] = float(risks[i])
        out[f"effect_retained@{int(c * 100)}"] = float(effret[i])
    # oracle AURC for normalisation (excess AURC)
    oe = np.sort(np.abs(y - mu))
    orisk = np.array([oe[:max(1, int(round(c * n)))].mean() for c in grid])
    out["E-AURC"] = out["AURC"] - float(np.trapezoid(orisk, grid) / (grid[-1] - grid[0]))
    return out, risks, effret


def cls_metrics(y, p, thr=None):
    y = np.asarray(y).astype(int)
    out = {"AUPRC": float(average_precision_score(y, p)), "AUROC": float(roc_auc_score(y, p)),
           "prevalence": float(y.mean()), "n": int(len(y))}
    if thr is not None:
        out["MCC"] = float(matthews_corrcoef(y, p >= thr))
    if np.all((p >= 0) & (p <= 1)):
        out["Brier"] = float(brier_score_loss(y, p))
    return out


def best_mcc_threshold(y, p):
    qs = np.unique(np.quantile(p, np.linspace(0.5, 0.995, 200)))
    m = [matthews_corrcoef(y, p >= q) for q in qs]
    return float(qs[int(np.argmax(m))])


def gene_bootstrap(df, stat_fn, group_col="gene_name", n_boot=1000, seed=0):
    """Percentile CI by resampling genes (not variants)."""
    rng = np.random.default_rng(seed)
    genes = df[group_col].unique()
    idx = {g: np.where(df[group_col].to_numpy() == g)[0] for g in genes}
    vals = []
    for _ in range(n_boot):
        s = rng.choice(genes, len(genes), replace=True)
        j = np.concatenate([idx[g] for g in s])
        vals.append(stat_fn(df.iloc[j]))
    vals = np.array(vals)
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))
