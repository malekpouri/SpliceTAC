"""Core experiment machinery: data tensors, ensemble fitting, out-of-gene residuals, UQ methods."""
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.linear_model import RidgeCV

import lightgbm as lgb
import nn_model as nm
import uq
from config import PROCESSED, SEED
from features import ASSAYS

N_SEEDS, N_INNER, N_INNER_SEEDS = 5, 5, 3


# ----------------------------------------------------------------------------- data
def load_all():
    df = pd.read_parquet(PROCESSED / "variants_v1.parquet")
    tab = pd.read_parquet(PROCESSED / "tabular_v1.parquet")
    pg = pd.read_parquet(PROCESSED / "pangolin_features_v1.parquet")
    win = np.load(PROCESSED / "windows_v1.npz", allow_pickle=True)
    assert (tab.sample_id.values == df.sample_id.values).all()
    assert (pg.sample_id.values == df.sample_id.values).all()
    assert (win["sample_id"] == df.sample_id.values).all()
    tab = tab.drop(columns="sample_id")
    pg = pg.drop(columns="sample_id")
    ctx_cols = [c for c in pg.columns]
    pgsum = [c for c in pg.columns if c.startswith("pg_")]
    df = pd.concat([df.reset_index(drop=True), tab.reset_index(drop=True), pg.reset_index(drop=True)], axis=1)
    df["pg_absmax"] = df[[c for c in pgsum if "_d_" in c or "gain" in c or "loss" in c]].abs().max(1)
    df["abs_dist_ss"] = np.log1p(np.minimum(np.abs(df.dist_acc), np.abs(df.dist_don)))
    # exon "splicing-context" embedding: Pangolin hidden state of the reference at both splice sites
    df_ctxemb = df[[f"pe_{k}" for k in range(128)]].to_numpy(np.float32)
    assay = df.study_id.map({a: i for i, a in enumerate(ASSAYS)}).fillna(2).astype(int).to_numpy()
    # per-study weights: inverse square-root of study size (normalised)
    n = df.study_id.map(df.study_id.value_counts())
    w = (1 / np.sqrt(n)).to_numpy(np.float32)
    w = w / w[df.study_id.isin(ASSAYS)].mean()
    T = {
        "acc": torch.from_numpy(win["acc"]).to(nm.DEV),
        "don": torch.from_numpy(win["don"]).to(nm.DEV),
        "var": torch.from_numpy(win["var"]).to(nm.DEV),
        "tab": torch.from_numpy(tab.to_numpy(np.float32)).to(nm.DEV),
        "ctx_raw": torch.from_numpy(pg[ctx_cols].to_numpy(np.float32)).to(nm.DEV),
        "assay": torch.from_numpy(assay).to(nm.DEV),
        "y": torch.from_numpy(df.delta_psi.fillna(0).to_numpy(np.float32)).to(nm.DEV),
        "w": torch.from_numpy(w).to(nm.DEV),
        "sdv": torch.from_numpy(df.disruption_label.to_numpy(np.float32)).to(nm.DEV),
    }
    return df, T, ctx_cols, pgsum, df_ctxemb


def standardize_ctx(T, train_idx):
    x = T["ctx_raw"]
    j = torch.as_tensor(train_idx, device=nm.DEV)
    mu, sd = x[j].mean(0), x[j].std(0) + 1e-6
    T["ctx"] = (x - mu) / sd


def gene_split(df, idx, fracs, seed, by="gene_name"):
    """Split rows `idx` into parts with disjoint genes (stratified by study)."""
    rng = np.random.default_rng(seed)
    parts = [[] for _ in fracs]
    for _, g in df.loc[idx].groupby("study_id"):
        genes = rng.permutation(g[by].unique())
        cut = np.cumsum(np.round(np.array(fracs) * len(genes)).astype(int))[:-1]
        for p, gs in zip(parts, np.split(genes, cut)):
            p.extend(g.index[g[by].isin(gs)].tolist())
    # genes shared across studies: assign all their rows to the first part that holds them
    gene_part = {}
    for k, p in enumerate(parts):
        for gname in df.loc[p, by].unique():
            gene_part.setdefault(gname, k)
    allrows = np.concatenate([np.asarray(p, int) for p in parts])
    final = [[] for _ in fracs]
    for r in allrows:
        final[gene_part[df.at[r, by]]].append(r)
    return [np.array(sorted(p), int) for p in final]


def random_split(idx, fracs, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    cut = np.cumsum(np.round(np.array(fracs) * len(idx)).astype(int))[:-1]
    return [np.sort(p) for p in np.split(idx, cut)]


# ----------------------------------------------------------------------------- ensembles
def fit_ensemble(T, tr, va, n_seeds, seed0, n_ctx, n_tab, **arch):
    return [nm.train(T, tr, va, n_ctx, n_tab, seed=seed0 + s, **arch)[0] for s in range(n_seeds)]


def ens_predict(models, T, idx, assay_override=None):
    P = [nm.predict(m, T, idx, assay_override) for m in models]
    mu = np.stack([p["mu"] for p in P])
    sig = np.stack([p["sig"] for p in P])
    return {"mu": mu.mean(0), "ens_std": mu.std(0), "ens_std3": mu[:3].std(0),
            "ale_sig": np.sqrt((sig ** 2).mean(0)), "p": np.stack([p["p"] for p in P]).mean(0),
            "z": np.stack([p["z"] for p in P]).mean(0), "mu_single": mu[0], "sig_single": sig[0]}


def transfer_features(df, idx, pred, ref_ctxemb, ref_z, ctxemb, unknown_assay, std_key="ens_std3"):
    F = pd.DataFrame(index=idx)
    F["ens_std"] = pred[std_key]
    F["ale_sig"] = pred["ale_sig"]
    F["abs_mu"] = np.abs(pred["mu"])
    F["knn_ctx"] = uq.knn_distance(ref_ctxemb, ctxemb[idx])
    F["knn_z"] = uq.knn_distance(ref_z, pred["z"])
    F["abs_dist_ss"] = df.loc[idx, "abs_dist_ss"].to_numpy()
    F["exon_len_l"] = df.loc[idx, "exon_len_l"].to_numpy()
    F["is_intronic"] = df.loc[idx, "is_intronic"].astype(float).to_numpy()
    F["pg_absmax"] = df.loc[idx, "pg_absmax"].to_numpy()
    F["assay_unknown"] = float(unknown_assay)
    return F


def exon_ref_embeddings(ctxemb, df, idx):
    """One embedding per training exon (avoid weighting exons by their variant count)."""
    e = pd.DataFrame(ctxemb[idx]).groupby(df.loc[idx, "exon_id"].to_numpy()).mean()
    return e.to_numpy(np.float32)


def out_of_gene_residuals(df, T, ctxemb, tr, va, split_fn, seed, n_ctx, n_tab, unknown_test_assay, **arch):
    """Inner K-fold over training genes -> residuals and transfer features measured under real
    gene shift. These train the difficulty model a(x)."""
    folds = split_fn(tr, [1 / N_INNER] * N_INNER, seed + 991)
    Fs, R = [], []
    for k in range(N_INNER):
        itr = np.concatenate([folds[j] for j in range(N_INNER) if j != k])
        ite = folds[k]
        models = fit_ensemble(T, itr, va, N_INNER_SEEDS, seed + 100 * (k + 1), n_ctx, n_tab, **arch)
        ov = 2 if unknown_test_assay else None
        p = ens_predict(models, T, ite, ov)
        ptr = ens_predict(models, T, itr[:: max(1, len(itr) // 4000)], ov)
        F = transfer_features(df, ite, p, exon_ref_embeddings(ctxemb, df, itr), ptr["z"], ctxemb,
                              unknown_test_assay)
        Fs.append(F)
        R.append(np.abs(df.loc[ite, "delta_psi"].to_numpy() - p["mu"]))
        del models
        torch.cuda.empty_cache()
    return pd.concat(Fs), np.concatenate(R)


# ----------------------------------------------------------------------------- UQ evaluation
def intervals(method, pc, pt, yc, gc, gt, Fc_a, Ft_a, rc, rt, alpha):
    """Return (lo, hi) on test for a UQ method. pc/pt: calibration/test prediction dicts."""
    A = lambda v: None if v is None else np.asarray(v, dtype=float)
    pc = {k: A(pc[k]) for k in ("mu", "ens_std", "ale_sig")}
    pt = {k: A(pt[k]) for k in ("mu", "ens_std", "ale_sig")}
    yc, Fc_a, Ft_a = A(yc), A(Fc_a), A(Ft_a)
    gc, rc, rt = np.asarray(gc), np.asarray(rc), np.asarray(rt)
    mu_c, mu_t = pc["mu"], pt["mu"]
    res = np.abs(yc - mu_c)
    if method == "ensemble_gaussian":
        s = np.sqrt(pt["ens_std"] ** 2 + pt["ale_sig"] ** 2)
        h = norm.ppf(1 - alpha / 2) * s
    elif method == "split_conformal":
        h = np.full_like(mu_t, uq.split_quantile(res, alpha))
    elif method == "scaled_conformal_ens":
        sc = np.sqrt(pc["ens_std"] ** 2 + pc["ale_sig"] ** 2)
        st = np.sqrt(pt["ens_std"] ** 2 + pt["ale_sig"] ** 2)
        h = uq.split_quantile(res / sc, alpha) * st
    elif method == "mondrian_region":
        q = uq.mondrian_quantiles(res, rc, alpha)
        h = np.array([q.get(r, np.inf) for r in rt])
    elif method == "group_conformal":
        h = np.full_like(mu_t, uq.group_quantile(res, gc, alpha))
    elif method == "TAC_split":
        h = uq.split_quantile(res / Fc_a, alpha) * Ft_a
    elif method == "TAC":
        h = uq.group_quantile(res / Fc_a, gc, alpha) * Ft_a
    else:
        raise ValueError(method)
    lo, hi = np.maximum(mu_t - h, -1), np.minimum(mu_t + h, 1)
    return lo, hi


UQ_METHODS = ["ensemble_gaussian", "split_conformal", "scaled_conformal_ens", "mondrian_region",
              "group_conformal", "TAC_split", "TAC"]


def tabular_matrix(df, pgsum):
    tcols = [c for c in df.columns if c.endswith("_sl") or c in ("exon_len_l", "rel_pos")
             or c.startswith("reg_") or c.startswith("ref_") or c.startswith("alt_")]
    tcols = [c for c in tcols if df[c].dtype != object]
    return df[tcols + pgsum + [f"pe_{k}" for k in range(384)]].to_numpy(np.float32)


def baseline_predictors(df, X, tr, va, te, pgsum):
    y = df.delta_psi.to_numpy()
    out = {"zero": np.zeros(len(te))}
    r = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(df.loc[tr, pgsum], y[tr])
    out["pangolin_ridge"] = r.predict(df.loc[te, pgsum])
    g = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31, min_child_samples=20,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.5, objective="huber",
                          alpha=0.1, verbose=-1, random_state=SEED)
    g.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
    out["lightgbm"] = g.predict(X[te])
    return out, g
