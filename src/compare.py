"""Paired gene-level bootstrap of differences between TAC and baseline UQ methods."""
import numpy as np
import pandas as pd

import analyze as A
import experiment as ex
import uq
from config import RESULTS

ALPHA = 0.1
BASELINES = ["split_conformal", "scaled_conformal_ens", "mondrian_region", "group_conformal",
             "ensemble_gaussian"]
CONF_BASE = {"ensemble_std": lambda d: -d.ens_std, "aleatoric_sigma": lambda d: -d.ale_sig,
             "total_std": lambda d: -np.sqrt(d.ens_std ** 2 + d.ale_sig ** 2)}


def pooled_intervals(p, method):
    out = []
    for fold, f in p.groupby("fold"):
        c, t = f[f.split == "cal"], f[f.split == "test"]
        lo, hi = ex.intervals(method, c, t, c.delta_psi, c.gene_name, t.gene_name, c.a_TAC, t.a_TAC,
                              c.region, t.region, ALPHA)
        out.append(t.assign(lo=np.asarray(lo), hi=np.asarray(hi)))
    t = pd.concat(out)
    y = t.delta_psi.to_numpy()
    t["cov"] = (y >= t.lo) & (y <= t.hi)
    t["iscore"] = (t.hi - t.lo) + 2 / ALPHA * ((t.lo - y) * (y < t.lo) + (y - t.hi) * (y > t.hi))
    return t.reset_index(drop=True)


def gene_stats(t):
    g = t.groupby("gene_name").agg(cov=("cov", "mean"), n=("cov", "size"), isc=("iscore", "sum"))
    big = t[t.delta_psi.abs() > 0.2].groupby("gene_name").agg(bc=("cov", "sum"), bn=("cov", "size"))
    return g.join(big).fillna(0)


N_COMPARISONS = 18   # comparisons per protocol (5 baselines x 3 interval metrics + 3 AURC)


def boot(stat, genes, B=2000, seed=0):
    """Percentile CIs: unadjusted 95% and Bonferroni-adjusted over N_COMPARISONS."""
    rng = np.random.default_rng(seed)
    vals = [stat(rng.choice(genes, len(genes), replace=True)) for _ in range(B)]
    a = 0.05 / N_COMPARISONS
    return np.percentile(vals, [2.5, 97.5, 100 * a / 2, 100 * (1 - a / 2)])


def compare_protocol(prot, fold=None):
    p = A.load(prot)
    if fold is not None:
        p = p[p.fold == fold]
        prot = f"{prot}:{fold}"
    rows = []
    T = {m: gene_stats(pooled_intervals(p, m)) for m in ["TAC"] + BASELINES}
    genes = T["TAC"].index.to_numpy()

    def metrics(G, gs):
        G = G.loc[gs]
        elig = G[G.n >= 10]
        return {"interval_score": G.isc.sum() / G.n.sum(),
                "cov_large_effect": G.bc.sum() / max(1, G.bn.sum()),
                "frac_genes_undercovered": float((elig["cov"] < 1 - ALPHA - 0.1).mean())}

    for m in BASELINES:
        for k in ("interval_score", "cov_large_effect", "frac_genes_undercovered"):
            f = lambda gs: metrics(T["TAC"], gs)[k] - metrics(T[m], gs)[k]
            lo, hi, blo, bhi = boot(f, genes)
            rows.append({"protocol": prot, "comparison": f"TAC - {m}", "metric": k, "diff": f(genes),
                         "ci_lo": lo, "ci_hi": hi, "bonf_lo": blo, "bonf_hi": bhi})
    # selective prediction: AURC differences (pooled test, genes resampled)
    t = p[p.split == "test"].reset_index(drop=True)
    idx = {g: np.where(t.gene_name.to_numpy() == g)[0] for g in t.gene_name.unique()}
    genes_t = np.array(list(idx))

    def aurc(conf, gs):
        j = np.concatenate([idx[g] for g in gs])
        return uq.risk_coverage(t.delta_psi.to_numpy()[j], t.mu.to_numpy()[j], conf[j])[0]["AURC"]

    c_tac = -t.a_TAC.to_numpy()
    for name, fn in CONF_BASE.items():
        cb = fn(t).to_numpy()
        f = lambda gs: aurc(c_tac, gs) - aurc(cb, gs)
        lo, hi, blo, bhi = boot(f, genes_t, B=500)
        rows.append({"protocol": prot, "comparison": f"TAC - {name}", "metric": "AURC", "diff": f(genes_t),
                     "ci_lo": lo, "ci_hi": hi, "bonf_lo": blo, "bonf_hi": bhi})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = []
    for prot in ("gene", "study", "random"):
        if (RESULTS / f"pred_{prot}.parquet").exists():
            res.append(compare_protocol(prot))
    for fold in ("MFASS->VexSeq", "VexSeq->MFASS"):     # per-direction (small calibration sets)
        res.append(compare_protocol("study", fold))
    d = pd.concat(res)
    d.to_csv(RESULTS / "paired_bootstrap.csv", index=False)
    pd.set_option("display.width", 200)
    print(d.round(4).to_string())
