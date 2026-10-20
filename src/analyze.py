"""Compute all metrics from saved predictions -> results/*.csv / *.json."""
import json

import numpy as np
import pandas as pd

import experiment as ex
import uq
from config import PROCESSED, RESULTS

DF = pd.read_parquet(PROCESSED / "variants_v1.parquet")
META = DF.set_index("sample_id")[["delta_psi", "gene_name", "region", "study_id", "disruption_label",
                                  "exon_id", "bench_readout", "measurement_uncertainty"]
                                 + [c for c in DF.columns if c.startswith("tool_")]]
CONF = {  # abstention scores: higher = more confident
    "random": None, "ensemble_std": lambda d: -d.ens_std, "aleatoric_sigma": lambda d: -d.ale_sig,
    "total_std": lambda d: -np.sqrt(d.ens_std ** 2 + d.ale_sig ** 2), "TAC": lambda d: -d.a_TAC,
    "TAC_noknn": lambda d: -d.a_TAC_noknn, "TAC_nostd": lambda d: -d.a_TAC_nostd,
    "TAC_onlystd": lambda d: -d.a_TAC_onlystd, "TAC_insample": lambda d: -d.a_TAC_insample,
}
TAC_VARIANTS = ["TAC", "TAC_noknn", "TAC_nostd", "TAC_onlystd", "TAC_insample"]


def load(protocol):
    p = pd.read_parquet(RESULTS / f"pred_{protocol}.parquet")
    return p.join(META, on="sample_id")


def point_table(p):
    rows = []
    t = p[p.split == "test"]
    preds = {"Two-branch ensemble": "mu", "Two-branch single": "mu_single"}
    preds.update({f"baseline: {c[5:]}": c for c in p.columns if c.startswith("base_")})
    preds.update({f"ablation: {c[4:-3]}": c for c in p.columns if c.startswith("abl_") and c.endswith("_mu")})
    for study, g in [("all", t)] + list(t.groupby("study_id")):
        for name, col in preds.items():
            m = uq.reg_metrics(g.delta_psi.to_numpy(), g[col].to_numpy())
            m["dir_acc@0.1"] = uq.direction_acc(g.delta_psi.to_numpy(), g[col].to_numpy())
            lo, hi = uq.gene_bootstrap(g, lambda d: np.mean(np.abs(d.delta_psi - d[col])), n_boot=300)
            rows.append({"study": study, "model": name, **m, "MAE_lo": lo, "MAE_hi": hi, "n": len(g)})
    return pd.DataFrame(rows)


def interval_table(p, extra_tac=True):
    rows, per_gene = [], []
    methods = ex.UQ_METHODS + ([m for m in TAC_VARIANTS if m != "TAC"] if extra_tac else [])
    for alpha in uq.ALPHAS:
        pooled = {m: [] for m in methods}
        for fold, f in p.groupby("fold"):
            c, t = f[f.split == "cal"], f[f.split == "test"]
            for m in methods:
                base = m if m in ex.UQ_METHODS else "TAC"
                acol = "a_" + (m if m.startswith("TAC") and m != "TAC_split" else "TAC")
                lo, hi = ex.intervals(base, c, t, c.delta_psi.to_numpy(), c.gene_name.to_numpy(),
                                      t.gene_name.to_numpy(), c[acol].to_numpy() if acol in c else None,
                                      t[acol].to_numpy() if acol in t else None,
                                      c.region.to_numpy(), t.region.to_numpy(), alpha)
                pooled[m].append(t.assign(lo=lo, hi=hi))
        for m in methods:
            t = pd.concat(pooled[m])
            for study, g in [("all", t)] + list(t.groupby("study_id")):
                r = uq.interval_metrics(g.delta_psi.to_numpy(), g.lo.to_numpy(), g.hi.to_numpy(),
                                        g.gene_name.to_numpy(), alpha, regions=g.region.to_numpy())
                rows.append({"alpha": alpha, "method": m, "study": study, **r})
            if alpha == 0.1:
                gc = t.assign(cov=(t.delta_psi >= t.lo) & (t.delta_psi <= t.hi), w=t.hi - t.lo) \
                      .groupby(["study_id", "gene_name"]).agg(cov=("cov", "mean"), width=("w", "mean"),
                                                              n=("cov", "size")).reset_index()
                per_gene.append(gc.assign(method=m))
    return pd.DataFrame(rows), pd.concat(per_gene)


def selective_table(p):
    rows, curves = [], {}
    t = p[p.split == "test"]
    rng = np.random.default_rng(0)
    for name, fn in CONF.items():
        if fn is not None and name.startswith("TAC") and "a_" + name not in t:
            continue
        for study, g in [("all", t)] + list(t.groupby("study_id")):
            per_fold = []
            for fold, f in g.groupby("fold"):
                conf = rng.random(len(f)) if fn is None else fn(f).to_numpy()
                r, risks, eff = uq.risk_coverage(f.delta_psi.to_numpy(), f.mu.to_numpy(), conf)
                per_fold.append(r)
                curves[(name, study, fold)] = (risks, eff)
            agg = pd.DataFrame(per_fold)
            row = {"score": name, "study": study, **agg.mean().to_dict()}
            row.update({k + "_sd": v for k, v in agg.std().to_dict().items()})
            rows.append(row)
    return pd.DataFrame(rows), curves


def classification_table(p):
    """MFASS SDV classification on gene-held-out test folds."""
    rows = []
    for fold, f in p.groupby("fold"):
        c = f[(f.split == "cal") & f.disruption_label.notna()]
        t = f[(f.split == "test") & f.disruption_label.notna()]
        if len(t) == 0 or t.disruption_label.nunique() < 2:
            continue
        for name, col, sgn in (("Two-branch p(SDV)", "p", 1), ("Two-branch -muPSI", "mu", -1),
                               ("LightGBM -muPSI", "base_lightgbm", -1),
                               ("Pangolin-ridge -muPSI", "base_pangolin_ridge", -1)):
            thr = (uq.best_mcc_threshold(c.disruption_label.to_numpy().astype(int), sgn * c[col].to_numpy())
                   if c.disruption_label.nunique() == 2 else None)   # calibration study may lack SDV labels
            m = uq.cls_metrics(t.disruption_label.to_numpy(), sgn * t[col].to_numpy(), thr)
            if sgn < 0:
                m.pop("Brier", None)
            rows.append({"fold": fold, "model": name, **m})
    d = pd.DataFrame(rows)
    return d.drop(columns="fold").groupby("model").agg(["mean", "std"]) if len(d) else d


TOOLS = ["tool_spliceai", "tool_pangolin", "tool_mmsplice", "tool_hal", "tool_scap", "tool_squirls",
         "tool_spanr", "tool_conspliceml"]


def bench_tables(p):
    """External benchmark: per-gene discrimination vs. published tools, selective AUROC, readout rank corr."""
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    t = p[p.split == "test"].copy()
    t["gene"] = t.study_id.str.replace("Bench_", "")
    t["s_pSDV"] = t.p
    t["s_negmu"] = -t.mu
    scores = ["s_pSDV", "s_negmu"] + TOOLS
    rows = []
    for gene, g in list(t.groupby("gene")) + [("pooled", t)]:
        for sc in scores:
            m = g[sc].notna()
            if m.sum() < 20 or g.loc[m, "disruption_label"].nunique() < 2:
                continue
            r = uq.cls_metrics(g.loc[m, "disruption_label"], g.loc[m, sc].to_numpy())
            r.pop("Brier", None)
            rows.append({"gene": gene, "score": sc, **r})
    disc = pd.DataFrame(rows)
    # selective discrimination: does retaining confident variants raise AUROC?
    rng = np.random.default_rng(0)
    conf = {"random": None, "ensemble_std": "ens_std", "total_std": None, "TAC": "a_TAC"}
    srows = []
    for gene, g in t.groupby("gene"):
        for name, col in conf.items():
            if name == "random":
                c = rng.random(len(g))
            elif name == "total_std":
                c = -np.sqrt(g.ens_std ** 2 + g.ale_sig ** 2).to_numpy()
            else:
                c = -g[col].to_numpy()
            o = np.argsort(-c)
            for cov in (0.5, 0.7, 0.9, 1.0):
                k = int(round(cov * len(g)))
                sub = g.iloc[o[:k]]
                if sub.disruption_label.nunique() < 2:
                    continue
                for sc in ("s_pSDV", "s_negmu"):
                    srows.append({"gene": gene, "conf": name, "coverage": cov, "score": sc,
                                  "AUROC": roc_auc_score(sub.disruption_label, sub[sc]),
                                  "prevalence": sub.disruption_label.mean(), "n": k})
    sel = pd.DataFrame(srows)
    crow = []
    for gene, g in t.groupby("gene"):
        m = g.bench_readout.notna()
        if m.sum() > 20:
            crow.append({"gene": gene, "n": int(m.sum()),
                         "spearman_negmu": spearmanr(g.loc[m, "bench_readout"], -g.loc[m, "mu"]).statistic,
                         "spearman_pSDV": spearmanr(g.loc[m, "bench_readout"], g.loc[m, "p"]).statistic,
                         "spearman_spliceai": spearmanr(g.loc[m, "bench_readout"], g.loc[m, "tool_spliceai"],
                                                        nan_policy="omit").statistic,
                         "spearman_pangolin": spearmanr(g.loc[m, "bench_readout"], g.loc[m, "tool_pangolin"],
                                                        nan_policy="omit").statistic})
    return disc, sel, pd.DataFrame(crow)


TRIAGE_METHODS = ["ensemble_gaussian", "split_conformal", "scaled_conformal_ens", "mondrian_region",
                  "group_conformal", "TAC"]


def triage_intervals(p, method, alpha=0.1):
    out = []
    for fold, f in p.groupby("fold"):
        c, t = f[f.split == "cal"], f[f.split == "test"]
        a_c = c.a_TAC if "a_TAC" in c else None
        a_t = t.a_TAC if "a_TAC" in t else None
        lo, hi = ex.intervals(method, c, t, c.delta_psi, c.gene_name, t.gene_name, a_c, a_t,
                              c.region, t.region, alpha)
        out.append(t.assign(lo=np.asarray(lo), hi=np.asarray(hi)))
    return pd.concat(out)


def triage_table(p, tau=0.2):
    """Decision-aware triage (Amendment 2): call |dPSI|>tau only when the interval excludes the boundary."""
    rows = []
    for m in TRIAGE_METHODS:
        t = triage_intervals(p, m)
        big_call = (t.hi < -tau) | (t.lo > tau)
        small_call = (t.lo > -tau) & (t.hi < tau)
        decided = big_call | small_call
        truth_big = t.delta_psi.abs() > tau
        for study, idx in [("all", np.ones(len(t), bool))] + [(s, (t.study_id == s).to_numpy())
                                                               for s in t.study_id.unique()]:
            d, bc, tb = decided[idx], big_call[idx], truth_big[idx]
            correct = (bc & tb) | (~bc & ~tb)
            rows.append({"method": m, "study": study, "frac_decided": d.mean(),
                         "accuracy_decided": correct[d].mean() if d.any() else np.nan,
                         "recall_large_confident": (bc & tb).sum() / max(1, tb.sum()),
                         "precision_large_confident": _ratio((bc & tb).sum(), bc.sum()),
                         "frac_large_abstained": (~d & tb).sum() / max(1, tb.sum()),
                         "n_large": int(tb.sum())})
    return pd.DataFrame(rows)


def _ratio(a, b):
    return float(a) / float(b) if b > 0 else np.nan


def bench_triage(p, tau=0.2):
    rows = []
    t0 = p[p.split == "test"]
    for m in TRIAGE_METHODS:
        t = triage_intervals(p, m)
        t["gene"] = t.study_id.str.replace("Bench_", "")
        for gene, g in list(t.groupby("gene")) + [("pooled", t)]:
            dis = g.hi < -tau
            neu = (g.lo > -tau) & (g.hi < tau)
            y = g.disruption_label.astype(bool)
            rows.append({"method": m, "gene": gene, "frac_decided": (dis | neu).mean(),
                         "frac_called_disruptive": dis.mean(),
                         "n_called_disruptive": int(dis.sum()), "n_called_neutral": int(neu.sum()),
                         "precision_disruptive": _ratio((dis & y).sum(), dis.sum()),
                         "npv_neutral": _ratio((neu & ~y).sum(), neu.sum()),
                         "recall_sdv_confident": (dis & y).sum() / max(1, y.sum()),
                         "n": len(g), "n_sdv": int(y.sum())})
    return pd.DataFrame(rows)


def p_large(d, tau=0.2):
    from scipy.stats import norm
    sd = np.sqrt(d.ens_std ** 2 + d.ale_sig ** 2).to_numpy()
    mu = d.mu.to_numpy()
    return norm.sf((tau - mu) / sd) + norm.cdf((-tau - mu) / sd)


def mondrian_triage(p, tau=0.2, alpha=0.1, gene_weighted=True):
    """Class-conditional conformal triage (Amendment 3). Returns test frame with call in {large, small, abstain}."""
    out = []
    for fold, f in p.groupby("fold"):
        c, t = f[f.split == "cal"], f[f.split == "test"]
        sc, st = p_large(c, tau), p_large(t, tau)
        zc = (c.delta_psi.abs() > tau).to_numpy()
        q = {}
        for k, nonc in ((1, 1 - sc), (0, sc)):
            m = zc == k
            q[k] = (uq.group_quantile(nonc[m], c.gene_name.to_numpy()[m], alpha) if gene_weighted
                    else uq.split_quantile(nonc[m], alpha))
        inc_large = (1 - st) <= q[1]
        inc_small = st <= q[0]
        call = np.where(inc_large & ~inc_small, "large", np.where(inc_small & ~inc_large, "small", "abstain"))
        out.append(t.assign(call=call, s_large=st))
    return pd.concat(out)


def mondrian_table(p, tau=0.2):
    rows = []
    for gw in (False, True):
        t = mondrian_triage(p, tau, gene_weighted=gw)
        tb = t.delta_psi.abs() > tau
        for study, g in [("all", t)] + list(t.groupby("study_id")):
            b = g.delta_psi.abs() > tau
            dec = g.call != "abstain"
            correct = ((g.call == "large") & b) | ((g.call == "small") & ~b)
            rows.append({"method": "Mondrian-" + ("gene" if gw else "split"), "study": study,
                         "frac_decided": dec.mean(), "accuracy_decided": correct[dec].mean(),
                         "recall_large_confident": ((g.call == "large") & b).sum() / max(1, b.sum()),
                         "precision_large_confident": _ratio(((g.call == "large") & b).sum(), (g.call == "large").sum()),
                         "frac_large_called_small": ((g.call == "small") & b).sum() / max(1, b.sum()),
                         "frac_small_called_large": ((g.call == "large") & ~b).sum() / max(1, (~b).sum()),
                         "frac_large_abstained": ((g.call == "abstain") & b).sum() / max(1, b.sum()),
                         "n_large": int(b.sum())})
    return pd.DataFrame(rows)


def bench_mondrian(p, tau=0.2):
    rows = []
    for gw in (False, True):
        t = mondrian_triage(p, tau, gene_weighted=gw)
        t["gene"] = t.study_id.str.replace("Bench_", "")
        for gene, g in list(t.groupby("gene")) + [("pooled", t)]:
            y = g.disruption_label.astype(bool)
            L, S = g.call == "large", g.call == "small"
            rows.append({"method": "Mondrian-" + ("gene" if gw else "split"), "gene": gene,
                         "frac_decided": (L | S).mean(), "n_called_large": int(L.sum()),
                         "n_called_small": int(S.sum()), "precision_large": _ratio((L & y).sum(), L.sum()),
                         "npv_small": _ratio((S & ~y).sum(), S.sum()),
                         "frac_sdv_called_small": (S & y).sum() / max(1, y.sum()),
                         "recall_sdv_confident": (L & y).sum() / max(1, y.sum()), "n_sdv": int(y.sum())})
    return pd.DataFrame(rows)


def main():
    out = {}
    for prot in ("random", "gene", "study"):
        f = RESULTS / f"pred_{prot}.parquet"
        if not f.exists():
            continue
        p = load(prot)
        point_table(p).to_csv(RESULTS / f"point_{prot}.csv", index=False)
        it, pg = interval_table(p)
        it.to_csv(RESULTS / f"intervals_{prot}.csv", index=False)
        pg.to_csv(RESULTS / f"pergene_cov_{prot}.csv", index=False)
        st, curves = selective_table(p)
        st.to_csv(RESULTS / f"selective_{prot}.csv", index=False)
        pd.to_pickle(curves, RESULTS / f"curves_{prot}.pkl")
        tt = triage_table(p)
        tt["frac_large_called_small"] = 1 - tt.recall_large_confident - tt.frac_large_abstained
        pd.concat([tt, mondrian_table(p)]).to_csv(RESULTS / f"triage_{prot}.csv", index=False)
        ct = classification_table(p)
        if len(ct):
            ct.to_csv(RESULTS / f"cls_{prot}.csv")
        out[prot] = "ok"
    if (RESULTS / "pred_bench.parquet").exists():
        disc, sel, corr = bench_tables(load("bench"))
        disc.to_csv(RESULTS / "bench_discrimination.csv", index=False)
        sel.to_csv(RESULTS / "bench_selective.csv", index=False)
        corr.to_csv(RESULTS / "bench_readout_corr.csv", index=False)
        pb = load("bench")
        pd.concat([bench_triage(pb), bench_mondrian(pb)]).to_csv(RESULTS / "bench_triage.csv", index=False)
        out["bench"] = "ok"
    if (RESULTS / "pred_ablation.parquet").exists():
        point_table(load("ablation")).to_csv(RESULTS / "point_ablation.csv", index=False)
        out["ablation"] = "ok"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
