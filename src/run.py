"""Run one evaluation protocol and save per-sample calibration/test predictions.

Protocols
  random      5-fold random variant split (diagnostic only)
  gene        5-fold gene-grouped CV over MFASS+Vex-seq (main)
  study       leave-one-study-out (MFASS->VexSeq, VexSeq->MFASS), assay token = unknown
  bench       train on all MFASS+Vex-seq, test on the locked external saturation benchmark
  ablation    gene CV with architecture ablations (no inner loop)
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

import experiment as ex
import uq
from config import RESULTS, SEED

ABL_FEATS = {
    "TAC": uq.TRANSFER_FEATS,
    "TAC_noknn": [f for f in uq.TRANSFER_FEATS if not f.startswith("knn")],
    "TAC_nostd": [f for f in uq.TRANSFER_FEATS if f not in ("ens_std", "ale_sig")],
    "TAC_onlystd": ["ens_std", "ale_sig"],
}


def run_fold(df, T, ctxemb, pgsum, X, dev, te, split_fn, seed, unknown, tag, inner=True, archs=None):
    t0 = time.time()
    tr, va, ca = split_fn(dev, [0.7, 0.1, 0.2], seed)
    ex.standardize_ctx(T, tr)
    n_ctx, n_tab = T["ctx_raw"].shape[1], T["tab"].shape[1]
    ov = 2 if unknown else None
    models = ex.fit_ensemble(T, tr, va, ex.N_SEEDS, seed, n_ctx, n_tab)
    pc, pt = ex.ens_predict(models, T, ca, ov), ex.ens_predict(models, T, te, ov)
    sub = tr[:: max(1, len(tr) // 4000)]
    ptr = ex.ens_predict(models, T, sub, None)
    ref_e = ex.exon_ref_embeddings(ctxemb, df, tr)
    Fc = ex.transfer_features(df, ca, pc, ref_e, ptr["z"], ctxemb, unknown)
    Ft = ex.transfer_features(df, te, pt, ref_e, ptr["z"], ctxemb, unknown)
    rows = []
    for split, idx, p, F in (("cal", ca, pc, Fc), ("test", te, pt, Ft)):
        d = pd.DataFrame({"sample_id": df.loc[idx, "sample_id"].values, "split": split, "fold": tag})
        for k in ("mu", "ens_std", "ens_std3", "ale_sig", "p", "mu_single", "sig_single"):
            d[k] = p[k]
        for k in uq.TRANSFER_FEATS:
            d["f_" + k] = F[k].to_numpy()
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)
    info = {"n_train": len(tr), "n_val": len(va), "n_cal": len(ca), "n_test": len(te)}

    if inner:
        Fo, Ro = ex.out_of_gene_residuals(df, T, ctxemb, tr, va, split_fn, seed, n_ctx, n_tab, unknown)
        for name, feats in ABL_FEATS.items():
            dm = uq.fit_difficulty(Fo, Ro, feats)
            out["a_" + name] = np.concatenate([uq.predict_difficulty(dm, Fc), uq.predict_difficulty(dm, Ft)])
        Fo.assign(abs_res=Ro).to_parquet(RESULTS / f"oog_{tag.replace('->', '_to_')}.parquet")
        # no-out-of-group ablation: difficulty learned from in-sample training residuals
        Fi = ex.transfer_features(df, sub, ptr, ref_e, ptr["z"], ctxemb, False)
        Fi["knn_ctx"] = uq.knn_distance(ref_e, ctxemb[sub], exclude_self=True)
        Fi["knn_z"] = uq.knn_distance(ptr["z"], ptr["z"], exclude_self=True)
        Ri = np.abs(df.loc[sub, "delta_psi"].to_numpy() - ptr["mu"])
        dm = uq.fit_difficulty(Fi, Ri)
        out["a_TAC_insample"] = np.concatenate([uq.predict_difficulty(dm, Fc), uq.predict_difficulty(dm, Ft)])

    # baseline predictors (on cal and test so they can be conformalised too)
    bp, _ = ex.baseline_predictors(df, X, tr, va, np.concatenate([ca, te]), pgsum)
    for k, v in bp.items():
        out["base_" + k] = v

    for name, arch in (archs or {}).items():
        ms = ex.fit_ensemble(T, tr, va, 3, seed + 7, n_ctx, n_tab, **arch)
        pc_, pt_ = ex.ens_predict(ms, T, ca, ov), ex.ens_predict(ms, T, te, ov)
        out["abl_" + name + "_mu"] = np.concatenate([pc_["mu"], pt_["mu"]])
        out["abl_" + name + "_std"] = np.concatenate([pc_["ens_std"], pt_["ens_std"]])
        del ms
    del models
    torch.cuda.empty_cache()
    info["minutes"] = (time.time() - t0) / 60
    print(tag, json.dumps(info), flush=True)
    return out, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("protocol")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_fold", type=int, default=None)
    a = ap.parse_args()
    df, T, ctx_cols, pgsum, ctxemb = ex.load_all()
    X = ex.tabular_matrix(df, pgsum)
    train_pool = np.where(df.study_id.isin(["MFASS", "VexSeq"]) & df.delta_psi.notna())[0]
    gsplit = lambda idx, fr, s: ex.gene_split(df, idx, fr, s)
    rsplit = lambda idx, fr, s: ex.random_split(idx, fr, s)
    results, infos = [], {}

    if a.protocol in ("gene", "random", "ablation"):
        split_fn = rsplit if a.protocol == "random" else gsplit
        folds = split_fn(train_pool, [1 / a.folds] * a.folds, SEED)
        archs = None
        if a.protocol == "ablation":
            archs = {"no_local": {"use_local": False}, "no_ctx": {"use_ctx": False},
                     "no_assay": {"use_assay": False}}
        for k in range(a.folds):
            if a.only_fold is not None and k != a.only_fold:
                continue
            te = folds[k]
            dev = np.concatenate([folds[j] for j in range(a.folds) if j != k])
            o, info = run_fold(df, T, ctxemb, pgsum, X, dev, te, split_fn, SEED + k, False,
                               f"{a.protocol}{k}", inner=(a.protocol != "ablation"), archs=archs)
            results.append(o); infos[k] = info
    elif a.protocol == "study":
        for src, tgt in (("MFASS", "VexSeq"), ("VexSeq", "MFASS")):
            dev = train_pool[df.loc[train_pool, "study_id"].to_numpy() == src]
            te = train_pool[df.loc[train_pool, "study_id"].to_numpy() == tgt]
            # remove target-study genes from the development set (gene AND study held out)
            tg = set(df.loc[te, "gene_name"])
            dev = dev[~df.loc[dev, "gene_name"].isin(tg).to_numpy()]
            o, info = run_fold(df, T, ctxemb, pgsum, X, dev, te, gsplit, SEED, True, f"{src}->{tgt}")
            results.append(o); infos[f"{src}->{tgt}"] = info
    elif a.protocol == "bench":
        te = np.where(df.study_id.str.startswith("Bench_"))[0]
        o, info = run_fold(df, T, ctxemb, pgsum, X, train_pool, te, gsplit, SEED, True, "bench")
        results.append(o); infos["bench"] = info
    suffix = "" if a.only_fold is None else f"_f{a.only_fold}"
    pd.concat(results, ignore_index=True).to_parquet(RESULTS / f"pred_{a.protocol}{suffix}.parquet")
    (RESULTS / f"info_{a.protocol}{suffix}.json").write_text(json.dumps(infos, indent=1))


if __name__ == "__main__":
    main()
