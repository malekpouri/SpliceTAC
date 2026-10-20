"""LaTeX tables for the manuscript (written to paper/tables/)."""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from config import PROCESSED, PROJECT, RESULTS

OUT = PROJECT / "results/latex_tables"
OUT.mkdir(parents=True, exist_ok=True)
DF = pd.read_parquet(PROCESSED / "variants_v1.parquet")
F3 = lambda v: "--" if v is None or not np.isfinite(v) else f"{v:.3f}"
F2 = lambda v: "--" if v is None or not np.isfinite(v) else f"{v:.2f}"


def write(name, body):
    (OUT / f"{name}.tex").write_text(body)


def table_data():
    rows = []
    for s, g in DF.groupby("study_id"):
        reg = g.region.value_counts(normalize=True)
        rep = ""
        if g.delta_psi_ctx1.notna().any():
            m = g.delta_psi_ctx1.notna() & g.delta_psi_ctx2.notna()
            rep = f"{pearsonr(g.delta_psi_ctx1[m], g.delta_psi_ctx2[m])[0]:.2f} / " \
                  f"{spearmanr(g.delta_psi_ctx1[m], g.delta_psi_ctx2[m])[0]:.2f}"
        rows.append([s.replace("Bench_", "Benchmark: ").replace("VexSeq", "Vex-seq"), f"{len(g):,}",
                     f"{g.exon_id.nunique():,}", f"{g.gene_id.nunique():,}",
                     F2(g.delta_psi.abs().mean()) if g.delta_psi.notna().any() else "--",
                     f"{100 * g.disruption_label.mean():.1f}" if g.disruption_label.notna().any() else "--",
                     f"{100 * reg.get('exon_deep', 0) + 100 * reg.get('exon_near', 0):.0f}", rep or "--"])
    body = ("\\begin{tabular}{lrrrrrrl}\n\\toprule\n"
            "Source & Variants & Exons & Genes & mean $|\\Delta\\Psi|$ & SDV (\\%) & Exonic (\\%) & "
            "Replicate $r$ / $\\rho$ \\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table1_data", body)


def table_point():
    rows = []
    names = {"Two-branch ensemble": "Two-branch ensemble (5 members)", "Two-branch single": "Two-branch, single member",
             "baseline: lightgbm": "LightGBM (same features)", "baseline: pangolin_ridge": "Pangolin + ridge",
             "baseline: zero": "$\\Delta\\Psi=0$"}
    for prot, lab in (("random", "Random"), ("gene", "Unseen gene"), ("study", "Unseen study")):
        f = RESULTS / f"point_{prot}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        for m, nm in names.items():
            r = []
            for s in ("MFASS", "VexSeq"):
                q = d[(d.study == s) & (d.model == m)]
                if len(q) == 0:
                    r += ["--"] * 3
                    continue
                q = q.iloc[0]
                r += [f"{q.MAE:.4f} [{q.MAE_lo:.4f}, {q.MAE_hi:.4f}]", F3(q.Spearman), F3(q.Pearson)]
            rows.append([lab, nm] + r)
    body = ("\\begin{tabular}{llllllll}\n\\toprule\n& & \\multicolumn{3}{c}{MFASS} & "
            "\\multicolumn{3}{c}{Vex-seq}\\\\\n\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\n"
            "Protocol & Model & MAE [95\\% CI] & $\\rho$ & $r$ & MAE [95\\% CI] & $\\rho$ & $r$\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table2_point", body)


def table_intervals(prot="gene", alpha=0.1):
    it = pd.read_csv(RESULTS / f"intervals_{prot}.csv")
    s = it[(it.alpha == alpha) & (it.study == "all")]
    order = ["ensemble_gaussian", "split_conformal", "scaled_conformal_ens", "mondrian_region",
             "group_conformal", "TAC_split", "TAC", "TAC_noknn", "TAC_nostd", "TAC_onlystd", "TAC_insample"]
    lab = {"ensemble_gaussian": "Ensemble Gaussian (uncalibrated)", "split_conformal": "Split conformal",
           "scaled_conformal_ens": "Scaled conformal (ensemble+aleatoric)",
           "mondrian_region": "Mondrian conformal (region)", "group_conformal": "Gene-weighted conformal",
           "TAC_split": "TAC, variant-level quantile", "TAC": "\\textbf{TAC (proposed)}",
           "TAC_noknn": "\\quad $-$ kNN distance features", "TAC_nostd": "\\quad $-$ ensemble/aleatoric features",
           "TAC_onlystd": "\\quad only ensemble/aleatoric features",
           "TAC_insample": "\\quad in-sample instead of out-of-gene residuals"}
    rows = []
    for m in order:
        q = s[s.method == m]
        if len(q) == 0:
            continue
        q = q.iloc[0]
        rows.append([lab[m], F3(q.coverage), F3(q.width_mean), F3(q.interval_score), F3(q.gene_cov_mean),
                     F3(q.gene_cov_p10), F3(q.frac_genes_undercovered)])
    body = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "Method & Coverage & Width & Interval score & Gene cov. (mean) & Gene cov. (P10) & "
            "Genes under-covered\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write(f"table_intervals_{prot}", body)


def table_selective(prot="gene"):
    st = pd.read_csv(RESULTS / f"selective_{prot}.csv")
    s = st[st.study == "all"]
    lab = {"random": "Random", "ensemble_std": "Ensemble SD", "aleatoric_sigma": "Aleatoric $\\sigma$",
           "total_std": "Total SD", "TAC": "\\textbf{TAC $a(x)$}", "TAC_noknn": "\\quad $-$ kNN features",
           "TAC_nostd": "\\quad $-$ ensemble/aleatoric", "TAC_onlystd": "\\quad only ensemble/aleatoric",
           "TAC_insample": "\\quad in-sample residuals"}
    rows = []
    for k, nm in lab.items():
        q = s[s.score == k]
        if len(q) == 0:
            continue
        q = q.iloc[0]
        rows.append([nm, f"{q.AURC:.4f}", f"{q['E-AURC']:.4f}", f"{q['risk@50']:.4f}", f"{q['risk@70']:.4f}",
                     f"{q['risk@90']:.4f}", F2(q["effect_retained@70"])])
    body = ("\\begin{tabular}{lrrrrrr}\n\\toprule\n"
            "Confidence score & AURC & E-AURC & MAE@50\\% & MAE@70\\% & MAE@90\\% & Large effects kept@70\\%\\\\\n"
            "\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write(f"table_selective_{prot}", body)


def table_bench():
    d = pd.read_csv(RESULTS / "bench_discrimination.csv")
    c = pd.read_csv(RESULTS / "bench_readout_corr.csv").set_index("gene")
    lab = {"s_pSDV": "Two-branch $p(\\mathrm{SDV})$ (primary)", "s_negmu": "Two-branch $-\\hat{\\Delta\\Psi}$",
           "tool_spliceai": "SpliceAI", "tool_pangolin": "Pangolin", "tool_mmsplice": "MMSplice",
           "tool_conspliceml": "ConSpliceML", "tool_hal": "HAL", "tool_scap": "S-CAP",
           "tool_squirls": "SQUIRLS", "tool_spanr": "SPANR"}
    genes = ["BRCA1", "FAS", "MLH1", "MST1R", "POU1F1", "WT1"]
    rows = []
    for k, nm in lab.items():
        q = d[d.score == k].set_index("gene")
        vals = [F3(q.AUROC.get(g, np.nan)) for g in genes]
        per = [q.AUROC.get(g, np.nan) for g in genes]
        macro = np.mean(per) if np.all(np.isfinite(per)) else np.nan
        rows.append([nm] + vals + [F3(macro), F3(q.AUPRC.get("pooled", np.nan))])
    body = ("\\begin{tabular}{l" + "r" * 8 + "}\n\\toprule\n"
            "Score & " + " & ".join(genes) + " & Macro AUROC & Pooled AUPRC\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    prev = d[(d.score == "s_pSDV")].set_index("gene")
    body += "\n\\midrule\nSDV prevalence & " + " & ".join(F2(prev.prevalence.get(g, np.nan)) for g in genes) \
            + f" & & {F2(prev.prevalence.get('pooled', np.nan))}\\\\"
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table_bench", body)
    rows = []
    for g in [x for x in genes if x in c.index]:
        r = c.loc[g]
        rows.append([g, str(int(r.n)), F3(r.spearman_negmu), F3(r.spearman_pSDV), F3(r.spearman_spliceai),
                     F3(r.spearman_pangolin)])
    body = ("\\begin{tabular}{lrrrrr}\n\\toprule\nGene & $n$ & Two-branch $-\\hat{\\Delta\\Psi}$ & "
            "Two-branch $p(\\mathrm{SDV})$ & SpliceAI & Pangolin\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table_bench_readout", body)


def table_triage():
    lab = {"split_conformal": "Split-conformal intervals", "scaled_conformal_ens": "Scaled-conformal intervals",
           "group_conformal": "Gene-weighted intervals", "TAC": "TAC intervals",
           "Mondrian-split": "Class-conditional (variant quantile)",
           "Mondrian-gene": "Class-conditional (gene-weighted)"}
    rows = []
    for prot, plab in (("gene", "Unseen gene"), ("random", "Random"), ("study", "Unseen study")):
        f = RESULTS / f"triage_{prot}.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        studies = ["all"] if prot != "study" else ["VexSeq", "MFASS"]
        for st in studies:
            for m, nm in lab.items():
                q = t[(t.method == m) & (t.study == st)]
                if len(q) == 0:
                    continue
                q = q.iloc[0]
                pl = plab if st == "all" else f"{plab} ($\\rightarrow${'Vex-seq' if st == 'VexSeq' else st})"
                rows.append([pl, nm, F2(q.frac_decided), F2(q.recall_large_confident),
                             F2(q.precision_large_confident), F2(q.frac_large_called_small),
                             F2(q.frac_large_abstained)])
    body = ("\\begin{tabular}{llrrrrr}\n\\toprule\nProtocol & Triage rule & Decided & Large: called large & "
            "Precision (large) & Large: called small & Large: abstained\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table_triage", body)


def table_bench_triage():
    t = pd.read_csv(RESULTS / "bench_triage.csv")
    rows = []
    for m, nm in (("TAC", "TAC interval"), ("Mondrian-gene", "Class-conditional (gene-weighted)")):
        for _, q in t[t.method == m].iterrows():
            if m == "TAC":
                rows.append([nm, q.gene, str(int(q.n_sdv)), F2(q.frac_decided), str(int(q.n_called_disruptive)),
                             F2(q.precision_disruptive), F2(q.recall_sdv_confident), str(int(q.n_called_neutral)),
                             F2(q.npv_neutral)])
            else:
                rows.append([nm, q.gene, str(int(q.n_sdv)), F2(q.frac_decided), str(int(q.n_called_large)),
                             F2(q.precision_large), F2(q.recall_sdv_confident), str(int(q.n_called_small)),
                             F2(q.npv_small)])
    body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\nRule & Gene & SDVs & Decided & Called disruptive & "
            "Precision & SDV recall & Called neutral & NPV\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("tableS2_bench_triage", body)


def table_paired():
    d = pd.read_csv(RESULTS / "paired_bootstrap.csv")
    plab = {"gene": "Unseen gene", "study": "Unseen study (pooled)", "random": "Random",
            "study:MFASS->VexSeq": "MFASS$\\rightarrow$Vex-seq", "study:VexSeq->MFASS": "Vex-seq$\\rightarrow$MFASS"}
    mlab = {"interval_score": "Interval score", "cov_large_effect": "Large-effect coverage",
            "frac_genes_undercovered": "Genes under-covered", "AURC": "AURC"}
    clab = {"split_conformal": "split conformal", "scaled_conformal_ens": "scaled conformal",
            "mondrian_region": "Mondrian (region)", "group_conformal": "gene-weighted conformal",
            "ensemble_gaussian": "ensemble Gaussian", "ensemble_std": "ensemble SD",
            "aleatoric_sigma": "aleatoric $\\sigma$", "total_std": "total SD"}
    rows = []
    for _, r in d.iterrows():
        comp = clab[r.comparison.replace("TAC - ", "")]
        rows.append([plab[r.protocol], mlab[r.metric], comp, f"{r['diff']:+.4f}",
                     f"[{r.ci_lo:+.4f}, {r.ci_hi:+.4f}]", f"[{r.bonf_lo:+.4f}, {r.bonf_hi:+.4f}]"])
    head = ("\\begin{tabular}{lllrll}\n\\toprule\nProtocol & Metric & TAC minus & Difference & 95\\% CI & "
            "Bonferroni CI\\\\\n\\midrule\n")
    for name, keep in (("tableS3a_paired", ("Unseen gene", "Random")),
                       ("tableS3b_paired", ("Unseen study (pooled)",)),
                       ("tableS3c_paired", ("MFASS$\\rightarrow$Vex-seq", "Vex-seq$\\rightarrow$MFASS"))):
        sub = [r for r in rows if r[0] in keep]
        write(name, head + "\n".join(" & ".join(r) + " \\\\" for r in sub) + "\n\\bottomrule\n\\end{tabular}\n")


def table_ablation():
    d = pd.read_csv(RESULTS / "point_ablation.csv")
    names = {"Two-branch ensemble": "Full model (5 members)", "Two-branch single": "Full model, single member",
             "ablation: no_local": "$-$ local sequence branch (3 members)",
             "ablation: no_ctx": "$-$ Pangolin context branch (3 members)",
             "ablation: no_assay": "$-$ assay embedding (3 members)",
             "baseline: lightgbm": "LightGBM (no raw sequence)"}
    rows = []
    for m, nm in names.items():
        r = [nm]
        for s in ("all", "MFASS", "VexSeq"):
            q = d[(d.study == s) & (d.model == m)].iloc[0]
            r += [f"{q.MAE:.4f} [{q.MAE_lo:.4f}, {q.MAE_hi:.4f}]", F3(q.Pearson)]
        rows.append(r)
    body = ("\\begin{tabular}{lllllll}\n\\toprule\n& \\multicolumn{2}{c}{All} & \\multicolumn{2}{c}{MFASS} & "
            "\\multicolumn{2}{c}{Vex-seq}\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
            "Model & MAE [95\\% CI] & $r$ & MAE [95\\% CI] & $r$ & MAE [95\\% CI] & $r$\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table_ablation", body)


UQ_LAB = {"ensemble_gaussian": "Ensemble Gaussian (uncalibrated)", "split_conformal": "Split conformal",
          "scaled_conformal_ens": "Scaled conformal", "weighted_scaled_conformal": "Weighted scaled conformal (cov.\\ shift)",
          "mondrian_region": "Mondrian (region)", "mondrian_pred_effect": "Mondrian (predicted effect)",
          "group_conformal": "Gene-weighted conformal", "CQR": "CQR", "CQR_gene_weighted": "CQR, gene-weighted",
          "TAC_split": "TAC, variant quantile", "TAC": "TAC"}


def table_uq_gene(alpha=0.1):
    """Main comparison on unseen genes: original + revision baselines, incl. predicted-effect coverage."""
    it = pd.read_csv(RESULTS / "intervals_gene.csv")
    it = it[(it.alpha == alpha) & (it.study == "all")].set_index("method")
    ex_ = pd.read_csv(RESULTS / "rev_extra_baselines_gene.csv")
    ex_ = ex_[ex_.alpha == alpha].set_index("method")
    pe = pd.read_csv(RESULTS / "rev_predicted_effect_coverage.csv")
    pe = pe[(pe.alpha == alpha) & (pe.protocol == "gene") & (pe.study == "all")].set_index("method")
    rows = []
    for m, nm in UQ_LAB.items():
        if m in ex_.index:
            q = ex_.loc[m]; cpl = q.cov_pred_large
        else:
            q = it.loc[m]; cpl = pe.loc[m, "cov_pred_large"]
        rows.append([nm, F3(q.coverage), F3(q.width_mean), F3(q.interval_score), F3(q.gene_cov_p10),
                     F3(q.frac_genes_undercovered), F3(q.region_cov_min), F3(cpl), F3(q.cov_large_effect)])
    body = ("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n& & & & \\multicolumn{2}{c}{Genes} & Regions & \\multicolumn{2}{c}{Large effects}\\\\\n"
            "\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-9}\n"
            "Method & Coverage & Width & Int.\\ score & P10 cov. & Under-cov. & Worst cov. & Predicted & True\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table_uq_gene", body)


def table_predictor_agnostic(alpha=0.1):
    d = pd.read_csv(RESULTS / "rev_predictor_agnostic.csv")
    d = d[d.alpha == alpha]
    mlab = {"split_conformal": "Split conformal", "group_conformal": "Gene-weighted conformal",
            "mondrian_pred_effect": "Mondrian (predicted effect)", "TAC": "TAC (out-of-gene scale)"}
    plab = {"pangolin_ridge": "Pangolin + ridge", "lightgbm": "LightGBM"}
    rows = []
    for pk, pn in plab.items():
        for m, nm in mlab.items():
            q = d[(d.predictor == pk) & (d.method == m)].iloc[0]
            rows.append([pn if m == "split_conformal" else "", nm, F3(q.coverage), F3(q.interval_score),
                         F3(q.frac_genes_undercovered), F3(q.region_cov_min), F3(q.cov_pred_large), F3(q.cov_large_effect)])
    body = ("\\begin{tabular}{llrrrrrr}\n\\toprule\nPredictor & Method & Coverage & Int.\\ score & Genes under-cov. & "
            "Worst region & Pred.\\ large & True large\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("table_predictor_agnostic", body)


def table_fewshot():
    d = pd.read_csv(RESULTS / "rev_fewshot_study.csv")
    mlab = {"split_conformal": "Split conformal", "scaled_conformal_ens": "Scaled conformal",
            "group_conformal": "Gene-weighted conformal", "TAC": "TAC"}
    rows = []
    for dirn in ("MFASS->VexSeq", "VexSeq->MFASS"):
        for m, nm in mlab.items():
            q = d[(d.direction == dirn) & (d.method == m)].iloc[0]
            rows.append([dirn.replace("->", "$\\rightarrow$").replace("VexSeq", "Vex-seq") if m == "split_conformal" else "",
                         nm, f"{q.coverage_mean:.3f} $\\pm$ {q.coverage_std:.3f}", F3(q.width_mean_mean),
                         F3(q.frac_genes_undercovered_mean), F3(q.region_cov_min_mean), F3(q.cov_large_effect_mean)])
    body = ("\\begin{tabular}{llrrrrr}\n\\toprule\nDirection & Method & Coverage (mean $\\pm$ SD) & Width & "
            "Genes under-cov. & Worst region & True large\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("tableS_fewshot", body)


def table_robustness():
    d = pd.read_csv(RESULTS / "rev_robustness.csv")
    mlab = {"split_conformal": "Split conformal", "scaled_conformal_ens": "Scaled conformal",
            "group_conformal": "Gene-weighted conformal", "TAC": "TAC"}
    rows = []
    for run, g in d.groupby("run", sort=False):
        first = True
        for _, q in g.iterrows():
            if q.method in mlab:
                rows.append([run if first else "", mlab[q.method], F3(q.MAE), F3(q.coverage),
                             F3(q.frac_genes_undercovered), F3(q.region_cov_min), F3(q.cov_pred_large),
                             F3(q.cov_large_effect), ""])
            else:
                rows.append([run if first else "", "Class-conditional triage", F3(q.MAE), "", "", "", "", "",
                             F3(q.frac_large_called_small)])
            first = False
    body = ("\\begin{tabular}{llrrrrrrr}\n\\toprule\nRun & Method & MAE & Coverage & Genes under-cov. & Worst region & "
            "Pred.\\ large & True large & Large called small\\\\\n\\midrule\n")
    body += "\n".join(" & ".join(r) + " \\\\" for r in rows)
    body += "\n\\bottomrule\n\\end{tabular}\n"
    write("tableS_robustness", body)


if __name__ == "__main__":
    table_data()
    for fn in (table_point, table_intervals, table_selective, table_bench, table_triage, table_bench_triage, table_paired, table_ablation,
               table_uq_gene, table_predictor_agnostic, table_fewshot, table_robustness):
        try:
            fn()
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)
    for prot in ("study", "random"):
        try:
            table_intervals(prot); table_selective(prot)
        except FileNotFoundError:
            pass
    print("tables written")
