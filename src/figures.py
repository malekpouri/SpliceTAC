"""Publication figures from results/*.csv (static, print-oriented, light theme)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import uq
from config import FIGS, RESULTS

# reference categorical palette (light); colour follows the method, never its rank
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
METHOD_COL = {"TAC": PAL[0], "split_conformal": PAL[1], "scaled_conformal_ens": PAL[2],
              "mondrian_region": PAL[3], "group_conformal": PAL[4], "TAC_split": PAL[6],
              "ensemble_gaussian": PAL[7]}
METHOD_LAB = {"TAC": "TAC (proposed)", "split_conformal": "Split conformal",
              "scaled_conformal_ens": "Scaled conformal (ens.)", "mondrian_region": "Mondrian (region)",
              "group_conformal": "Gene-weighted conformal", "TAC_split": "TAC, variant quantile",
              "ensemble_gaussian": "Ensemble Gaussian"}
CONF_COL = {"TAC": PAL[0], "ensemble_std": PAL[1], "aleatoric_sigma": PAL[2], "total_std": PAL[3],
            "random": GRAY}
CONF_LAB = {"TAC": "TAC difficulty a(x)", "ensemble_std": "Ensemble SD", "aleatoric_sigma": "Aleatoric σ",
            "total_std": "Total SD", "random": "Random"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "savefig.dpi": 300, "savefig.bbox": "tight", "pdf.fonttype": 42})


def panel(ax, letter):
    """Prefix the panel letter to a left-aligned title (avoids letter/title collisions)."""
    t = ax.get_title()
    ax.set_title("")
    ax.set_title(f"$\\bf{{{letter}}}$   {t}", loc="left")


def save(fig, name):
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)


def fig_shift_penalty():
    rows = []
    for prot, lab in (("random", "Random"), ("gene", "Gene"), ("study", "Study")):
        f = RESULTS / f"point_{prot}.csv"
        if f.exists():
            d = pd.read_csv(f)
            rows.append(d[d.study != "all"].assign(protocol=lab))
    d = pd.concat(rows)
    models = {"Two-branch ensemble": ("Two-branch ensemble", PAL[0]), "baseline: lightgbm": ("LightGBM", PAL[1]),
              "baseline: pangolin_ridge": ("Pangolin ridge", PAL[2]), "baseline: zero": ("ΔΨ = 0", GRAY)}
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4))
    for j, (study, metric) in enumerate([("MFASS", "MAE"), ("VexSeq", "MAE"), ("MFASS", "Pearson"),
                                         ("VexSeq", "Pearson")]):
        ax = axes[j]
        sub = d[d.study == study]
        prots = sub.protocol.unique()
        w = 0.8 / len(models)
        for k, (m, (lab, col)) in enumerate(models.items()):
            if metric != "MAE" and m == "baseline: zero":
                continue
            vals = [sub[(sub.protocol == p) & (sub.model == m)][metric].mean() for p in prots]
            ax.bar(np.arange(len(prots)) + (k - 1.5) * w, vals, w * 0.92, color=col, label=lab)
        ax.set_xticks(range(len(prots)), prots, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_title(f"{'Vex-seq' if study == 'VexSeq' else study}: {metric}")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.28), ncol=4)
    for a, l in zip(axes, "abcd"):
        panel(a, l)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    save(fig, "fig2_shift_penalty")


def fig_intervals(prot="gene"):
    it = pd.read_csv(RESULTS / f"intervals_{prot}.csv")
    pg = pd.read_csv(RESULTS / f"pergene_cov_{prot}.csv")
    methods = [m for m in METHOD_COL if m in it.method.unique()]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.5]})
    ax = axes[0]
    for m in methods:
        s = it[(it.method == m) & (it.study == "all")].sort_values("alpha")
        ax.plot(1 - s.alpha, s.coverage, marker="o", ms=4, color=METHOD_COL[m], lw=1.5)
    ax.plot([0.78, 0.97], [0.78, 0.97], color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Nominal coverage"); ax.set_ylabel("Empirical coverage (variants)")
    ax.set_title("Marginal coverage")
    ax = axes[1]
    s = it[(it.alpha == 0.1) & (it.study == "all")].set_index("method")
    for m in methods:
        ax.scatter(s.loc[m, "width_mean"], s.loc[m, "gene_cov_p10"], s=40, color=METHOD_COL[m],
                   edgecolor="white", linewidth=1.5, zorder=3)
    ax.axhline(0.9, color=MUTED, lw=1, ls="--")
    ax.set_xlabel("Mean interval width"); ax.set_ylabel("10th pct. gene-level coverage")
    ax.set_title("Width vs. worst genes (α = 0.1)")
    ax = axes[2]
    data = [pg[(pg.method == m) & (pg.n >= 10)]["cov"].to_numpy() for m in methods]
    bp = ax.boxplot(data, vert=False, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.2))
    for patch, m in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COL[m]); patch.set_alpha(0.55); patch.set_edgecolor(METHOD_COL[m])
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        ax.scatter(v, i + 1 + rng.uniform(-0.18, 0.18, len(v)), s=3, color=METHOD_COL[methods[i]], alpha=0.5,
                   lw=0)
    ax.axvline(0.9, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(1, len(methods) + 1), [METHOD_LAB[m] for m in methods])
    ax.set_xlabel("Per-gene coverage (genes with ≥10 test variants)")
    ax.set_title("Gene-level coverage (α = 0.1)")
    ax.grid(axis="y", visible=False)
    for a, l in zip(axes, "abc"):
        panel(a, l)
    handles = [plt.Line2D([], [], color=METHOD_COL[m], marker="o", lw=2, ms=5) for m in methods]
    fig.legend(handles, [METHOD_LAB[m] for m in methods], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, f"fig3_intervals_{prot}")


def fig_risk_coverage():
    grid = np.linspace(0.05, 1, 96)
    prots = [p for p in ("gene", "study") if (RESULTS / f"curves_{p}.pkl").exists()]
    fig, axes = plt.subplots(2, len(prots), figsize=(3.6 * len(prots), 4.6), squeeze=False)
    for j, prot in enumerate(prots):
        curves = pd.read_pickle(RESULTS / f"curves_{prot}.pkl")
        for name in CONF_COL:
            ks = [k for k in curves if k[0] == name and k[1] == "all"]
            if not ks:
                continue
            r = np.mean([curves[k][0] for k in ks], 0)
            e = np.mean([curves[k][1] for k in ks], 0)
            axes[0, j].plot(grid, r, color=CONF_COL[name], label=CONF_LAB[name],
                            lw=2.5 if name == "TAC" else 1.6)
            axes[1, j].plot(grid, e, color=CONF_COL[name], lw=2.5 if name == "TAC" else 1.6)
        axes[0, j].set_title({"gene": "Unseen gene", "study": "Unseen study"}[prot])
        axes[0, j].set_ylabel("MAE of accepted variants"); axes[1, j].set_ylabel("Large effects retained")
        axes[1, j].set_xlabel("Coverage (fraction accepted)")
        axes[1, j].plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    axes[0, 0].legend(loc="upper left")
    for a, l in zip(axes.flat, "abcd"):
        panel(a, l)
    fig.tight_layout()
    save(fig, "fig4_risk_coverage")


def fig_benchmark():
    disc = pd.read_csv(RESULTS / "bench_discrimination.csv")
    sel = pd.read_csv(RESULTS / "bench_selective.csv")
    lab = {"s_pSDV": "Two-branch p(SDV)", "s_negmu": "Two-branch −ΔΨ̂", "tool_spliceai": "SpliceAI",
           "tool_pangolin": "Pangolin", "tool_mmsplice": "MMSplice", "tool_hal": "HAL", "tool_scap": "S-CAP",
           "tool_squirls": "SQUIRLS", "tool_spanr": "SPANR", "tool_conspliceml": "ConSpliceML"}
    genes = ["FAS", "MST1R", "POU1F1", "WT1", "BRCA1", "MLH1"]
    piv = disc.pivot(index="score", columns="gene", values="AUROC").reindex(index=list(lab), columns=genes)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axes[0]
    im = ax.imshow(piv.to_numpy(), cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.82 else INK)
    ax.set_xticks(range(len(genes)), genes, rotation=0)
    ax.set_yticks(range(len(lab)), [lab[k] for k in piv.index])
    ax.grid(False)
    ax.set_title("External benchmark: AUROC per gene")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax = axes[1]
    s = sel[sel.score == "s_pSDV"]
    # average only over genes evaluable (both classes present) at every coverage level for that score
    full = s.groupby(["conf", "gene"]).coverage.nunique()
    keep = full[full == s.coverage.nunique()].reset_index()[["conf", "gene"]]
    s = s.merge(keep, on=["conf", "gene"]).groupby(["conf", "coverage"]).AUROC.mean().reset_index()
    for name in ("TAC", "ensemble_std", "total_std", "random"):
        q = s[s.conf == name]
        ax.plot(q.coverage, q.AUROC, marker="o", ms=4, color=CONF_COL[name], label=CONF_LAB[name],
                lw=2.5 if name == "TAC" else 1.6)
    ax.set_xlabel("Coverage (fraction retained)"); ax.set_ylabel("Mean per-gene AUROC")
    ax.set_title("Selective discrimination")
    ax.legend()
    panel(axes[0], "a"); panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "fig5_benchmark")


def fig_difficulty(prot="gene"):
    p = pd.read_parquet(RESULTS / f"pred_{prot}.parquet")
    from analyze import META
    p = p.join(META[["delta_psi"]], on="sample_id")
    t = p[p.split == "test"].copy()
    t["res"] = np.abs(t.delta_psi - t.mu)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    for name, col in (("TAC", "a_TAC"), ("ensemble_std", "ens_std"), ("total_std", None)):
        sc = np.sqrt(t.ens_std ** 2 + t.ale_sig ** 2) if col is None else t[col]
        b = pd.qcut(sc.rank(method="first"), 10, labels=False)
        m = t.groupby(b).res.mean()
        ax.plot(np.arange(1, 11), m.to_numpy(), marker="o", ms=4, color=CONF_COL[name], label=CONF_LAB[name])
    ax.set_xlabel("Decile of uncertainty score"); ax.set_ylabel("Mean |ΔΨ − ΔΨ̂| on unseen genes")
    ax.set_title("Does the score rank errors?")
    ax.legend()
    ax = axes[1]
    import glob
    imps = []
    for f in sorted(glob.glob(str(RESULTS / f"oog_{prot}*.parquet"))):
        o = pd.read_parquet(f)
        dm = uq.fit_difficulty(o, o.abs_res.to_numpy())
        imps.append(pd.Series(dm.booster_.feature_importance("gain"), index=dm.feats_))
    imp = pd.concat(imps, axis=1)
    imp = (imp / imp.sum()).mean(1).sort_values()
    names = {"ens_std": "Ensemble SD", "ale_sig": "Aleatoric σ", "abs_mu": "|ΔΨ̂|",
             "knn_ctx": "kNN dist. (Pangolin exon emb.)", "knn_z": "kNN dist. (network emb.)",
             "abs_dist_ss": "Dist. to splice site", "exon_len_l": "Exon length", "is_intronic": "Intronic",
             "pg_absmax": "max |Pangolin Δ|", "assay_unknown": "Unknown assay"}
    ax.barh([names[k] for k in imp.index], imp.to_numpy(), color=PAL[0], height=0.6)
    ax.set_xlabel("Relative gain importance in a(x)")
    ax.set_title("What predicts out-of-gene error")
    ax.grid(axis="y", visible=False)
    panel(axes[0], "a"); panel(axes[1], "b")
    fig.tight_layout()
    save(fig, "fig6_difficulty")


def fig_triage():
    lab = {"split_conformal": "Split-conformal interval", "scaled_conformal_ens": "Scaled-conformal interval",
           "TAC": "TAC interval", "Mondrian-split": "Class-conditional", "Mondrian-gene": "Class-cond., gene-weighted"}
    panels = [("gene", "all", "Unseen gene"), ("study", "VexSeq", "MFASS → Vex-seq"),
              ("study", "MFASS", "Vex-seq → MFASS")]
    if (RESULTS / "triage_random.csv").exists():
        panels.insert(0, ("random", "all", "Random split"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.6), sharey=True)
    cols = {"called large": PAL[0], "abstained": "#c3c2b7", "called small": PAL[7]}
    for ax, (prot, st, title) in zip(axes, panels):
        t = pd.read_csv(RESULTS / f"triage_{prot}.csv")
        t = t[t.study == st].set_index("method").reindex(list(lab))
        parts = {"called large": t.recall_large_confident, "abstained": t.frac_large_abstained,
                 "called small": t.frac_large_called_small}
        left = np.zeros(len(t))
        y = np.arange(len(t))[::-1]
        for k, v in parts.items():
            ax.barh(y, v.to_numpy(), left=left, color=cols[k], height=0.62, label=k, edgecolor="white", lw=1)
            left += v.to_numpy()
        ax.axvline(0.1, color=INK, lw=0.8, ls=":")
        ax.set_xlim(0, 1)
        ax.set_title(title)
        ax.set_xlabel("Fraction of large effects")
        ax.grid(axis="y", visible=False)
        ax.set_yticks(y, [lab[m] for m in t.index])
    for a, l in zip(axes, "abcd"):
        panel(a, l)
    axes[0].legend(loc="lower center", bbox_to_anchor=(len(panels) / 2, -0.42), ncol=3)
    fig.tight_layout()
    save(fig, "fig7_triage")


if __name__ == "__main__":
    for fn in (fig_shift_penalty, fig_intervals, fig_risk_coverage, fig_benchmark, fig_difficulty, fig_triage):
        try:
            fn()
            print("ok", fn.__name__)
        except Exception as e:  # keep going so partial results still produce figures
            print("FAILED", fn.__name__, repr(e))
