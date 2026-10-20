"""Graphical abstract (Elsevier: >= 1328 x 531 px, 2.5:1, readable at 13 x 5 cm). Numbers read from results/."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import pandas as pd

from config import PROJECT, RESULTS

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "red": "#e34948", "gray": "#8a8985",
       "ink": "#0b0b0b", "muted": "#52514e", "grid": "#e6e5e1", "light": "#eef4fc"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": PAL["muted"], "xtick.color": PAL["muted"],
                     "ytick.color": PAL["muted"]})

# --- numbers from the analysis outputs
pe = pd.read_csv(RESULTS / "rev_extra_baselines_gene.csv")
pe = pe[pe.alpha == 0.1].set_index("method")
pc = pd.read_csv(RESULTS / "rev_predicted_effect_coverage.csv")
pc = pc[(pc.alpha == 0.1) & (pc.protocol == "gene") & (pc.study == "all")].set_index("method")
cov = {"Split\nconformal": pc.loc["split_conformal", "cov_pred_large"], "CQR": pe.loc["CQR", "cov_pred_large"],
       "Scaled\nconformal": pc.loc["scaled_conformal_ens", "cov_pred_large"], "TAC": pe.loc["TAC", "cov_pred_large"]}
tri = pd.read_csv(RESULTS / "triage_gene.csv"); tri = tri[tri.study == "all"].set_index("method")
missed = {"Interval\ntriage (TAC)": 1 - tri.loc["TAC", "recall_large_confident"] - tri.loc["TAC", "frac_large_abstained"],
          "Class-\nconditional": tri.loc["Mondrian-gene", "frac_large_called_small"]}

fig = plt.figure(figsize=(13.28, 5.31), dpi=150)
# panel 1: schematic
ax0 = fig.add_axes([0.01, 0.05, 0.31, 0.88]); ax0.axis("off"); ax0.set_xlim(0, 10); ax0.set_ylim(0, 10)
ax0.text(0.2, 9.5, "Splicing predictor meets\nunseen genes & assays", fontsize=15, weight="bold", color=PAL["ink"], va="top")
def box(x, y, w, h, txt, fc, ec):
    ax0.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.3", fc=fc, ec=ec, lw=1.5))
    ax0.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=12, color=PAL["ink"])
box(0.3, 4.6, 3.6, 2.3, "MPSA training\ngenes\n(29,669 SNVs)", "#f3f3f1", PAL["muted"])
box(5.9, 4.6, 3.8, 2.3, "Unseen gene /\nassay + locked\nbenchmark", PAL["light"], PAL["blue"])
ax0.add_patch(FancyArrowPatch((4.1, 5.75), (5.75, 5.75), arrowstyle="-|>", mutation_scale=18, lw=1.8, color=PAL["muted"]))
ax0.text(0.3, 3.4, "11 uncertainty methods\n3 point predictors\ngene-held-out evaluation", fontsize=12, color=PAL["muted"], va="top")

# panel 2: coverage where a large effect is predicted
ax1 = fig.add_axes([0.40, 0.20, 0.27, 0.62])
names = list(cov); vals = [cov[k] for k in names]
cols = [PAL["gray"], PAL["orange"], PAL["aqua"], PAL["blue"]]
ax1.bar(range(len(vals)), vals, color=cols, width=0.62)
for i, v in enumerate(vals):
    ax1.text(i, v + 0.03, f"{v:.0%}", ha="center", fontsize=13, weight="bold", color=PAL["ink"])
ax1.axhline(0.9, ls="--", lw=1.2, color=PAL["muted"])
ax1.text(0, 1.0, "dashed line: 90% target", transform=ax1.transAxes, fontsize=10.5, color=PAL["muted"], va="bottom")
ax1.set_ylim(0, 1.12); ax1.set_yticks([0, 0.5, 1]); ax1.set_yticklabels(["0", "50%", "100%"])
ax1.set_xticks(range(len(vals))); ax1.set_xticklabels(names, fontsize=11)
ax1.set_title("Coverage where the model\npredicts a large effect", fontsize=14, weight="bold", color=PAL["ink"], loc="left", pad=22)

# panel 3: confidently missed large effects
ax2 = fig.add_axes([0.76, 0.20, 0.22, 0.62])
names2 = list(missed); vals2 = [missed[k] for k in names2]
ax2.bar(range(2), vals2, color=[PAL["red"], PAL["blue"]], width=0.55)
for i, v in enumerate(vals2):
    ax2.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=13, weight="bold", color=PAL["ink"])
ax2.axhline(0.1, ls="--", lw=1.2, color=PAL["muted"])
ax2.text(0, 1.0, "dashed line: nominal 10%", transform=ax2.transAxes, fontsize=10.5, color=PAL["muted"], va="bottom")
ax2.set_ylim(0, 0.6); ax2.set_yticks([0, 0.25, 0.5]); ax2.set_yticklabels(["0", "25%", "50%"])
ax2.set_xticks(range(2)); ax2.set_xticklabels(names2, fontsize=11)
ax2.set_title("Large effects confidently\ncalled small", fontsize=14, weight="bold", color=PAL["ink"], loc="left", pad=22)

out = PROJECT / "figures"
fig.savefig(out / "Graphical_Abstract.png", dpi=150)
fig.savefig(out / "Graphical_Abstract.pdf")
print("saved", out / "Graphical_Abstract.png")
