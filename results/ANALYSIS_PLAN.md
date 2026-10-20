# Pre-specified analysis plan (frozen 2026-10-19, before any model was trained on real features)

Data version: `processed/variants_v1.parquet` (provenance in `variants_v1_provenance.json`).
Split seed: 20261019. Splits are generated deterministically from this seed before fitting.

## Primary endpoints
1. **Unseen-gene interval quality** (protocol `gene`, alpha = 0.1): gene-level coverage
   (10th percentile across genes with >=10 test variants, fraction of genes under-covered by >10 points)
   and mean interval width, TAC vs. every baseline UQ method.
2. **Unseen-gene selective prediction** (protocol `gene`): AURC / E-AURC and MAE at 70% acceptance,
   TAC difficulty vs. ensemble SD, aleatoric sigma and total SD, with large-effect retention reported alongside.
3. **External benchmark** (protocol `bench`, evaluated once): per-gene AUROC/AUPRC of the
   pre-specified primary score `p(SDV)` (MFASS-trained disruption head); `-mu` is secondary.

## Secondary
- Point accuracy (MAE, Spearman) across random / gene / study protocols (to quantify the shift penalty).
- Leave-one-study-out intervals and selective prediction (assay token = unknown).
- Ablations: predictor branches (no local / no Pangolin context / no assay) and TAC components
  (no kNN distances, no ensemble/aleatoric features, only ensemble/aleatoric, in-sample residuals,
  ordinary instead of gene-weighted quantile).

## Decision rules (from roadmap section 12)
- If TAC does not beat the best *calibrated* baseline (scaled / Mondrian / gene-weighted conformal) on the
  primary endpoints, the methodological contribution is reported as negative / not established.
- If transferability features add nothing over ensemble variance (TAC vs. TAC_onlystd), report that the
  added complexity is not justified.
- Uncertainty of summary statistics: gene-level bootstrap.

## Amendment 1 (2026-10-19, after the `gene` protocol finished, before `study`/`bench`/`random`/`ablation` were analysed)
Added two interval diagnostics that are listed in the roadmap (section 8: coverage by region; performance on
effectful variants) but had been omitted from the metric code: coverage within each variant region
(and its minimum over regions) and coverage/width for large-effect variants (|dPSI| > 0.2).
These are reported for all protocols and all methods; they are labelled in the paper as roadmap-specified
secondary endpoints that were added after inspecting the primary `gene` results.
Primary-endpoint outcome on `gene`: TAC did not beat the best calibrated baseline on gene-level coverage
(gene-weighted conformal) or on AURC (aleatoric sigma); the decision rule therefore reports the learned
transferability component as not established on unseen genes.

## Amendment 2 (2026-10-19 ~06:00; written after inspecting `gene`, BEFORE any `study`/`bench`/`random`/`ablation` output was inspected)
Observation on `gene` (exploratory): every uncertainty score preferentially discards large-effect variants
(at 70% acceptance ~30% of |dPSI|>0.2 variants are kept vs. 70% under random abstention), i.e. magnitude-based
abstention improves MAE by hiding the variants that matter (roadmap section 8 warning).
New, pre-specified secondary analysis -- **decision-aware triage** (roadmap output: reportable / caution / abstain):
for interval method M at alpha = 0.1 and effect threshold tau = 0.2,
  * "confident large effect"  if the interval lies entirely outside [-tau, tau];
  * "confident small effect"  if the interval lies entirely inside (-tau, tau);
  * "abstain" otherwise.
Metrics: fraction decided; accuracy of decisions; recall of true large effects among all large effects
(confidently called large); precision of "confident large". Benchmark: "confident disruptive" if hi < -tau,
"confident neutral" if interval inside (-tau, tau); report precision / NPV / fraction decided vs. SDV labels.
`gene` results for this analysis are exploratory (hypothesis-generating); `study`, `random` and `bench` are
the confirmatory evaluations.

## Amendment 3 (2026-10-19 ~06:10; after Amendment-2 triage on `gene`, BEFORE inspecting `study`/`bench`/`random`/`ablation`)
Exploratory finding on `gene`: interval-based triage confidently calls ~40-45% of true large effects "small"
for every interval method (marginal validity does not protect the minority class).
New pre-specified analysis -- **class-conditional (Mondrian) conformal triage** (Sadinle et al. 2019):
binary decision Z = 1[|dPSI| > tau], tau = 0.2; score s(x) = P(|Y| > tau) under N(mu, total SD).
Nonconformity: 1 - s for Z=1, s for Z=0; separate calibration quantiles per class at alpha = 0.1,
computed (i) with the ordinary and (ii) with the gene-weighted quantile. Prediction set {large}, {small}
or abstain (both / empty). Target: <= alpha of true large effects confidently called small.
Same metrics as Amendment 2; benchmark: calibrate on MFASS+Vex-seq, evaluate vs. SDV labels.
`gene` = exploratory; `study`, `random`, `bench` = confirmatory.
