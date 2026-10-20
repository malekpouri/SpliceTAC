# SpliceTAC — transfer-aware calibration of splicing variant effects

Code, harmonised data pipeline, per-variant predictions and analysis outputs for

> M. Malekpouri, S. Lotfi. *Transfer-aware calibration and selective prediction of splicing variant effects
> across genes and assays.* (submitted to *Computers in Biology and Medicine*)

The study asks whether a splicing variant-effect predictor can tell when it is unreliable on genes and assays whose
experimental effects were never used for training. It provides

* a harmonised variant–exon–assay table built from two massively parallel splicing assays (MFASS, Vex-seq;
  29,669 SNVs, 1,677 genes) and a locked six-gene saturation benchmark (Smith & Kitzman 2023; 3,912 variants);
* **SpliceNet**, a two-branch predictor (local CNN + frozen Pangolin features) trained as a 5-member deep ensemble;
* **TAC** (transfer-aware conformal prediction): a difficulty model learned from *out-of-gene* residuals,
  calibrated with a gene-weighted conformal quantile;
* baseline UQ methods (split, scaled, Mondrian, gene-weighted conformal; ensemble-Gaussian), selective-prediction
  analyses and **class-conditional conformal triage** for the decision |ΔΨ| > 0.2;
* gene-held-out, random, unseen-study and external-benchmark evaluation protocols.

## Repository layout

| Path | Content |
|---|---|
| `src/config.py` | paths (`SPLICETAC_DATA`, default `./data`) and constants (split seed 20261019) |
| `src/prep.py` | builds `data/processed/variants_v1.parquet` (+ provenance json) from the raw sources |
| `src/features.py` | acceptor / donor / variant sequence windows and tabular position features |
| `src/pangolin_features.py` | frozen Pangolin features on hg19 context (GPU recommended) |
| `src/nn_model.py` | SpliceNet architecture and training loop |
| `src/experiment.py` | gene-disjoint splits, ensembles, out-of-gene residuals, interval methods |
| `src/uq.py` | split and gene-weighted conformal quantiles, difficulty model, metrics |
| `src/run.py` | runs one protocol: `gene`, `random`, `study`, `bench`, `ablation` |
| `src/analyze.py`, `compare.py`, `tables.py`, `figures.py` | metrics, paired gene bootstrap, LaTeX tables, figures |
| `scripts/download_data.sh` | downloads all public inputs into `data/` |
| `results/` | per-variant predictions (`pred_*.parquet`), out-of-gene residuals (`oog_*.parquet`), all metric CSVs, `ANALYSIS_PLAN.md` (pre-specified plan with time-stamped amendments), data provenance |
| `figures/` | all manuscript figures (PDF + PNG) |

## Quick start

```bash
# 1. environment (Python 3.12; pick the torch wheel matching your CUDA, e.g. cu128 for RTX 50-series)
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 2. public data (~1.2 GB download: hg19, GENCODE v19, MFASS, Vex-seq, benchmark tables, Pangolin weights)
bash scripts/download_data.sh            # writes ./data (or set SPLICETAC_DATA=/path)

# 3. full pipeline (~5 h on an 8 GB GPU); protocols with existing results/pred_*.parquet are skipped
bash run_all.sh

# 4. statistics, tables, figures
cd src && python compare.py && python tables.py && python figures.py
```

All metrics can be recomputed from the shipped predictions without retraining:
`cd src && python analyze.py && python compare.py` (requires steps 1–2 and `python prep.py` for the variant table).

## Reproducibility notes

* Splits are deterministic (seed 20261019) and gene-disjoint; benchmark genes are removed from training data.
* The analysis plan was frozen before training; three amendments (region/large-effect coverage, interval triage,
  class-conditional triage) were added after the unseen-gene protocol and before any other protocol was inspected
  (see `results/ANALYSIS_PLAN.md`).
* GPU nondeterminism causes small run-to-run differences; an independent retraining reproduced the main
  unseen-gene MAE (0.0707).

## Data sources

MFASS (Chong et al. 2019, <https://github.com/KosuriLab/MFASS>), Vex-seq (Adamson et al. 2018,
<https://github.com/scottiadamson/Vex-seq>), saturation benchmark (Smith & Kitzman 2023, Genome Biology,
doi:10.1186/s13059-023-03144-z, Additional files 1–4), UCSC hg19, GENCODE v19, Pangolin (Zeng & Li 2022,
<https://github.com/tkzeng/Pangolin>). These remain under their original licences and are downloaded, not redistributed.

## Contact

Mohammad Malekpouri — m.malekpouri@iau.ir — Department of Computer Engineering, BA.C., Islamic Azad University,
Bandar Abbas, Iran.

## License

Code: MIT (see `LICENSE`). Results files: CC BY 4.0.
