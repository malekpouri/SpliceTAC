"""Central paths and constants. Set SPLICETAC_DATA to relocate the datasets folder."""
import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SPLICETAC_DATA", PROJECT / "data"))
RAW_VEXSEQ = DATA / "vexseq/repo"
RAW_MFASS = DATA / "mfass/processed_data/snv"
RAW_BENCH = DATA / "benchmark2023/13059_2023_3144_MOESM2_ESM.xlsx"
HG19 = DATA / "reference/hg19.fa"
GTF19 = DATA / "reference/gencode.v19.annotation.gtf.gz"
PANGOLIN_DIR = DATA / "models/Pangolin/pangolin/models"

PROCESSED = DATA / "processed"           # versioned standardized tables
RESULTS = PROJECT / "results"
FIGS = PROJECT / "figures"
for p in (PROCESSED, RESULTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)

# genes of the external saturation benchmark: never used for supervised training
BENCH_GENES = ["FAS", "MST1R", "POU1F1", "WT1", "BRCA1", "MLH1"]

SEED = 20261019
