#!/usr/bin/env bash
# Download all public inputs used by SpliceTAC into $SPLICETAC_DATA (default: <repo>/data).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
D="${SPLICETAC_DATA:-$ROOT/data}"
mkdir -p "$D"/{vexseq,mfass,benchmark2023,reference,models,processed}

# Vex-seq processed data
[ -d "$D/vexseq/repo" ] || git clone --depth 1 https://github.com/scottiadamson/Vex-seq "$D/vexseq/repo"

# MFASS processed SNV tables (only the files used; the full repository is ~1.5 GB)
for p in processed_data/snv/snv_data_clean.txt processed_data/snv/sdv_simple_list.txt \
         processed_data/snv/snp_positions_hg19.bed; do
  mkdir -p "$D/mfass/$(dirname "$p")"
  [ -f "$D/mfass/$p" ] || curl -sSL -o "$D/mfass/$p" "https://raw.githubusercontent.com/KosuriLab/MFASS/master/$p"
done

# Saturation benchmark supplementary tables (Smith & Kitzman 2023, Genome Biology)
for f in MOESM1_ESM.pdf MOESM2_ESM.xlsx MOESM3_ESM.xlsx MOESM4_ESM.xlsx; do
  out="$D/benchmark2023/13059_2023_3144_$f"
  [ -f "$out" ] || curl -sSL -A "Mozilla/5.0" -o "$out" \
    "https://static-content.springer.com/esm/art%3A10.1186%2Fs13059-023-03144-z/MediaObjects/13059_2023_3144_$f"
done

# Reference genome and annotation (hg19 / GENCODE v19)
if [ ! -f "$D/reference/hg19.fa" ]; then
  curl -sSL -C - -o "$D/reference/hg19.fa.gz" https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
  gunzip -f "$D/reference/hg19.fa.gz"
fi
[ -f "$D/reference/gencode.v19.annotation.gtf.gz" ] || curl -sSL -o "$D/reference/gencode.v19.annotation.gtf.gz" \
  https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.annotation.gtf.gz

# Pangolin pretrained weights
[ -d "$D/models/Pangolin" ] || git clone --depth 1 https://github.com/tkzeng/Pangolin "$D/models/Pangolin"

echo "Data ready in $D"
