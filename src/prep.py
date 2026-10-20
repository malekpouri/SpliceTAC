"""Build the standardized variant-exon-assay table from Vex-seq, MFASS and the
2023 saturation benchmark (all hg19).

Output: PROCESSED/variants_v1.parquet (+ a provenance json with counts per filter).
Missing information is stored as NaN, never imputed.
"""
import gzip
import json
import re

import numpy as np
import pandas as pd
import pyfaidx

from config import (BENCH_GENES, GTF19, HG19, PROCESSED, RAW_BENCH, RAW_MFASS,
                    RAW_VEXSEQ)

COMP = str.maketrans("ACGTN", "TGCAN")
LOG = {}


def load_gtf_exons():
    cache = PROCESSED / "gencode_v19_exons.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    attr = re.compile(r'(\w+) "?([^";]+)"?;')
    with gzip.open(GTF19, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if f[2] != "exon":
                continue
            a = dict(attr.findall(f[8]))
            rows.append((f[0], int(f[3]), int(f[4]), f[6], a["gene_id"].split(".")[0],
                         a["gene_name"], a["gene_type"], a["transcript_id"].split(".")[0],
                         int(a["exon_number"]), a["exon_id"].split(".")[0]))
    ex = pd.DataFrame(rows, columns=["chrom", "start", "end", "strand", "gene_id", "gene_name",
                                     "gene_type", "transcript_id", "exon_number", "exon_id"])
    ex.to_parquet(cache)
    return ex


def region_annotation(df):
    """Signed distances (transcript orientation) from variant to acceptor (3'SS) and donor (5'SS).

    dist_acc = variant - first exon base (negative: upstream intron)
    dist_don = variant - last exon base (positive: downstream intron)
    """
    plus = df.strand == "+"
    first = np.where(plus, df.exon_start, df.exon_end)
    last = np.where(plus, df.exon_end, df.exon_start)
    sgn = np.where(plus, 1, -1)
    df["dist_acc"] = (df.position - first) * sgn
    df["dist_don"] = (df.position - last) * sgn
    df["exon_len"] = df.exon_end - df.exon_start + 1
    intronic = (df.dist_acc < 0) | (df.dist_don > 0)
    d_int = np.where(df.dist_acc < 0, -df.dist_acc, df.dist_don)          # 1-based intron offset
    d_ex = np.minimum(df.dist_acc, -df.dist_don)                           # 0 = first/last base
    reg = np.where(intronic,
                   np.where(d_int <= 2, "canonical_ss",
                            np.where(d_int <= 10, "intron_near", "intron_deep")),
                   np.where(d_ex <= 2, "exon_near", "exon_deep"))
    df["region"] = reg
    df["is_intronic"] = intronic
    return df


def check_ref(df, fa):
    ok = []
    for c, p, r in zip(df.chromosome, df.position, df.ref):
        ok.append(str(fa[c][p - 1:p]).upper() == r)
    return np.array(ok)


def prep_vexseq(ex):
    seqs = pd.read_csv(RAW_VEXSEQ / "preprocess_data/68_97_final_sequences_fixed.tsv", sep="\t")
    seqs = seqs[seqs.reference != "Control"].drop_duplicates(
        ["chromosome", "position", "reference", "alternative"])
    seqs["variant"] = (seqs.chromosome + "_" + seqs.position.astype(str) + "_" +
                       seqs.reference + "_" + seqs.alternative)
    d = pd.read_csv(RAW_VEXSEQ / "processed_files/delta_PSI_values.tsv", sep="\t")
    rc = pd.read_csv(RAW_VEXSEQ / "processed_files/read_counts_PSI_values.tsv", sep="\t")
    LOG["vexseq_raw_variants_with_dpsi"] = len(d)
    df = d.merge(seqs, on="variant", how="inner")
    # replicate-level delta-PSI vs. the matched reference ("Original") construct
    rc["exon_key"] = rc.variant.str.split("_").str[:2].str.join("_")
    reps = ["H1", "H2", "H3", "K1", "K2", "K3"]
    refpsi = rc[rc["class"] == "Reference"].set_index("exon_key")[reps + ["mean_PSI"]]
    rc = rc[~rc.variant.str.contains("Control")].set_index("variant")
    # reference constructs are keyed by an arbitrary position inside the construct;
    # match them to variants by construct coordinates
    keys = refpsi.index.to_series().str.split("_", expand=True)
    keys.columns = ["chrom", "pos"]
    keys["pos"] = keys.pos.astype(int)
    df["psi_ref"] = np.nan
    df["meas_sd"] = np.nan
    df["psi_alt"] = df.mean_PSI / 100
    for i, r in df.iterrows():
        k = keys[(keys.chrom == r.chromosome) & (keys.pos >= r["Upstream Int Adj"]) &
                 (keys.pos <= r["Down Int Adj"])]
        if len(k) != 1 or r.variant not in rc.index:
            continue
        rp = refpsi.loc[k.index[0]]
        vp = rc.loc[r.variant]
        reps_d = np.array([vp[x] - rp[x] for x in reps], dtype=float) / 100
        df.at[i, "psi_ref"] = rp["mean_PSI"] / 100
        df.at[i, "meas_sd"] = np.nanstd(reps_d, ddof=1) / np.sqrt(np.isfinite(reps_d).sum())
    out = pd.DataFrame({
        "study_id": "VexSeq", "assay_type": "minigene_RNAseq_PSI",
        "cell_context": "HepG2+K562", "genome_build": "hg19",
        "chromosome": df.chromosome, "position": df.position,
        "ref": df.reference, "alt": df.alternative, "strand": df.strand,
        "exon_start": df["Exon up"], "exon_end": df["Exon down"],
        "construct_start": df["Upstream Int Adj"], "construct_end": df["Down Int Adj"],
        "psi_ref": df.psi_ref, "psi_alt": df.psi_alt,
        "delta_psi": df.delta_PSI / 100,
        "delta_psi_ctx1": df.delta_psi_H / 100, "delta_psi_ctx2": df.delta_psi_K / 100,
        "measurement_uncertainty": df.meas_sd, "replicate_count": 6,
    })
    out["disruption_label"] = np.nan   # Vex-seq has no published SDV definition
    LOG["vexseq_merged"] = len(out)
    snv = (out.ref.str.len() == 1) & (out.alt.str.len() == 1)
    LOG["vexseq_indels_removed"] = int((~snv).sum())
    out = out[snv]
    return out


def prep_mfass(ex):
    m = pd.read_csv(RAW_MFASS / "snv_data_clean.txt", sep="\t", low_memory=False)
    LOG["mfass_rows"] = len(m)
    m = m[(m.category == "mutant") & m.v2_dpsi.notna() & m.ref_allele.notna()].copy()
    m = m[(m.ref_allele.str.len() == 1) & (m.alt_allele.str.len() == 1)]
    LOG["mfass_snv_with_v2_dpsi"] = len(m)
    # intron1 is the genomic-left intron on both strands; verified against GENCODE exon ids
    # (exact match for all 925 exons whose ENSE id exists in v19)
    cstart = m.start
    ex_s = m.start + m.intron1_len
    ex_e = m.end - m.intron2_len
    reps = m[["v2_dpsi_R1", "v2_dpsi_R2"]].to_numpy(float)
    out = pd.DataFrame({
        "study_id": "MFASS", "assay_type": "minigene_flowsort_index",
        "cell_context": "HEK293T", "genome_build": "hg19",
        "chromosome": m.chr, "position": m.snp_position.astype(int),
        "ref": m.ref_allele, "alt": m.alt_allele, "strand": m.strand,
        "exon_start": ex_s, "exon_end": ex_e,
        "construct_start": cstart, "construct_end": m.end,
        "psi_ref": m.nat_v2_index, "psi_alt": m.v2_index,
        "delta_psi": m.v2_dpsi,
        "delta_psi_ctx1": reps[:, 0], "delta_psi_ctx2": reps[:, 1],
        "measurement_uncertainty": np.abs(reps[:, 0] - reps[:, 1]) / 2,
        "replicate_count": 2,
        "disruption_label": m.strong_lof.map({True: 1.0, False: 0.0, "TRUE": 1.0, "FALSE": 0.0}),
        "mfass_exon_id": m.ensembl_id,
    })
    return out


BENCH_SHEETS = {
    # sheet: (gene, column holding a continuous readout, sign so that larger = more skipping)
    "FAS exon 6": ("FAS", "Enrichment score (measured)", -1.0),
    "RON exon 11": ("MST1R", "SKIP (measured)", 1.0),
    "POU1F1 exon 2": ("POU1F1", "SKIP (measured)", 1.0),
    "WT1 exon 9": ("WT1", "SKIP (measured)", 1.0),
    "BRCA1": ("BRCA1", "RNA (measured)", -1.0),
    "MLH1": ("MLH1", None, 1.0),
}
TOOL_COLS = {
    "SpliceAI delta max": "tool_spliceai", "Pangolin delta max (abs)": "tool_pangolin",
    "MMSplice logit PSI change (abs)": "tool_mmsplice", "HAL PSI change (abs)": "tool_hal",
    "S-Cap sens minimum (rev)": "tool_scap", "SQUIRLS score": "tool_squirls",
    "SPANR zPSI change (abs)": "tool_spanr", "ConSpliceML": "tool_conspliceml",
}


def prep_bench(ex):
    xl = pd.read_excel(RAW_BENCH, sheet_name=list(BENCH_SHEETS))
    outs = []
    for sheet, (gene, readout, sign) in BENCH_SHEETS.items():
        b = xl[sheet].copy()
        b = b.loc[:, ~b.columns.duplicated()]
        b.columns = [c.strip() for c in b.columns]
        tcol = [c for c in b.columns if c.startswith("Transcript ID")][0]
        b["tx"] = b[tcol].astype(str).str.split(".").str[0]
        b["Exon"] = b["Exon"].astype(int)
        # Exon numbering/isoforms in the benchmark transcripts do not always match GENCODE v19, so for
        # every (gene, exon) block we pick the v19 exon of the gene that best reproduces the benchmark's
        # own exonic/intronic variant classes.
        ge = ex[ex.gene_name == gene].drop_duplicates(["start", "end"])
        b[["start", "end", "strand", "chrom"]] = np.nan
        b = b.astype({"strand": object, "chrom": object})
        for exn, blk in b.groupby("Exon"):
            pos = blk["HG19 pos"].to_numpy()
            is_ex = blk["Variant class"].astype(str).str.contains("Exon").to_numpy()
            cand = ge[(ge.end >= pos.min() - 5000) & (ge.start <= pos.max() + 5000)]
            best, best_acc = None, -1
            for r in cand.itertuples():
                acc = np.mean(((pos >= r.start) & (pos <= r.end)) == is_ex)
                if acc > best_acc or (acc == best_acc and r.end - r.start < best.end - best.start):
                    best, best_acc = r, acc
            b.loc[blk.index, ["start", "end", "strand", "chrom"]] = [best.start, best.end, best.strand, best.chrom]
            LOG[f"bench_{gene}_exon{exn}"] = f"{best.chrom}:{best.start}-{best.end} class-agreement={best_acc:.3f}"
        LOG[f"bench_{gene}_rows"] = len(b)
        LOG[f"bench_{gene}_no_exon_match"] = int(b.start.isna().sum())
        lab = b["Splice disruptive"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0})
        o = pd.DataFrame({
            "study_id": f"Bench_{gene}", "assay_type": "benchmark_" + gene,
            "cell_context": np.nan, "genome_build": "hg19",
            "chromosome": b.chrom, "position": b["HG19 pos"].astype(int),
            "ref": b.ref, "alt": b.alt, "strand": b.strand,
            "exon_start": b.start, "exon_end": b.end,
            "construct_start": np.nan, "construct_end": np.nan,
            "psi_ref": np.nan, "psi_alt": np.nan, "delta_psi": np.nan,
            "delta_psi_ctx1": np.nan, "delta_psi_ctx2": np.nan,
            "measurement_uncertainty": np.nan, "replicate_count": np.nan,
            "disruption_label": lab,
            "bench_readout": sign * pd.to_numeric(b[readout], errors="coerce") if readout else np.nan,
            "bench_variant_class": b["Variant class"],
        })
        for src, dst in TOOL_COLS.items():
            o[dst] = pd.to_numeric(b[src], errors="coerce") if src in b else np.nan
        outs.append(o)
    out = pd.concat(outs, ignore_index=True)
    out = out[out.exon_start.notna() & out.disruption_label.notna()]
    return out


def attach_gene(df, ex):
    """Map each exon to a GENCODE v19 gene by exact coordinates, else by overlap (protein coding first)."""
    exs = ex.drop_duplicates(["chrom", "start", "end", "gene_id"])
    key = df[["chromosome", "exon_start", "exon_end"]].drop_duplicates()
    rows = []
    for c, s, e in key.itertuples(index=False):
        sub = exs[(exs.chrom == c) & (exs.end >= s) & (exs.start <= e)]
        exact = sub[(sub.start == s) & (sub.end == e)]
        pick = exact if len(exact) else sub
        if len(pick) == 0:
            rows.append((c, s, e, np.nan, np.nan, False))
            continue
        pick = pick.assign(pc=(pick.gene_type == "protein_coding")).sort_values("pc", ascending=False)
        rows.append((c, s, e, pick.gene_id.iloc[0], pick.gene_name.iloc[0], len(exact) > 0))
    m = pd.DataFrame(rows, columns=["chromosome", "exon_start", "exon_end", "gene_id", "gene_name",
                                    "exon_exact_match"])
    return df.merge(m, on=["chromosome", "exon_start", "exon_end"], how="left")


def main():
    ex = load_gtf_exons()
    fa = pyfaidx.Fasta(str(HG19))
    parts = [prep_vexseq(ex), prep_mfass(ex), prep_bench(ex)]
    df = pd.concat(parts, ignore_index=True)
    df["exon_start"] = df.exon_start.astype(int)
    df["exon_end"] = df.exon_end.astype(int)
    df = attach_gene(df, ex)
    for s in df.study_id.unique():
        LOG[f"{s}_no_gene"] = int(df[df.study_id == s].gene_id.isna().sum())
    df["ref_ok"] = check_ref(df, fa)
    for s in df.study_id.unique():
        LOG[f"{s}_ref_mismatch"] = int((~df[df.study_id == s].ref_ok).sum())
    df = df[df.ref_ok & df.gene_id.notna()].drop(columns="ref_ok")
    df = region_annotation(df)
    # benchmark gene names come from the benchmark itself (the exon may overlap other loci)
    is_b = df.study_id.str.startswith("Bench_")
    df.loc[is_b, "gene_name"] = df.loc[is_b, "study_id"].str.replace("Bench_", "")
    # leakage control: remove training-study rows on benchmark genes
    leak = (~is_b) & df.gene_name.isin(BENCH_GENES)
    LOG["train_rows_removed_on_benchmark_genes"] = int(leak.sum())
    df = df[~leak]
    df["exon_id"] = df.chromosome + ":" + df.exon_start.astype(str) + "-" + df.exon_end.astype(str)
    df["variant_id"] = df.chromosome + ":" + df.position.astype(str) + ":" + df.ref + ">" + df.alt
    df["sample_id"] = df.variant_id + "|" + df.exon_id + "|" + df.study_id
    LOG["duplicate_sample_ids"] = int(df.sample_id.duplicated().sum())
    df = df.drop_duplicates("sample_id").reset_index(drop=True)
    # variants measured by more than one study (kept, flagged)
    multi = df.groupby("variant_id").study_id.nunique()
    df["n_studies_variant"] = df.variant_id.map(multi)
    for s, g in df.groupby("study_id"):
        LOG[f"final_{s}"] = {"n": len(g), "exons": g.exon_id.nunique(), "genes": g.gene_id.nunique(),
                             "sdv_rate": float(g.disruption_label.mean()) if g.disruption_label.notna().any() else None}
    df.to_parquet(PROCESSED / "variants_v1.parquet")
    (PROCESSED / "variants_v1_provenance.json").write_text(json.dumps(LOG, indent=1, default=str))
    print(json.dumps(LOG, indent=1, default=str))


if __name__ == "__main__":
    main()
