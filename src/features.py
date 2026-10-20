"""Local-branch sequence windows and tabular features.

Three windows in transcript orientation, each with 6 channels (A,C,G,T, exon mask, variant mark):
  acceptor-centred (140 nt), donor-centred (140 nt), variant-centred (101 nt); ref and alt.
"""
import numpy as np
import pandas as pd
import pyfaidx

from config import HG19, PROCESSED

WIN = {"acc": 140, "don": 140, "var": 101}
RC = str.maketrans("ACGTN", "TGCAN")
LUT = np.full(256, 4, dtype=np.int64)
for k, v in zip(b"ACGT", range(4)):
    LUT[k] = v
EYE = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)])
REGIONS = ["canonical_ss", "intron_near", "intron_deep", "exon_near", "exon_deep"]
ASSAYS = ["MFASS", "VexSeq"]


def window(fa, chrom, strand, center, size):
    """Genomic window of `size` around `center` in transcript orientation.
    Returns sequence and the genomic coordinate of each transcript-ordered position."""
    half = size // 2
    if strand == "+":
        lo = center - half
        coords = np.arange(lo, lo + size)
        s = str(fa[chrom][lo - 1: lo - 1 + size]).upper()
    else:
        hi = center + half
        coords = np.arange(hi, hi - size, -1)
        s = str(fa[chrom][hi - size: hi]).upper().translate(RC)[::-1]
    return s, coords


def encode(seq, coords, es, ee, vpos, alt=None, strand="+"):
    s = np.frombuffer(seq.encode(), dtype=np.uint8).copy()
    vidx = np.where(coords == vpos)[0]
    if alt is not None and len(vidx):
        b = alt if strand == "+" else alt.translate(RC)
        s[vidx[0]] = ord(b)
    x = np.zeros((6, len(seq)), np.float32)
    x[:4] = EYE[LUT[s]].T
    x[4] = (coords >= es) & (coords <= ee)
    if len(vidx):
        x[5, vidx[0]] = 1.0
    return x


def build_windows(df):
    fa = pyfaidx.Fasta(str(HG19))
    out = {k: np.zeros((len(df), 2, 6, w), np.float16) for k, w in WIN.items()}
    for i, r in enumerate(df.itertuples(index=False)):
        plus = r.strand == "+"
        acc = r.exon_start if plus else r.exon_end
        don = r.exon_end if plus else r.exon_start
        for k, c in (("acc", acc), ("don", don), ("var", r.position)):
            seq, coords = window(fa, r.chromosome, r.strand, c, WIN[k])
            out[k][i, 0] = encode(seq, coords, r.exon_start, r.exon_end, r.position, None, r.strand)
            out[k][i, 1] = encode(seq, coords, r.exon_start, r.exon_end, r.position, r.alt, r.strand)
    return out


def tabular(df):
    t = pd.DataFrame(index=df.index)
    for c in ("dist_acc", "dist_don"):
        t[c + "_sl"] = np.sign(df[c]) * np.log1p(np.abs(df[c]))
    t["exon_len_l"] = np.log(df.exon_len)
    t["rel_pos"] = np.clip(df.dist_acc / df.exon_len, -1, 2)
    for r in REGIONS:
        t["reg_" + r] = (df.region == r).astype(float)
    minus = df.strand == "-"
    ref = df.ref.where(~minus, df.ref.str.translate(RC))      # transcript orientation
    alt = df.alt.where(~minus, df.alt.str.translate(RC))
    for b in "ACGT":
        t["ref_" + b] = (ref == b).astype(float)
        t["alt_" + b] = (alt == b).astype(float)
    return t.astype(np.float32)


def main():
    df = pd.read_parquet(PROCESSED / "variants_v1.parquet")
    w = build_windows(df)
    np.savez_compressed(PROCESSED / "windows_v1.npz", **w, sample_id=df.sample_id.values)
    tabular(df).assign(sample_id=df.sample_id.values).to_parquet(PROCESSED / "tabular_v1.parquet")
    print({k: v.shape for k, v in w.items()})


if __name__ == "__main__":
    main()
