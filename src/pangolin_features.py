"""Frozen pretrained context branch: Pangolin (Zeng & Li 2022) features on hg19 genomic context.

For every variant we score ref and alt sequences (5 kb flank each side) with all 8 Pangolin heads
(4 tissues x {splice-site probability, usage}; 3 replicate models each) and keep:
  * ref/alt/delta at the target exon's acceptor (first exon base) and donor (last exon base)
  * max gain / max loss of the delta track over the exon span (Pangolin-style summary)
  * 32-d hidden ("skip") representations at acceptor, donor and the variant for two models,
    as ref and alt-ref, used by the network and by the transferability distance.
Pangolin was trained on genome annotation / RNA-seq splice-site usage, never on MPSA labels.
"""
import sys

import numpy as np
import pandas as pd
import pyfaidx
import torch
import torch.nn.functional as F
from tqdm import tqdm

from config import HG19, PANGOLIN_DIR, PROCESSED

sys.path.insert(0, str(PANGOLIN_DIR.parents[1]))
from pangolin.model import AR, W, L, Pangolin  # noqa: E402

FLANK = 5000
PAD = 25
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HEAD_CH = [1, 2, 4, 5, 7, 8, 10, 11]          # group i -> output channel (P-site / usage per tissue)
HEAD_NAMES = [f"{t}_{k}" for t in ("heart", "liver", "brain", "testis") for k in ("p", "u")]
EMB_GROUPS = (0, 1)                            # heart P-site and heart usage, replicate 1


def skip_forward(m, x):
    conv = m.conv1(x)
    skip = m.skip(conv)
    j = 0
    for i in range(len(W)):
        conv = m.resblocks[i](conv)
        if ((i + 1) % 4 == 0) or ((i + 1) == len(W)):
            skip = skip + m.convs[j](conv)
            j += 1
    CL = 2 * int(np.sum(AR * (W - 1)))
    skip = F.pad(skip, (-CL // 2, -CL // 2))
    return skip


def load_models():
    models = []
    for i in range(8):                       # file naming: final.{rep}.{group}.3.v2
        reps = []
        for rep in (1, 2, 3):
            m = Pangolin(L, W, AR).to(DEV)
            m.load_state_dict(torch.load(PANGOLIN_DIR / f"final.{rep}.{i}.3.v2", map_location=DEV))
            m.eval()
            reps.append(m)
        models.append(reps)
    return models


LUT = np.zeros(256, dtype=np.int64)
for k, v in zip(b"ACGTN", (0, 1, 2, 3, 4)):
    LUT[k] = v
EYE = np.vstack([np.eye(4, dtype=np.float32), np.zeros((1, 4), np.float32)])
RC = str.maketrans("ACGTN", "TGCAN")


def onehot(seqs):
    idx = LUT[np.frombuffer("".join(seqs).encode(), dtype=np.uint8)].reshape(len(seqs), -1)
    return torch.from_numpy(EYE[idx]).permute(0, 2, 1).contiguous()


@torch.no_grad()
def run_exon(models, chrom, strand, lo, hi, variants, fa):
    """lo/hi: genomic output span (1-based inclusive). variants: list of (pos, ref, alt)."""
    seq = str(fa[chrom][lo - 1 - FLANK: hi + FLANK]).upper()
    seqs = [seq]
    for pos, ref, alt in variants:
        k = pos - lo + FLANK
        assert seq[k] == ref, (chrom, pos, ref, seq[k])
        seqs.append(seq[:k] + alt + seq[k + 1:])
    if strand == "-":
        seqs = [s.translate(RC)[::-1] for s in seqs]
    x = onehot(seqs).to(DEV)
    tracks, embs = [], []
    for gi, reps in enumerate(models):
        acc = 0
        for ri, m in enumerate(reps):
            if gi in EMB_GROUPS and ri == 0:
                sk = skip_forward(m, x)
                out = getattr(m, f"conv_last{2 * (gi // 2) + 1 + (gi % 2)}")(sk)
                out = F.softmax(out, 1)[:, 1] if gi % 2 == 0 else torch.sigmoid(out)[:, 0]
                embs.append(sk)
            else:
                out = m(x)[:, HEAD_CH[gi]]
            acc = acc + out
        tracks.append(acc / len(reps))
    tr = torch.stack(tracks, 1).float().cpu().numpy()          # (n, 8, span)
    em = torch.cat(embs, 1).float().cpu().numpy()               # (n, 64, span)
    if strand == "-":
        tr, em = tr[..., ::-1], em[..., ::-1]
    return tr, em                                               # index 0 <-> genomic lo


def main():
    df = pd.read_parquet(PROCESSED / "variants_v1.parquet")
    fa = pyfaidx.Fasta(str(HG19))
    models = load_models()
    feats = {}
    KEY = ["chromosome", "strand", "exon_start", "exon_end", "position", "ref", "alt"]
    old = PROCESSED / "old_v1_benchfix"
    if (old / "pangolin_features_v1.parquet").exists():
        # reuse features of unchanged (exon, variant) pairs from the previous build
        od = pd.read_parquet(old / "variants_v1.parquet")[KEY]
        of = pd.read_parquet(old / "pangolin_features_v1.parquet").drop(columns="sample_id")
        for k, row in zip(od.itertuples(index=False, name=None), of.to_dict("records")):
            feats[k] = row
        todo = ~pd.Series(list(df[KEY].itertuples(index=False, name=None))).isin(feats.keys()).to_numpy()
        print("cached", len(feats), "to compute", int(todo.sum()))
    else:
        todo = np.ones(len(df), bool)
    groups = df[todo].groupby(["chromosome", "strand", "exon_start", "exon_end"])
    for (chrom, strand, es, ee), g in tqdm(groups, total=groups.ngroups):
        lo = int(min(es, g.position.min())) - PAD
        hi = int(max(ee, g.position.max())) + PAD
        uniq = g[["position", "ref", "alt"]].drop_duplicates()
        var = list(uniq.itertuples(index=False, name=None))
        res_tr, res_em = [], []
        for b in range(0, len(var), 48):
            tr, em = run_exon(models, chrom, strand, lo, hi, var[b:b + 48], fa)
            res_tr.append(tr[1:]); res_em.append(em[1:])
            ref_tr, ref_em = tr[0], em[0]
        alt_tr = np.concatenate(res_tr); alt_em = np.concatenate(res_em)
        a_i = (es if strand == "+" else ee) - lo
        d_i = (ee if strand == "+" else es) - lo
        e0, e1 = es - lo, ee - lo
        for j, (pos, ref, alt) in enumerate(var):
            v_i = pos - lo
            dl = alt_tr[j] - ref_tr
            f = {}
            for h, hn in enumerate(HEAD_NAMES):
                f[f"pg_acc_ref_{hn}"] = ref_tr[h, a_i]
                f[f"pg_acc_alt_{hn}"] = alt_tr[j, h, a_i]
                f[f"pg_acc_d_{hn}"] = dl[h, a_i]
                f[f"pg_don_ref_{hn}"] = ref_tr[h, d_i]
                f[f"pg_don_alt_{hn}"] = alt_tr[j, h, d_i]
                f[f"pg_don_d_{hn}"] = dl[h, d_i]
                win = dl[h, max(0, min(e0, v_i) - PAD): max(e1, v_i) + PAD + 1]
                f[f"pg_gain_{hn}"] = win.max()
                f[f"pg_loss_{hn}"] = win.min()
            emb = np.concatenate([ref_em[:, a_i], ref_em[:, d_i], ref_em[:, v_i],
                                  alt_em[j, :, a_i] - ref_em[:, a_i],
                                  alt_em[j, :, d_i] - ref_em[:, d_i],
                                  alt_em[j, :, v_i] - ref_em[:, v_i]])
            for k, val in enumerate(emb):
                f[f"pe_{k}"] = val
            feats[(chrom, strand, es, ee, pos, ref, alt)] = f
    keys = list(zip(df.chromosome, df.strand, df.exon_start, df.exon_end, df.position, df.ref, df.alt))
    out = pd.DataFrame([feats[k] for k in keys]).astype(np.float32)
    out.insert(0, "sample_id", df.sample_id.values)
    tmp = PROCESSED / "pangolin_features_v1.tmp.parquet"
    out.to_parquet(tmp)
    tmp.rename(PROCESSED / "pangolin_features_v1.parquet")
    print(out.shape)


if __name__ == "__main__":
    main()
