"""Two-branch variant-effect network with mean, aleatoric-scale and disruption heads."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResConv(nn.Module):
    def __init__(self, c, k, d):
        super().__init__()
        self.n1, self.n2 = nn.BatchNorm1d(c), nn.BatchNorm1d(c)
        self.c1 = nn.Conv1d(c, c, k, dilation=d, padding=d * (k - 1) // 2)
        self.c2 = nn.Conv1d(c, c, k, dilation=d, padding=d * (k - 1) // 2)

    def forward(self, x):
        h = self.c1(F.gelu(self.n1(x)))
        return x + self.c2(F.gelu(self.n2(h)))


class LocalEncoder(nn.Module):
    """Multi-scale CNN shared between ref and alt of one window type."""

    def __init__(self, c=64):
        super().__init__()
        self.stem = nn.ModuleList([nn.Conv1d(6, c // 2, k, padding=k // 2) for k in (5, 11)])
        self.blocks = nn.Sequential(ResConv(c, 5, 1), ResConv(c, 5, 2), ResConv(c, 5, 4))
        self.out_dim = 2 * c

    def forward(self, x):
        h = torch.cat([s(x) for s in self.stem], 1)
        h = self.blocks(h)
        return torch.cat([h.mean(-1), h.amax(-1)], 1)


class SpliceNet(nn.Module):
    def __init__(self, n_ctx, n_tab, use_local=True, use_ctx=True, use_assay=True, n_assay=3, hid=256,
                 drop=0.2):
        super().__init__()
        self.use_local, self.use_ctx, self.use_assay = use_local, use_ctx, use_assay
        dims = [n_tab]
        if use_local:
            self.enc = nn.ModuleDict({k: LocalEncoder() for k in ("acc", "don", "var")})
            d = sum(e.out_dim for e in self.enc.values())
            self.local_proj = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU())
            dims.append(128)
        if use_ctx:
            self.ctx = nn.Sequential(nn.Linear(n_ctx, 256), nn.GELU(), nn.Dropout(drop),
                                     nn.Linear(256, 128), nn.GELU())
            dims.append(128)
        if use_assay:
            self.assay = nn.Embedding(n_assay, 8)
            dims.append(8)
        self.fuse = nn.Sequential(nn.Linear(sum(dims), hid), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hid, hid), nn.GELU())
        self.mu, self.logsig, self.sdv = nn.Linear(hid, 1), nn.Linear(hid, 1), nn.Linear(hid, 1)

    def forward(self, b):
        parts = [b["tab"]]
        if self.use_local:
            hr = torch.cat([self.enc[k](b[k][:, 0]) for k in self.enc], 1)
            ha = torch.cat([self.enc[k](b[k][:, 1]) for k in self.enc], 1)
            parts.append(self.local_proj(torch.cat([ha - hr, (ha + hr) / 2], 1)))
        if self.use_ctx:
            parts.append(self.ctx(b["ctx"]))
        if self.use_assay:
            parts.append(self.assay(b["assay"]))
        z = self.fuse(torch.cat(parts, 1))
        mu = torch.tanh(self.mu(z)).squeeze(-1)
        sig = (F.softplus(self.logsig(z)) + 1e-3).squeeze(-1)
        return mu, sig, self.sdv(z).squeeze(-1), z


def batches(T, idx, bs, shuffle, rng=None):
    idx = np.asarray(idx)
    if shuffle:
        idx = rng.permutation(idx)
    for s in range(0, len(idx), bs):
        j = torch.as_tensor(idx[s:s + bs], device=DEV)
        yield {k: (v[j].float() if v.dtype == torch.float16 else v[j]) for k, v in T.items()}


def loss_fn(mu, sig, logit, b, lam_nll=0.5, lam_cls=0.5):
    y, w = b["y"], b["w"]
    hub = F.huber_loss(mu, y, delta=0.1, reduction="none")
    nll = 0.5 * ((y - mu.detach()) / sig) ** 2 + torch.log(sig)
    m = torch.isfinite(b["sdv"])
    cls = (F.binary_cross_entropy_with_logits(logit[m], b["sdv"][m], reduction="sum") / max(1, m.sum())
           if m.any() else 0.0)
    return ((hub + lam_nll * nll) * w).sum() / w.sum() + lam_cls * cls


@torch.no_grad()
def predict(model, T, idx, assay_override=None, bs=1024):
    model.eval()
    out = {"mu": [], "sig": [], "p": [], "z": []}
    for b in batches(T, idx, bs, False):
        if assay_override is not None:
            b["assay"] = torch.full_like(b["assay"], assay_override)
        mu, sig, lg, z = model(b)
        out["mu"].append(mu); out["sig"].append(sig); out["p"].append(torch.sigmoid(lg)); out["z"].append(z)
    return {k: torch.cat(v).float().cpu().numpy() for k, v in out.items()}


def train(T, tr_idx, va_idx, n_ctx, n_tab, seed, max_epochs=40, patience=6, lr=1e-3, bs=256,
          unknown_assay_p=0.2, **arch):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SpliceNet(n_ctx, n_tab, **arch).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epochs)
    best, best_state, bad = np.inf, None, 0
    for ep in range(max_epochs):
        model.train()
        for b in batches(T, tr_idx, bs, True, rng):
            # assay-token dropout teaches the network an "unknown assay" mode used for unseen studies
            drop = torch.rand(len(b["assay"]), device=DEV) < unknown_assay_p
            b["assay"] = torch.where(drop, torch.full_like(b["assay"], 2), b["assay"])
            mu, sig, lg, _ = model(b)
            loss = loss_fn(mu, sig, lg, b)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        pv = predict(model, T, va_idx)
        yv = T["y"][torch.as_tensor(va_idx, device=DEV)].cpu().numpy()
        wv = T["w"][torch.as_tensor(va_idx, device=DEV)].cpu().numpy()
        val = float(np.sum(wv * np.abs(pv["mu"] - yv)) / wv.sum())
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best
