"""EURS Track 2 (DeepSeaSight) - the single exported network: one lightweight CNN encoder (ImageNet-pretrained,
`timm`) feeding a segmentation head and a physics-shaped restoration head, <= 1.0 M parameters and <= 12 GFLOPs
at the exported resolution. Trimmed, inference/export-only extract of the team's own `eurs3_t2.py` (the
degradation bank, GPU augmentation, loss functions, training job/EMA wrapper, and the training-only auxiliary
(water-type/depth) head are all removed - `export_onnx.py` only needs the forward graph that was actually
uploaded to the leaderboard). No code or weights are copied from any other project; the only third-party
component is the pretrained ImageNet encoder loaded by `timm.create_model(..., pretrained=True)` during
TRAINING (see METHOD.md) - at export/inference time weights are always loaded from the local checkpoint with
`pretrained=False`, so no network access is required or attempted.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

NCLS = 5


class Expose(nn.Module):
    """Per-channel exposure gain, INSIDE the exported graph: pure tensor ops (AveragePool, TopK, Clip, Mul), no
    learned parameters. Lifts the ~99.5th percentile of the block-averaged image to `target` (clamped to
    [1, maxg]); for very dark frames (gain > g0) a 3x3 box blur is blended in first, because an 8-bit staircase
    multiplied by a large gain shows up as posterised plateaus."""

    def __init__(self, k=8, q=0.005, target=0.85, maxg=128.0, g0=10.0):
        super().__init__()
        self.k, self.q, self.target, self.maxg, self.g0 = k, q, target, maxg, g0

    def forward(self, x):
        p = F.avg_pool2d(x, self.k)
        n = int(p.shape[2]) * int(p.shape[3])
        kk = max(1, int(round(self.q * n)))
        s = p.reshape(-1, 3, n).topk(kk, dim=2).values.amin(2)
        g = (self.target / (s + 1e-4)).clamp(1.0, self.maxg)
        gm = g.amax(1, keepdim=True)
        w = ((torch.log2(gm) - math.log2(self.g0)) / 3.0).clamp(0.0, 1.0) * 0.85
        bl = F.avg_pool2d(x, 3, 1, 1, count_include_pad=False)
        xs = x + w.view(-1, 1, 1, 1) * (bl - x)
        return (xs * g.view(-1, 3, 1, 1)).clamp(0.0, 1.0), g, s


def dwsep(i, o):
    return nn.Sequential(nn.Conv2d(i, i, 3, padding=1, groups=i, bias=False), nn.BatchNorm2d(i),
                          nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


def up(t, f):
    """Bilinear x f upsampling with a CONSTANT scale factor (static graph: no Shape / Gather ops when exported)."""
    return F.interpolate(t, scale_factor=float(f), mode="bilinear", align_corners=False, recompute_scale_factor=False)


class RestTrunk(nn.Module):
    """Half-resolution detail/restoration refiner ("v4"): stride-2 conv -> residual blocks -> bilinear x2, plus a
    thin full-resolution tail that sees the exposed input and the gain/offset estimate again (so high-frequency
    detail can still be written). Plain convs / ReLU / Resize only. The last conv of the tail is zero-initialised
    at construction, so an untrained model starts from the identity (`restored = o0`); the real, trained values
    come from the checkpoint."""

    def __init__(self, c=40, nb=2, cf=12):
        super().__init__()
        self.inp = nn.Conv2d(6, c, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c, c, 3, padding=1)) for _ in range(nb)])
        self.proj = nn.Conv2d(c, cf, 1)
        self.fin1 = nn.Conv2d(cf + 6, cf, 3, padding=1)
        self.fin2 = nn.Conv2d(cf, 3, 3, padding=1)

    def forward(self, xe, o0):
        z = torch.cat([xe, o0], 1)
        h = F.relu(self.inp(z))
        for b in self.blocks:
            h = F.relu(h + b(h))
        return self.fin2(F.relu(self.fin1(torch.cat([up(self.proj(h), 2), z], 1))))


class StudentNet(nn.Module):
    """x: (B,3,H,W) RGB float in [0,1], H and W multiples of 32  ->  (restored (B,3,H,W) in [0,1],
    seg_logits (B,5,H,W)). ImageNet normalisation, exposure normalisation and the calibrated per-class logit
    bias are all inside the graph - nothing extra has to happen after `model(x)`.

    The reported submission ("new_alt") uses `enc="lcnet_075.ra2_in1k"`, `d=56`, half-res trunk `(c=40, nb=2)`,
    exported at S=448: 0.784 M parameters, 10.30 GFLOPs (2xMAC)."""

    def __init__(self, enc="lcnet_075.ra2_in1k", d=56, c_ref=16, trunk_c=40, trunk_nb=2, trunk_cf=12, pretrained=False):
        super().__init__()
        import timm
        self.expose = Expose()
        self.enc = timm.create_model(enc, pretrained=pretrained, features_only=True, out_indices=(1, 2, 3, 4))
        chs = self.enc.feature_info.channels()
        self.chs, self.d = chs, d
        self.lat = nn.ModuleList([nn.Conv2d(c, d, 1) for c in chs])
        self.ctx = nn.Sequential(nn.Conv2d(chs[-1], d, 1), nn.ReLU(inplace=True))
        self.smooth = nn.ModuleList([dwsep(d, d) for _ in chs])
        self.fuse = nn.Sequential(nn.Conv2d(4 * d, d, 1, bias=False), nn.BatchNorm2d(d), nn.ReLU(inplace=True))
        self.fuse2 = dwsep(d, d)
        self.cls = nn.Conv2d(d, NCLS, 1)
        self.seg_ref = nn.Sequential(nn.Conv2d(3 + NCLS, c_ref, 3, padding=1), nn.ReLU(inplace=True),
                                      nn.Conv2d(c_ref, c_ref, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(c_ref, NCLS, 3, padding=1))
        self.film = nn.Sequential(nn.Conv2d(chs[-1] + 6, 64, 1), nn.ReLU(inplace=True), nn.Conv2d(64, 2 * d, 1))
        self.r1, self.r2 = dwsep(d + 3, d), dwsep(d, d)
        self.r_head = nn.Conv2d(d, 6, 3, padding=1)
        self.trunk = RestTrunk(trunk_c, trunk_nb, trunk_cf)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("bias", torch.zeros(NCLS))   # calibrated per-class logit offsets, baked in from the checkpoint

    def forward(self, x):
        xe, g, s = self.expose(x)
        feats = [f if f.shape[1] == c else f.permute(0, 3, 1, 2) for f, c in zip(self.enc((xe - self.mean) / self.std), self.chs)]
        p = self.smooth[3](self.lat[3](feats[3]) + self.ctx(feats[-1].mean((2, 3), keepdim=True)))
        ps = [p]
        for i in (2, 1, 0):
            p = self.smooth[i](self.lat[i](feats[i]) + up(p, 2))
            ps.append(p)
        f4 = self.fuse2(self.fuse(torch.cat([ps[3], up(ps[2], 2), up(ps[1], 4), up(ps[0], 8)], 1)))

        lg = up(self.cls(f4), 4)
        lg = lg + self.seg_ref(torch.cat([xe, lg], 1)) + self.bias.view(1, NCLS, 1, 1)

        gv = feats[-1].mean((2, 3), keepdim=True)
        gvec = torch.cat([gv, (torch.log(g) / 5.0).view(-1, 3, 1, 1), s.view(-1, 3, 1, 1)], 1)
        fl = self.film(gvec)
        sc, sh = fl[:, :self.d], fl[:, self.d:]
        r = self.r2(self.r1(torch.cat([f4 * (1.0 + sc) + sh, F.avg_pool2d(xe, 4)], 1)))
        ab = up(self.r_head(r), 4)
        o0 = xe * torch.exp(ab[:, :3].clamp(-4.0, 4.0)) + ab[:, 3:]
        out = (o0 + self.trunk(xe, o0)).clamp(0.0, 1.0)
        return out, lg


class ExportWrap(nn.Module):
    """Plain (restored_image, seg_logits) wrapper so the exported graph has exactly two, unambiguously-named
    outputs in the required order."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        r, s = self.m(x)
        return r, s
