"""
AIM (Adaptive Intersection Maximization; Ma et al., Sci. Adv. 10, eadm7765, 2024) reimplemented in PyTorch,
following github.com/YangLiuLab/AIM (AIM.m / IntersectionMax.m; the MEX intersection counter replaced by searchsorted).

  - positions quantized on a grid of `intersect_d` (paper: 20 nm = 0.2 px at 100 nm/px)
  - per segment of `interval` frames: count coinciding grid cells with the reference for all shifts in [-R, R]^2
    (R = 3 grid units), around the previous segment's estimate (adaptive tracking)
  - peak position from the phase of the first Fourier harmonic of the (2R+1)^2 count map
  - stage 1: reference = first segment; stage 2: reference = full stage-1 corrected dataset
  - drift at segment centers -> cubic spline per frame
2D only (xy). Units of positions: px.
"""
import numpy as np
import torch
from scipy.interpolate import CubicSpline


def _keys(xy, d, W):
    q = torch.round(xy / d).long()
    return q[:, 1] * W + q[:, 0]


def _unique_counts(k):
    u, c = torch.unique(k, return_counts=True)   # sorted
    return u, c


def _phase_peak(cc, R):
    """First-harmonic phase localization of the peak in a (2R+1)-length profile (as in AIM)."""
    L = 2 * R + 1
    f1 = np.fft.fft(cc)[1]
    ang = np.angle(f1)
    ang = ang - 2 * np.pi * (ang > 0)
    return (abs(ang) / (2 * np.pi / L) + 1) - (L + 1) / 2


def _intersection_max(xy, seg, nseg, ref_xy, d, W, R, dev):
    ru, rc = _unique_counts(_keys(ref_xy, d, W))
    shifts = torch.arange(-R, R + 1, device=dev)
    sh2 = (shifts[:, None] * W + shifts[None, :]).reshape(-1)          # [dy, dx] -> key offset
    keys = _keys(xy, d, W)
    order = torch.argsort(seg)
    bounds = torch.searchsorted(seg[order], torch.arange(nseg + 1, device=dev)).tolist()
    drift = np.zeros((nseg, 2))
    ref = np.zeros(2)
    for s in range(1, nseg):
        k = keys[order[bounds[s]:bounds[s + 1]]]
        if len(k) == 0:
            drift[s] = drift[s - 1]
            continue
        su, sc = _unique_counts(k)
        base = su + int(round(ref[1])) * W + int(round(ref[0]))
        q = base[None, :] + sh2[:, None]                                 # [(2R+1)^2, M]
        idx = torch.searchsorted(ru, q).clamp(max=len(ru) - 1)
        hit = ru[idx] == q
        cnt = (torch.where(hit, rc[idx], 0) * sc[None, :]).sum(1).double().view(2 * R + 1, 2 * R + 1).cpu().numpy()
        px = _phase_peak(cnt.sum(0), R)   # profile along dx
        py = _phase_peak(cnt.sum(1), R)   # profile along dy
        ref = np.array([round(ref[0]) + px, round(ref[1]) + py])
        drift[s] = -ref
    return drift * d


def aim(positions, framenum, interval=20, intersect_d=0.2, R=3, numframes=None, device='cuda'):
    """Returns per-frame drift [F, 2] in px (zero mean), same sign convention as DME (corrected = pos - drift)."""
    dev = torch.device(device)
    xy = torch.as_tensor(np.asarray(positions)[:, :2], dtype=torch.float64, device=dev)
    fr = np.asarray(framenum).astype(np.int64)
    F = int(numframes or fr.max() + 1)
    nseg = max(2, F // interval)
    seg_np = np.minimum(fr // interval, nseg - 1)
    seg = torch.as_tensor(seg_np, device=dev)
    xy = xy - xy.min(0).values + 10 * intersect_d
    W = int((xy.max().item() + 10) / intersect_d) + 2 * R + 10
    centers = (np.arange(nseg) + 0.5) * interval

    def per_frame(ds):
        return CubicSpline(centers, ds, axis=0, bc_type='natural')(np.clip(np.arange(F), centers[0], centers[-1]))

    first = torch.as_tensor(seg_np == 0, device=dev)
    d1 = per_frame(_intersection_max(xy, seg, nseg, xy[first], intersect_d, W, R, dev))
    xy1 = xy - torch.as_tensor(d1[fr], device=dev)
    d2 = per_frame(_intersection_max(xy1, seg, nseg, xy1, intersect_d, W, R, dev))
    d = d1 + d2
    return (d - d.mean(0)).astype(np.float32)
