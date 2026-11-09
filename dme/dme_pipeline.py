"""
GPU-only drop-in for dme.dme_estimate: RCC init (torch FFT) -> optional coarse DME -> fine DME -> split-half precision.
PyTorch + CuPy runtime-compiled kernels; no build step.
"""
import time
import numpy as np
import torch

from .dme_gpu import DMEGPU, cupy_available


# ----------------------------------------------------------------------------- RCC
def _render_bins(xy, bin_idx, nbins, shape, sigma_px, dev):
    """Histogram per time bin, blurred with a Gaussian (via FFT). Returns FFT of images [nbins, H, W]."""
    H, W = shape
    ix = xy[:, 0].round().long().clamp(0, W - 1)
    iy = xy[:, 1].round().long().clamp(0, H - 1)
    img = torch.zeros(nbins * H * W, device=dev)
    img.index_add_(0, (bin_idx * H + iy) * W + ix, torch.ones(len(ix), device=dev))
    img = img.view(nbins, H, W)
    fy = torch.fft.fftfreq(H, device=dev)[:, None]
    fx = torch.fft.fftfreq(W, device=dev)[None, :]
    g = torch.exp(-2 * (torch.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
    return torch.fft.fft2(img) * g


def _peak_subpixel(cc, hw):
    """cc: [P, H, W] fftshifted. Argmax in central window +-hw, then 3-point Gaussian fit per axis."""
    P, H, W = cc.shape
    cy, cx = H // 2, W // 2
    win = cc[:, cy - hw:cy + hw + 1, cx - hw:cx + hw + 1]
    am = win.reshape(P, -1).argmax(1)
    py = (am // win.shape[2]).clamp(1, win.shape[1] - 2)
    px = (am % win.shape[2]).clamp(1, win.shape[2] - 2)
    ar = torch.arange(P, device=cc.device)

    def fit(m, c, p):
        lm, lc, lp = [torch.log(t.clamp(min=1e-12)) for t in (m, c, p)]
        den = lm - 2 * lc + lp
        return torch.where(den.abs() > 1e-12, 0.5 * (lm - lp) / den, torch.zeros_like(den)).clamp(-1, 1)

    dy = fit(win[ar, py - 1, px], win[ar, py, px], win[ar, py + 1, px])
    dx = fit(win[ar, py, px - 1], win[ar, py, px], win[ar, py, px + 1])
    return torch.stack([px + dx - hw, py + dy - hw], 1)


def rcc_torch(xy, framenum, timebins=10, zoom=2, sigma=1.0, maxdrift_px=None, numframes=None, device='cuda'):
    """Redundant cross-correlation, all pairs of time bins, batched on GPU. Returns per-frame drift [F, 2] (px)."""
    dev = torch.device(device)
    xy = torch.as_tensor(np.asarray(xy)[:, :2], dtype=torch.float32, device=dev)
    fr = torch.as_tensor(np.asarray(framenum), dtype=torch.long, device=dev)
    F = int(numframes or fr.max().item() + 1)
    fpb = F / timebins
    b = ((fr + 0.5) / fpb).long().clamp(max=timebins - 1)
    lo = xy.min(0).values
    p = (xy - lo) * zoom
    W, H = [max(int(v) + 2, 16) for v in p.max(0).values.tolist()]   # >= 16 px: room for the peak search
    Hf, Wf = H + H % 2, W + W % 2
    Fimg = _render_bins(p, b, timebins, (Hf, Wf), sigma, dev)
    i, j = torch.triu_indices(timebins, timebins, 1, device=dev)
    hw = int(min(Hf, Wf) // 2 - 2) if maxdrift_px is None else int(min(maxdrift_px * zoom, min(Hf, Wf) // 2 - 2))
    shifts = []
    bs = max(1, int(1.5e9 // (Hf * Wf * 16)))   # batch pairs to bound memory (complex64 product + real cc)
    for s in range(0, len(i), bs):
        a, c = i[s:s + bs], j[s:s + bs]
        cc = torch.fft.ifft2(Fimg[a].conj() * Fimg[c]).real
        cc = torch.fft.fftshift(cc, (-2, -1))
        shifts.append(_peak_subpixel(cc, hw))
    sh = torch.cat(shifts).double() / zoom                 # shift of bin j relative to bin i
    A = torch.zeros(len(i), timebins, dtype=torch.float64, device=dev)
    A[torch.arange(len(i)), j] = 1
    A[torch.arange(len(i)), i] = -1
    A = torch.cat([A, torch.ones(1, timebins, dtype=torch.float64, device=dev)])  # zero-mean constraint
    rhs = torch.cat([sh, torch.zeros(1, 2, dtype=torch.float64, device=dev)])
    d_bins = torch.linalg.lstsq(A, rhs).solution.float()  # [timebins, 2]
    # piecewise-linear interpolation between bin centers, constant extrapolation
    t = (torch.arange(timebins, device=dev) + 0.5) * fpb
    f = torch.arange(F, device=dev, dtype=torch.float32)
    k = torch.searchsorted(t, f).clamp(1, timebins - 1)
    w = ((f - t[k - 1]) / (t[k] - t[k - 1])).clamp(0, 1)[:, None]
    d = d_bins[k - 1] * (1 - w) + d_bins[k] * w
    return (d - d.mean(0)).cpu().numpy()


def rcc3d_torch(xyz, framenum, timebins=10, zoom=2, sigma=1.0, numframes=None, device='cuda'):
    """XY by RCC; Z by RCC on a sheared projection (y + z), as in dme.rcc.rcc3D."""
    xyz = np.asarray(xyz)
    dxy = rcc_torch(xyz[:, :2], framenum, timebins, zoom, sigma, numframes=numframes, device=device)
    sh = xyz[:, :2] - dxy[framenum]
    sh[:, 1] += xyz[:, 2]
    dz = rcc_torch(sh, framenum, timebins, zoom, sigma, numframes=numframes, device=device)[:, 1]
    d = np.concatenate([dxy, dz[:, None]], 1)
    return d - d.mean(0)


# ----------------------------------------------------------------------------- DME driver
def _run(positions, framenum, crlb, fpb, init, numframes, max_iter, optimizer, device, verbose, label):
    t = time.time()
    if cupy_available(device):
        from .dme_cupy import DMECupy
        e = DMECupy(positions, framenum, crlb, fpb, init, numframes, device=device)
    else:
        # pure PyTorch fallback (stored pair list, more memory): CPU, or CUDA without cupy
        e = DMEGPU(positions, framenum, crlb, fpb, init, numframes, device=device)
    d = e.run_lbfgs(max_iter=max_iter) if optimizer == 'lbfgs' else e.run_original_ascent(max_iter=max_iter)
    if verbose:
        print(f"DME[{label}] N={e.N} fpb={fpb} knots={e.K} evals={e.iterations} score={e.final_score:.5f} "
              f"time={time.time() - t:.2f}s", flush=True)
    return d, e


def dme_estimate_gpu(positions, framenum, crlb, framesperbin, coarseFramesPerBin=None, coarseSigma=None,
                     perSpotCRLB=False, initializeWithRCC=10, initialEstimate=None, rccZoom=2,
                     estimatePrecision=True, maxIterations=200, coarseMaxIterations=100, numframes=None,
                     optimizer='lbfgs', device='cuda', verbose=True):
    """Mirrors dme.dme_estimate (minus plotting). Returns drift [F,D] or (drift, (drift_set1, drift_set2))."""
    positions = np.asarray(positions, np.float32)
    framenum = np.asarray(framenum, np.int32)
    crlb = np.asarray(crlb, np.float32)
    D = positions.shape[1]
    F = int(numframes if numframes is not None else framenum.max() + 1)

    t0 = time.time()
    if initialEstimate is not None:
        init = np.asarray(initialEstimate, np.float32)[:F]
    elif initializeWithRCC:
        nb = 10 if initializeWithRCC is True else int(initializeWithRCC)
        init = (rcc3d_torch if D == 3 else rcc_torch)(positions, framenum, nb, rccZoom, numframes=F, device=device)
        if verbose:
            print(f"RCC ({nb} bins) {time.time() - t0:.2f}s", flush=True)
    else:
        init = np.zeros((F, D), np.float32)

    if not perSpotCRLB:
        crlb = crlb.reshape(-1, D).mean(0) if crlb.ndim == 2 else crlb[:D]

    def estimate(sel, label):
        p, f = positions[sel], framenum[sel]
        c = crlb[sel] if crlb.ndim == 2 else crlb
        d = init
        if coarseFramesPerBin is not None:
            d, _ = _run(p, f, np.asarray(coarseSigma, np.float32), coarseFramesPerBin, d, F, coarseMaxIterations,
                        optimizer, device, verbose, label + '/coarse')
        d, _ = _run(p, f, c, framesperbin, d, F, maxIterations, optimizer, device, verbose, label)
        return d

    drift = estimate(slice(None), 'full')
    if not estimatePrecision:
        return drift
    ax = int(np.argmax(np.var(positions[:, :2], 0)))
    s1 = positions[:, ax] > np.median(positions[:, ax])
    d1, d2 = estimate(s1, 'set1'), estimate(~s1, 'set2')
    rmsd = np.sqrt(((d1 - d1.mean(0) - d2 + d2.mean(0)) ** 2).mean(0))
    if verbose:
        print(f"RMSD of drift traces on split dataset: {rmsd}  (total {time.time() - t0:.1f}s)", flush=True)
    return drift, (d1, d2)
