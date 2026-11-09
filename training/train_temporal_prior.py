"""
Training of the learned temporal prior used by DME v2 (dme/dme_prior.py, weights dme/weights/temporal_prior.pt).

The network (dme.dme_prior.Smoother3) is trained only on synthetic sequences: drift traces from a broad generator
(polynomials, slow sinusoids, random walks, vibrations, steps, Gaussian-process drift, damped oscillations,
piecewise-linear drift) plus measurement noise with a random per-segment scale, random miscalibration (0.5-3x), AR(1)
correlation (up to 0.85, as measured for multi-frame binding events), occasional outliers and missing segments. The
loss is the per-sequence RMS error of the denoised trace (mean removed). Sequence lengths 32-12000 segments.

    python training/train_temporal_prior.py --steps 15000 --out dme/weights/temporal_prior.pt    # ~1 h on an RTX 3090
"""
import os, math, time, argparse
import numpy as np
import torch
import torch.nn as nn

from dme.dme_prior import Smoother3

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(os.path.dirname(HERE), 'dme', 'weights', 'temporal_prior.pt')


def _drift_families(B, T, g, dev):
    """[B, T] drift sequences in units of the median measurement noise. Broad mixture of families + amplitudes."""
    t = torch.linspace(0, 1, T, device=dev)[None]
    x = torch.zeros(B, T, device=dev)
    def U(lo, hi, shape=(B, 1)):
        return lo + (hi - lo) * torch.rand(*shape, device=dev, generator=g)
    def logU(lo, hi, shape=(B, 1)):
        return torch.exp(U(math.log(lo), math.log(hi), shape))
    on = lambda p: (torch.rand(B, 1, device=dev, generator=g) < p).float()
    # smooth: polynomial + few low-frequency sinusoids
    A = logU(0.1, 3000)
    x += A * (U(-1, 1) * t + U(-1, 1) * t ** 2 + U(-1, 1) * t ** 3)
    for _ in range(3):
        x += on(0.6) * logU(0.05, 1000) * torch.sin(6.283 * U(0.2, 6) * t + U(0, 6.283))
    # random walk
    x += on(0.5) * logU(0.01, 3.0) * torch.cumsum(torch.randn(B, T, device=dev, generator=g), 1)
    # vibration: sinusoid with period 2..T/3 segments, possibly amplitude-modulated
    per = logU(2.0, max(3.0, T / 3))
    vib = logU(0.05, 30) * torch.sin(6.283 * torch.arange(T, device=dev)[None] / per + U(0, 6.283))
    vib = vib * (1 + on(0.3) * U(-1, 1) * torch.sin(6.283 * U(0.2, 3) * t))
    x += on(0.4) * vib
    # steps
    nst = torch.randint(0, 6, (B,), device=dev, generator=g)
    for k in range(5):
        pos = torch.randint(1, T, (B, 1), device=dev, generator=g)
        h = logU(0.1, 50) * torch.sign(torch.randn(B, 1, device=dev, generator=g))
        x += (k < nst)[:, None].float() * h * (torch.arange(T, device=dev)[None] >= pos).float()
    return x - x.mean(1, keepdim=True)


def random_drift(B, T, g, dev):
    x = _drift_families(B, T, g, dev)
    t = torch.linspace(0, 1, T, device=dev)[None]
    U = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, device=dev, generator=g)
    logU = lambda lo, hi: torch.exp(U(math.log(lo), math.log(hi)))
    on = lambda p: (torch.rand(B, 1, device=dev, generator=g) < p).float()
    # GP with RBF kernel via random Fourier features
    ell = logU(3.0 / T, 0.5)
    w = torch.randn(B, 64, device=dev, generator=g) / ell
    b = torch.rand(B, 64, device=dev, generator=g) * 6.283
    a = torch.randn(B, 64, device=dev, generator=g) * math.sqrt(2 / 64)
    gp = (a[:, :, None] * torch.cos(w[:, :, None] * t[:, None, :] * 6.283 + b[:, :, None])).sum(1)
    x += on(0.4) * logU(0.05, 300) * gp
    # damped oscillation
    per = logU(4.0, max(5.0, T / 2)); tau = logU(0.02, 1.0); t0 = U(0, 0.8)
    tt = (t - t0).clamp(min=0)
    x += on(0.2) * logU(0.1, 30) * torch.exp(-tt / tau) * torch.sin(6.283 * tt * T / per) * (t >= t0).float()
    # piecewise linear (random kinks)
    nk = 6
    kinks = torch.sort(torch.rand(B, nk, device=dev, generator=g), 1).values
    slopes = torch.randn(B, nk, device=dev, generator=g)
    pl = (slopes[:, :, None] * (t[:, None, :] - kinks[:, :, None]).clamp(min=0)).sum(1)
    x += on(0.3) * logU(0.1, 300) * pl
    return x - x.mean(1, keepdim=True)


def ar1(w, rho):
    """AR(1) filter along dim 1 with per-row rho, unit stationary variance (chunked recursion)."""
    e = torch.empty_like(w)
    e[:, 0] = w[:, 0]
    sq = torch.sqrt(1 - rho ** 2)
    for i in range(1, w.shape[1]):
        e[:, i] = rho * e[:, i - 1] + sq * w[:, i]
    return e


def make_batch(B, T, g, dev):
    x = random_drift(B, T, g, dev)
    lvl = torch.exp(0.6 * torch.randn(B, 1, device=dev, generator=g))
    s = torch.exp(lvl * 0.3 * torch.randn(B, T, device=dev, generator=g))
    tt = torch.linspace(0, 1, T, device=dev)[None]
    s = s * torch.exp(0.5 * torch.sin(6.283 * torch.rand(B, 1, device=dev, generator=g) * 3 * tt + 6.283 * torch.rand(B, 1, device=dev, generator=g)))
    big = torch.rand(B, T, device=dev, generator=g) < 0.01 * torch.rand(B, 1, device=dev, generator=g)
    s = torch.where(big, s * 30, s)
    s = s / s.median(1, keepdim=True).values
    valid = torch.rand(B, T, device=dev, generator=g) > 0.03 * torch.rand(B, 1, device=dev, generator=g)
    k = torch.exp(torch.empty(B, 1, device=dev).uniform_(math.log(0.5), math.log(3.0), generator=g))
    rho = torch.rand(B, device=dev, generator=g) * 0.85 * (torch.rand(B, device=dev, generator=g) < 0.7).float()
    e = ar1(torch.randn(B, T, device=dev, generator=g), rho) * s * k.sqrt()
    out = torch.rand(B, T, device=dev, generator=g) < 0.002
    e = torch.where(out, e * 8, e)
    return x + e, s, valid, x


def train(steps, out, lr=1e-3, seed=2):
    dev = torch.device('cuda')
    g = torch.Generator(device=dev).manual_seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Smoother3().to(dev)
    nn.init.zeros_(net.out.weight); nn.init.zeros_(net.out.bias)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    t0, hist = time.time(), []
    for it in range(steps):
        T = int(np.exp(rng.uniform(np.log(32), np.log(12000))))
        B = int(np.clip(200000 // T, 8, 64))
        with torch.no_grad():
            z, s, valid, x = make_batch(B, T, g, dev)
        xh = net(z, s, valid)
        err = (xh - x) - (xh - x).mean(1, keepdim=True)
        loss = torch.sqrt((err ** 2).mean(1) + 1e-8).mean()
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sch.step()
        hist.append(loss.item())
        if (it + 1) % 250 == 0:
            print(f"step {it + 1} rmse {np.mean(hist[-250:]):.4f} {time.time() - t0:.0f}s", flush=True)
        if (it + 1) % 1000 == 0 or it + 1 == steps:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torch.save(net.state_dict(), out)
    return net


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--out', default=WEIGHTS)
    a = ap.parse_args()
    train(a.steps, a.out)
