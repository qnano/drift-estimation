"""
DME v2 with a learned temporal prior (dme_estimate(method='v2') when framesperbin is not given).

With a fixed framesperbin, drift is a spline with one knot per framesperbin frames, so the user has to choose the
temporal resolution. Without it, the resolution is chosen from the data and the spline is replaced by a learned
smoother:

  1. Time is cut into segments of L frames. For a current drift estimate, the DME objective gives, for every segment,
     the gradient g_t and the DxD curvature H_t with respect to shifting that segment alone (all other localizations
     act as reference). This gives a Newton measurement of the segment's drift, z_t = d_t + (-H_t)^-1 g_t, with
     information J_t = -N H_t.
  2. A small 1D convolutional network (dilated residual blocks, ~0.3M parameters, trained on simulated drift traces
     and correlated measurement noise) turns the noisy per-segment measurements into a drift trace. This is repeated
     at kernel widths 4, 2, 1, 1, 1 x CRLB.
  3. The segment length L (6-192 frames) and the frames per bin of the DME v2 start (the ~20k-localizations-per-bin
     rule and 10x finer) are chosen by a spatial cross-fit: the drift estimated from one half of a checkerboard split
     of the field of view is scored by the DME objective of the other half, and vice versa. The fixed-spline
     estimate (no smoother) is one of the candidates, so it falls back to that when the smoother does not help.

The network weights are in dme/weights/temporal_prior.pt. Uses the mean of the given CRLBs.
"""
import itertools
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .dme_grid import DMEGrid, dme_grid_estimate, choose_backend
from .dme_gpu import cupy_available, mem_scale

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'temporal_prior.pt')
LAGS = (1, 2, 4, 8)
_nets = {}


# ---- the network --------------------------------------------------------------------------------------------------

class _Block(nn.Module):
    def __init__(self, C, dil):
        super().__init__()
        self.n = nn.GroupNorm(8, C)
        self.c1 = nn.Conv1d(C, C, 3, padding=dil, dilation=dil)
        self.c2 = nn.Conv1d(C, C, 1)

    def forward(self, x):
        return x + self.c2(Fn.gelu(self.c1(Fn.gelu(self.n(x)))))


def _noise_feats(zf, s_used):
    """log robust noise-scale estimates from second differences at several lags -> [B, len(LAGS)]."""
    out = []
    for m in LAGS:
        if zf.shape[1] > 2 * m:
            v2 = (zf[:, m:-m] - 0.5 * (zf[:, :-2 * m] + zf[:, 2 * m:])) ** 2 / (1.5 * s_used[:, m:-m] ** 2)
            out.append(torch.log((v2.median(1).values / 0.4549).clamp(1e-3, 1e4)))
        else:
            out.append(out[-1] if out else torch.zeros(zf.shape[0], device=zf.device))
    return torch.stack(out, 1)


class Smoother3(nn.Module):
    """Input: per-segment measurements z [B, T], their standard deviations s [B, T] and validity [B, T].
    Output: smoothed drift [B, T]."""

    def __init__(self, C=64, dils=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512)):
        super().__init__()
        self.inp = nn.Conv1d(3 + len(LAGS), C, 1)
        self.blocks = nn.Sequential(*[_Block(C, d) for d in dils])
        self.out = nn.Conv1d(C, 1, 1)

    @staticmethod
    def features(z, s, valid):
        B, T = z.shape
        smed = torch.stack([s[b][valid[b]].median() if valid[b].any() else s[b].median() for b in range(B)])[:, None]
        valid = valid & (s < 10 * smed)
        zf = z.clone()
        idx = np.arange(T, dtype=np.float64)
        for b in torch.nonzero((~valid).any(1)).flatten().tolist():
            v = valid[b]
            if v.sum() >= 2:
                vn = v.cpu().numpy()
                zf[b] = torch.as_tensor(np.interp(idx, idx[vn], z[b, v].cpu().numpy()), device=z.device, dtype=z.dtype)
        s_used = torch.where(valid, s, smed * 30)
        nf = _noise_feats(zf, s_used)
        dz = torch.diff(zf, dim=1, prepend=zf[:, :1])
        f = torch.cat([torch.stack([torch.asinh(dz / smed), torch.log(s_used / smed), valid.float()], 1),
                       nf[:, :, None].expand(-1, -1, T)], 1)
        return f, zf, s_used

    def forward(self, z, s, valid):
        f, zf, s_used = self.features(z, s, valid)
        e = self.out(self.blocks(self.inp(f)))[:, 0]
        return zf - e * s_used


def _get_net(device):
    key = str(device)
    if key not in _nets:
        net = Smoother3()
        net.load_state_dict(torch.load(WEIGHTS, map_location='cpu', weights_only=True))
        _nets[key] = net.to(device).eval()
    return _nets[key]


# ---- per-segment DME measurements ---------------------------------------------------------------------------------

def _min_max_eig(M):
    """M [T,D,D] symmetric -> (min eig, max eig) [T]. Closed form for D=2, chunked eigvalsh otherwise."""
    if M.shape[1] == 2:
        a, b, c = M[:, 0, 0], M[:, 0, 1], M[:, 1, 1]
        m = 0.5 * (a + c)
        r = torch.sqrt((0.5 * (a - c)) ** 2 + b * b)
        return m - r, m + r
    lo, hi = [], []
    for i in range(0, len(M), 4096):
        ev = torch.linalg.eigvalsh(M[i:i + 4096])
        lo.append(ev[:, 0]); hi.append(ev[:, -1])
    return torch.cat(lo), torch.cat(hi)


@torch.no_grad()
def _pair_sums(u, fr, R, w=None, need_hess=False, budget=1 << 23):
    """Spatial hash sums over other-frame neighbours j of each point i, |d| < R with d = u_i - u_j, K = exp(-|d|^2/2):
    S0_i = sum w_j K, S1_i = sum w_j dK/du_i = -sum w_j K d, and (need_hess) S2_i = sum K (d d^T - I).
    Points are sorted by cell, the 3^D neighbour cells are found with searchsorted, and query points are processed in
    chunks of bounded candidate count, so memory is O(N + budget) however many pairs there are."""
    N, D = u.shape
    dev = u.device
    budget = max(int(budget * mem_scale(dev)), 1 << 16)
    cell = torch.floor((u - u.min(0).values) / R).long() + 1
    dims = (cell.max(0).values + 2).tolist()
    mult = [int(math.prod(dims[:d])) for d in range(D)]
    key = (cell * torch.tensor(mult, device=dev)).sum(1)
    skey, order = torch.sort(key)
    okeys = [sum(o[d] * mult[d] for d in range(D)) for o in itertools.product([-1, 0, 1], repeat=D)]
    S0 = torch.zeros(N, device=dev)
    S1 = torch.zeros(N, D, device=dev)
    S2 = torch.zeros(N, D, D, device=dev) if need_hess else None
    eye = torch.eye(D, device=dev)
    start, chunk = 0, min(N, 4096)
    while start < N:
        end = min(N, start + chunk)
        qi, qk = order[start:end], skey[start:end]
        total = 0
        for ok in okeys:
            lo = torch.searchsorted(skey, qk + ok, side='left')
            cnt = torch.searchsorted(skey, qk + ok, side='right') - lo
            tot = int(cnt.sum())
            if tot == 0:
                continue
            total += tot
            a = torch.repeat_interleave(torch.arange(end - start, device=dev), cnt)
            j = order[torch.repeat_interleave(lo - (cnt.cumsum(0) - cnt), cnt) + torch.arange(tot, device=dev)]
            i = qi[a]
            d = u[i] - u[j]
            d2 = (d * d).sum(1)
            m = (d2 < R * R) & (fr[i] != fr[j])
            i, j, d, d2 = i[m], j[m], d[m], d2[m]
            K = torch.exp(-0.5 * d2)
            if need_hess:
                S2.index_add_(0, i, (d[:, :, None] * d[:, None, :] - eye) * K[:, None, None])
            if w is not None:
                K = K * w[j]
            S0.index_add_(0, i, K)
            S1.index_add_(0, i, -d * K[:, None])
        chunk = int(max(256, min(1 << 20, budget * (end - start) / max(total, 1))))
        start = end
    return S0, S1, S2


class DMEPairs:
    """Exact DME objective (spatial hash, no grid) with one drift value per frame, and the same score_and_grad
    interface as DMEGrid(framesperbin=1). Used where a grid would be large and mostly empty, e.g. 3D data with a
    z range of many CRLBs."""

    def __init__(self, positions, framenum, crlb, initial_drift, numframes, device='cuda', kernel_radius=4.0):
        dev = torch.device(device)
        self.scale = torch.as_tensor(np.asarray(crlb, np.float32), device=dev)
        self.v = torch.as_tensor(np.asarray(positions, np.float32), device=dev) / self.scale
        self.fr = torch.as_tensor(np.asarray(framenum), dtype=torch.long, device=dev)
        self.N, self.D = self.v.shape
        self.F = int(numframes)
        self.state = torch.as_tensor(np.asarray(initial_drift, np.float32)[:self.F], device=dev) / self.scale
        self.R = kernel_radius

    @torch.no_grad()
    def score_and_grad(self, state, need_grad=True, need_hess=False):
        u = self.v - state[self.fr]
        S0, gG, Hg = _pair_sums(u, self.fr, self.R, need_hess=need_hess)
        Z = 1.0 + S0
        score = torch.log(Z.double()).mean().float()
        if not need_grad:
            return score, None
        norm = 1.0 / Z
        _, Gw, _ = _pair_sums(u, self.fr, self.R, w=norm)      # other-end term: sum_j norm_j dK(u_i - u_j)/du_i
        dSdu = norm[:, None] * gG + Gw
        grad = torch.zeros(self.F, self.D, device=u.device).index_add_(0, self.fr, -dSdu) / self.N
        hess = None
        if need_hess:     # as DMEGrid: own-end curvature of log Z_i, x2 for the other end
            Hi = Hg * norm[:, None, None] - gG[:, :, None] * gG[:, None, :] * (norm ** 2)[:, None, None]
            hess = torch.zeros(self.F, self.D, self.D, device=u.device).index_add_(0, self.fr, 2 * Hi) / self.N
        return score, grad, hess


# Without CuPy the exact pair sums are plain PyTorch (DMEPairs), much slower per pair than the fused CuPy kernel that
# choose_backend's cost model was measured with
PAIR_COST_FACTOR_TORCH = 50.0


class _CupyFrames:
    """Exact pair sums with the fused CuPy kernels (dme_cupy), one drift value per frame; DMEGrid interface."""

    def __init__(self, pos, fr, sig, d_frames, F, device):
        from .dme_cupy import DMECupy
        # cutoff 4 sigma, as the kernel radius of DMEGrid, so the objective does not depend on the evaluator chosen
        self.e = DMECupy(pos, fr, sig, 1, d_frames, F, device=device, search_mult=4.0)
        self.state, self.N = self.e.state, self.e.N

    def score_and_grad(self, state, need_grad=True, need_hess=False):
        if need_hess:
            return self.e.score_grad_hess(state)
        if not need_grad:
            return self.e.score_and_grad(state, need_grad=False)
        return (*self.e.score_and_grad(state), None)


def per_frame_evaluator(pos, fr, sig, F, d_frames, device, backend='auto'):
    """DME objective with one drift value per frame: the density grid, exact pair sums (spatial hash), or whichever
    the cost model says is cheaper (backend='auto', default). The grid's cost grows with the area or volume covered (in
    CRLB units), the pair sum's with the number of neighbour pairs: dense data with a large CRLB uses the grid; very
    precise localizations over a large field of view, sparse data, and thick 3D data use the pair sums."""
    fused = cupy_available(device)
    if backend == 'auto':
        _, c = choose_backend(pos, sig, device=device)
        backend = 'grid' if c['t_grid'] < (1.0 if fused else PAIR_COST_FACTOR_TORCH) * c['t_exact'] else 'exact'
    if backend == 'grid':
        return DMEGrid(pos, fr, sig, 1, d_frames, F, device=device)
    if fused:
        return _CupyFrames(pos, fr, sig, d_frames, F, device)
    return DMEPairs(pos, fr, sig, d_frames, F, device=device)


def seg_measurements(pos, fr, sig, F, d_frames, L, device, trust=1.5, backend='auto'):
    """Newton measurements of the drift of each L-frame segment at the current drift d_frames [F, D] (px).
    Returns z [T, D] px and information J [T, D, D] px^-2 (0 for segments without a valid measurement)."""
    e = per_frame_evaluator(pos, fr, sig, F, d_frames, device, backend)
    _, g, H = e.score_and_grad(e.state, need_hess=True)
    T = (F + L - 1) // L
    seg = torch.arange(F, device=g.device) // L
    D = g.shape[1]
    gs = torch.zeros(T, D, device=g.device, dtype=torch.float64).index_add_(0, seg, g.double())
    Hs = torch.zeros(T, D, D, device=g.device, dtype=torch.float64).index_add_(0, seg, H.double())
    J = -Hs * e.N
    eye = torch.eye(D, device=J.device, dtype=J.dtype)
    Jr = J + eye * (J.diagonal(dim1=1, dim2=2).mean(1).clamp(min=1e-9) * 1e-3 + 1e-9)[:, None, None]
    step = torch.linalg.solve(Jr, (gs * e.N)[..., None])[..., 0]
    lo, hi = _min_max_eig(-Hs)
    valid = lo > 1e-6 * hi.clamp(min=1e-30)
    nrm = step.norm(dim=1, keepdim=True)
    step = step * torch.clamp(trust / nrm.clamp(min=1e-12), max=1.0)
    J = torch.where(valid[:, None, None], J, torch.zeros_like(J))
    dseg = torch.as_tensor(d_frames, device=g.device, dtype=torch.float64)
    cnt = torch.bincount(seg, minlength=T).double()
    dcur = torch.zeros(T, D, device=g.device, dtype=torch.float64).index_add_(0, seg, dseg) / cnt[:, None]
    s = torch.as_tensor(sig, device=g.device, dtype=torch.float64)
    z = dcur + step * s
    info = J / (s[:, None] * s[None, :])
    return z.cpu().numpy(), info.cpu().numpy()


@torch.no_grad()
def smooth_net(z, info, device):
    net = _get_net(device)
    T, D = z.shape
    if D > 2:
        valid_all = np.array([np.linalg.det(m) > 0 for m in info])
    else:
        valid_all = info[:, 0, 0] * info[:, 1, 1] - info[:, 0, 1] ** 2 > 0
    var = np.full((D, T), np.nan)
    if valid_all.any():
        inv = np.linalg.inv(info[valid_all])
        for d in range(D):
            var[d, valid_all] = inv[:, d, d]
    valid = np.isfinite(var) & (var > 0)
    s = np.sqrt(np.where(valid, var, np.nanmax(np.where(valid, var, np.nan)) if valid.any() else 1.0))
    zt = torch.as_tensor(z.T.copy(), device=device, dtype=torch.float32)
    st = torch.as_tensor(s, device=device, dtype=torch.float32)
    vt = torch.as_tensor(valid, device=device)
    off = torch.stack([zt[d][vt[d]].mean() if vt[d].any() else zt[d].mean() for d in range(D)])[:, None]
    x = net(zt - off, st, vt) + off
    return x.T.double().cpu().numpy()


def seg_to_frames(xseg, L, F):
    T = len(xseg)
    tc = (np.arange(T) + 0.5) * L - 0.5
    f = np.arange(F)
    return np.stack([np.interp(f, tc, xseg[:, d]) for d in range(xseg.shape[1])], 1)


def refine(pos, frame, sig, F, d, L, scales, device, backend='auto'):
    """Learned-prior refinement of drift d [F, D] with segment length L, coarse to fine over kernel widths."""
    if (F + L - 1) // L < 24:
        return d
    for sc in scales:
        z, info = seg_measurements(pos, frame, sig * sc, F, d, L, device, backend=backend)
        d = seg_to_frames(smooth_net(z, info, device), L, F)
        d = d - d.mean(0)
    return d


def dme_score(pos, frame, sig, F, d, device, backend='auto'):
    e = per_frame_evaluator(pos, frame, sig, F, d, device, backend)
    return e.score_and_grad(e.state, need_grad=False)[0].item()


def checkerboard(pos, ntiles=8, min_frac=0.3):
    """Spatial checkerboard split of the field of view; tries tile offsets / counts until both halves hold at least
    min_frac of the localizations."""
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    best = None
    for n in (ntiles, ntiles + 1, ntiles - 1, ntiles + 3):
        ts = (hi - lo).max() / max(n, 2)
        for off in (0.0, 0.5, 0.25):
            t = np.floor((pos[:, :2] - lo) / ts + off).astype(np.int64)
            A = ((t[:, 0] + t[:, 1]) % 2) == 0
            frac = min(A.mean(), 1 - A.mean())
            if frac >= min_frac:
                return A
            if best is None or frac > best[0]:
                best = (frac, A)
    return best[1]


# ---- estimator ----------------------------------------------------------------------------------------------------

def dme_prior_estimate(positions, framenum, crlb, numframes=None, device='cuda', scales=(4, 2, 1, 1, 1),
                    L_frames=(1, 2, 3, 6, 12, 24, 48, 96, 192), fpb_divs=(1, 10), ntiles=8, min_locs_per_seg=30,
                    verbose=True, return_info=False, isotropic3d=False):
    """Drift [F, D] in the units of positions (corrected = positions - drift[framenum]).
    crlb: [N, D] or [D]; the mean CRLB is used.
    Every evaluation of the objective uses the density grid or exact pair sums (spatial hash), whichever is cheaper.
    Candidates of the cross-fit: the fixed spline (two bin lengths), the learned prior on segments of 1-192 frames, and,
    with at least min_locs_per_seg localizations per frame, a drift value per frame without prior.
    3D (default): x, y are estimated on the x-y projection and then z on the x-z projection.
    isotropic3d=True: the objective is evaluated in 3D, with a 3D grid or exact pair sums (spatial hash), whichever is
    cheaper. Only worth it when binding sites at different depths overlap in projection (very dense 3D samples): then
    it is up to 2-3x more precise, but 10-30x slower."""
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    frame = np.ascontiguousarray(framenum, dtype=np.int32)
    D = pos.shape[1]
    F = int(numframes if numframes is not None else frame.max() + 1)
    crlb = np.asarray(crlb, np.float32)
    crlb = np.broadcast_to(crlb[:D], pos.shape) if crlb.ndim == 1 else crlb.reshape(-1, D)
    if D == 3 and not isotropic3d:
        kw = dict(numframes=F, device=device, scales=scales, L_frames=L_frames, fpb_divs=fpb_divs, ntiles=ntiles,
                  min_locs_per_seg=min_locs_per_seg, verbose=verbose, return_info=True)
        dxy, info_xy = dme_prior_estimate(pos[:, :2], frame, crlb[:, :2], **kw)
        s = float(crlb[:, 0].mean() / crlb[:, 2].mean())   # z in units where its precision equals x's
        xz = np.stack([pos[:, 0] - dxy[frame, 0], pos[:, 2] * s], 1)
        dxz, info_z = dme_prior_estimate(xz, frame, np.stack([crlb[:, 0], crlb[:, 2] * s], 1), **kw)
        d = np.concatenate([dxy, dxz[:, 1:2] / s], 1).astype(np.float32)
        return (d, dict(xy=info_xy, z=info_z)) if return_info else d
    N = len(pos)
    sig = crlb.mean(0).astype(np.float32)
    be = 'auto'
    rule = int(np.clip(round(20000 * F / N), 10, 2000))
    fpbs = sorted({max(10, rule // dv) for dv in fpb_divs}, reverse=True)
    Ls = sorted({L for L in L_frames if L * N / F >= min_locs_per_seg and (F + L - 1) // L >= 24})
    A = checkerboard(pos, ntiles)
    halves = [(pos[A], frame[A]), (pos[~A], frame[~A])]
    v2 = lambda p, f, fpb, init='auto': dme_grid_estimate(p, f, sig, fpb, init=init, estimatePrecision=False,
                                                         numframes=F, device=device, verbose=False)
    cv, starts = {}, {}
    prev = None          # the multi-start (AIM, RCC, zero) runs once per half; later fits start from its result
    if N / F >= min_locs_per_seg:
        # enough localizations per frame for a drift value per frame: candidate without temporal prior
        d1 = [v2(p, f, 1) for p, f in halves]
        starts[1], prev = d1, d1
        cv[('dme', 1, 0)] = (dme_score(*halves[1], sig, F, d1[0], device, be) +
                             dme_score(*halves[0], sig, F, d1[1], device, be))
    for fpb in fpbs:
        d0 = [v2(p, f, fpb, 'auto' if prev is None else np.asarray(pv, np.float32))
              for (p, f), pv in zip(halves, prev or [None, None])]
        starts[fpb], prev = d0, d0
        cv[('dme', fpb, 0)] = (dme_score(*halves[1], sig, F, d0[0], device, be) +
                               dme_score(*halves[0], sig, F, d0[1], device, be))
        for L in Ls:
            dh = [refine(p, f, sig, F, d, L, scales, device, be) for (p, f), d in zip(halves, d0)]
            cv[('smooth', fpb, L)] = (dme_score(*halves[1], sig, F, dh[0], device, be) +
                                      dme_score(*halves[0], sig, F, dh[1], device, be))
    kind, fpb, L = max(cv, key=cv.get)
    if verbose:
        print("DME v2: drift per frame, no temporal prior (enough localizations per frame)" if fpb == 1 else
              f"DME v2: start = fixed spline at {fpb} frames/bin, " +
              (f"learned prior with {L}-frame segments" if kind == 'smooth' else "no learned prior (the fixed spline is best)"),
              flush=True)
    d = v2(pos, frame, fpb, np.asarray(0.5 * (starts[fpb][0] + starts[fpb][1]), np.float32))   # same drift in both halves
    if kind == 'smooth':
        d = refine(pos, frame, sig, F, d, L, scales, device, be)
    d = np.asarray(d, np.float32)
    if return_info:
        return d, dict(start_framesperbin=fpb, segment_frames=L if kind == 'smooth' else None, cv=cv)
    return d
