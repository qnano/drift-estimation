"""
DME v2: same objective as DME (constant CRLB), evaluated through a blurred density grid instead of pair sums.

    score = mean_i log Z_i,   Z_i = 1 + sum_{j: frame_j != frame_i} K(u_i - u_j),   K(d) = exp(-|d|^2/2),  u = (x - drift)/sigma

Evaluation (all PyTorch, O(N + grid cells) per evaluation, independent of the number of neighbors):
  - G(x)  = sum_j K(x - u_j)            : cloud-in-cell splat of all points onto a grid (spacing h sigma), separable
                                          Gaussian blur (variance 1 - h^2/3, so splat+blur+interp ~ unit Gaussian),
                                          linear interpolation at u_i. Gradient grids use the derivative kernel.
  - Gw(x) = sum_j K(x - u_j) / Z_j      : same with weights 1/Z_j (the "other end" term of the gradient).
  - self term (j = i) is subtracted exactly through the discrete kernel; same-frame pairs (drift independent,
    precomputed once) are subtracted exactly.
  - large fields of view are processed in tiles with a halo; empty tiles are skipped.
Gradient:  dS/du_k = (1/N) sum_j (1/Z_k + 1/Z_j) grad K(u_k - u_j)  =  (1/N) [ grad G(u_k)/Z_k + grad Gw(u_k) ]  (minus self/same-frame)

Optimization:
  - coarse-to-fine over kernel width (scales 4, 2, 1 x CRLB); coarse levels use coarser grids, so they are cheap
  - L-BFGS preconditioned with a per-knot DxD block Hessian ("segment vs rest" curvature, AIM-like decomposition)
  - initialization by AIM-style intersection tracking (aim_torch) or RCC
"""
import math
import time
import itertools
import numpy as np
import torch
import torch.nn.functional as Fn

from .dme_gpu import catmull_rom_basis, sync, cupy_available


def _kernels(h, rk, dev):
    """1D kernels as function of offset (eval node - source node), in the tensor order conv1d expects."""
    sb2 = 1.0 - h * h / 3.0
    A = 1.0 / math.sqrt(sb2)
    delta = torch.arange(-rk, rk + 1, device=dev, dtype=torch.float32)
    x = delta * h
    k0 = A * torch.exp(-x * x / (2 * sb2))
    k1 = -x / sb2 * k0
    k2 = (x * x / sb2 ** 2 - 1.0 / sb2) * k0
    # conv1d: out[n] = sum_m in[n + m - rk] k[m]  ->  offset (n - j) = rk - m  ->  reverse
    rev = lambda k: torch.flip(k, [0])
    return dict(f0=k0, f1=k1, f2=k2, c0=rev(k0), c1=rev(k1), c2=rev(k2))


def _blur(grid, kerns, axes_kind):
    """grid [B, n1..nD]; axes_kind: tuple of 'c0'/'c1'/'c2' per axis. Separable 'same' convolution (zero padded)."""
    D = grid.dim() - 1
    out = grid
    for ax in range(D):
        k = kerns[axes_kind[ax]]
        rk = (len(k) - 1) // 2
        x = out.movedim(ax + 1, -1)
        shp = x.shape
        x = Fn.conv1d(x.reshape(-1, 1, shp[-1]), k.view(1, 1, -1), padding=rk)
        out = x.view(shp).movedim(-1, ax + 1)
    return out


class DMEGrid:
    def __init__(self, positions, framenum, crlb, framesperbin=1, initial_drift=None, numframes=None, device='cuda',
                 h=0.5, kernel_radius=4.0, max_tile_cells=1 << 26, batch_cells=1 << 26):
        dev = torch.device(device)
        self.dev = dev
        pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32, device=dev)
        self.N, self.D = pos.shape
        self.fr = torch.as_tensor(np.asarray(framenum), dtype=torch.long, device=dev)
        self.F = int(numframes if numframes is not None else self.fr.max().item() + 1)
        crlb = np.asarray(crlb, np.float32)
        assert crlb.ndim == 1, "DMEGrid supports constant CRLB only (use DMECupy for per-spot CRLB)"
        self.scale = torch.as_tensor(crlb, device=dev)
        self.v = pos / self.scale
        self.h = h
        self.rk = int(math.ceil(kernel_radius * math.sqrt(1 - h * h / 3) / h))
        self.kern = _kernels(h, self.rk, dev)
        self.halo = self.rk + 2
        self.max_tile_cells, self.batch_cells = max_tile_cells, batch_cells
        self._cornerset = torch.tensor(list(itertools.product([0, 1], repeat=self.D)), device=dev)
        self._tileoffs = torch.tensor(list(itertools.product([-1, 0, 1], repeat=self.D)), device=dev)

        self.fpb = framesperbin
        if framesperbin > 1:
            self.bidx, self.bw, self.K = catmull_rom_basis(self.F, framesperbin, dev)
        else:
            self.K = self.F
        d0 = np.zeros((self.F, self.D), np.float32) if initial_drift is None else np.asarray(initial_drift, np.float32)[:self.F]
        d0 = torch.as_tensor(d0, device=dev) / self.scale
        if framesperbin > 1:
            k = torch.arange(self.K, device=dev)
            st = (k * framesperbin - framesperbin // 2).clamp(0, self.F - 1)
            en = (st + framesperbin).clamp(max=self.F)
            cs = torch.cat([torch.zeros(1, self.D, device=dev), d0.cumsum(0)])
            self.state = (cs[en] - cs[st]) / (en - st)[:, None].float()
        else:
            self.state = d0.clone()
        self._same_frame_pairs(kernel_radius)
        self.stats = dict(t_eval=0.0, evals=0)

    # ------------------------------------------------------------------ drift representation
    def drift_per_frame(self, state):
        if self.fpb == 1:
            return state
        return (state[self.bidx] * self.bw[..., None]).sum(1)

    def frame_to_state(self, gf):
        if self.fpb == 1:
            return gf
        g = torch.zeros(self.K, *gf.shape[1:], device=self.dev, dtype=gf.dtype)
        w = self.bw.reshape(self.F, 4, *([1] * (gf.dim() - 1)))
        g.index_add_(0, self.bidx.reshape(-1), (gf[:, None] * w).reshape(-1, *gf.shape[1:]))
        return g

    def frame_to_state_sq(self, hf):
        """Diagonal-block mapping of per-frame Hessian blocks to knots (uses w^2; ignores knot coupling)."""
        if self.fpb == 1:
            return hf
        g = torch.zeros(self.K, *hf.shape[1:], device=self.dev, dtype=hf.dtype)
        w = (self.bw ** 2).reshape(self.F, 4, 1, 1)
        g.index_add_(0, self.bidx.reshape(-1), (hf[:, None] * w).reshape(-1, *hf.shape[1:]))
        return g

    # ------------------------------------------------------------------ same-frame pairs (drift independent)
    @torch.no_grad()
    def _same_frame_pairs(self, R):
        v, fr, D, dev = self.v, self.fr, self.D, self.dev
        c = torch.floor((v - v.min(0).values) / R).long() + 1
        dims = c.max(0).values + 2
        mult = [1]
        for d in range(1, D):
            mult.append(mult[-1] * int(dims[d - 1]))
        ncell = mult[-1] * int(dims[-1])
        key = fr * ncell + (c * torch.tensor(mult, device=dev)).sum(1)
        skey, order = torch.sort(key)
        offs = [sum(o[d] * mult[d] for d in range(D)) for o in itertools.product([-1, 0, 1], repeat=D)]
        offs = [o for o in offs if o >= 0]
        I, J = [], []
        ar = torch.arange(self.N, device=dev)
        for o in offs:
            nk = skey + o
            lo = torch.searchsorted(skey, nk, side='left')
            hi = torch.searchsorted(skey, nk, side='right')
            if o == 0:
                lo = ar + 1
            cnt = (hi - lo).clamp(min=0)
            tot = int(cnt.sum())
            if tot == 0:
                continue
            ii = torch.repeat_interleave(ar, cnt)
            jj = torch.repeat_interleave(lo - (cnt.cumsum(0) - cnt), cnt) + torch.arange(tot, device=dev)
            ii, jj = order[ii], order[jj]
            d = v[ii] - v[jj]
            m = (d * d).sum(1) < R * R
            I.append(ii[m]); J.append(jj[m])
        if I:
            I, J = torch.cat(I), torch.cat(J)
        else:
            I = J = torch.zeros(0, dtype=torch.long, device=dev)
        d = self.v[I] - self.v[J]
        self.sf_I, self.sf_J = I, J
        self.sf_K = torch.exp(-0.5 * (d * d).sum(1))
        self.sf_gK = -d * self.sf_K[:, None]          # grad_{u_i} K(u_i - u_j)
        # constant per-point sums
        self.sf_sumK = torch.zeros(self.N, device=dev).index_add_(0, I, self.sf_K).index_add_(0, J, self.sf_K)
        self.sf_sumgK = torch.zeros(self.N, D, device=dev).index_add_(0, I, self.sf_gK).index_add_(0, J, -self.sf_gK)
        self.n_sf = len(I)

    # ------------------------------------------------------------------ grid evaluation
    def _corners(self, p, slot, gcells, gmult):
        """CIC corner flat indices [M, 2^D] and weights [M, 2^D] for local continuous coords p [M, D]."""
        b = torch.floor(p)
        f = p - b
        b = b.long()
        flat0 = (b * gmult).sum(1)
        if slot is not None:
            flat0 = flat0 + slot * gcells
        offs = (self._cornerset * gmult).sum(1)                      # [2^D], itertools order (last dim fastest)
        w = torch.ones(len(p), 1, device=self.dev)
        for d in range(self.D):
            w = (w[:, :, None] * torch.stack([1 - f[:, d], f[:, d]], 1)[:, None, :]).reshape(len(p), -1)
        return flat0[:, None] + offs[None], w

    def _splat_blur_interp(self, B, gshape, m_idx, m_w, m_pt, o_idx, o_w, o_pt, weights_list, kinds_list, outs):
        gcells = int(math.prod(gshape))
        for wi, (wt, kinds) in enumerate(zip(weights_list, kinds_list)):
            g = torch.zeros(B * gcells, device=self.dev)
            src = m_w if m_pt is None else m_w
            vals = src * (wt[:, None] if m_pt is None else wt[m_pt][:, None])
            g.index_add_(0, m_idx.reshape(-1), vals.reshape(-1))
            g = g.view(B, *gshape)
            for ki, kd in enumerate(kinds):
                bl = _blur(g, self.kern, kd).reshape(-1)
                v = (bl[o_idx] * o_w).sum(1)
                if o_pt is None:
                    outs[wi][:, ki] = v
                else:
                    outs[wi][o_pt, ki] = v

    def _eval_grids(self, u, weights_list, kinds_list, need_self=True):
        """
        For each (weights, kinds) evaluate interpolated blurred grids at all points.
        Returns list (per weights) of [N, len(kinds)] tensors.
        """
        D, H, h = self.D, self.halo, self.h
        c = (u - u.min(0).values) / h + H
        ext = torch.ceil(c.max(0).values).long() + H + 2
        outs = [torch.empty(self.N, len(k), device=self.dev) for k in kinds_list]
        extl = ext.tolist()
        if float(math.prod(extl)) <= self.max_tile_cells:
            # single grid, no tiling bookkeeping
            gshape = extl
            gmult = torch.tensor([int(math.prod(gshape[d + 1:])) for d in range(D)], device=self.dev)
            idx, w = self._corners(c, None, 0, gmult)
            self._splat_blur_interp(1, gshape, idx, w, None, idx, w, None, weights_list, kinds_list, outs)
            return outs
        # banded along axis 0: sort by c0 once; each band's members (band + halo) and owners are contiguous slices
        other = int(math.prod(extl[1:]))
        T = max(self.max_tile_cells // other - 2 * H, 2 * H)
        order = torch.argsort(c[:, 0])
        cs = c[order]
        c0 = cs[:, 0].contiguous()
        nb = (extl[0] + T - 1) // T
        edges = torch.arange(nb + 1, device=self.dev, dtype=torch.float32) * T
        m_lo = torch.searchsorted(c0, edges[:-1] - H).tolist()
        m_hi = torch.searchsorted(c0, edges[1:] + H - 1).tolist()
        o_lo = torch.searchsorted(c0, edges[:-1]).tolist()
        o_hi = torch.searchsorted(c0, edges[1:]).tolist()
        gshape = [T + 2 * H] + extl[1:]
        gmult = torch.tensor([int(math.prod(gshape[d + 1:])) for d in range(D)], device=self.dev)
        wsorted = [w[order] for w in weights_list]
        souts = [torch.empty(self.N, len(k), device=self.dev) for k in kinds_list]
        for bi in range(nb):
            if o_hi[bi] <= o_lo[bi]:
                continue
            shift = torch.zeros(D, device=self.dev)
            shift[0] = bi * T - H
            mr = slice(m_lo[bi], m_hi[bi])
            orr = slice(o_lo[bi], o_hi[bi])
            midx, mw = self._corners(cs[mr] - shift, None, 0, gmult)
            oidx, ow = self._corners(cs[orr] - shift, None, 0, gmult)
            bouts = [torch.empty(o_hi[bi] - o_lo[bi], len(k), device=self.dev) for k in kinds_list]
            self._splat_blur_interp(1, gshape, midx, mw, None, oidx, ow, None,
                                    [ws[mr] for ws in wsorted], kinds_list, bouts)
            for so, bo in zip(souts, bouts):
                so[orr] = bo
        for o, so in zip(outs, souts):
            o[order] = so
        return outs

    def _self_terms(self, u, kinds):
        """Exact contribution of a point's own splat to its own interpolated value, closed form (separable):
        per dim  s_k(f) = ((1-f)^2 + f^2) k[0] + 2 f (1-f) k[1]; the odd derivative kernel gives s_1 = 0."""
        D, h, rk = self.D, self.h, self.rk
        f = (u - u.min(0).values) / h          # same cell phase as in _eval_grids
        f = f - torch.floor(f)
        a = (1 - f) ** 2 + f ** 2
        b = 2 * f * (1 - f)
        s = {'0': a * self.kern['f0'][rk] + b * self.kern['f0'][rk + 1],
             '2': a * self.kern['f2'][rk] + b * self.kern['f2'][rk + 1]}
        res = torch.zeros(self.N, len(kinds), device=self.dev)
        for ki, kd in enumerate(kinds):
            if any(k[1] == '1' for k in kd):
                continue
            val = torch.ones(self.N, device=self.dev)
            for d in range(D):
                val = val * s[kd[d][1]][:, d]
            res[:, ki] = val
        return res

    # ------------------------------------------------------------------ objective
    def _kinds(self, order):
        D = self.D
        v0 = tuple('c0' for _ in range(D))
        g = [tuple('c1' if a == d else 'c0' for a in range(D)) for d in range(D)]
        hs = []
        if order >= 2:
            for d in range(D):
                for e in range(d, D):
                    hs.append(tuple(('c2' if d == e else 'c1') if a in (d, e) else 'c0' for a in range(D)))
        return [v0] + g + hs

    @torch.no_grad()
    def score_and_grad(self, state, need_grad=True, need_hess=False):
        t0 = time.time()
        D = self.D
        dpf = self.drift_per_frame(state)
        u = self.v - dpf[self.fr]
        kinds = self._kinds(2 if need_hess else 1)
        ones = torch.ones(self.N, device=self.dev)
        (G,) = self._eval_grids(u, [ones], [kinds])
        S = self._self_terms(u, kinds)
        G = G - S
        Z = 1.0 + G[:, 0] - self.sf_sumK
        Z = Z.clamp(min=1.0)
        score = torch.log(Z.double()).mean().float()
        if not need_grad:
            sync(self.dev); self.stats['t_eval'] += time.time() - t0; self.stats['evals'] += 1
            return score, None
        norm = 1.0 / Z
        gG = G[:, 1:1 + D] - self.sf_sumgK
        # weighted density gradient (other end): grad Gw(u_k) = sum_j norm_j grad K(u_k - u_j)
        gkinds = kinds[1:1 + D]
        (Gw,) = self._eval_grids(u, [norm], [gkinds])
        Gw = Gw - norm[:, None] * S[:, 1:1 + D]
        sfw = torch.zeros(self.N, D, device=self.dev)
        sfw.index_add_(0, self.sf_I, self.sf_gK * norm[self.sf_J, None])
        sfw.index_add_(0, self.sf_J, -self.sf_gK * norm[self.sf_I, None])
        Gw = Gw - sfw
        dSdu = norm[:, None] * gG + Gw               # x N
        gf = torch.zeros(self.F, D, device=self.dev).index_add_(0, self.fr, -dSdu)
        grad = self.frame_to_state(gf) / self.N
        hess = None
        if need_hess:
            # d^2 log Z_i / du_i^2 = hess G / Z - grad G grad G^T / Z^2 ; x2 for the other-end term (approximation)
            Hg = torch.zeros(self.N, D, D, device=self.dev)
            k = 1 + D
            for d in range(D):
                for e in range(d, D):
                    Hg[:, d, e] = Hg[:, e, d] = G[:, k]
                    k += 1
            Hi = Hg * norm[:, None, None] - gG[:, :, None] * gG[:, None, :] * (norm ** 2)[:, None, None]
            hf = torch.zeros(self.F, D, D, device=self.dev).index_add_(0, self.fr, 2 * Hi)
            hess = self.frame_to_state_sq(hf) / self.N
        sync(self.dev)
        self.stats['t_eval'] += time.time() - t0
        self.stats['evals'] += 1
        return score, grad, hess

    # ------------------------------------------------------------------ optimizer
    def run_lbfgs(self, max_iter=200, precondition=True, tol=1e-10, history=20):
        x0 = self.state.clone()
        if precondition:
            _, _, Hs = self.score_and_grad(x0, need_hess=True)
            # SPD block preconditioner from -H (ascent problem), eigenvalues clamped
            ev, evec = torch.linalg.eigh(-Hs)
            floor = ev.clamp(min=0).mean() * 1e-2 + 1e-12
            ev = ev.clamp(min=float(floor))
            L = evec * ev.rsqrt()[:, None, :] @ evec.transpose(1, 2)      # P^{-1/2}
        else:
            L = None
        tf = (lambda y: x0 + torch.einsum('kab,kb->ka', L, y)) if L is not None else (lambda y: x0 + y)
        y = torch.zeros_like(x0, requires_grad=True)
        opt = torch.optim.LBFGS([y], lr=1, max_iter=max_iter, history_size=history, tolerance_grad=1e-12,
                                tolerance_change=tol, line_search_fn='strong_wolfe')
        self.nevals = 0

        def closure():
            opt.zero_grad()
            s, g, _ = self.score_and_grad(tf(y.detach()))
            gy = torch.einsum('kba,kb->ka', L, g) if L is not None else g
            y.grad = -gy
            self.nevals += 1
            return -s

        opt.step(closure)
        self.state = tf(y.detach())
        self.final_score = self.score_and_grad(self.state, need_grad=False)[0].item()
        self.iterations = self.nevals
        return self.result()

    def result(self):
        d = self.drift_per_frame(self.state) * self.scale
        d = d - d.mean(0)
        return d.cpu().numpy()


# ---------------------------------------------------------------------- backend choice
@torch.no_grad()
def choose_backend(positions, sigma, h=0.5, r=3.0, device='cuda'):
    """
    Cost model (RTX 3090, measured): exact pair kernel ~1.6e-11 s per candidate pair (3^D cells of size r sigma),
    grid ~1e-8 s per localization + ~2e-9 s per grid cell (value + D gradient blurs).
    Returns ('grid'|'exact', info).
    """
    u = torch.as_tensor(np.asarray(positions), dtype=torch.float32, device=device) / torch.as_tensor(sigma, device=device)
    D = u.shape[1]
    ext = (u.max(0).values - u.min(0).values)
    cells_grid = float(torch.prod(ext / h + 2 * 10))
    c = torch.floor((u - u.min(0).values) / r).long() + 1
    dims = (c.max(0).values + 2).tolist()
    mult = [int(math.prod(dims[:d])) for d in range(D)]
    key = (c * torch.tensor(mult, device=device)).sum(1)
    cnt = torch.bincount(key, minlength=int(math.prod(dims)))
    nb = torch.zeros_like(key, dtype=torch.float32)
    for o in itertools.product([-1, 0, 1], repeat=D):
        nb += cnt[key + sum(o[d] * mult[d] for d in range(D))].float()
    cand = float(nb.sum())
    t_exact = 1.6e-11 * cand * (2 if D == 2 else 1.3)
    t_grid = 1e-8 * len(u) + 2e-9 * cells_grid * (1 + D) / 3
    return ('grid' if t_grid < t_exact else 'exact'), dict(cand_per_loc=cand / len(u), cells=cells_grid,
                                                            t_exact=t_exact, t_grid=t_grid)


# ---------------------------------------------------------------------- pipeline
def dme_grid_estimate(positions, framenum, crlb, framesperbin, init='auto', scales=(4, 2, 1), coarse_fpb_mult=None,
                      init_interval=None, maxIterations=200, estimatePrecision=True, numframes=None, precondition=True,
                      backend='auto', device='cuda', verbose=True):
    """
    Drop-in style estimator. init: 'auto' (default: multi-start -- AIM, zero and RCC initializations are each run
    through the coarsest level and the one reaching the highest DME score at that level is continued; AIM alone can
    fail badly on sparse data with long binding events), 'aim' (intersection tracking), 'rcc', None/'zero', or an
    initial drift array [F, D].
    scales: kernel width multipliers of the CRLB, coarse to fine. coarse_fpb_mult: frames-per-bin multiplier
    per scale (default: max(1, scale) so coarse levels also use coarser time bins).
    """
    from .aim_torch import aim
    from .dme_pipeline import rcc_torch, rcc3d_torch
    positions = np.asarray(positions, np.float32)
    framenum = np.asarray(framenum, np.int32)
    crlb = np.asarray(crlb, np.float32)
    D = positions.shape[1]
    F = int(numframes if numframes is not None else framenum.max() + 1)
    sig = crlb.reshape(-1, D).mean(0) if crlb.ndim == 2 else crlb[:D]
    coarse_fpb_mult = coarse_fpb_mult or [max(1, s) for s in scales]
    t0 = time.time()

    def initial(kind, p, f):
        if kind == 'aim':
            iv = init_interval or int(np.clip(round(5000 * F / len(p)), framesperbin, max(framesperbin, F // 8)))
            d = np.zeros((F, D), np.float32)
            d[:, :2] = aim(p, f, interval=iv, intersect_d=2 * float(sig[:2].mean()), numframes=F, device=device)
            return d
        if kind == 'rcc':
            return (rcc3d_torch if D == 3 else rcc_torch)(p, f, 10, 2, numframes=F, device=device)
        return np.zeros((F, D), np.float32)

    def level(p, f, d, s, m, label):
        t = time.time()
        fpb_s = max(1, framesperbin * m) if framesperbin > 1 else 1
        be = backend if backend != 'auto' else choose_backend(p, sig * s, device=device)[0]
        if be == 'grid':
            e = DMEGrid(p, f, sig * s, fpb_s, d, F, device=device)
            d = e.run_lbfgs(maxIterations, precondition=precondition)
        else:
            if cupy_available(device):
                from .dme_cupy import DMECupy as Exact
            else:
                from .dme_gpu import DMEGPU as Exact      # pure PyTorch pair list (CPU, or no cupy)
            e = Exact(p, f, sig * s, fpb_s, d, F, device=device)
            d = e.run_lbfgs(maxIterations)
        if verbose:
            print(f"DMEv2[{label}] scale={s} backend={be} fpb={e.fpb} knots={e.K} evals={e.iterations} "
                  f"score={e.final_score:.5f} {time.time() - t:.2f}s", flush=True)
        return d, e.final_score

    def estimate(sel, label):
        p, f = positions[sel], framenum[sel]
        if len(p) < 2:
            raise ValueError(f'dme_grid_estimate: too few localizations ({len(p)})')
        levels = list(zip(scales, coarse_fpb_mult))
        if isinstance(init, np.ndarray):
            d = init[:F]
        elif init == 'auto':
            best = None
            for kind in ('aim', 'zero', 'rcc'):
                try:
                    dk, sk = level(p, f, initial(kind, p, f), *levels[0], f'{label}/init={kind}')
                except Exception as ex:          # e.g. RCC on degenerate data
                    if verbose:
                        print(f"DMEv2[{label}] init {kind} failed: {ex}")
                    continue
                if best is None or sk > best[1]:
                    best = (dk, sk, kind)
            if best is None:
                raise RuntimeError('dme_grid_estimate: all initializations failed (see messages with verbose=True)')
            d = best[0]
            if verbose:
                print(f"DMEv2[{label}] multi-start: using init={best[2]}", flush=True)
            levels = levels[1:]
        else:
            d = initial(init if init in ('aim', 'rcc') else 'zero', p, f)
        for s, m in levels:
            d, _ = level(p, f, d, s, m, label)
        return d

    drift = estimate(slice(None), 'full')
    if not estimatePrecision:
        return drift
    ax = int(np.argmax(np.var(positions[:, :2], 0)))
    s1 = positions[:, ax] > np.median(positions[:, ax])
    d1, d2 = estimate(s1, 'set1'), estimate(~s1, 'set2')
    if verbose:
        r = np.sqrt(((d1 - d1.mean(0) - d2 + d2.mean(0)) ** 2).mean(0))
        print(f"split-half RMSD {r} total {time.time() - t0:.1f}s")
    return drift, (d1, d2)
