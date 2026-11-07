"""
GPU-resident DME (drift estimation by entropy minimization), PyTorch implementation.

Same objective as the C++/CUDA DME (Cnossen et al. 2021):
    score = mean_i log Z_i,  Z_i = 1 + sum_{j in nb(i), frame_j != frame_i} exp(-KL(i,j))
maximized w.r.t. drift (per frame, or Catmull-Rom spline knots).

Differences from the original implementation:
 - Neighbor search uses a spatial hash (sorted cell keys + searchsorted) on the GPU instead of a CPU KD-tree.
   Everything stays on the GPU; only the drift trace is copied back at the end.
 - Pair list is a Verlet list (search radius + skin). It is rebuilt when drift moved more than skin/2,
   so the evaluated objective is exact (original rebuilds at 0.5 * search range, i.e. stale neighbors).
 - No maxNeighbors truncation (original truncates in arbitrary KD-tree order).
 - Pairs stored once (i<j); contributions scattered to both ends.
"""
import os
import time
import math
import itertools
import numpy as np
import torch


def sync(dev):
    """Wait for queued GPU work (for timing); no-op on CPU."""
    if torch.device(dev).type == 'cuda':
        torch.cuda.synchronize(dev)


def default_device(device=None):
    if device is not None:
        return device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


_cupy_ok = None


def cupy_available(device='cuda'):
    """True if the fused CuPy pair kernels can be used (CUDA device and a working cupy install)."""
    global _cupy_ok
    if torch.device(device).type != 'cuda' or os.environ.get('DME_NO_CUPY'):
        return False
    if _cupy_ok is None:
        try:
            import cupy
            cupy.cuda.runtime.getDeviceCount()
            _cupy_ok = True
        except Exception:
            _cupy_ok = False
    return _cupy_ok


def catmull_rom_basis(nframes, fpb, device):
    """Returns (idx[F,4], w[F,4], nknots) so that drift[f] = sum_j w[f,j] * knots[idx[f,j]]"""
    nknots = (nframes + fpb - 1) // fpb
    f = torch.arange(nframes, device=device, dtype=torch.float32)
    t = f / fpb
    b = torch.floor(t)
    t = t - b
    t2, t3 = t * t, t * t * t
    w = 0.5 * torch.stack([-t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2, -3 * t3 + 4 * t2 + t, t3 - t2], 1)
    idx = (b.long()[:, None] + torch.arange(-1, 3, device=device)[None]).clamp(0, nknots - 1)
    return idx, w, nknots


class DMEGPU:
    def __init__(self, positions, framenum, crlb, framesperbin=1, initial_drift=None, numframes=None,
                 device='cuda', search_mult=3.0, skin=0.5, pair_chunk=2_000_000):
        dev = torch.device(device)
        self.dev = dev
        pos = torch.as_tensor(np.asarray(positions), dtype=torch.float32, device=dev)
        self.N, self.D = pos.shape
        self.fr = torch.as_tensor(np.asarray(framenum), dtype=torch.long, device=dev)
        self.F = int(numframes if numframes is not None else self.fr.max().item() + 1)

        crlb = np.asarray(crlb, dtype=np.float32)
        self.const_crlb = crlb.ndim == 1
        if self.const_crlb:
            self.scale = torch.as_tensor(crlb, device=dev)
        else:
            c = torch.as_tensor(crlb, device=dev)
            self.scale = c.mean(0)
        # work in normalized coordinates: v = x / scale
        self.v = pos / self.scale
        if not self.const_crlb:
            self.s = torch.as_tensor(crlb, device=dev) / self.scale  # per spot sigma in normalized units
            self.inv_s2 = 1.0 / self.s ** 2
            self.logs2sum = torch.log(self.s ** 2).sum(1)

        self.r = search_mult
        self.skin = skin
        self.pair_chunk = pair_chunk

        self.fpb = framesperbin
        if framesperbin > 1:
            self.bidx, self.bw, self.K = catmull_rom_basis(self.F, framesperbin, dev)
        else:
            self.K = self.F

        d0 = np.zeros((self.F, self.D), np.float32) if initial_drift is None else np.asarray(initial_drift, np.float32)[:self.F]
        d0 = torch.as_tensor(d0, device=dev) / self.scale
        if framesperbin > 1:
            # knot k sits at frame k*fpb; take mean of initial drift around it
            k = torch.arange(self.K, device=dev)
            st = (k * framesperbin - framesperbin // 2).clamp(0, self.F - 1)
            en = (st + framesperbin).clamp(max=self.F)
            cs = torch.cat([torch.zeros(1, self.D, device=dev), d0.cumsum(0)])
            self.state = (cs[en] - cs[st]) / (en - st)[:, None].float()
        else:
            self.state = d0.clone()

        self.pairs = None
        self.pair_drift = None
        self.stats = dict(rebuilds=0, t_build=0.0, t_eval=0.0, npairs=0)

    # --- drift representation -------------------------------------------------
    def drift_per_frame(self, state):
        if self.fpb == 1:
            return state
        return (state[self.bidx] * self.bw[..., None]).sum(1)

    def frame_grad_to_state(self, gf):
        if self.fpb == 1:
            return gf
        g = torch.zeros(self.K, self.D, device=self.dev)
        g.index_add_(0, self.bidx.reshape(-1), (gf[:, None, :] * self.bw[..., None]).reshape(-1, self.D))
        return g

    # --- spatial hash neighbor search -----------------------------------------
    @torch.no_grad()
    def build_pairs(self, u):
        t0 = time.time()
        R = self.r + self.skin
        cell = torch.floor(u / R).long()
        cell -= cell.min(0).values
        dims = cell.max(0).values + 3  # +3 margin so that offsets -1..1 never wrap
        cell += 1
        mult = torch.ones(self.D, dtype=torch.long, device=self.dev)
        for d in range(1, self.D):
            mult[d] = mult[d - 1] * dims[d - 1]
        key = (cell * mult).sum(1)
        skey, order = torch.sort(key)

        offsets = torch.tensor(list(itertools.product([-1, 0, 1], repeat=self.D)), device=self.dev)
        okey = (offsets * mult).sum(1)
        # only half the offsets + own cell (i<j handled within own cell) -> each pair found once
        okey = okey[okey >= 0]

        R2 = R * R
        out_i, out_j = [], []
        N = self.N
        # process query points in chunks, sized so candidate pair arrays stay bounded
        start = 0
        chunk = max(1, min(N, 1 << 20))
        while start < N:
            end = min(N, start + chunk)
            qi = order[start:end]           # query points in sorted order (locality)
            qk = skey[start:end]
            cand_i, cand_j = [], []
            total = 0
            for ok in okey.tolist():
                nk = qk + ok
                lo = torch.searchsorted(skey, nk, side='left')
                hi = torch.searchsorted(skey, nk, side='right')
                if ok == 0:
                    # same cell: only partners after the query point in sorted order
                    lo = torch.arange(start, end, device=self.dev) + 1
                cnt = (hi - lo).clamp(min=0)
                tot = int(cnt.sum())
                if tot == 0:
                    continue
                ii = torch.repeat_interleave(torch.arange(len(qi), device=self.dev), cnt)
                base = torch.repeat_interleave(lo - (cnt.cumsum(0) - cnt), cnt)
                jj = order[base + torch.arange(tot, device=self.dev)]
                ii = qi[ii]
                m = (self.fr[ii] != self.fr[jj])
                dd = u[ii] - u[jj]
                m &= (dd * dd).sum(1) < R2
                cand_i.append(ii[m].int()); cand_j.append(jj[m].int())
                total += tot
            if cand_i:
                out_i.append(torch.cat(cand_i)); out_j.append(torch.cat(cand_j))
            # adapt chunk size to keep candidate arrays around pair_chunk*4
            if total > 0:
                per = total / (end - start)
                chunk = int(max(1024, min(1 << 22, 4 * self.pair_chunk / max(per, 1e-3))))
            start = end
        if out_i:
            self.pairs = (torch.cat(out_i).long(), torch.cat(out_j).long())
        else:
            e = torch.zeros(0, dtype=torch.long, device=self.dev)
            self.pairs = (e, e)
        self.stats['rebuilds'] += 1
        self.stats['npairs'] = len(self.pairs[0])
        sync(self.dev)
        self.stats['t_build'] += time.time() - t0

    # --- objective + gradient -------------------------------------------------
    @torch.no_grad()
    def score_and_grad(self, state, need_grad=True):
        dpf = self.drift_per_frame(state)
        if self.pair_drift is None or (dpf - self.pair_drift).abs().max() > self.skin * 0.5 / math.sqrt(self.D):
            self.build_pairs(self.v - dpf[self.fr])
            self.pair_drift = dpf.clone()
        t0 = time.time()
        u = self.v - dpf[self.fr]
        I, J = self.pairs
        r2 = self.r * self.r
        Z = torch.ones(self.N, device=self.dev)
        C = self.pair_chunk * 4
        P = len(I)
        cache = []
        for s in range(0, P, C):
            i, j = I[s:s + C], J[s:s + C]
            diff = u[i] - u[j]
            if self.const_crlb:
                d2 = (diff * diff).sum(1)
                e = torch.exp(-0.5 * d2) * (d2 < r2)
                Z.index_add_(0, i, e); Z.index_add_(0, j, e)
                cache.append((e,))
            else:
                si2, sj2 = self.s[i] ** 2, self.s[j] ** 2
                inside = ((diff * diff).sum(1) < r2).float()
                # KL(i||j) as in original ComputeKLDivergence(posA=i, crlbA=i, posB=j, crlbB=j)
                d2 = diff * diff
                ls = self.logs2sum[i] - self.logs2sum[j]
                kl_ij = 0.5 * ((si2 / sj2).sum(1) - ls + (d2 / sj2).sum(1) - self.D)
                kl_ji = 0.5 * ((sj2 / si2).sum(1) + ls + (d2 / si2).sum(1) - self.D)
                e_ij = torch.exp(-kl_ij) * inside
                e_ji = torch.exp(-kl_ji) * inside
                Z.index_add_(0, i, e_ij); Z.index_add_(0, j, e_ji)
                cache.append((e_ij, e_ji))
        score = torch.log(Z).mean()
        grad = None
        if need_grad:
            norm = 1.0 / Z
            g = torch.zeros(self.N, self.D, device=self.dev)
            for k, s in enumerate(range(0, P, C)):
                i, j = I[s:s + C], J[s:s + C]
                diff = u[i] - u[j]
                if self.const_crlb:
                    (e,) = cache[k]
                    pd = diff * ((norm[i] + norm[j]) * e)[:, None]
                else:
                    e_ij, e_ji = cache[k]
                    # d score/d drift contribution for spot i (and -for j), matches original UpdateDeltas_PerSpotCRLB
                    pd = diff * (self.inv_s2[j] * (norm[i] * e_ij)[:, None] + self.inv_s2[i] * (norm[j] * e_ji)[:, None])
                g.index_add_(0, i, pd); g.index_add_(0, j, -pd)
            gf = torch.zeros(self.F, self.D, device=self.dev)
            gf.index_add_(0, self.fr, g)
            grad = self.frame_grad_to_state(gf) / self.N
        sync(self.dev)
        self.stats['t_eval'] += time.time() - t0
        return score, grad

    # --- optimizers -----------------------------------------------------------
    def run_original_ascent(self, max_iter=2000, step=1e-6, callback=None):
        """Same adaptive step rule as the C++ code (gradient is not divided by N there, so scale step by N)."""
        state = self.state.clone()
        step = step * self.N
        last, prev_state, prev_delta, rejects = None, None, None, 0
        it = 0
        for it in range(max_iter):
            score, delta = self.score_and_grad(state)
            score = score.item()
            if last is None or score > last:
                prev_state, prev_delta, last = state.clone(), delta, score
                step *= 1.2; rejects = 0
            else:
                if step < 1e-16 or rejects == 10:
                    break
                state, delta = prev_state.clone(), prev_delta
                step *= 0.5; rejects += 1
            state = state + delta * step
            if callback: callback(it, last)
        self.state = prev_state
        self.iterations = it + 1
        self.final_score = last
        return self.result()

    def run_lbfgs(self, max_iter=200, tol=1e-9, history=20, callback=None):
        x = self.state.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([x], lr=1, max_iter=max_iter, history_size=history,
                                tolerance_grad=1e-12, tolerance_change=tol, line_search_fn='strong_wolfe')
        self.nevals = 0

        def closure():
            opt.zero_grad()
            s, g = self.score_and_grad(x.detach())
            x.grad = -g
            self.nevals += 1
            if callback: callback(self.nevals, s.item())
            return -s

        opt.step(closure)
        self.state = x.detach()
        self.final_score = self.score_and_grad(self.state, need_grad=False)[0].item()
        self.iterations = self.nevals
        return self.result()

    def result(self):
        d = self.drift_per_frame(self.state) * self.scale
        d = d - d.mean(0)
        return d.cpu().numpy()


def dme_gpu(positions, framenum, crlb, framesperbin=1, initial_drift=None, optimizer='lbfgs',
            coarse_framesperbin=None, coarse_sigma=None, max_iter=None, numframes=None, device='cuda', **kw):
    """Convenience wrapper mirroring dme_estimate's (optional) coarse-then-fine schedule."""
    if numframes is None:
        numframes = int(np.max(framenum)) + 1
    d = initial_drift
    def run(e):
        if optimizer == 'lbfgs':
            return e.run_lbfgs(max_iter=max_iter or 200)
        return e.run_original_ascent(max_iter=max_iter or 2000)
    if coarse_framesperbin is not None:
        e = DMEGPU(positions, framenum, np.asarray(coarse_sigma, np.float32), coarse_framesperbin, d, numframes, device, **kw)
        d = run(e)
    e = DMEGPU(positions, framenum, crlb, framesperbin, d, numframes, device, **kw)
    return run(e), e
