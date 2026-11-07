"""
Scalable GPU DME: spatial hash in PyTorch + fused pair kernels compiled at runtime by CuPy (NVRTC).
No stored pair list, O(N) memory. No nvcc / build step, works on Linux and Windows
(requires `pip install cupy-cuda12x[ctk]` or `cupy-cuda13x[ctk]` matching the driver).

Per evaluation:
  1. u = (x - drift[frame]) / sigma_mean      (normalized coordinates, cutoff radius r = 3)
  2. cell = floor(u / cellsize) on a dense padded grid, points sorted by cell, cell start/count via bincount+cumsum
  3. kernel Z:    one thread per point; a CUDA block holds points of a single cell, so all threads of a block walk
                  the same 3^D neighbor-cell ranges (broadcast loads).
                  Z_i = 1 + sum_j exp(-KL(i,j)) [|u_i-u_j| < r, frame_i != frame_j]
  4. kernel grad: same loop, accumulates d score / d u_i using 1/Z of both ends
The hash is rebuilt every evaluation (a sort of N keys), so the objective is exact at every step.
Each pair is visited from both ends (2x flops, but no atomics and no pair storage).
"""
import time
import itertools
import numpy as np
import torch
import cupy as cp

from .dme_gpu import DMEGPU, sync

_SRC = r'''
template<int D, bool PERSPOT, bool GRAD>
__global__ void pair_kernel(const float* __restrict__ u, const int* __restrict__ fr, const float* __restrict__ s,
                            const int* __restrict__ cstart, const int* __restrict__ ccount,
                            const int* __restrict__ blk_cell, const int* __restrict__ blk_start, const int* __restrict__ blk_end,
                            const long long* __restrict__ offs, int noff,
                            const float* __restrict__ norm, float* __restrict__ out, float r2)
{
    const int b = blockIdx.x;
    const long long cell = blk_cell[b];
    const int i = blk_start[b] + threadIdx.x;
    const bool active = i < blk_end[b];
    const int ii = active ? i : blk_start[b];   // inactive threads still loop (no divergence), store nothing

    float ui[D], si2[D], isi[D];
    for (int d = 0; d < D; d++) ui[d] = u[ii * D + d];
    const int fi = fr[ii];
    float lsi = 0.f;
    if (PERSPOT) for (int d = 0; d < D; d++) { float sv = s[ii * D + d]; si2[d] = sv * sv; isi[d] = 1.f / si2[d]; lsi += logf(si2[d]); }
    const float ni = GRAD ? norm[ii] : 0.f;

    float acc[D];
    for (int d = 0; d < D; d++) acc[d] = 0.f;
    float zsum = 0.f;

    for (int o = 0; o < noff; o++) {
        const long long nc = cell + offs[o];
        const int j0 = cstart[nc], j1 = j0 + ccount[nc];
        for (int j = j0; j < j1; j++) {
            float diff[D], d2 = 0.f;
            for (int d = 0; d < D; d++) { diff[d] = ui[d] - u[j * D + d]; d2 += diff[d] * diff[d]; }
            if (d2 >= r2 || fr[j] == fi) continue;
            if (!PERSPOT) {
                const float e = __expf(-0.5f * d2);
                if (GRAD) { const float w = (ni + norm[j]) * e; for (int d = 0; d < D; d++) acc[d] += diff[d] * w; }
                else zsum += e;
            } else {
                float sj2[D], isj[D], lsj = 0.f, tr_ij = 0.f, m_j = 0.f;
                for (int d = 0; d < D; d++) {
                    float sv = s[j * D + d]; sj2[d] = sv * sv; isj[d] = 1.f / sj2[d]; lsj += logf(sj2[d]);
                    tr_ij += si2[d] * isj[d]; m_j += diff[d] * diff[d] * isj[d];
                }
                const float dls = lsi - lsj;
                // KL(i||j) = 0.5*(sum s_i^2/s_j^2 - log prod(s_i^2/s_j^2) + sum d^2/s_j^2 - D)  (as ComputeKLDivergence)
                const float e_ij = __expf(-0.5f * (tr_ij - dls + m_j - D));
                if (GRAD) {
                    float tr_ji = 0.f, m_i = 0.f;
                    for (int d = 0; d < D; d++) { tr_ji += sj2[d] * isi[d]; m_i += diff[d] * diff[d] * isi[d]; }
                    const float e_ji = __expf(-0.5f * (tr_ji + dls + m_i - D));
                    const float a = ni * e_ij, c = norm[j] * e_ji;
                    for (int d = 0; d < D; d++) acc[d] += diff[d] * (a * isj[d] + c * isi[d]);
                } else zsum += e_ij;
            }
        }
    }
    if (!active) return;
    if (GRAD) for (int d = 0; d < D; d++) out[i * D + d] = acc[d];
    else out[i] = 1.f + zsum;
}
'''

_NAMES = [f'pair_kernel<{D},{p},{g}>' for D in (2, 3) for p in ('false', 'true') for g in ('false', 'true')]
_module = None


def _kernel(D, perspot, grad):
    global _module
    if _module is None:
        _module = cp.RawModule(code=_SRC, options=('-std=c++14', '--use_fast_math'), name_expressions=_NAMES)
    return _module.get_function(f'pair_kernel<{D},{"true" if perspot else "false"},{"true" if grad else "false"}>')


def _cp(t):
    return cp.from_dlpack(t)


class DMECupy(DMEGPU):
    """Same interface/objective as DMEGPU, but pair sums are computed by fused CuPy kernels (no pair list)."""

    def __init__(self, *args, block=64, max_cells=1 << 26, **kw):
        kw.setdefault('skin', 0.0)
        super().__init__(*args, **kw)
        self.block, self.max_cells = block, max_cells
        self.fr32 = self.fr.int()
        self.stats.update(t_hash=0.0, t_kernel=0.0, evals=0)

    @torch.no_grad()
    def _hash(self, u):
        D, BI = self.D, self.block
        lo = u.min(0).values
        ext = u.max(0).values - lo
        cs = float(self.r)
        # enlarge cells if the dense grid would be too large (still correct: cells >= r)
        while float(torch.prod(torch.ceil(ext / cs) + 3)) > self.max_cells:
            cs *= 1.25
        c = torch.floor((u - lo) / cs).long() + 1
        dims = torch.ceil(ext / cs).long() + 3
        mult = [1]
        for d in range(1, D):
            mult.append(mult[-1] * int(dims[d - 1]))
        ncell = mult[-1] * int(dims[D - 1])
        key = (c * torch.tensor(mult, device=self.dev)).sum(1)
        skey, order = torch.sort(key)
        count = torch.bincount(skey, minlength=ncell)
        start = torch.cumsum(count, 0) - count
        nonempty = torch.nonzero(count).squeeze(1)
        nb = (count[nonempty] + BI - 1) // BI
        blk_cell = torch.repeat_interleave(nonempty, nb)
        first = torch.cumsum(nb, 0) - nb
        k = torch.arange(len(blk_cell), device=self.dev) - torch.repeat_interleave(first, nb)
        blk_start = start[blk_cell] + k * BI
        blk_end = start[blk_cell] + count[blk_cell]
        offs = torch.tensor([sum(o[d] * mult[d] for d in range(D)) for o in itertools.product([-1, 0, 1], repeat=D)],
                            device=self.dev, dtype=torch.long)
        return order, count.int(), start.int(), blk_cell.int(), blk_start.int(), blk_end.int(), offs

    def _launch(self, H, us, frs, ss, norm, grad):
        order, count, start, blk_cell, blk_start, blk_end, offs = H
        N, D = us.shape[0], self.D
        out = torch.empty((N, D) if grad else (N,), device=self.dev, dtype=torch.float32)
        k = _kernel(D, not self.const_crlb, grad)
        k((len(blk_cell),), (self.block,),
          (_cp(us), _cp(frs), _cp(ss), _cp(start), _cp(count), _cp(blk_cell), _cp(blk_start), _cp(blk_end),
           _cp(offs), np.int32(len(offs)), _cp(norm if norm is not None else us), _cp(out), np.float32(self.r ** 2)))
        return out

    @torch.no_grad()
    def score_and_grad(self, state, need_grad=True):
        t0 = time.time()
        dpf = self.drift_per_frame(state)
        u = self.v - dpf[self.fr]
        H = self._hash(u)
        order = H[0]
        us = u[order].contiguous()
        frs = self.fr32[order].contiguous()
        ss = self.s[order].contiguous() if not self.const_crlb else us
        # torch and cupy both use the legacy default stream here, so kernels are ordered; sync only for timing
        sync(self.dev)
        t1 = time.time()
        Z = self._launch(H, us, frs, ss, None, False)
        score = torch.log(Z.double()).mean().float()
        grad = None
        if need_grad:
            gs = self._launch(H, us, frs, ss, (1.0 / Z).contiguous(), True)
            g = torch.empty_like(gs)
            g[order] = gs
            gf = torch.zeros(self.F, self.D, device=self.dev)
            gf.index_add_(0, self.fr, g)
            grad = self.frame_grad_to_state(gf) / self.N
        sync(self.dev)
        t2 = time.time()
        self.stats['t_hash'] += t1 - t0
        self.stats['t_kernel'] += t2 - t1
        self.stats['t_eval'] += t2 - t0
        self.stats['evals'] += 1
        return score, grad
