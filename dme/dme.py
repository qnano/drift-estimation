"""
Drift estimation by entropy minimization (DME), GPU implementation in PyTorch (+ optional CuPy kernels).

    from dme import DME
    est = DME().fit(positions, framenum, crlb)          # DME v2
    corrected = est.apply(positions, framenum)

    from dme import dme_estimate                                    # function interface (compatibility, 'v1')
    drift, (drift_set1, drift_set2) = dme_estimate(positions, framenum, crlb)

method='v2' (default): density-grid evaluation of the DME objective, multi-start initialization (AIM, RCC, zero;
                       the best by DME score), coarse-to-fine kernel widths and preconditioned L-BFGS.
                       framesperbin=None (default): the temporal resolution is chosen from the data and the drift
                       trace comes from a small learned temporal prior applied to per-segment DME measurements
                       (dme/dme_prior.py). With a given framesperbin: a cubic spline with that knot spacing.
method='v1':           the algorithm of the paper (Cnossen et al., Opt. Express 2021): exact pairwise objective,
                       RCC initialization, optional coarse pass. The neighbor search uses a GPU spatial hash instead
                       of the original CPU KD-tree, and optimization uses L-BFGS.
The original C++/CUDA implementation of the paper is in the repository history (github.com/qnano/drift-estimation,
commit 2ee39f7).
"""
import time
import warnings
import numpy as np

from . import dme_gpu
from .dme_gpu import default_device

class DME:
    """DME v2: drift estimation by entropy minimization with a learned temporal prior. Nothing to tune.

        est = DME()                       # device=None: CUDA if available
        est.fit(positions, framenum, crlb)           # positions [N, 2 or 3], framenum [N], crlb [N, D] or [D]
        corrected = est.apply(positions, framenum)   # positions - drift[framenum]
        est.drift                                    # [F, D], zero mean, same units as positions
        est.precision()                              # optional: RMS difference of two half-data estimates, per axis
        est.plot(pixelsize=108.3)                    # optional: matplotlib figure of the trace(s)

    device:         torch device; default 'cuda' if available, else 'cpu'
    gpu_memory_gb:  GPU memory the estimation may use (or what is free on the device, if less). Everything stays on
                    the GPU; the work is split into smaller chunks when less memory is allowed
    isotropic3d:    3D data: evaluate the objective in 3D instead of on the x-y and x-z projections. Only worth it for
                    very dense 3D samples where structures at different depths overlap in projection; much slower
    verbose:        print progress
    """

    def __init__(self, device=None, gpu_memory_gb=8.0, isotropic3d=False, verbose=False):
        self.device = default_device(device)
        self.gpu_memory_gb = float(gpu_memory_gb)
        self.isotropic3d = isotropic3d
        self.verbose = verbose
        self.drift = None           # [F, D] after fit()
        self.drift_halves = None    # (drift_set1, drift_set2) after precision()
        self.runtime = None         # seconds of the last fit()
        self.numframes = None
        self._data = None

    def _estimate(self, p, f, c, verbose):
        from .dme_prior import dme_prior_estimate
        dme_gpu.GPU_MEMORY_LIMIT_GB = self.gpu_memory_gb
        return dme_prior_estimate(p, f, c, numframes=self.numframes, device=self.device, verbose=verbose,
                                  isotropic3d=self.isotropic3d)

    def fit(self, positions, framenum, crlb, numframes=None, max_spots=None):
        """Estimate the drift. positions [N, D] (D = 2 or 3; e.g. x, y in pixels and z in microns), framenum [N]
        (integers from 0), crlb [N, D] or [D]: localization precision in the units of positions.
        numframes: number of frames of the movie (default: max(framenum) + 1).
        max_spots: if set, use only the max_spots localizations with the smallest CRLB.
        Returns self; the drift is in self.drift."""
        positions = np.ascontiguousarray(positions, dtype=np.float32)
        framenum = np.ascontiguousarray(framenum, dtype=np.int32)
        D = positions.shape[1]
        crlb = np.asarray(crlb, dtype=np.float32)
        if crlb.ndim == 1:
            crlb = np.broadcast_to(crlb[:D], positions.shape)
        crlb = np.ascontiguousarray(crlb[:, :D])
        self.numframes = int(numframes if numframes is not None else framenum.max() + 1)
        if max_spots is not None and max_spots < len(positions):
            best = np.sort(np.argsort(np.prod(crlb, 1))[:max_spots])
            if self.verbose:
                print(f"DME: using the {max_spots}/{len(positions)} localizations with the smallest CRLB")
            positions, framenum, crlb = positions[best], framenum[best], crlb[best]
        self._data = positions, framenum, crlb
        self.drift_halves = None
        t0 = time.time()
        self.drift = self._estimate(positions, framenum, crlb, self.verbose)
        self.runtime = time.time() - t0
        if self.verbose:
            print(f"DME (device={self.device}): {self.runtime:.1f}s", flush=True)
        return self

    def apply(self, positions, framenum):
        """Drift-corrected copy of positions [N, D'] (D' <= D of the fit): positions - drift[framenum]."""
        self._check()
        positions = np.asarray(positions)
        return positions - self.drift[np.asarray(framenum), :positions.shape[1]].astype(positions.dtype)

    def precision(self):
        """Estimate the drift again on two halves of the data (split at the median along the widest of x, y) and
        return the RMS difference of the two traces per axis: an indication of the precision. About doubles the
        run time. The two traces are kept in self.drift_halves."""
        self._check()
        p, f, c = self._data
        if self.drift_halves is None:
            ax = int(np.argmax(np.var(p[:, :2], 0)))
            s1 = p[:, ax] > np.median(p[:, ax])
            self.drift_halves = (self._estimate(p[s1], f[s1], c[s1], False),
                                 self._estimate(p[~s1], f[~s1], c[~s1], False))
        return precision(*self.drift_halves)

    def plot(self, pixelsize=None):
        """Matplotlib figure of the drift trace (and of the two half-data traces if precision() was called).
        pixelsize: nm per pixel, to show x and y in nm."""
        self._check()
        d1, d2 = self.drift_halves if self.drift_halves is not None else (None, None)
        return plot_drift(self.drift, d1, d2, pixelsize)

    def _check(self):
        if self.drift is None:
            raise RuntimeError("DME: call fit(positions, framenum, crlb) first")


_IGNORED_LEGACY_ARGS = ('useCuda', 'maxNeighbors', 'useDebugLibrary', 'traces')


def dme_estimate(positions, framenum, crlb, framesperbin=None, imgshape=None,
                 method='v2',
                 perSpotCRLB=False,
                 initializeWithRCC=10,
                 initialEstimate=None,
                 rccZoom=2,
                 coarseFramesPerBin=None,
                 coarseSigma=None,
                 estimatePrecision=True,
                 maxIterations=200,
                 coarseMaxIterations=100,
                 maxspots=None,
                 numframes=None,
                 display=False,
                 pixelsize=None,
                 device=None,
                 verbose=True,
                 isotropic3d=False,
                 gpuMemoryGB=8,
                 **legacy_kwargs):
    """
    Function interface, kept for compatibility with earlier versions and for the 2021 algorithm (method='v1').
    New code: use DME (DME v2) instead.

    Estimate drift by entropy minimization.

    positions:     [N, D] localizations (D = 2 or 3), e.g. x, y in pixels and z in microns
    framenum:      [N] integer frame numbers
    crlb:          [N, D] localization precision (CRLB), same units as positions, or a constant [D] vector
    framesperbin:  frames per spline knot (1 = drift per frame, otherwise cubic Catmull-Rom spline). Required for
                   'v1'. For 'v2', None (default) chooses the temporal resolution automatically and uses the
                   learned temporal prior (most precise; slower)
    imgshape:      ignored (kept for compatibility with the original API)

    method:        'v2' (default) or 'v1' (algorithm of the paper), see the module docstring
    perSpotCRLB:   use per-localization CRLBs in the objective ('v1' only; 'v2' uses the mean CRLB)
    initializeWithRCC: number of RCC time bins for the initial estimate ('v1'; 'v2' tries AIM, RCC and zero
                   drift and continues with the one with the best DME score)
    initialEstimate: [F, D] initial drift, replaces RCC / AIM initialization
    coarseFramesPerBin, coarseSigma: optional coarse pass before the fine one ('v1'; 'v2' is always coarse-to-fine)
    estimatePrecision: also estimate drift on two halves of the data (split along x or y); the RMSD between the two
                   traces indicates the precision
    maxIterations: L-BFGS function evaluations per optimization
    maxspots:      if set, use only the maxspots localizations with the best (smallest) CRLB
    numframes:     number of frames (default: max(framenum) + 1)
    display:       plot the result with matplotlib
    pixelsize:     nm per pixel, for the plot title
    device:        torch device, default 'cuda' if available, else 'cpu'
    isotropic3d:   3D data, 'v2' without framesperbin: evaluate the objective in 3D (3D grid or spatial hash, whichever
                   is cheaper) instead of on the x-y and x-z projections. Only worth it for very dense 3D samples where
                   binding sites at different depths overlap in projection; much slower.

    gpuMemoryGB:   GPU memory the estimation may use (default 8, or what is free on the device if that is less).
                   All work stays on the GPU; the chunk sizes of the pair sums, density grids and per-frame
                   decompositions are scaled to this budget, so less memory means more, smaller chunks

    Returns drift [F, D] (corrected positions = positions - drift[framenum]), and if estimatePrecision is True
    also (drift_set1, drift_set2).
    """
    for k in legacy_kwargs:
        if k not in _IGNORED_LEGACY_ARGS:
            raise TypeError(f"dme_estimate() got an unexpected keyword argument '{k}'")
    if legacy_kwargs and verbose:
        print(f"dme_estimate: ignoring arguments of the C++ implementation: {', '.join(legacy_kwargs)}")

    positions = np.ascontiguousarray(positions, dtype=np.float32)
    framenum = np.ascontiguousarray(framenum, dtype=np.int32)
    D = positions.shape[1]
    crlb = np.asarray(crlb, dtype=np.float32)
    if crlb.ndim == 1:
        crlb = np.broadcast_to(crlb[:D], positions.shape)
    crlb = np.ascontiguousarray(crlb[:, :D])
    F = int(numframes if numframes is not None else framenum.max() + 1)
    device = default_device(device)
    dme_gpu.GPU_MEMORY_LIMIT_GB = float(gpuMemoryGB)

    if maxspots is not None and maxspots < len(positions):
        best = np.sort(np.argsort(np.prod(crlb, 1))[:maxspots])
        if verbose:
            print(f"dme_estimate: using the {maxspots}/{len(positions)} localizations with the smallest CRLB")
        positions, framenum, crlb = positions[best], framenum[best], crlb[best]

    init = None if initialEstimate is None else np.asarray(initialEstimate, np.float32)[:F]
    if method == 'v1' and framesperbin is None:
        raise TypeError("method='v1' needs framesperbin")
    if method == 'v2' and perSpotCRLB:
        warnings.warn("method='v2' uses the mean CRLB; use method='v1' for per-spot CRLB")
    t0 = time.time()

    if method == 'v2' and framesperbin is None:
        if init is not None:
            warnings.warn("initialEstimate is ignored when framesperbin is None")
        e = DME(device=device, gpu_memory_gb=gpuMemoryGB, isotropic3d=isotropic3d, verbose=verbose)
        e.fit(positions, framenum, crlb, numframes=F)
        if estimatePrecision:
            e.precision()
            r = e.drift, e.drift_halves
        else:
            r = e.drift
    elif method == 'v2':
        from .dme_grid import dme_grid_estimate
        r = dme_grid_estimate(positions, framenum, crlb.mean(0), framesperbin,
                              init='auto' if init is None else init,
                              maxIterations=maxIterations, estimatePrecision=estimatePrecision,
                              numframes=F, device=device, verbose=verbose)
    elif method == 'v1':
        from .dme_pipeline import dme_estimate_gpu
        r = dme_estimate_gpu(positions, framenum, crlb, framesperbin,
                             coarseFramesPerBin=coarseFramesPerBin, coarseSigma=coarseSigma,
                             perSpotCRLB=perSpotCRLB, initializeWithRCC=initializeWithRCC, initialEstimate=init,
                             rccZoom=rccZoom, estimatePrecision=estimatePrecision, maxIterations=maxIterations,
                             coarseMaxIterations=coarseMaxIterations, numframes=F, device=device, verbose=verbose)
    else:
        raise ValueError(f"method must be 'v1' or 'v2', got {method!r}")

    if verbose:
        print(f"dme_estimate (method={method}, device={device}): {time.time() - t0:.1f}s", flush=True)

    if estimatePrecision:
        drift, (d1, d2) = r
    else:
        drift, d1, d2 = r, None, None
    if display:
        plot_drift(drift, d1, d2, pixelsize)
    return (drift, (d1, d2)) if estimatePrecision else drift


def precision(drift_set1, drift_set2):
    """RMSD between the drift traces of the two halves of the data (per dimension)."""
    a = drift_set1 - drift_set1.mean(0)
    b = drift_set2 - drift_set2.mean(0)
    return np.sqrt(((a - b) ** 2).mean(0))


def plot_drift(drift, drift_set1=None, drift_set2=None, pixelsize=None):
    import matplotlib.pyplot as plt
    D = drift.shape[1]
    names, units = ['X', 'Y', 'Z'], ['px', 'px', 'um']
    fig, ax = plt.subplots(D, 1, sharex=True, figsize=(8, 2.2 * D))
    for i in range(D):
        if drift_set1 is not None:
            ax[i].plot(drift_set1[:, i], '--', lw=0.8, label='set 1')
            ax[i].plot(drift_set2[:, i], '--', lw=0.8, label='set 2')
        ax[i].plot(drift[:, i], 'k', lw=1, label='full')
        ax[i].set_ylabel(f'{names[i]} [{units[i]}]')
    ax[0].legend(fontsize=8)
    ax[-1].set_xlabel('Frame')
    if drift_set1 is not None:
        p = precision(drift_set1, drift_set2)
        sc = [pixelsize or 1, pixelsize or 1, 1000]
        u = ['nm' if pixelsize else 'px', 'nm' if pixelsize else 'px', 'nm']
        fig.suptitle('Split-half RMSD: ' + ', '.join(f'{names[i]} {p[i] * sc[i]:.3g} {u[i]}' for i in range(D)))
    fig.tight_layout()
    return fig
