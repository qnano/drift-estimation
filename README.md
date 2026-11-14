DME: drift correction for localization microscopy by entropy minimization
==========================================================================

Fiducial-free drift estimation for single-molecule localization microscopy (2D and 3D), from the localizations alone.

J. Cnossen, T. J. Cui, C. Joo, C. Smith, "Drift correction in localization microscopy using entropy minimization",
[Opt. Express 29, 27961 (2021)](https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245).

**DME v2 (2026)** is a rewrite in PyTorch: no compilation, runs on any CUDA GPU (Linux, Windows) or on the CPU. It
maximizes the same entropy objective as the paper, and adds a small learned temporal prior: instead of a spline with a
user-chosen number of frames per knot, it measures the drift of short time segments with the DME objective and turns
these measurements into a drift trace with a trained 1D network.

**No frames per bin to choose.** Every other method, including the 2021 DME code, needs the temporal resolution
(frames per bin or window) set by hand. Too short and the drift estimate is noisy; too long and fast drift is smoothed
away, and the best value depends on the dataset (see the default-settings results in
[docs/comparison.md](docs/comparison.md)). DME v2 chooses it from the data itself, so there are no settings to tune.
How it works: [docs/method.md](docs/method.md).

![comparison](docs/method_comparison.png)

On simulated DNA-PAINT data DME v2 is 1.6–2.1× more precise than COMET, the next best method, and 2–5× more precise
than AIM, RCC and the 2021 DME code, although each of those was run at its best settings, picked with the true drift
(DME v2 has no settings). On a real DNA-PAINT nanoruler (the dataset in `example_data/`) it is also the most precise.
On raw 3D dSTORM data with real drift (EPFL challenge microtubules, 6.7M localizations) its split-half error is
0.3–0.4 nm in x and y and 1 nm in z. COMET at its best window length reaches the same x-y error (on the 1M most
precise localizations, 4.5× slower); the 2021 code gives 0.5–0.8 nm, AIM 1.5–1.8 nm and RCC 4–5 nm (figure below).
It is slower than AIM and RCC: 3 s for 10k localizations, 20 s for 1M, 3 minutes for 10M on an RTX 3090.
Details, tables and the settings searched: [docs/comparison.md](docs/comparison.md).

![tubulin](docs/epfl_tubulin.png)

Installation
------------

```
pip install git+https://github.com/qnano/drift-estimation.git
```

This needs PyTorch. For GPU support, install the CUDA build of PyTorch first, see [pytorch.org](https://pytorch.org).
Optionally, install CuPy (`pip install cupy-cuda12x`, matching your CUDA version) for faster exact pair kernels.

Usage
-----

```python
from dme import DME

# positions: [N, 2] or [N, 3] (e.g. x, y in pixels, z in microns), framenum: [N], crlb: [N, D] localization precision
est = DME().fit(positions, framenum, crlb)
corrected = est.apply(positions, framenum)        # positions - est.drift[framenum]
```

The class `DME` runs DME v2; there is no temporal resolution or other setting to choose.

- `est.drift`: the drift per frame, `[F, D]`, in the units of the positions.
- `est.precision()`: estimates the drift again on two halves of the data and returns their RMS difference per axis, an
  indication of the precision (about doubles the run time). The two traces are in `est.drift_halves`.
- `est.plot(pixelsize=...)`: figure of the trace.
- `DME(device=..., gpu_memory_gb=8, isotropic3d=False, verbose=False)`: `gpu_memory_gb` is the GPU memory the
  estimation may use (or what is free, if less). Everything stays on the GPU; the work is split into smaller chunks
  when less memory is allowed. The localizations themselves must fit (about 1.5 GB for 7M localizations).
- `fit(..., numframes=..., max_spots=...)`: the movie length, if the last frames have no localizations; `max_spots`
  uses only the most precise localizations.

The function `dme_estimate(positions, framenum, crlb, ...)` of earlier versions still works with the same arguments;
by default it runs DME v2 and returns `drift, (drift_set1, drift_set2)`. It is also the way to run the algorithm of
the 2021 paper: `dme_estimate(positions, framenum, crlb, framesperbin=100, method='v1')` (exact pairwise objective, RCC
start, optional per-localization CRLBs with `perSpotCRLB=True`).

Examples:
- [dme_example.py](dme_example.py): 3D simulation with known drift, DME v2 and v1.
- [dme_example_picasso_hdf5.py](dme_example_picasso_hdf5.py): drift correction of a Picasso HDF5 file (the nanoruler in
  `example_data/`), writes a corrected file that opens in Picasso. Needs `h5py`.
- [dme_example_colab.ipynb](dme_example_colab.ipynb): notebook with both
  ([open in Google Colab](https://colab.research.google.com/github/qnano/drift-estimation/blob/master/dme_example_colab.ipynb)).

MATLAB
------

[matlab/dme_estimate.m](matlab/dme_estimate.m) calls the Python package through MATLAB's Python interface
(R2022a or newer), so MATLAB gets exactly the same DME v2, on the GPU. Point MATLAB once to a Python environment with
PyTorch and this package:

```matlab
pyenv('Version', 'C:\path\to\env\python.exe', 'ExecutionMode', 'OutOfProcess')
addpath('matlab')
[drift, set1, set2] = dme_estimate(positions, framenum, crlb);      % framenum starts at 0
corrected = positions - drift(framenum + 1, :);
```

Options: `'Method', 'v1'` with `'FramesPerBin'` (the 2021 algorithm), `'EstimatePrecision'`, `'Device'`. Example:
[dme_example.m](dme_example.m).

Repository
----------

| path | contents |
|---|---|
| `dme/` | the package: `dme_estimate`; `dme_prior.py` learned prior; `dme_grid.py` density-grid DME; `dme_pipeline.py` v1 and RCC; `aim_torch.py` AIM (2D and 3D) |
| `dme/weights/` | weights of the learned temporal prior |
| `docs/` | [method.md](docs/method.md), [comparison.md](docs/comparison.md), figures |
| `benchmark/` | the comparison: scripts and results |
| `training/` | training of the temporal prior (synthetic data only) |
| `tests/` | `python -m pytest tests` |
| `matlab/` | `dme_estimate.m`: DME from MATLAB (calls the Python package) |

Original C++/CUDA version
-------------------------

The 2021 C++/CUDA implementation and its MATLAB wrapper are no longer part of the package; everything is PyTorch
now. They remain available at commit [2ee39f7](https://github.com/qnano/drift-estimation/tree/2ee39f7), which is
the code of the paper. `dme_estimate` still accepts the arguments of the old Python API (and ignores the ones that only
applied to the C++ code). Known problems of that code, absent in this version: its neighbor list is not updated often
enough, so it can stop before reaching the optimum; above a few million localizations an integer overflow makes it
return zero drift; and its `maxspots` option kept the least precise localizations instead of the most precise.
