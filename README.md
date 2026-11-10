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
It is slower than AIM and RCC: 3 s for 10k localizations, 20 s for 1M, 3 minutes for 10M on an RTX 3090.
Details, tables and the settings searched: [docs/comparison.md](docs/comparison.md).

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
from dme import dme_estimate

# positions: [N, 2] or [N, 3] (e.g. x, y in pixels, z in microns), framenum: [N], crlb: [N, D] localization precision
drift, (drift_set1, drift_set2) = dme_estimate(positions, framenum, crlb)
corrected = positions - drift[framenum]
```

- `dme_estimate(positions, framenum, crlb)`: DME v2. The temporal resolution is chosen automatically; there is no
  `framesperbin` to set.
- `method='v1', framesperbin=100`: the algorithm of the 2021 paper (exact pairwise objective, RCC start), on the GPU.
  Supports per-localization CRLBs (`perSpotCRLB=True`).
- `drift_set1`/`drift_set2` are estimates from two halves of the data; `dme.precision(drift_set1, drift_set2)` gives
  their RMS difference as an indication of the precision. Disable with `estimatePrecision=False`.

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
| `dme/` | the package: `dme_estimate`; `dme_prior.py` learned prior; `dme_grid.py` density-grid DME; `dme_pipeline.py` v1 and RCC; `aim_torch.py` AIM |
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
