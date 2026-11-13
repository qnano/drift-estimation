Split-half drift comparison on the raw EPFL SMLM challenge 2016 3D dSTORM movies (results in docs/comparison.md).

Input: `<EPFL_DATA>/<dataset>/locs/<dataset>_locs.csv` with columns frame, x, y (px), z (nm), crlb_x, crlb_y (px),
crlb_z (nm), flags; localizations of the raw movies, not drift corrected (ours came from a spline-PSF fit calibrated on
the bead stacks of the dataset). Output: `<EPFL_DATA>/<dataset>/drift/*.npz` and a log with the split-half errors.

```
python tub_drift.py new        # DME v2, AIM, RCC
DME2021_PATH=... python tub_drift.py legacy     # DME 2021 and its RCC (C++/CUDA build of commit 2ee39f7)
python npc_drift.py new; python npc_drift.py legacy
```
