"""
Drift correction of localization data stored in Picasso HDF5 format (needs h5py).
Writes a drift-corrected copy next to the input file.
"""
import os
import h5py
import numpy as np
from dme import dme_estimate

fn = 'example_data/gattaquant 80nm RY.hdf5'
with h5py.File(fn, 'r') as f:
    locs = f['locs'][:]
print(f"{len(locs)} localizations")

pos = np.stack([locs['x'], locs['y']], 1)
crlb = np.stack([locs['lpx'], locs['lpy']], 1)

drift, (set1, set2) = dme_estimate(pos, locs['frame'], crlb, pixelsize=108.3, display=True)

corrected = locs.copy()
corrected['x'] -= drift[locs['frame'], 0]
corrected['y'] -= drift[locs['frame'], 1]
out = os.path.splitext(fn)[0] + '-drift-corrected.hdf5'
with h5py.File(out, 'w') as f:
    f.create_dataset('locs', data=corrected)
print(f"Saved {out} (copy the .yaml file from the input to open it in Picasso)")
