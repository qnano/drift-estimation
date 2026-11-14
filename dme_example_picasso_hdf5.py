"""
Drift correction of localization data stored in Picasso HDF5 format (needs h5py).
Writes a drift-corrected copy (with its .yaml, so it opens in Picasso) next to the input file.
"""
import os
import shutil
import h5py
import numpy as np
from dme import DME

fn = 'example_data/gattaquant 80nm RY.hdf5'
with h5py.File(fn, 'r') as f:
    locs = f['locs'][:]
print(f"{len(locs)} localizations")

pos = np.stack([locs['x'], locs['y']], 1)
crlb = np.stack([locs['lpx'], locs['lpy']], 1)

est = DME(verbose=True).fit(pos, locs['frame'], crlb)
print('split-half precision [px]:', est.precision())      # optional: estimates again on two halves of the data
est.plot(pixelsize=108.3)                                  # pixelsize in nm, for the axis units

corrected = locs.copy()
corrected['x'], corrected['y'] = est.apply(pos, locs['frame']).T
base = os.path.splitext(fn)[0]
out = base + '-drift-corrected'
with h5py.File(out + '.hdf5', 'w') as f:
    f.create_dataset('locs', data=corrected)
if os.path.exists(base + '.yaml'):
    shutil.copy(base + '.yaml', out + '.yaml')
print(f"Saved {out}.hdf5")
