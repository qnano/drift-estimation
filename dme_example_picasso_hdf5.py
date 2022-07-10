# -*- coding: utf-8 -*-
"""
How to process localization data from Picasso, stored in HDF5 format.

Photonpy is required for this (but no CUDA is required for the HDF5 loader):
    
    pip install photonpy=1.0.39

"""

import os
from photonpy import Dataset
from dme.dme import dme_estimate
import matplotlib.pyplot as plt

fn = 'example_data/gattaquant 80nm RY.hdf5'
ds = Dataset.load(fn)
print(ds) # prints number of localizations

drift_trace, (set1,set2)  = dme_estimate(ds.pos, ds.frame, ds.crlb.pos,
             framesperbin = 10,
             imgshape=ds.imgshape,  # size of field of view in pixels (same units as ds.pos)
             useCuda=True,
             maxspots=500000, 
             pixelsize=108.3,
             display=True)

"""
# Already using 'display=True' above

fig,ax=plt.subplots(2,1,figsize=(8,6),sharex=True)
for i in range(2):
    ax[i].plot(drift_trace[:,i], label='Estimated drift')
    if i==0:
        ax[i].legend()
        
    axname=['X', 'Y'][i]
    ax[i].set_title(f'{axname} drift')
    ax[i].set_ylabel('Drift [pixels]')
ax[1].set_xlabel('Frame')
plt.tight_layout()
"""

ds.applyDrift(drift_trace)

ds.save(os.path.splitext(fn)[0]+"-drift-corrected.hdf5")
