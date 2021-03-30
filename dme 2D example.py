""" 
Example that uses the photonpy library to
- simulate an SMLM acquisition and run localization on it
- perform drift estimation using the minimum entropy method
- compare with the ground truth drift data

See https://github.com/qnano/photonpy/blob/master/cpp/SMLMLib/DriftEstimation.cu for the actual algorithm
"""
from photonpy import Context,Dataset,GaussianPSFMethods
import numpy as np
import matplotlib.pyplot as plt
import os

from photonpy.smlm.blinking_spots import blinking
from photonpy.smlm.process_movie import Localizer2D


tiff_fn = os.path.abspath('simulated-clusters.tif')
output_fn = os.path.splitext(tiff_fn)[0]+ "_locs.hdf5"

# Generate a bunch of ground truth points  ('binding sites for super resolution people')
W = 300
N = 1000
gt = np.random.uniform(0,W,size=(N,2))
intensity = 1000
numframes = 2000
# Ground truth drift
gt_drift = np.cumsum(np.random.normal(0.0002, 0.02, size=(numframes,2)),0)
gt_drift -= gt_drift.mean(0)

def generate_data():
    ds = Dataset(N, 2, [W,W])
    
    ds.pos = gt
    ds.photons = intensity
    
    os.makedirs(os.path.split(tiff_fn)[0],exist_ok=True)
    
    blink_generator = blinking(len(ds), numframes=numframes, avg_on_time = 10, on_fraction=0.05, subframe_blink=4)

    # Simulate SMLM acquisition
    with Context() as ctx:
        psf = GaussianPSFMethods(ctx).CreatePSF_XYIBg(10, 2, cuda=True)
        
        blinkds = ds.simulateBlinking(numframes, blink_generator)
        blinkds.simulateMovie( psf, tiff_fn, background=10, drift=gt_drift)
            
    # Run localization
    cfg = {
        'roisize': 9,
        'threshold': 5,
        'sigmaframesperbin': 500,
        'gain': 1, 
        'offset': 0,
        'maxframes': 0,
        'startframe': 0,
        'pixelsize': 100, 
        'spotdetectsigma': 2,
        'sumframes': 1,
        'maxchisq': 1.5
    }
        
    loc = Localizer2D()
    result_ds = loc.process(tiff_fn, cfg, output_file=output_fn)
    
    return result_ds, gt_drift
    
        

ds, gt_drift = generate_data()

# Run drift estimation
# The parameters coarseFramesPerBin and coarseSigma can be used to do an initial 
drift_trace, estimated_precision = ds.estimateDriftMinEntropy(
    framesPerBin=1, initializeWithRCC=True, display=True, 
    coarseFramesPerBin=50, coarseSigma=ds.crlb.pos.mean()*4)

rmsd = np.sqrt(np.mean((drift_trace-gt_drift)**2, 0))
print(f"RMSD of drift estimation: {rmsd} pixels")

# Compare with ground truth
fig,ax=plt.subplots(2,1,figsize=(8,6))
for i in range(2):
    ax[i].plot(drift_trace[:,i], label='Estimated drift')
    ax[i].plot(gt_drift[:,i]+0.1, label='Ground truth')
    if i==0:
        ax[i].legend()
        
    axname=['X', 'Y'][i]
    ax[i].set_title(f'{axname} drift')
plt.tight_layout()

