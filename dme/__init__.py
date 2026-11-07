"""
DME: drift estimation by entropy minimization for single-molecule localization microscopy.

    from dme import dme_estimate
    drift, (set1, set2) = dme_estimate(positions, framenum, crlb, framesperbin=500)            # DME v2
    drift, (set1, set2) = dme_estimate(positions, framenum, crlb, framesperbin=500, method='v1')  # paper algorithm
"""
__version__ = '2.0.0'

from .dme import dme_estimate, precision, plot_drift
from .aim_torch import aim
from .dme_pipeline import rcc_torch, rcc3d_torch
