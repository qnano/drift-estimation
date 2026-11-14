"""
DME: drift estimation by entropy minimization for single-molecule localization microscopy.

    from dme import DME
    est = DME().fit(positions, framenum, crlb)                            # DME v2 (learned temporal prior)
    corrected = est.apply(positions, framenum)

Function interface (compatibility with earlier versions; also the 2021 algorithm):
    from dme import dme_estimate
    drift, (set1, set2) = dme_estimate(positions, framenum, crlb)                    # DME v2
    drift, (set1, set2) = dme_estimate(positions, framenum, crlb, framesperbin=100)  # DME v2 with fixed bins
    drift, (set1, set2) = dme_estimate(positions, framenum, crlb, 100, method='v1')  # algorithm of the 2021 paper
"""
__version__ = '2.0.0'

from .dme import DME, dme_estimate, precision, plot_drift
from .aim_torch import aim
from .dme_pipeline import rcc_torch, rcc3d_torch
