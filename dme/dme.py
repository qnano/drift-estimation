# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import os
from .native_api import NativeAPI

from .rcc import rcc
        
def dme_estimate(positions, framenum, crlb, framesperbin, imgshape, 
          maxdrift=3, 
          debugMode=False,
          coarseFramesPerBin=None,coarseSigma=None, perSpotCRLB=False,
          useCuda=False,
          display=True, # make a plot
          pixelsize=None,
          maxspots=None, 
          initializeWithRCC=True, initialEstimate=None, 
          rccZoom=2,estimatePrecision=True,
          maxNeighbors=1000):
    """
    Maximize localization correlation
    """
    ndims = positions.shape[1]
    numframes = np.max(framenum)+1

    initial_drift = np.zeros((numframes,ndims))
    
    with NativeAPI(useCuda) as dll:

        if initialEstimate is not None:
            initial_drift = np.ascontiguousarray(initialEstimate,dtype=np.float32)
            assert initial_drift.shape[1] == ndims
            
        elif initializeWithRCC:
            if type(initializeWithRCC) == bool:
                initializeWithRCC = 10
    
            xyI = np.ones((len(positions),3)) 
            xyI[:,:2] = positions[:,:2]
            initial_drift[:,:2],_,imgs = rcc(xyI, framenum, initializeWithRCC, 
                                             np.max(imgshape), dll, zoom=rccZoom)
            del imgs
        
            
        if maxspots is not None and maxspots < len(positions):
            print(f"Drift correction: Limiting spot count to {maxspots}/{len(positions)} spots.")
            bestspots = np.argsort(np.prod(crlb,1))
            indices = bestspots[-maxspots:]
            crlb = crlb[indices]
            positions = positions[indices]
            framenum = framenum[indices]
        
        if not perSpotCRLB:
            crlb = np.mean(crlb,0)[:ndims]
            
        numIterations = 10000
        step = 0.000001

        splitAxis = np.argmax( np.var(positions,0) )
        splitValue = np.median(positions[:,splitAxis])
        
        set1 = positions[:,splitAxis] > splitValue
        set2 = np.logical_not(set1)
        
        if perSpotCRLB:
            print("Using drift correction with per-spot CRLB")
            crlb_set1 = crlb[set1]
            crlb_set2 = crlb[set2]
        else:
            crlb_set1 = crlb
            crlb_set2 = crlb
                            
        if coarseFramesPerBin is not None:
            print(f"Computing initial coarse drift estimate... ({coarseFramesPerBin} frames/bin)",flush=True)
            with tqdm.tqdm() as pbar:
                def update_pbar(i,info): 
                    pbar.set_description(info.decode("utf-8")); pbar.update(1)
                    return 1
    
                initial_drift,score = dll.MinEntropyDriftEstimate(
                    positions, framenum, initial_drift*1, coarseSigma, numIterations, step, maxdrift, 
                    framesPerBin=coarseFramesPerBin, cuda=useCuda,progcb=update_pbar)
                
        print(f"\nEstimating drift... ({framesperbin} frames/bin)",flush=True)
        with tqdm.tqdm() as pbar:
            def update_pbar(i,info): 
                pbar.set_description(info.decode("utf-8"));pbar.update(1)
                return 1
            drift,score = dll.MinEntropyDriftEstimate(
                positions, framenum, initial_drift*1, crlb, numIterations, step, maxdrift, framesPerBin=framesperbin, maxneighbors=maxNeighbors,
                cuda=useCuda, progcb=update_pbar)
                
        if estimatePrecision:
            print(f"\nComputing drift estimation precision... (Splitting axis={splitAxis})",flush=True)
            with tqdm.tqdm() as pbar:
                def update_pbar(i,info): 
                    pbar.set_description(info.decode("utf-8"));pbar.update(1)
                    return 1
                drift_set1,score_set1 = dll.MinEntropyDriftEstimate(
                    positions[set1], framenum[set1], initial_drift*1, crlb_set1, numIterations, step, maxdrift, 
                    framesPerBin=framesperbin,cuda=useCuda, progcb=update_pbar)
    
                drift_set2,score_set2 = dll.MinEntropyDriftEstimate(
                    positions[set2], framenum[set2], initial_drift*1, crlb_set2, numIterations, step, maxdrift, 
                    framesPerBin=framesperbin,cuda=useCuda,progcb=update_pbar)
    
        drift -= np.mean(drift,0)

        if estimatePrecision:
            drift_set1 -= np.mean(drift_set1,0)
            drift_set2 -= np.mean(drift_set2,0)

            # both of these use half the data (so assuming precision scales with sqrt(N)), 2x variance
            # subtracting two 2x variance signals gives a 4x variance estimation of precision
            # this seems to be very accurate if framesperbin is 1, a lot less if higher
            L = min(len(drift_set1),len(drift_set2))
            diff = drift_set1[:L] - drift_set2[:L]
            est_precision = np.std(diff,0)/2
            print(f"\nEstimated precision: {est_precision}",flush=True)

        if display:
            L=len(drift)
            fig,ax=plt.subplots(ndims,1,sharex=True,figsize=(10,8),dpi=100)
            
            axnames = ['X', 'Y', 'Z']
            axunits = ['px', 'px', 'um']
            for i in range(ndims):
                axname=axnames[i]
                axunit = axunits[i]
                if estimatePrecision:
                    ax[i].plot(drift_set1[:L,i], '--', label=f'{axname} - set1')
                    ax[i].plot(drift_set2[:L,i], '--', label=f'{axname} - set2')
                ax[i].plot(drift[:L,i], label=f'{axname} - full')
                ax[i].plot(initial_drift[:L,i], label=f'Initial value {axname}')
                ax[i].set_ylabel(f'Drift {axname} [{axunit}]')
                ax[i].set_xlabel('Frame number')
                if i==0: ax[i].legend(fontsize=12)
            
            if estimatePrecision:
                if pixelsize is not None:
                    p=est_precision
                    scale = [pixelsize, pixelsize, 1000]
                    info = ';'.join([ f'{axnames[i]}: {p[i]*scale[i]:.1f} nm ({p[i]:.3f} {axunits[i]})' for i in range(ndims)])
                    
                    plt.suptitle(f'Drift trace. Approx. precision of drift estimate: {info}')
                else:
                    plt.suptitle(f'Drift trace. Est. Precision: X/Y={est_precision[1]:.3f}/{est_precision[1]:.3f} pixels')
                                
        return drift


