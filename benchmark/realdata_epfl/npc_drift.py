"""Drift estimation on the EPFL NPC-A647-3D localizations of the spline-PSF session (flags == 0, not drift corrected).
Each method runs on all data and on each half of a 2 um checkerboard split; rms(dA - dB) / sqrt(2) estimates the error
of a half-data estimate (no ground truth needed). x, y in px, z in um.
python npc_drift.py new|legacy  -> drift/npc_<mode>.npz (separate processes: both packages are called dme)"""
import os, sys, time
import numpy as np
import pandas as pd

mode = sys.argv[1]
D = os.path.join(os.environ.get('EPFL_DATA', '/mnt/data/SMLM_challenge'), 'NPC-A647-3D')
OUT = f'{D}/drift/npc_{mode}.npz'
PXX, PXY = 134.9, 142.0                      # nm/px, scale of the Ries-lab Cspline submission (see locs/README.md)
df = pd.read_csv(f'{D}/locs/NPC-A647-3D_locs.csv')
df = df[df['flags'] == 0]
p = np.ascontiguousarray(np.stack([df.x, df.y, df.z / 1000], 1), dtype=np.float32)
c = np.ascontiguousarray(np.stack([df.crlb_x, df.crlb_y, df.crlb_z / 1000], 1), dtype=np.float32)
f = np.ascontiguousarray(df.frame, dtype=np.int32)
F = 150_000
blk = 2000.0 / ((PXX + PXY) / 2)
half = ((np.floor(p[:, 0] / blk) + np.floor(p[:, 1] / blk)) % 2).astype(bool)
print(len(p), 'localizations; halves', (~half).sum(), half.sum(), flush=True)
out = dict(np.load(OUT)) if os.path.exists(OUT) else {}

if mode == 'legacy':
    sys.path.insert(0, os.environ['DME2021_PATH'])      # checkout of the 2021 code (commit 2ee39f7), built
    from dme.legacy import dme_estimate as leg

    def legacy(fpb):
        return lambda pp, ff, cc: leg(pp[:, :2], ff, cc[:, :2], fpb, None, useCuda=True, display=False,
                                      estimatePrecision=False, initializeWithRCC=10, numframes=F)
    methods = {f'dme2021_{fpb}': legacy(fpb) for fpb in (100, 250, 500, 1000)}
    from dme.rcc import rcc as rcc_orig
    from dme.native_api import NativeAPI

    def rcc_o(fb):
        def run(pp, ff, cc):
            with NativeAPI(True) as dll:
                return rcc_orig(pp[:, :2], ff, F // fb, dll, zoom=2, sigma=1)[0]
        return run
    for fb in (1000, 2500, 5000):
        methods[f'rccorig_{fb}'] = rcc_o(fb)
    from dme.rcc import rcc3D as rcc3d_orig

    def rcc3_o(fb):
        def run(pp, ff, cc):
            with NativeAPI(True) as dll:
                return rcc3d_orig(pp, ff, F // fb, 2, dll, sigma=1)
        return run
    for fb in (5000, 10000, 15000):
        methods[f'rccorig3d_{fb}'] = rcc3_o(fb)

    def legacy3(fpb):
        return lambda pp, ff, cc: leg(pp, ff, cc, fpb, None, useCuda=True, display=False,
                                      estimatePrecision=False, initializeWithRCC=10, numframes=F)
    for fpb in (250, 1000, 2500):
        methods[f'dme2021_3d_{fpb}'] = legacy3(fpb)
else:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from dme import dme_estimate, aim, rcc_torch, rcc3d_torch
    methods = {
        'dme_v2_3d': lambda pp, ff, cc: dme_estimate(pp, ff, cc, estimatePrecision=False, verbose=False, numframes=F),
        'dme_v2_2d': lambda pp, ff, cc: dme_estimate(pp[:, :2], ff, cc[:, :2], estimatePrecision=False, verbose=False,
                                                     numframes=F),
    }
    for iv in (100, 250, 500, 1000, 2000, 4000):
        for d in (0.1, 0.2, 0.3):
            methods[f'aim_{iv}_{d}'] = (lambda iv, d: lambda pp, ff, cc: aim(pp, ff, interval=iv, intersect_d=d,
                                                                             numframes=F))(iv, d)
    for fb in (500, 1000, 2500, 5000, 7500, 10000, 15000):
        methods[f'rcc_{fb}'] = (lambda fb: lambda pp, ff, cc: rcc_torch(pp[:, :2], ff, timebins=F // fb, zoom=2,
                                                                        numframes=F))(fb)
        methods[f'rccw_{fb}'] = (lambda fb: lambda pp, ff, cc: rcc_torch(pp[:, :2], ff, timebins=F // fb, zoom=2,
                                                                         maxdrift_px=10, numframes=F))(fb)
        methods[f'rcc3d_{fb}'] = (lambda fb: lambda pp, ff, cc: rcc3d_torch(pp, ff, timebins=F // fb, zoom=2,
                                                                            numframes=F))(fb)

UNIT = np.array([PXX, PXY, 1000.0])          # -> nm
for name, fn in methods.items():
    for part, sel in (('h0', ~half), ('h1', half), ('all', None)):
        key = f'{name}_{part}'
        if key in out or key + '_failed' in out:
            continue
        pp, ff, cc = (p, f, c) if sel is None else (p[sel], f[sel], c[sel])
        t = time.time()
        try:
            out[key] = np.asarray(fn(np.ascontiguousarray(pp), np.ascontiguousarray(ff), np.ascontiguousarray(cc)),
                                  np.float32)
        except Exception as ex:
            print(key, 'failed:', repr(ex)[:300], flush=True)
            out[key + '_failed'] = np.array(repr(ex)[:300])
        out[key + '_time'] = time.time() - t
        np.savez(OUT, **out)
        print(f'{key:22s} {out[key + "_time"]:7.1f} s', flush=True)
    if f'{name}_h0' in out and f'{name}_h1' in out:
        a, b = out[f'{name}_h0'], out[f'{name}_h1']
        e = ((a - a.mean(0)) - (b - b.mean(0))) * UNIT[:a.shape[1]]
        r = np.sqrt(np.mean(e ** 2, 0) / 2)
        print(f'== {name:16s} split-half error per half [nm]: ' + ' '.join(f'{v:.2f}' for v in r), flush=True)
print('DONE', flush=True)
