"""Drift estimation on the EPFL Tubulin-A647-3D localizations of the spline-PSF session (flags == 0, not drift
corrected). Same protocol as npc_drift.py: each method on all data and on each half of a 2 um checkerboard split;
rms(dA - dB) / sqrt(2) estimates the error of a half-data estimate. x, y in px, z in um. DME 2021 is limited to 1M
localizations: it gets the 1M with the smallest CRLB of each subset.
python tub_drift.py new|legacy|comet  -> drift/tub_<mode>.npz (separate processes: both packages are called dme)"""
import os, sys, time
import numpy as np
import pandas as pd

mode = sys.argv[1]
D = os.path.join(os.environ.get('EPFL_DATA', '/mnt/data/SMLM_challenge'), 'Tubulin-A647-3D')
os.makedirs(f'{D}/drift', exist_ok=True)
OUT = f'{D}/drift/tub_{mode}.npz'
PXX, PXY = 117.0, 118.2                      # nm/px, scale of the Ries-lab Cspline submission (see locs/README.md)
df = pd.read_csv(f'{D}/locs/Tubulin-A647-3D_locs.csv')
df = df[df['flags'] == 0]
p = np.ascontiguousarray(np.stack([df.x, df.y, df.z / 1000], 1), dtype=np.float32)
c = np.ascontiguousarray(np.stack([df.crlb_x, df.crlb_y, df.crlb_z / 1000], 1), dtype=np.float32)
f = np.ascontiguousarray(df.frame, dtype=np.int32)
F = int(f.max()) + 1
blk = 2000.0 / ((PXX + PXY) / 2)
half = ((np.floor(p[:, 0] / blk) + np.floor(p[:, 1] / blk)) % 2).astype(bool)
print(len(p), 'localizations', F, 'frames; halves', (~half).sum(), half.sum(), flush=True)
out = dict(np.load(OUT)) if os.path.exists(OUT) else {}


def best1m(pp, ff, cc):
    if len(pp) <= 1_000_000:
        return pp, ff, cc
    s = np.sort(np.argsort(cc[:, 0] * cc[:, 1], kind='stable')[:1_000_000])
    return pp[s], ff[s], cc[s]


if mode == 'legacy':
    sys.path.insert(0, os.environ['DME2021_PATH'])      # checkout of the 2021 code (commit 2ee39f7), built
    from dme.legacy import dme_estimate as leg
    from dme.rcc import rcc as rcc_orig, rcc3D as rcc3d_orig
    from dme.native_api import NativeAPI

    def legacy(fpb, dims):
        def run(pp, ff, cc):
            pp, ff, cc = best1m(pp, ff, cc)
            return leg(np.ascontiguousarray(pp[:, :dims]), ff, np.ascontiguousarray(cc[:, :dims]), fpb, None,
                       useCuda=True, display=False, estimatePrecision=False, initializeWithRCC=10, numframes=F)
        return run

    def rcc_o(fb, three):
        def run(pp, ff, cc):
            with NativeAPI(True) as dll:
                return rcc3d_orig(pp, ff, F // fb, 2, dll, sigma=1) if three else \
                    rcc_orig(pp[:, :2], ff, F // fb, dll, zoom=2, sigma=1)[0]
        return run
    methods = {f'dme2021_{fpb}': legacy(fpb, 2) for fpb in (50, 100, 250, 500, 1000)}
    methods.update({f'rccorig_{fb}': rcc_o(fb, False) for fb in (1000, 2500, 5000)})
    methods.update({f'rccorig3d_{fb}': rcc_o(fb, True) for fb in (2500, 5000, 10000)})
    methods.update({f'dme2021_3d_{fpb}': legacy(fpb, 3) for fpb in (250, 1000)})
elif mode == 'comet':                       # pip install comet-smlm; 2D, at most 2000 time windows
    from comet import comet_run_kd

    def comet(fpb, tsig):
        def run(pp, ff, cc):
            ds = np.zeros((len(pp), 4))
            ds[:, 0], ds[:, 1], ds[:, 3] = pp[:, 0] * PXX, pp[:, 1] * PXY, ff
            d = np.asarray(comet_run_kd(ds, segmentation_mode=2, segmentation_var=int(fpb), max_drift_nm=300,
                                        target_sigma_nm=tsig, min_max_frames=(0, F - 1), random_state=0))[:F, :2]
            return d / np.array([PXX, PXY])
        return run
    methods = {f'comet_{fpb}_10': comet(fpb, 10.0) for fpb in (250, 100, 500, 50, 1000, 2500)}
else:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    from dme import dme_estimate, aim, rcc_torch
    methods = {'dme_v2_3d': lambda pp, ff, cc: dme_estimate(pp, ff, cc, estimatePrecision=False, verbose=False,
                                                            numframes=F)}
    for iv in (20, 50, 100, 250, 500, 1000, 2000):
        for d in (0.1, 0.2, 0.3):
            methods[f'aim_{iv}_{d}'] = (lambda iv, d: lambda pp, ff, cc: aim(pp, ff, interval=iv, intersect_d=d,
                                                                             numframes=F))(iv, d)
    for fb in (250, 500, 1000, 2500, 5000, 10000):         # RCC port with the original's +-20 zoomed-px peak search
        methods[f'rccw_{fb}'] = (lambda fb: lambda pp, ff, cc: rcc_torch(pp[:, :2], ff, timebins=F // fb, zoom=2,
                                                                         maxdrift_px=10, numframes=F))(fb)

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
        np.savez(OUT + '.tmp.npz', **out)
        os.replace(OUT + '.tmp.npz', OUT)                   # readers never see a half-written file
        print(f'{key:22s} {out[key + "_time"]:7.1f} s', flush=True)
    if f'{name}_h0' in out and f'{name}_h1' in out:
        a, b = out[f'{name}_h0'], out[f'{name}_h1']
        e = ((a - a.mean(0)) - (b - b.mean(0))) * UNIT[:a.shape[1]]
        r = np.sqrt(np.mean(e ** 2, 0) / 2)
        print(f'== {name:16s} split-half error per half [nm]: ' + ' '.join(f'{v:.2f}' for v in r), flush=True)
print('DONE', flush=True)
