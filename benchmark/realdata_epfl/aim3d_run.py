"""3D AIM (port) split-half z error on tubulin and nuclear pores; which start DME v2 picks on tubulin."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import numpy as np, pandas as pd
from dme import aim
from dme.dme_grid import dme_grid_estimate
for name, px, F in (('Tubulin-A647-3D', (117.0, 118.2), None), ('NPC-A647-3D', (134.9, 142.0), 150000)):
    D = os.path.join(os.environ.get('EPFL_DATA', '/mnt/data/SMLM_challenge'), name)
    df = pd.read_csv(f'{D}/locs/{name}_locs.csv'); df = df[df['flags'] == 0]
    p = np.ascontiguousarray(np.stack([df.x, df.y, df.z / 1000], 1), dtype=np.float32)
    c = np.stack([df.crlb_x, df.crlb_y, df.crlb_z / 1000], 1).astype(np.float32)
    f = np.ascontiguousarray(df.frame, dtype=np.int32)
    F = F or int(f.max()) + 1
    U = np.array([px[0], px[1], 1000.0])
    blk = 2000.0 / np.mean(px)
    half = ((np.floor(p[:, 0] / blk) + np.floor(p[:, 1] / blk)) % 2).astype(bool)
    out = {}
    for iv in ((500, 1000, 2000, 4000) if 'Tub' in name else (1000, 2000, 4000, 8000)):
        for dz in (0.02, 0.04, 0.08):
            t = time.time()
            d = [aim(p[s], f[s], interval=iv, intersect_d=0.1, intersect_dz=dz, numframes=F) for s in (~half, half)]
            dt = (time.time() - t) / 2
            da = aim(p, f, interval=iv, intersect_d=0.1, intersect_dz=dz, numframes=F)
            e = np.sqrt(np.mean((((d[0] - d[0].mean(0)) - (d[1] - d[1].mean(0))) * U) ** 2, 0) / 2)
            out[f'aim3d_{iv}_{dz}_h0'], out[f'aim3d_{iv}_{dz}_h1'], out[f'aim3d_{iv}_{dz}_all'] = d[0], d[1], da
            print(f'{name[:7]} AIM 3D interval {iv:5d} dz {dz * 1000:3.0f} nm: split-half x y z [nm] {e.round(2)}  ({dt:.1f} s per half)', flush=True)
    np.savez(f'{D}/drift/aim3d.npz', **out)
    if 'Tub' in name:
        print('--- DME v2 start on tubulin x-y (verbose):', flush=True)
        dme_grid_estimate(p[:, :2], f, c[:, :2].mean(0), 275, init='auto', estimatePrecision=False, numframes=F, verbose=True)
print('DONE')
