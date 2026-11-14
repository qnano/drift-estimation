"""Run time, peak GPU memory (of this process, from nvidia-smi, CUDA context included) and peak RAM of one method at its
best setting on the tubulin data. One method per process: python tub_bench.py dme_v2|dme2021|comet|aim|rcc
Appends a line to drift/tub_bench.txt."""
import os, sys, time, threading, subprocess, resource
import numpy as np
import pandas as pd

m = sys.argv[1]
D = os.path.join(os.environ.get('EPFL_DATA', '/mnt/data/SMLM_challenge'), 'Tubulin-A647-3D')
PXX, PXY = 117.0, 118.2
df = pd.read_csv(f'{D}/locs/Tubulin-A647-3D_locs.csv')
df = df[df['flags'] == 0]
p = np.ascontiguousarray(np.stack([df.x, df.y, df.z / 1000], 1), dtype=np.float32)
c = np.ascontiguousarray(np.stack([df.crlb_x, df.crlb_y, df.crlb_z / 1000], 1), dtype=np.float32)
f = np.ascontiguousarray(df.frame, dtype=np.int32)
F = int(f.max()) + 1
del df
if m in ('dme2021', 'comet'):                       # 1M most precise localizations
    s = np.sort(np.argsort(c[:, 0] * c[:, 1], kind='stable')[:1_000_000])
    p, c, f = np.ascontiguousarray(p[s]), np.ascontiguousarray(c[s]), np.ascontiguousarray(f[s])
ram0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6          # GB, data loaded
peak, stop = [0.0], [False]


def poll():
    while not stop[0]:
        try:
            o = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True).stdout
            for line in o.splitlines():
                pid, mem = line.split(',')
                if int(pid) == os.getpid():
                    peak[0] = max(peak[0], float(mem) / 1024)
        except Exception:
            pass
        time.sleep(0.25)


if m == 'dme_v2':
    sys.path.insert(0, os.environ.get('DME_PATH', '/tmp/fast/wt'))
    from dme import dme_estimate
    run = lambda: dme_estimate(p, f, c, estimatePrecision=False, verbose=False, numframes=F)
elif m == 'aim':
    sys.path.insert(0, os.environ.get('DME_PATH', '/tmp/fast/wt'))
    from dme import aim
    run = lambda: aim(p, f, interval=4000, intersect_d=0.1, intersect_dz=0.02, numframes=F)     # 3D
elif m == 'rcc':
    sys.path.insert(0, os.environ.get('DME_PATH', '/tmp/fast/wt'))
    from dme import rcc_torch
    run = lambda: rcc_torch(p[:, :2], f, timebins=F // 10000, zoom=2, maxdrift_px=10, numframes=F)
elif m == 'dme2021':
    sys.path.insert(0, os.environ.get('DME2021_PATH', os.path.expanduser('~/dme_v2_test')))
    from dme.legacy import dme_estimate as leg
    run = lambda: leg(p, f, c, 250, None, useCuda=True, display=False, estimatePrecision=False, initializeWithRCC=10,
                      numframes=F)
elif m == 'comet':
    from comet import comet_run_kd
    ds = np.zeros((len(p), 4))
    ds[:, 0], ds[:, 1], ds[:, 3] = p[:, 0] * PXX, p[:, 1] * PXY, f
    run = lambda: comet_run_kd(ds, segmentation_mode=2, segmentation_var=2500, max_drift_nm=300, target_sigma_nm=10.0,
                               min_max_frames=(0, F - 1), random_state=0)
th = threading.Thread(target=poll, daemon=True)
th.start()
t = time.time()
run()
dt = time.time() - t
time.sleep(0.6)
stop[0] = True
ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
line = f'{m:8s} N {len(p):8d}  time {dt:7.1f} s  GPU {peak[0]:5.2f} GB  RAM peak {ram:5.1f} GB (data loaded: {ram0:.1f} GB)'
print(line, flush=True)
open(f'{D}/drift/tub_bench.txt', 'a').write(line + '\n')
