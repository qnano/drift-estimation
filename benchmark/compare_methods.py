"""
Comparison of drift estimation methods (results in benchmark/results, overview in docs/comparison.md).

Methods
  dme_v2        DME v2 with the learned temporal prior: dme_estimate(..., framesperbin=None). No settings.
  dme_v2_fixed  DME v2 with fixed bins: dme_estimate(..., framesperbin=fpb). Also the reference for real data.
  dme2021       the original C++/CUDA code of the 2021 paper (dme.legacy), single stage
  dme2021_c2f   the same code run coarse to fine: 100k most precise localizations at 4x kernel width and 4x
                longer bins, then 300k at 2x, then the final setting on at most 1M localizations
  comet         COMET (Reinkensmeier et al., bioRxiv 2026; pip install comet-smlm)
  aim           AIM (Ma et al., Sci. Adv. 2024), PyTorch port in dme.aim_torch
  rcc           redundant cross-correlation (Wang et al. 2014), dme.rcc_torch

Every method except dme_v2 is run over a grid of settings:
  sim   smooth drift, 10k-10M localizations, 10,000 frames, 3 seeds. Error = RMS difference to the true drift; the
        best setting per case is picked with the true drift (an advantage no user has).
  real  a Picasso hdf5 file (default: example_data/gattaquant 80nm RY.hdf5). The localizations are split into two
        halves by an 8x8 spatial checkerboard (different binding sites, same drift). Error = difference between a
        method's drift from one half and a fine reference (dme_v2_fixed at 10 and 20 frames per bin) from the other
        half, with the reference's own noise subtracted; the best setting is picked by this error and then timed
        on all localizations.
DME 2021 uses at most the 1M most precise localizations. COMET: at most 2000 time windows, 300 nm max. drift.

    python benchmark/compare_methods.py --part sim         # resumable, writes benchmark/results/sim.json
    python benchmark/compare_methods.py --part real        # writes benchmark/results/real.json
    python benchmark/compare_plots.py                       # figures in docs/
"""
import os, json, time, argparse, contextlib, io
import numpy as np
import torch

from dme import dme_estimate
from dme.aim_torch import aim
from dme.dme_pipeline import rcc_torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
EXAMPLE = os.path.join(os.path.dirname(HERE), 'example_data', 'gattaquant 80nm RY.hdf5')
F_SIM, PX_NM_SIM = 10000, 100.0
SIZES = [1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7]
SEEDS = [1, 2, 3]
LOCS_PER_SITE, DENSITY = 200, 0.02          # DNA-PAINT-like: localizations per site, sites per px^2
MAXSPOTS = 1_000_000
FPB = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]   # used where F / fpb >= 5
METHODS = ['rcc', 'aim', 'comet', 'dme2021', 'dme2021_c2f', 'dme_v2_fixed', 'dme_v2']


def rule(N, F):
    """~20k localizations per bin, 10-2000 frames (used as the default bin length)."""
    return int(np.clip(round(20000 * F / N), 10, 2000))


def simulate(n, seed):
    """Sites at constant density, ~200 localizations per site at random frames; per-localization precision from
    a log-normal photon count (mean ~9 nm); smooth drift: linear + two slow sinusoids (periods F/1.3, F/3.1)."""
    rng = np.random.default_rng(seed)
    F = F_SIM
    t = np.arange(F) / F
    drift = np.zeros((F, 2))
    for a, lin in enumerate((0.8, -0.6)):
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        drift[:, a] = lin * (t - 0.5) + 0.4 * np.sin(2 * np.pi * 1.3 * t + p1) + 0.2 * np.sin(2 * np.pi * 3.1 * t + p2)
    drift -= drift.mean(0)
    nsites = int(n / LOCS_PER_SITE)
    fov = np.sqrt(nsites / DENSITY)
    sites = rng.uniform(0, fov, (nsites, 2))
    sid = np.repeat(np.arange(nsites), rng.poisson(LOCS_PER_SITE, nsites))
    fr = rng.integers(0, F, len(sid))
    sig = np.clip(0.08 / np.sqrt(rng.lognormal(0, 0.6, len(sid))), 0.03, 0.3)
    pos = sites[sid] + drift[fr] + rng.normal(size=(len(sid), 2)) * sig[:, None]
    o = np.argsort(fr, kind='stable')
    crlb = np.repeat(sig[o, None], 2, 1)
    return pos[o].astype(np.float32), fr[o].astype(np.int32), crlb.astype(np.float32), drift, fov


def rmsd(d, gt):
    e = (d - d.mean(0)) - (gt - gt.mean(0))
    return float(np.sqrt((e ** 2).sum(1).mean()))


# ---- methods: f(pos, frame, crlb, F, setting, px_nm) -> drift [F, 2] px ---------------------------------------------

def most_precise(p, f, c, n):
    keep = np.sort(np.argsort(np.prod(c, 1), kind='stable')[:n])
    return p[keep], f[keep], c[keep]


def run_rcc(p, f, c, F, s, px):
    fpb, zoom = s
    return rcc_torch(p, f, timebins=max(2, int(round(F / fpb))), zoom=zoom, numframes=F)


def run_aim(p, f, c, F, s, px):
    fpb, d = s
    return aim(p, f, interval=fpb, intersect_d=d, numframes=F)


def run_comet(p, f, c, F, s, px):
    from comet import comet_run_kd
    fpb, tsig = s
    ds = np.zeros((len(p), 4))
    ds[:, :2] = np.asarray(p)[:, :2] * px
    ds[:, 3] = f
    d = comet_run_kd(ds, segmentation_mode=2, segmentation_var=int(fpb), max_drift_nm=300, target_sigma_nm=tsig,
                     min_max_frames=(0, F - 1), random_state=0)
    return np.asarray(d)[:F, :2] / px


def run_dme2021(p, f, c, F, s, px):
    from dme.legacy import dme_estimate as legacy
    fpb, perspot, rccbins = (list(s) + [10])[:3]
    if len(p) > MAXSPOTS:
        p, f, c = most_precise(p, f, c, MAXSPOTS)
    return legacy(np.ascontiguousarray(p), np.ascontiguousarray(f), np.ascontiguousarray(c), fpb, None, useCuda=True,
                  display=False, estimatePrecision=False, initializeWithRCC=rccbins, perSpotCRLB=perspot, numframes=F)


C2F_STAGES = ((100_000, 4), (300_000, 2), (MAXSPOTS, 1))   # (max localizations, kernel / CRLB and bin-length factor)


def run_dme2021_c2f(p, f, c, F, s, px):
    from dme.legacy import dme_estimate as legacy
    fpb, perspot, rccbins = (list(s) + [20])[:3]
    order = np.argsort(np.prod(c, 1), kind='stable')
    d = None
    for i, (nmax, k) in enumerate(C2F_STAGES):
        keep = np.sort(order[:min(nmax, len(p))])
        pp, ff, cc = (np.ascontiguousarray(a[keep]) for a in (p, f, c))
        last = i == len(C2F_STAGES) - 1
        d = legacy(pp, ff, cc * (1 if last else k), max(4, int(fpb * k)), None, useCuda=True, display=False,
                   estimatePrecision=False, initializeWithRCC=rccbins if d is None else None,
                   initialEstimate=None if d is None else np.asarray(d, np.float32),
                   perSpotCRLB=perspot and last, numframes=F)
        d = pad(np.asarray(d), F)
    return d


def run_dme_v2_fixed(p, f, c, F, s, px):
    return dme_estimate(p, f, c, framesperbin=s, estimatePrecision=False, numframes=F, verbose=False)


def run_dme_v2(p, f, c, F, s, px):
    return dme_estimate(p, f, c, estimatePrecision=False, numframes=F, verbose=False)


RUN = dict(rcc=run_rcc, aim=run_aim, comet=run_comet, dme2021=run_dme2021, dme2021_c2f=run_dme2021_c2f,
           dme_v2_fixed=run_dme_v2_fixed, dme_v2=run_dme_v2)


def grid(m, N, F):
    """Settings searched per method (followed by followups())."""
    fpbs = [x for x in FPB if F / x >= 5]
    return {'rcc': [(fpb, z) for fpb in fpbs if fpb >= 100 for z in (1, 2, 4)],
            'aim': [(fpb, d) for fpb in fpbs for d in (0.05, 0.1, 0.2, 0.3)],
            'comet': [(fpb, 10.0) for fpb in fpbs if F / fpb <= 2000],
            'dme2021': [(fpb, False) for fpb in fpbs if fpb >= 20],
            'dme2021_c2f': [(fpb, False, 20) for fpb in fpbs if fpb >= 50],
            'dme_v2_fixed': fpbs}.get(m, [None])


def followups(m, scored):
    """More settings once the grid is done, from [(error, setting)]: COMET's final kernel width 1, 5, 20 nm at its
    best window length; DME 2021's RCC start with 20 instead of 10 time bins near its best bin length (10 bins can
    leave it in a local optimum)."""
    if not scored:
        return []
    if m == 'comet':
        have = {json.dumps(list(s)) for _, s in scored}
        st1 = [(e, list(s)) for e, s in scored if s[1] == 10.0]
        b = min(st1)[1][0] if st1 else None
        return [] if b is None else [s for s in ([b, t] for t in (1.0, 5.0, 20.0)) if json.dumps(s) not in have]
    if m == 'dme2021':
        norm = lambda s: (list(s) + [10])[:3]
        have = {json.dumps(norm(s)) for _, s in scored}
        st1 = [(e, norm(s)) for e, s in scored if norm(s)[2] == 10]
        if not st1:
            return []
        b = min(st1)[1][0]
        return [s for s in ([fpb, False, 20] for _, (fpb, _, _) in st1 if b / 4 <= fpb <= b * 4)
                if json.dumps(s) not in have]
    return []


EARLY_STOP = {'comet', 'dme2021_c2f'}     # slow methods: stop lengthening the bins once clearly past the optimum


def past_optimum(m, s, scored):
    """True if a shorter bin length with the same other settings already gave > 3x the best error so far."""
    if m not in EARLY_STOP or not scored:
        return False
    s = list(s)
    best = min(e for e, _ in scored)
    shorter = [(list(t)[0], e) for e, t in scored if list(t)[1:] == s[1:] and list(t)[0] < s[0]]
    return bool(shorter) and max(shorter)[1] > 3 * best


def default(m, N, F):
    """Settings a user would start from: RCC 10 time bins at zoom 2, AIM 0.2 px intersection distance, COMET as in
    its README example (60-frame windows, 10 nm), ~20k localizations per bin for AIM and DME with fixed bins."""
    r = rule(N, F)
    return {'rcc': (max(1, F // 10), 2), 'aim': (r, 0.2), 'comet': (60, 10.0), 'dme2021': (r, False),
            'dme2021_c2f': (r, False, 20), 'dme_v2_fixed': r, 'dme_v2': None}[m]


def key(m, s):
    return f'{m}|{json.dumps(s)}'


def pad(d, F):
    """The C++ code returns max(frame)+1 frames: hold the last value."""
    d = np.asarray(d, np.float64)[:F, :2]
    return d if len(d) >= F else np.concatenate([d, np.repeat(d[-1:], F - len(d), 0)])


def _timed_inproc(m, p, f, c, F, s, px):
    torch.cuda.synchronize()
    t = time.time()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            d = pad(RUN[m](p, f, c, F, s, px), F)
        torch.cuda.synchronize()
        return d, time.time() - t, bool(np.isfinite(d).all() and np.abs(d).max() > 0), ''
    except Exception as ex:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None, time.time() - t, False, f'{type(ex).__name__}: {str(ex)[:120]}'


def _child(m, p, f, c, F, s, px, conn):
    conn.send(_timed_inproc(m, p, f, c, F, s, px))
    conn.close()


# The C++/CUDA code can fail with a CUDA error that leaves the CUDA context unusable, so it runs in a fresh process
# per call (timing is measured inside that process).
ISOLATE = {'dme2021', 'dme2021_c2f'}


def timed(m, p, f, c, F, s, px):
    if m not in ISOLATE:
        return _timed_inproc(m, p, f, c, F, s, px)
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    rx, tx = ctx.Pipe(duplex=False)
    pr = ctx.Process(target=_child, args=(m, p, f, c, F, s, px, tx))
    pr.start()
    tx.close()
    try:
        res = rx.recv()
    except EOFError:
        res = None
    pr.join()
    return res or (None, 0.0, False, f'process died (exit code {pr.exitcode})')


def load_json(fn):
    return json.load(open(fn)) if os.path.exists(fn) else {}


def save_json(obj, fn):
    json.dump(obj, open(fn + '.tmp', 'w'), separators=(',', ':'))
    os.replace(fn + '.tmp', fn)


def tune(m, N, F, run_setting, scored):
    """Run the grid, skipping settings past the optimum, then the followups."""
    todo = list({key(m, s): s for s in grid(m, N, F) + [default(m, N, F)]}.values())
    while todo:
        for s in todo:
            if not past_optimum(m, s, scored()):
                run_setting(s)
        todo = followups(m, scored())


# ---- simulations --------------------------------------------------------------------------------------------------

def part_sim(methods, sizes, seeds):
    fn = os.path.join(RES, 'sim.json')
    R = load_json(fn)
    R.setdefault('meta', dict(gpu=torch.cuda.get_device_name(), F=F_SIM, px_nm=PX_NM_SIM, maxspots=MAXSPOTS,
                              drift='linear + sinusoids, periods F/1.3 and F/3.1', sizes=sizes, seeds=seeds))
    runs = R.setdefault('runs', {})
    p, f, c, _, _ = simulate(2e4, 99)                                       # warm-up (compilation, FFT plans)
    for m in methods:
        timed(m, p, f, c, F_SIM, default(m, len(p), F_SIM), PX_NM_SIM)
    for n in sizes:
        for seed in seeds:
            p, f, c, gt, fov = simulate(n, seed)
            N, case = len(p), f'{int(n)}:{seed}'
            print(f'== N={N} seed={seed} FOV={fov:.0f} px', flush=True)
            for m in methods:
                def run_setting(s):
                    rk = f'{case}|{key(m, s)}'
                    if rk in runs:
                        return
                    d, t, ok, err = timed(m, p, f, c, F_SIM, s, PX_NM_SIM)
                    runs[rk] = dict(N=N, n=n, seed=seed, method=m, setting=s, time=t, ok=ok, err=err,
                                    nm=rmsd(d, gt) * PX_NM_SIM if ok else None)
                    print(f'  {m:12s} {json.dumps(s):16s} {t:8.2f} s  ' +
                          (f'{runs[rk]["nm"]:7.3f} nm' if ok else f'FAILED {err}'), flush=True)
                    save_json(R, fn)
                scored = lambda: [(r['nm'], r['setting']) for r in runs.values()
                                  if r['method'] == m and r['n'] == n and r['seed'] == seed and r['ok']]
                tune(m, N, F_SIM, run_setting, scored)
    save_json(R, fn)


# ---- real data ----------------------------------------------------------------------------------------------------

def load_picasso(path):
    import h5py
    L = h5py.File(path, 'r')['locs'][:]
    pos = np.stack([L['x'], L['y']], 1).astype(np.float32)
    crlb = np.stack([L['lpx'], L['lpy']], 1).astype(np.float32)
    ok = np.isfinite(pos).all(1) & np.isfinite(crlb).all(1) & (crlb > 0).all(1)
    return pos[ok], L['frame'][ok].astype(np.int32), crlb[ok]


def checkerboard(pos, ntiles=8):
    lo, hi = pos[:, :2].min(0), pos[:, :2].max(0)
    t = np.floor((pos[:, :2] - lo) / ((hi - lo).max() / ntiles)).astype(np.int64)
    return ((t[:, 0] + t[:, 1]) % 2) == 0


BANDS = [(2, 100), (100, 500), (500, 5000), (5000, 10 ** 9)]   # drift periods in frames


def band_power(x):
    """Power per period band of a drift trace x [F, D] (mean removed, per axis)."""
    F = len(x)
    X = np.fft.rfft(x - x.mean(0), axis=0)
    per = F / np.maximum(np.arange(X.shape[0]), 1e-9)
    out = []
    for lo, hi in BANDS:
        sel = (per >= lo) & (per < hi)
        sel[0] = False
        out.append(np.sum(np.abs(X[sel]) ** 2) * 2 / F ** 2 / x.shape[1])
    return np.array(out)


def held_out_error(dA, dB, rA, rB):
    """Error of a method (drift dA from half A, dB from half B) against a reference (rA, rB): the method's half-A
    drift is compared with the reference from half B and vice versa; the reference's own noise, estimated from
    rA - rB, is subtracted. Returns the RMS error over all bands (same units as the inputs)."""
    e2 = (band_power(dA - rB) + band_power(dB - rA)) / 2 - band_power(rA - rB) / 2
    return float(np.sqrt(np.maximum(e2, 0).sum()))


def part_real(methods, files, px_nm):
    fn = os.path.join(RES, 'real.json')
    R = load_json(fn)
    R.setdefault('meta', dict(gpu=torch.cuda.get_device_name(), maxspots=MAXSPOTS, px_nm=px_nm,
                              metric='held-out-reference half-data error, all bands, mean over references at '
                                     '10 and 20 frames per bin'))
    for path in files:
        name = os.path.basename(path)
        pos, fr, crlb = load_picasso(path)
        F = int(fr.max()) + 1
        A = checkerboard(pos)
        halves = [(pos[A], fr[A], crlb[A]), (pos[~A], fr[~A], crlb[~A])]
        Nh = len(halves[0][0])
        print(f'== {name}: N={len(pos)} F={F}', flush=True)
        info = R.setdefault(name, dict(N=len(pos), F=F, halves={}, full={}))
        D = {}            # drift traces of the halves, nm (not stored)

        def half_run(m, s):
            kk = key(m, s)
            if kk not in D:
                res = [timed(m, *h, F, s, px_nm) for h in halves]
                ok = all(r[2] for r in res)
                info['halves'][kk] = dict(method=m, setting=s, ok=ok, time=sum(r[1] for r in res),
                                          err=' / '.join(r[3] for r in res if r[3]))
                if ok:
                    D[kk] = (res[0][0] * px_nm, res[1][0] * px_nm)
                print(f'  {m:12s} {json.dumps(s):16s} halves {info["halves"][kk]["time"]:7.1f} s' +
                      ('' if ok else f'  FAILED {info["halves"][kk]["err"]}'), flush=True)
            return kk

        refs = [half_run('dme_v2_fixed', fpb) for fpb in (10, 20)]

        def score(kk):
            es = [held_out_error(*D[kk], *D[rk]) for rk in refs if rk != kk and rk in D and kk in D]
            return float(np.mean(es)) if es else None

        for m in methods:
            scored = lambda: [(score(kk), v['setting']) for kk, v in info['halves'].items()
                              if v['method'] == m and score(kk) is not None]
            tune(m, Nh, F, lambda s: half_run(m, s), scored)
            sc = {kk: score(kk) for kk, v in info['halves'].items() if v['method'] == m and score(kk) is not None}
            info.setdefault('scores', {}).update(sc)
            if not sc:
                continue
            for tag, kk in (('best', min(sc, key=sc.get)), ('default', key(m, default(m, Nh, F)))):
                s = info['halves'][kk]['setting'] if kk in info['halves'] else default(m, Nh, F)
                _, t, ok, err = timed(m, pos, fr, crlb, F, s, px_nm)
                info['full'][f'{tag}|{m}'] = dict(method=m, setting=s, half_err_nm=sc.get(kk), time=t, ok=ok, err=err)
                print(f'  -> {tag:7s} {m:12s} {json.dumps(s):16s} error {sc.get(kk)} nm, all data {t:.1f} s',
                      flush=True)
            save_json(R, fn)
    save_json(R, fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--part', choices=['sim', 'real'], required=True)
    ap.add_argument('--methods', nargs='*', default=METHODS)
    ap.add_argument('--sizes', type=float, nargs='*', default=SIZES)
    ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS)
    ap.add_argument('--files', nargs='*', default=[EXAMPLE], help='Picasso hdf5 localization files')
    ap.add_argument('--pixelsize', type=float, default=108.3, help='nm per pixel of the real data')
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.part == 'sim':
        part_sim(a.methods, a.sizes, a.seeds)
    else:
        part_real(a.methods, a.files, a.pixelsize)
