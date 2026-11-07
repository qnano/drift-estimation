"""
Figure for the main README: runtime vs. number of localizations on simulated DNA-PAINT-like data
(known drift), for the original C++/CUDA DME, DME v1 (GPU, exact objective) and DME v2.

    python docs/readme_plots.py --run      # simulate + run all methods, writes results json
    python docs/readme_plots.py            # plot from the json
"""
import os, sys, time, json, argparse, contextlib, io
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, 'docs')
RES = os.path.join(OUT, 'benchmark_results.json')

F, FPB, SIGMA, LOCS_PER_SITE, DENSITY = 10000, 100, 0.1, 200, 0.02   # frames, frames/bin, px, locs/site, sites/px^2
SIZES = [1e5, 3e5, 1e6, 3e6, 1e7]
TRACE_N = 1e6


def simulate(n_locs, seed=0):
    """Binding sites at constant density, each giving ~LOCS_PER_SITE localizations at random frames.
    Drift: random walk + slow linear term (px)."""
    rng = np.random.default_rng(seed)
    nsites = int(n_locs / LOCS_PER_SITE)
    fov = np.sqrt(nsites / DENSITY)
    sites = rng.uniform(0, fov, (nsites, 2))
    n = rng.poisson(LOCS_PER_SITE, nsites)
    sid = np.repeat(np.arange(nsites), n)
    fr = rng.integers(0, F, len(sid))
    drift = np.cumsum(rng.normal(0, 0.01, (F, 2)), 0) + np.linspace(-1, 1, F)[:, None] * [1.0, -0.7]
    drift -= drift.mean(0)
    pos = sites[sid] + drift[fr] + rng.normal(0, SIGMA, (len(sid), 2))
    o = np.argsort(fr, kind='stable')
    return pos[o].astype(np.float32), fr[o].astype(np.int32), drift, fov


def rmsd(d, gt):
    e = (d - d.mean(0)) - (gt - gt.mean(0))
    return float(np.sqrt((e ** 2).sum(1).mean()))   # 2D RMSD (px)


def run_method(name, pos, fr, crlb):
    import torch
    from dme import dme_estimate
    torch.cuda.synchronize()
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if name == 'legacy':
            from dme.legacy import dme_estimate as legacy
            d = legacy(pos, fr, crlb, FPB, None, useCuda=True, display=False, estimatePrecision=False,
                       initializeWithRCC=10, numframes=F)
        else:
            d = dme_estimate(pos, fr, crlb, FPB, method=name, estimatePrecision=False, numframes=F)
    torch.cuda.synchronize()
    return np.asarray(d), time.time() - t


def run(methods, max_legacy_time):
    import torch
    rows, traces = [], {}
    pos, fr, gt, _ = simulate(2e4, seed=99)          # warm-up: kernel compilation, cuFFT plans
    for m in methods:
        run_method(m, pos, fr, np.full_like(pos, SIGMA))
    skip = set()
    for n in SIZES:
        pos, fr, gt, fov = simulate(n, seed=1)
        crlb = np.full_like(pos, SIGMA)
        print(f"N={len(pos)} FOV={fov:.0f}px", flush=True)
        for m in methods:
            if m in skip:
                continue
            try:
                d, t = run_method(m, pos, fr, crlb)
                r = rmsd(d, gt)
                ok = bool(np.abs(d).max() > 0)
            except Exception as ex:
                print(f"  {m}: FAILED {type(ex).__name__}: {str(ex)[:100]}", flush=True)
                torch.cuda.empty_cache()
                rows.append(dict(method=m, N=len(pos), time=None, rmsd=None, failed=True))
                continue
            print(f"  {m:7s} {t:8.2f}s  RMSD {r:.4f} px{'' if ok else '  (returned zero drift)'}", flush=True)
            rows.append(dict(method=m, N=len(pos), time=t, rmsd=r, failed=not ok))
            if n == TRACE_N:
                traces[m] = d.tolist()
            if m == 'legacy' and t > max_legacy_time:
                skip.add(m)
        if n == TRACE_N:
            traces['gt'] = gt.tolist()
    gpu = torch.cuda.get_device_name()
    if os.path.exists(RES):        # keep results of methods that were not rerun
        old = json.load(open(RES))
        rows = [r for r in old['rows'] if r['method'] not in methods] + rows
        traces = {**{k: v for k, v in old['traces'].items() if k not in methods}, **traces}
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(gpu=gpu, F=F, fpb=FPB, sigma=SIGMA, rows=rows, traces=traces, trace_N=TRACE_N), open(RES, 'w'),
              separators=(',', ':'))


STYLE = {'legacy': ('Original C++/CUDA code (paper, 2021)', '#9a9a9a', 's'),
         'v1': ('DME v1 (same algorithm, PyTorch)', '#4c72b0', 'o'),
         'v2': ('DME v2', '#dd8452', 'D')}


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = json.load(open(RES))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3), gridspec_kw=dict(width_ratios=[1, 1.6]))
    for m, (label, c, mk) in STYLE.items():
        rows = [r for r in R['rows'] if r['method'] == m and r['time'] is not None]
        if not rows:
            continue
        N = np.array([r['N'] for r in rows])
        good = np.array([not r['failed'] for r in rows])
        ax[0].loglog(N, [r['time'] for r in rows], '-', color=c, marker=mk, ms=5, label=label)
        if (~good).any():
            ax[0].loglog(N[~good], [r['time'] for r, g in zip(rows, good) if not g], 'x', color='r', ms=9, mew=2,
                         label='returned zero drift')
    ax[0].set(xlabel='localizations', ylabel='runtime [s]', title='Runtime (incl. initialization)')
    ax[0].set_ylim(top=ax[0].get_ylim()[1] * 20)      # room for the legend
    ax[0].legend(fontsize=7.5, frameon=False, loc='upper left')
    ax[0].grid(True, which='major', alpha=0.3)

    T = R['traces']
    gt = np.array(T['gt']) * 100
    fr = np.arange(len(gt))
    for i, (axn, off) in enumerate([('x', 0), ('y', 0)]):
        ax[1].plot(fr, gt[:, i], color='k', lw=3, alpha=0.25, label='true drift' if i == 0 else None)
        for m in ('v1', 'v2'):
            if m in T:
                d = np.array(T[m]) * 100
                d = d - d.mean(0) + gt[:, i].mean()
                ax[1].plot(fr, d[:, i], color=STYLE[m][1], lw=0.8, ls='-' if m == 'v2' else '--',
                           label=STYLE[m][0] if i == 0 else None)
        ax[1].text(fr[-1] * 1.01, gt[-1, i], axn, va='center')
    n = int(R['trace_N'])
    ax[1].set(xlabel='frame', ylabel='drift [nm]', title=f'Example trace ({n / 1e6:g}M localizations)')
    ax[1].legend(fontsize=7.5, frameon=False)
    fig.text(0.995, 0.01, f"Simulated, 100 nm px, {R['sigma'] * 100:g} nm precision, {R['F']} frames, "
                          f"{R['fpb']} frames/bin. {R['gpu']}", ha='right', fontsize=7, color='gray')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fn = os.path.join(OUT, 'dme_v2_benchmark.png')
    fig.savefig(fn, dpi=130)
    print('saved', fn)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true')
    ap.add_argument('--methods', nargs='*', default=['legacy', 'v1', 'v2'])
    ap.add_argument('--max-legacy-time', type=float, default=900)
    a = ap.parse_args()
    if a.run:
        run(a.methods, a.max_legacy_time)
    plot()
