"""
Figures and tables from benchmark/results (see compare_methods.py).

    python benchmark/compare_plots.py              # docs/method_comparison.png (README), docs/method_overview.png
    python benchmark/compare_plots.py --markdown   # also print the tables used in docs/comparison.md
"""
import os, json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
OUT_PNG = os.path.join(ROOT, 'docs', 'method_comparison.png')
OUT_OVERVIEW = os.path.join(ROOT, 'docs', 'method_overview.png')
F_SIM = 10000
README_METHODS = ['dme_v2', 'dme2021x', 'comet', 'aim', 'rcc']

ORDER = ['dme_v2', 'dme_v2_fixed', 'dme2021x', 'dme2021_c2f', 'dme2021', 'comet', 'aim', 'rcc']
STYLE = {   # validated categorical palette (fixed order), marker as secondary encoding
    'dme_v2':        ('DME v2 (2026, learned temporal prior)', '#2a78d6', 'o'),
    'dme_v2_fixed':        ('DME v2 with fixed bins (no learned prior)', '#eb6834', 's'),
    'dme2021x':  ('DME 2021 (published C++/CUDA, ≤1M localizations)', '#1baf7a', 'D'),
    'dme2021_c2f': ('DME 2021, coarse to fine', '#1baf7a', 'D'),
    'dme2021':   ('DME 2021, single stage', '#1baf7a', 'd'),
    'comet':     ('COMET (Reinkensmeier et al. 2026)', '#008300', 'P'),
    'aim':       ('AIM (Ma et al. 2024)', '#eda100', '^'),
    'rcc':       ('RCC (Wang et al. 2014)', '#e87ba4', 'v'),
}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
CAP_TEXT = 'above 1M: DME 2021 uses the\n1M most precise localizations'

NOTES = [  # method, settings searched (best = lowest error vs. the true drift), limits / caps
    ('DME v2', 'none: chooses its temporal resolution itself (segments of 6–192 frames, by a spatial cross-fit)', ''),
    ('DME 2021', 'bins 20–2000 frames; RCC start 10 or 20 bins; single stage or coarse to fine '
                 '(100k loc. at 4× kernel → 300k at 2× → final)', 'at most the 1M most precise localizations'),
    ('COMET', 'windows 10–2000 frames at 10 nm final kernel, then 1, 5, 20 nm at the best window',
     'max. drift 300 nm (default); real data: ≤ 2000 windows'),
    ('AIM', 'bins 10–2000 frames × intersection distance 5–30 nm', ''),
    ('RCC', 'bins 100–2000 frames × render zoom 1, 2, 4', 'zoom 4 does not fit in 24 GB GPU memory at 3M–10M'),
    ('all', 'bin sweeps of COMET and DME 2021 stop once the error exceeds 3× the best; runtime = best setting, '
            'incl. initialization', ''),
]
DEFAULTS_TEXT = ('Defaults: RCC 10 bins, zoom 2 · AIM 0.2 px, ~20k loc./bin · COMET 60-frame windows, 10 nm '
                 '(README example) · DME 2021 single stage, ~20k loc./bin, RCC 10 bins · DME v2 has no settings')


def rule(N, F):
    return int(np.clip(round(20000 * F / N), 10, 2000))


def default_setting(m, N, F=F_SIM):
    r = rule(N, F)
    return {'rcc': [max(1, F // 10), 2], 'aim': [r, 0.2], 'comet': [60, 10.0], 'dme2021': [r, False],
            'dme_v2_fixed': r, 'dme_v2': None}.get(m, 'none')


def summarize_sim(kind='best'):
    """Mean over seeds per (method, size). kind='best': best setting per case; 'default': the default setting.
    'dme2021x' = DME 2021 run either way (single stage or coarse to fine), best of both."""
    R = json.load(open(os.path.join(RES, 'sim.json')))
    by = collections.defaultdict(list)
    for r in R['runs'].values():
        by[(r['n'], r['seed'], r['method'])].append(r)
        if r['method'] in ('dme2021', 'dme2021_c2f'):
            by[(r['n'], r['seed'], 'dme2021x')].append(r)
    S = collections.defaultdict(lambda: dict(nm=[], time=[], setting=[], N=[]))
    for (n, seed, m), rs in by.items():
        ok = [r for r in rs if r['ok']]
        if kind == 'default':
            dm = 'dme2021' if m == 'dme2021x' else m
            ok = [r for r in ok if r['method'] == dm and json.dumps(r['setting']) == json.dumps(default_setting(dm, r['N']))]
        if not ok:
            continue
        b = min(ok, key=lambda r: r['nm'])
        s = S[(m, n)]
        s['nm'].append(b['nm']); s['time'].append(b['time']); s['N'].append(b['N'])
        s['setting'].append((b['method'], b['setting']) if m == 'dme2021x' else b['setting'])
    out = {}
    for (m, n), s in S.items():
        out.setdefault(m, {})[str(int(n))] = dict(nm=float(np.mean(s['nm'])), nm_min=float(np.min(s['nm'])),
                                                  nm_max=float(np.max(s['nm'])), time=float(np.mean(s['time'])),
                                                  N=float(np.mean(s['N'])), seeds=len(s['nm']), settings=s['setting'])
    return R['meta'], out


REAL_FILE = os.path.join(RES, 'real.json')
REAL_TITLE = 'GATTAquant 80 nm nanoruler, DNA-PAINT (638k loc., 5000 frames)'


def summarize_real():
    """{method: dict(best_nm, best_t, best_setting, def_nm, def_t)} for the (single) real dataset; errors are the
    held-out-reference half-data error, runtime is on the full dataset."""
    if not os.path.exists(REAL_FILE):
        return None
    R = json.load(open(REAL_FILE))
    name = [k for k in R if k != 'meta'][0]
    full = R[name]['full']
    out = {}
    for k, v in full.items():
        tag, m = k.split('|')
        if v.get('half_err_nm') is None or not v['ok']:
            continue
        o = out.setdefault(m, {})
        p = 'best' if tag == 'best' else 'def'
        o[p + '_nm'], o[p + '_t'], o[p + '_setting'] = v['half_err_nm'], v['time'], v['setting']
    cand = [m for m in ('dme2021', 'dme2021_c2f') if m in out and 'best_nm' in out[m]]
    if cand:
        b = min(cand, key=lambda m: out[m]['best_nm'])
        out['dme2021x'] = dict(out[b], **{k: out['dme2021'][k] for k in ('def_nm', 'def_t', 'def_setting')
                                          if 'dme2021' in out and k in out['dme2021']})
    return dict(name=name, gpu=R['meta']['gpu'], N=R[name]['N'], F=R[name]['F'], m=out)


def _real_bars(ax, RR, methods, key, xlabel, title, log=False, show_default=True):
    ms = [m for m in ORDER if m in methods and m in RR['m'] and f'best_{key}' in RR['m'][m]]
    for i, m in enumerate(ms):
        label, col, mk = STYLE[m]
        v = RR['m'][m][f'best_{key}']
        ax.barh(i, v, height=0.58, color=col, zorder=3)
        dv = RR['m'][m].get(f'def_{key}') if show_default else None
        if dv is not None and m != 'dme_v2':
            ax.plot(dv, i, mk, mfc='white', mec=col, mew=1.6, ms=7, zorder=4)
        txt = (f'{v:.2f} nm' if key == 'nm' else (f'{v:.2g} s' if v < 10 else f'{v:.0f} s'))
        xl = v if dv is None or m == 'dme_v2' or dv > 1.6 * v else max(v, dv) * 1.03
        ax.text(xl, i, '  ' + txt, va='center', fontsize=7.5, color=INK, zorder=5)
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels([STYLE[m][0].split(' (')[0] for m in ms])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc='left', fontweight='bold', color=INK)
    ax.grid(True, axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
    if log:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.margins(x=0.25)


def _style():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                         'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                         'font.family': 'DejaVu Sans'})


def _lines(ax, S, methods, key):
    for m in ORDER[::-1]:
        if m not in S or m not in methods:
            continue
        label, c, mk = STYLE[m]
        ns = sorted(S[m], key=float)
        ax.plot([float(n) for n in ns], [S[m][n][key] for n in ns], '-', color=c, lw=2.6 if m == 'dme_v2' else 2,
                marker=mk, ms=6.5, mec='white', mew=1.0, label=label, zorder=4 if m == 'dme_v2' else 3)


def _axes(a, ylabel, title):
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('localizations (10,000 frames)'); a.set_ylabel(ylabel)
    a.set_title(title, loc='left', fontweight='bold', color=INK)
    a.grid(True, which='major', color=GRID, lw=0.8); a.set_axisbelow(True)
    a.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}'))
    a.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: {1e4: '10k', 1e5: '100k', 1e6: '1M', 1e7: '10M'}.get(v, '')))


def _cap_marker(a, text=None):
    a.axvline(1e6, color=MUTED, lw=1, ls=':', zorder=1)
    if text:
        a.text(1.12e6, 0.03, text, transform=a.get_xaxis_transform(), fontsize=7, color=MUTED, va='bottom')


def _legend(fig, ax, y):
    h, l = ax.get_legend_handles_labels()
    idx = [l.index(STYLE[m][0]) for m in ORDER if STYLE[m][0] in l]
    fig.legend([h[i] for i in idx], [l[i] for i in idx], loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(0.5, y), fontsize=8.5, handlelength=2.2, columnspacing=1.6)


def plot(S, meta, methods, out, RR=None):
    _style()
    if RR is not None:
        fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.1), gridspec_kw=dict(width_ratios=[1, 1, 0.85]))
        _real_bars(ax[2], RR, methods, 'nm', 'half-data error vs. held-out reference (nm)', 'Real data')
        ax[2].text(0, -0.24, REAL_TITLE + chr(10) + 'bars: best settings · hollow: default settings', transform=ax[2].transAxes,
                   fontsize=7, color=MUTED, va='top')
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.1))
    _lines(ax[0], S, methods, 'nm'); _lines(ax[1], S, methods, 'time')
    _axes(ax[0], 'drift error vs. true drift (nm RMS)', 'Precision (simulations)')
    _axes(ax[1], 'runtime (s)', 'Compute time (simulations)')
    if any(m.startswith('dme2021') for m in methods):
        _cap_marker(ax[0]); _cap_marker(ax[1], CAP_TEXT)
    _legend(fig, ax[0], 1.04)
    fig.text(0.99, -0.01, f"Simulated DNA-PAINT, smooth drift, mean of {meta.get('nseeds', 3)} seeds. All methods except "
             f"DME v2 at their best settings (picked with the true drift). {meta['gpu']}.",
             ha='right', fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    fig.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', out)


def overview(S, D, meta, methods, out, RR=None):
    """Best settings (precision, runtime), default settings (precision), real data, and a table of what was
    tuned / capped."""
    _style()
    fig = plt.figure(figsize=(13, 14.6))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 0.72, 0.62], hspace=0.42, wspace=0.3)
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    _lines(a, S, methods, 'nm'); _lines(b, S, methods, 'time'); _lines(c, D, methods, 'nm')
    _axes(a, 'drift error vs. true drift (nm RMS)', 'a  Precision, best settings')
    _axes(b, 'runtime (s)', 'b  Compute time, best settings')
    _axes(c, 'drift error vs. true drift (nm RMS)', 'c  Precision, default settings')
    for ax_ in (a, b, c):
        _cap_marker(ax_, CAP_TEXT if ax_ is b else None)
    # d: ratio of each method's best-setting error to DME v2's (internal key 'dme_v2' = DME v2 with learned prior)
    d = fig.add_subplot(gs[1, 1])
    for m in ORDER[::-1]:
        if m not in S or m not in methods or m == 'dme_v2':
            continue
        label, col, mk = STYLE[m]
        ns = sorted(set(S[m]) & set(S['dme_v2']), key=float)
        d.plot([float(n) for n in ns], [S[m][n]['nm'] / S['dme_v2'][n]['nm'] for n in ns], '-', color=col, lw=2,
               marker=mk, ms=6.5, mec='white', mew=1.0)
    d.axhline(1, color=STYLE['dme_v2'][1], lw=2.6)
    _axes(d, 'error relative to DME v2 (best settings)', 'd  How much larger than DME v2')
    d.set_yticks([1, 2, 3, 5, 10, 20]); d.yaxis.set_minor_formatter(mt.NullFormatter())
    d.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f'{v:g}×'))
    _cap_marker(d)
    if RR is not None:
        e, f = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
        _real_bars(e, RR, methods, 'nm', 'half-data error vs. held-out reference (nm)',
                   'e  Real data: precision (bars best, hollow default)')
        _real_bars(f, RR, methods, 't', 'runtime on all localizations (s)', 'f  Real data: compute time, best settings',
                   log=True, show_default=False)
        e.text(0, -0.28, REAL_TITLE + '. Error: drift from one half of a spatial checkerboard split vs. a fine '
               'DME reference from the other half (no ground truth).', transform=e.transAxes, fontsize=7, color=MUTED,
               va='top')
    # notes: method | settings searched | limits, as wrapped text rows
    import textwrap
    t = fig.add_subplot(gs[3, :]); t.axis('off')
    xs, widths = (0.0, 0.085, 0.66), (12, 92, 48)
    y = 1.0
    for i, row in enumerate([('method', 'settings searched (best setting: simulations by the true drift, real data by the held-out-reference error)', 'limits')] + NOTES):
        cols = [textwrap.wrap(txt, w) or [''] for txt, w in zip(row, widths)]
        for x, lines in zip(xs, cols):
            t.text(x, y, chr(10).join(lines), transform=t.transAxes, va='top', fontsize=8.3, color=INK,
                   fontweight='bold' if i == 0 or x == 0.0 else 'normal', linespacing=1.35)
        y -= 0.075 * max(len(l) for l in cols) + 0.035
        t.plot([0, 1], [y + 0.02, y + 0.02], color=GRID, lw=0.8, transform=t.transAxes, clip_on=False)
    t.text(0, y - 0.02, DEFAULTS_TEXT, transform=t.transAxes, fontsize=8, color=MUTED, va='top')
    _legend(fig, a, 0.965)
    fig.suptitle('Drift estimation methods. a–d: simulated DNA-PAINT (smooth drift, 10,000 frames, '
                 f"mean of {meta.get('nseeds', 3)} seeds); e–f: real DNA-PAINT. {meta['gpu']}", x=0.01, y=0.995, ha='left',
                 fontsize=11.5, fontweight='bold', color=INK)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('wrote', out)


def markdown(S, D, RR):
    """Tables for docs/comparison.md."""
    names = [('dme_v2', 'DME v2'), ('comet', 'COMET'), ('dme2021x', 'DME 2021'), ('aim', 'AIM'), ('rcc', 'RCC'),
             ('dme_v2_fixed', 'DME v2, fixed bins')]
    sizes = sorted({n for m in S.values() for n in m}, key=float)
    lab = lambda n: {1e4: '10k', 3e4: '30k', 1e5: '100k', 3e5: '300k', 1e6: '1M', 3e6: '3M', 1e7: '10M'}[float(n)]
    ts = lambda t: f'{t:.2f}' if t < 1 else (f'{t:.1f}' if t < 10 else f'{t:.0f}')
    for title, X in (('best settings', S), ('default settings', D)):
        print(f'{chr(10)}**Simulations, {title}**: drift error (nm) · runtime (s){chr(10)}')
        print('| method | ' + ' | '.join(lab(n) for n in sizes) + ' |')
        print('|---|' + '---|' * len(sizes))
        for m, nm in names:
            if m in X:
                print(f'| {nm} | ' + ' | '.join(f"{X[m][n]['nm']:.3g} · {ts(X[m][n]['time'])}" if n in X[m] else '–'
                                                 for n in sizes) + ' |')
    if RR:
        print(f'{chr(10)}**Real data** ({RR["name"]}): half-data error (nm) and runtime on all localizations (s){chr(10)}')
        print('| method | best settings | default settings | runtime (best) | best setting |')
        print('|---|---|---|---|---|')
        for m, nm in names:
            v = RR['m'].get(m)
            if v:
                d = '–' if m == 'dme_v2' else f"{v.get('def_nm', float('nan')):.2f}"
                print(f"| {nm} | {v['best_nm']:.2f} | {d} | {ts(v['best_t'])} | {v.get('best_setting')} |")


if __name__ == '__main__':
    import sys
    meta, S = summarize_sim('best')
    _, D = summarize_sim('default')
    meta['nseeds'] = max(v['seeds'] for m in S.values() for v in m.values())
    RR = summarize_real()
    plot(S, meta, README_METHODS, OUT_PNG, RR)
    overview(S, D, meta, README_METHODS, OUT_OVERVIEW, RR)
    if '--markdown' in sys.argv:
        markdown(S, D, RR)
