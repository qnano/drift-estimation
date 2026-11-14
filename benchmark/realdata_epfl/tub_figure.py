"""Figure: EPFL Tubulin-A647-3D (raw dSTORM, real drift). Overview, a crop without and with DME v2 drift correction,
drift traces (whole movie and a short part with all methods), table of split-half errors.
Writes drift/tub_figure.png."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter

D = os.path.join(os.environ.get('EPFL_DATA', '/mnt/data/SMLM_challenge'), 'Tubulin-A647-3D')
U = np.array([117.0, 118.2, 1000.0])
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e6e5e0'
COL = {'dme_v2': '#2a78d6', 'dme2021': '#1baf7a', 'aim': '#eda100', 'rcc': '#8f8e88', 'comet': '#d1432b'}
R = {}
for m in ('new', 'legacy', 'comet'):
    z = np.load(f'{D}/drift/tub_{m}.npz')
    R.update({k: z[k] for k in z.files})
KEY = {'dme_v2': 'dme_v2_3d', 'dme2021': 'dme2021_3d_250', 'aim': 'aim_2000_0.1', 'rcc': 'rccw_10000'}
NAME = {'dme_v2': 'DME v2', 'dme2021': 'DME 2021', 'aim': 'AIM', 'rcc': 'RCC', 'comet': 'COMET'}


from scipy.interpolate import CubicSpline
F_ = len(R['dme_v2_3d_all'])
for k in [k for k in R if k.startswith('rccw_') and not k.endswith('_time')]:
    fb = int(k.split('_')[1]); nb = F_ // fb
    tc = (np.arange(nb) + 0.5) * F_ / nb                     # bin centres; the linear trace is exact there
    kn = R[k][np.clip(np.round(tc).astype(int), 0, F_ - 1)]
    R[k.replace('rccw_', 'rccs_')] = CubicSpline(tc, kn, bc_type='natural')(np.clip(np.arange(F_), tc[0], tc[-1]))


def nm(k):
    d = R[k] - R[k].mean(0)
    return d * U[:d.shape[1]]


def split(k):
    return np.sqrt(np.mean((nm(k + '_h0') - nm(k + '_h1')) ** 2, 0) / 2)




rs = {fb: split(f'rccs_{fb}') for fb in (250, 500, 1000, 2500, 5000, 10000)}
for fb, e in rs.items():
    print('RCC cubic', fb, e.round(2), 'linear', split(f'rccw_{fb}').round(2), flush=True)
RFB = min(rs, key=lambda fb: np.sqrt(np.mean(rs[fb] ** 2)))
KEY['rcc'] = f'rccs_{RFB}'
# COMET: best of the finished settings (window length in frames, 10 nm kernel)
cs = {int(k.split('_')[1]): split(k[:-4]) for k in R if k.startswith('comet_') and k.endswith('_all')
      and k[:-4] + '_h0' in R and k[:-4] + '_h1' in R}
CFB = min(cs, key=lambda fb: np.sqrt(np.mean(cs[fb] ** 2)))
KEY['comet'] = f'comet_{CFB}_10'
for fb, e in sorted(cs.items()):
    print('COMET', fb, e.round(2), flush=True)
a = pd.read_csv(f'{D}/locs/Tubulin-A647-3D_locs.csv')
a = a[(a['flags'] == 0) & (a.crlb_x < 0.1) & (a.crlb_y < 0.1)]
P = np.stack([a.x * U[0], a.y * U[1]], 1)
fr = a.frame.values
dv = nm('dme_v2_3d_all')
Q = P - dv[fr, :2]
F = len(dv)


def render(q, x0, y0, L, rb, sm=1.0):
    s = (q[:, 0] > x0) & (q[:, 0] < x0 + L) & (q[:, 1] > y0) & (q[:, 1] < y0 + L)
    img, _, _ = np.histogram2d(q[s, 1], q[s, 0], bins=int(round(L / rb)), range=[[y0, y0 + L], [x0, x0 + L]])
    return gaussian_filter(img, sm)


# crop: 3 um window in the top part of the field covering the most filament area; zoom: 700 nm with most filament
from scipy.ndimage import uniform_filter
B, ZL = 3000.0, 700.0
fov = max(np.ptp(Q[:, 0]), np.ptp(Q[:, 1]))
x00, y00 = Q[:, 0].min(), Q[:, 1].min()
g = 100.0
H, _, _ = np.histogram2d(Q[:, 0], Q[:, 1], bins=int(fov / g), range=[[x00, x00 + fov], [y00, y00 + fov]])
occ = np.minimum(gaussian_filter(H, 0.7), np.percentile(H[H > 0], 97))      # bright filaments, capped
w = int(B / g)
cov = uniform_filter(occ, w, mode='constant')
n_ = cov.shape[0]
cov[:, int(0.27 * n_):] = 0; cov[:, :int(0.10 * n_)] = 0        # top part of the field (small y)
cov[:int(0.40 * n_)] = 0; cov[int(0.80 * n_):] = 0               # centre-right in x: the fan of filaments
i, j = np.unravel_index(np.argmax(cov), cov.shape)
cx, cy = x00 + (i - w // 2) * g, y00 + (j - w // 2) * g
s_ = (Q[:, 0] > cx) & (Q[:, 0] < cx + B) & (Q[:, 1] > cy) & (Q[:, 1] < cy + B)
h, ex, ey = np.histogram2d(Q[s_, 0], Q[s_, 1], bins=60, range=[[cx, cx + B], [cy, cy + B]])
o2 = uniform_filter(np.minimum(gaussian_filter(h, 0.7), np.percentile(h[h > 0], 95)), 14, mode='constant')
o2[:8] = 0; o2[-8:] = 0; o2[:, :8] = 0; o2[:, -8:] = 0
i, j = np.unravel_index(np.argmax(o2), o2.shape)
zx, zy = ex[i] - ZL / 2, ey[j] - ZL / 2

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'text.color': INK, 'font.family': 'DejaVu Sans'})
fig = plt.figure(figsize=(14, 9.4), dpi=150)
fig.suptitle('Drift correction of raw 3D dSTORM microtubules', fontsize=16, fontweight='bold', x=0.015, ha='left',
             y=0.975)
fig.text(0.015, 0.93, f'EPFL SMLM challenge 2016, Tubulin-A647-3D (Ries lab): {F:,} frames, 6.7M localizations, '
         f'localized from the raw movie, no fiducials. Drift over the movie: {np.ptp(dv[:, 0]):.0f} nm in x, '
         f'{np.ptp(dv[:, 1]):.0f} nm in y, {np.ptp(dv[:, 2]):.0f} nm in z.\nImages: localizations with precision better '
         'than 0.1 px; each image auto-scaled to its own brightness.', fontsize=10, color=MUTED, ha='left', va='top',
         linespacing=1.4)
gtop = fig.add_gridspec(1, 5, left=0.015, right=0.985, top=0.83, bottom=0.50, wspace=0.04)
panels = [('Whole field, DME v2', render(Q, x00, y00, fov, 20.0, 0.8), fov, None),
          ('No correction', render(P, cx, cy, B, 5.0), B, None), ('DME v2', render(Q, cx, cy, B, 5.0), B, 'dme_v2'),
          ('No correction, zoom', render(P, zx, zy, ZL, 2.5, 1.2), ZL, None),
          ('DME v2, zoom', render(Q, zx, zy, ZL, 2.5, 1.2), ZL, 'dme_v2')]
for k, (title, img, L, hl) in enumerate(panels):
    ax = fig.add_subplot(gtop[0, k])
    ax.imshow(img, cmap='inferno', vmin=0, vmax=np.percentile(img, 99.7 if k else 99.5), interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight='bold', loc='left', pad=4, color=COL['dme_v2'] if hl else INK)
    n = img.shape[0]
    bar = {fov: 5000.0, B: 500.0, ZL: 100.0}[L]
    lb = bar / L * n
    ax.plot([n * 0.95 - lb, n * 0.95], [n * 0.95, n * 0.95], color='white', lw=3)
    ax.text(n * 0.95 - lb / 2, n * 0.93, f'{bar / 1000:g} µm' if bar >= 500 else f'{bar:.0f} nm', color='white',
            ha='center', va='bottom', fontsize=8.5)
    if k == 0:
        ax.add_patch(Rectangle(((cx - x00) / 20, (cy - y00) / 20), B / 20, B / 20, fill=False, ec='white', lw=1.2))
    if k in (1, 2):
        ax.add_patch(Rectangle(((zx - cx) / 5, (zy - cy) / 5), ZL / 5, ZL / 5, fill=False, ec='white', lw=1.0, ls='--'))

gbot = fig.add_gridspec(1, 3, left=0.055, right=0.985, top=0.405, bottom=0.10, wspace=0.26, width_ratios=[1, 1, 1.15])
t = np.arange(F)
T0, T1 = 56000, 68000
ax1 = fig.add_subplot(gbot[0, 0])
for j, (lab, ls) in enumerate((('x', '-'), ('y', '--'), ('z', ':'))):
    ax1.plot(t, dv[:, j], color=COL['dme_v2'], ls=ls, lw=1.4, label=lab)
ax1.axvspan(T0, T1, color=GRID, zorder=0)
ax1.set_xlabel('frame'); ax1.set_ylabel('drift [nm]'); ax1.set_xlim(0, F)
ax1.set_title('Drift estimated by DME v2', fontsize=12, fontweight='bold', loc='left')
ax1.legend(frameon=False, fontsize=9, ncol=3, loc='upper right')
ax2 = fig.add_subplot(gbot[0, 1])
for m in ('rcc', 'aim', 'dme2021', 'comet', 'dme_v2'):
    ax2.plot(t[T0:T1], nm(KEY[m] + '_all')[T0:T1, 0], color=COL[m], lw=2.0 if m == 'dme_v2' else 1.3, label=NAME[m])
ax2.set_xlim(T0, T1); ax2.set_xlabel('frame'); ax2.set_ylabel('x drift [nm]')
ax2.set_title('Part of the x trace, all methods', fontsize=12, fontweight='bold', loc='left')
hh, ll = ax2.get_legend_handles_labels()
ax2.legend(hh[::-1], ll[::-1], frameon=False, fontsize=9, loc='upper right')
for ax in (ax1, ax2):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

# table of split-half errors
ax3 = fig.add_subplot(gbot[0, 2]); ax3.axis('off')
ax3.set_title('Remaining drift error [nm RMS]', fontsize=12, fontweight='bold', loc='left')
rows = [('dme_v2', 'DME v2', 'dme_v2_3d', 'none'), ('dme2021', 'DME 2021', 'dme2021_3d_250', '250 frames/bin'),
        ('comet', 'COMET', KEY['comet'], f'{CFB} frames/window'),
        ('aim', 'AIM', 'aim_2000_0.1', '2000 frames/bin'), ('rcc', 'RCC', KEY['rcc'], f'{RFB} frames/bin')]
zrcc = split('rccorig3d_5000')[2]
cols = [0.0, 0.40, 0.53, 0.65, 0.79]
ax3.text(cols[1], 0.90, 'x', fontweight='bold', ha='right', transform=ax3.transAxes)
ax3.text(cols[2], 0.90, 'y', fontweight='bold', ha='right', transform=ax3.transAxes)
ax3.text(cols[3], 0.90, 'z', fontweight='bold', ha='right', transform=ax3.transAxes)
ax3.text(cols[4] - 0.08, 0.90, 'setting (best)', fontweight='bold', transform=ax3.transAxes)
ax3.plot([0, 1], [0.86, 0.86], color=MUTED, lw=0.8, transform=ax3.transAxes, clip_on=False)
for r, (m, name, key, setting) in enumerate(rows):
    e = split(key)
    y = 0.77 - r * 0.118
    fw = 'bold' if m == 'dme_v2' else 'normal'
    ax3.add_patch(Rectangle((0, y - 0.02), 0.022, 0.07, color=COL[m], transform=ax3.transAxes, clip_on=False))
    ax3.text(0.04, y, name, fontweight=fw, transform=ax3.transAxes, fontsize=11)
    zs = f'{e[2]:.1f}' if len(e) > 2 else (f'{zrcc:.0f}*' if m == 'rcc' else '–')
    for cpos, sv in zip(cols[1:4], (f'{e[0]:.2f}', f'{e[1]:.2f}', zs)):
        ax3.text(cpos, y, sv, ha='right', fontweight=fw, transform=ax3.transAxes, fontsize=11)
    ax3.text(cols[4] - 0.08, y, setting, color=MUTED, transform=ax3.transAxes, fontsize=9.5)
    print(name, e.round(2), flush=True)
ax3.text(0, 0.02, 'Split-half test: the field is split into two halves by a 2 µm\ncheckerboard, each method estimates '
         'the drift on both halves\nseparately; error = RMS difference / √2. Noise only: smooth\n(long-bin) estimates '
         'score well even if they miss fast drift.', fontsize=8.6, color=MUTED, transform=ax3.transAxes, va='bottom',
         linespacing=1.4)
fig.text(0.015, 0.012, f'AIM, RCC, DME 2021 and COMET at the setting with the smallest error (21, 6, 2 and {len(cs)} settings tried); '
         'DME v2 has no settings. AIM and RCC: 2D PyTorch ports of this package, RCC with cubic-spline interpolation.\n'
         '* z of the 2021 RCC code (5000 frames/bin). DME 2021 (its limit) and COMET (run time) use the 1M most precise localizations. '
         'z in stage units of the bead calibration. Code: github.com/jcnossen/drift-estimation', fontsize=8.2,
         color=MUTED, ha='left', va='bottom', linespacing=1.45)
fig.savefig(f'{D}/drift/tub_figure.png', dpi=150, facecolor='white')
print('saved')
