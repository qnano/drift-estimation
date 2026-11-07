"""
3D drift estimation example on a simulated dataset.
Units for X, Y, Z are pixels, pixels and microns.
Runs on a CUDA GPU if available, otherwise on the CPU (slower).
"""
import numpy as np
import matplotlib.pyplot as plt

from dme import dme_estimate, precision


def smlm_simulation(drift_trace, fov_width, loc_error, n_sites, n_frames, on_prob=0.1, seed=0):
    """Binding sites that blink on and off; each localization is shifted by the drift of its frame."""
    rng = np.random.default_rng(seed)
    sites = rng.uniform([0, 0, -1], [fov_width, fov_width, 1], size=(n_sites, 3))
    on = rng.random((n_frames, n_sites)) < on_prob
    frame, site = np.nonzero(on)
    locs = sites[site] + drift_trace[frame] + rng.normal(0, loc_error, size=(len(frame), 3))
    return locs.astype(np.float32), frame.astype(np.int32)


n_frames = 4000
fov_width = 200
loc_error = np.array([0.1, 0.1, 0.03])  # px, px, um

rng = np.random.default_rng(1)
drift_trace = np.cumsum(rng.normal([0.001, 0, 0], [0.02, 0.02, 0.02], size=(n_frames, 3)), 0)
drift_trace -= drift_trace.mean(0)

localizations, framenum = smlm_simulation(drift_trace, fov_width, loc_error, n_sites=400, n_frames=n_frames)
crlb = np.tile(loc_error, (len(localizations), 1)).astype(np.float32)
print(f"Total localizations: {len(localizations)}")

# DME v2 (default)
drift, (set1, set2) = dme_estimate(localizations, framenum, crlb, framesperbin=20)
# Algorithm of the paper (exact pairwise objective, RCC init), still on the GPU
drift_v1 = dme_estimate(localizations, framenum, crlb, framesperbin=20, method='v1', estimatePrecision=False)

for name, d in [('v2', drift), ('v1', drift_v1)]:
    rmsd = np.sqrt(np.mean((d - d.mean(0) - drift_trace) ** 2, 0))
    print(f"DME {name}: RMSD vs true drift = {rmsd} (px, px, um)")
print(f"Split-half precision estimate (v2): {precision(set1, set2)}")

fig, ax = plt.subplots(3, figsize=(7, 6), sharex=True)
for i in range(3):
    ax[i].plot(drift_trace[:, i], 'k', lw=2, label='True drift')
    ax[i].plot(drift[:, i], label='DME v2')
    ax[i].plot(drift_v1[:, i], '--', label='DME v1')
    ax[i].set_ylabel(f"{'xyz'[i]} [{['px', 'px', 'um'][i]}]")
ax[0].legend()
ax[-1].set_xlabel('Frame')
plt.tight_layout()
plt.show()
