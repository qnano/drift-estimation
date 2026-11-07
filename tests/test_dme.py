"""Smoke tests on small simulations: python -m pytest tests (runs on CPU, and on CUDA if available)."""
import numpy as np
import pytest
import torch

from dme import dme_estimate

DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


def simulate(D, n_frames=1000, n_sites=150, fov=60, seed=0):
    rng = np.random.default_rng(seed)
    err = np.array([0.1, 0.1, 0.03])[:D]
    drift = np.cumsum(rng.normal(0, 0.02, (n_frames, D)), 0) + np.linspace(-0.5, 0.5, n_frames)[:, None]
    drift -= drift.mean(0)
    sites = rng.uniform([0, 0, -1][:D], [fov, fov, 1][:D], (n_sites, D))
    f, s = np.nonzero(rng.random((n_frames, n_sites)) < 0.1)
    pos = sites[s] + drift[f] + rng.normal(0, err, (len(f), D))
    return pos.astype(np.float32), f.astype(np.int32), np.tile(err, (len(f), 1)).astype(np.float32), drift


def rmsd(d, gt):
    return np.sqrt((((d - d.mean(0)) - gt) ** 2).mean(0))


@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('method', ['v2', 'v1'])
@pytest.mark.parametrize('D', [2, 3])
def test_simulation(device, method, D):
    pos, fr, crlb, gt = simulate(D)
    drift, (d1, d2) = dme_estimate(pos, fr, crlb, framesperbin=20, method=method, device=device, verbose=False)
    assert drift.shape == gt.shape
    assert (rmsd(drift, gt) < 0.05).all(), rmsd(drift, gt)
    assert (rmsd(d1, gt) < 0.08).all() and (rmsd(d2, gt) < 0.08).all()


@pytest.mark.parametrize('device', DEVICES)
def test_perspot_v1(device):
    pos, fr, crlb, gt = simulate(2)
    crlb = crlb * np.random.default_rng(1).uniform(0.7, 1.4, (len(pos), 1)).astype(np.float32)
    drift = dme_estimate(pos, fr, crlb, 20, method='v1', perSpotCRLB=True, estimatePrecision=False,
                         device=device, verbose=False)
    assert (rmsd(drift, gt) < 0.05).all()
