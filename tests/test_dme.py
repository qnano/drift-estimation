"""Smoke tests on small simulations: python -m pytest tests (runs on CPU, and on CUDA if available)."""
import numpy as np
import pytest
import torch

from dme import DME, dme_estimate

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
@pytest.mark.parametrize('method,fpb', [('v2', None), ('v2', 20), ('v1', 20)])
@pytest.mark.parametrize('D', [2, 3])
def test_simulation(device, method, fpb, D):
    pos, fr, crlb, gt = simulate(D)
    drift, (d1, d2) = dme_estimate(pos, fr, crlb, framesperbin=fpb, method=method, device=device, verbose=False)
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


@pytest.mark.parametrize('device', DEVICES)
def test_dme_class(device):
    pos, fr, crlb, gt = simulate(2)
    est = DME(device=device)
    with pytest.raises(RuntimeError):
        est.apply(pos, fr)
    assert est.fit(pos, fr, crlb) is est
    assert est.drift.shape == gt.shape and (rmsd(est.drift, gt) < 0.05).all()
    assert np.allclose(est.apply(pos, fr), pos - est.drift[fr])
    assert est.precision().shape == (2,) and (est.precision() < 0.1).all()
    # same result through the function interface
    drift = dme_estimate(pos, fr, crlb, estimatePrecision=False, device=device, verbose=False)
    assert np.allclose(drift, est.drift, atol=1e-4)



@pytest.mark.parametrize('device', DEVICES)
def test_aim_3d(device):
    from dme import aim
    pos, fr, crlb, gt = simulate(3, n_sites=400)
    d2 = aim(pos[:, :2], fr, interval=20, intersect_d=0.2, device=device)
    d3 = aim(pos, fr, interval=20, intersect_d=0.2, intersect_dz=0.1, device=device)
    assert d2.shape == (len(gt), 2) and d3.shape == gt.shape
    assert np.allclose(d2, d3[:, :2], atol=1e-5)          # x-y is not changed by the z step
    e = rmsd(d3, gt)
    assert (e < 0.05).all(), e                              # uncorrected: 0.33, 0.62 px and 0.17 um
