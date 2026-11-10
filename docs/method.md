# DME v2: method

DME estimates drift from the localizations alone. A drift trace is good when, after correcting for it, the
localizations of each binding site collapse onto each other; DME measures this with an entropy bound and maximizes it.
DME v2 keeps the objective of the 2021 paper, evaluates it much faster, and replaces the fixed-resolution spline by a
learned temporal prior.

## 1. Objective

Every localization *i* is treated as a Gaussian *p<sub>i</sub>* with its drift-corrected position as mean and its
localization precision (CRLB) as standard deviation. DME minimizes an upper bound on the entropy of the mixture of all
these Gaussians (the pairwise-KL bound). Up to drift-independent terms this means maximizing

S(drift) = mean<sub>i</sub> log( 1 + Σ<sub>j: frame j ≠ frame i</sub> exp(−KL(p<sub>i</sub> ‖ p<sub>j</sub>)) )

with KL = |u<sub>i</sub> − u<sub>j</sub>|² / 2σ² for a constant CRLB σ (u = drift-corrected position / σ). Pairs
within the same frame are excluded: they do not depend on the drift.

## 2. Fast evaluation: density grid

The 2021 code (and `method='v1'`) evaluates S as an explicit sum over pairs: for every localization it finds the
neighbours within ~3σ (a KD-tree in the C++ code, a GPU spatial hash in v1) and sums the kernel over them. The cost
grows with the number of neighbour pairs, which in dense DNA-PAINT data runs into billions.

But the sum over *j* of a Gaussian around every localization is a blurred density, evaluated at localization *i*.
DME v2 computes that density once on a regular grid, instead of forming pairs:

- all localizations are splatted onto a grid (spacing σ/2, cloud-in-cell), blurred with a separable Gaussian whose
  width makes splat + blur + linear interpolation equal to the unit Gaussian kernel, and read back at each
  localization;
- the self term and the same-frame pairs are subtracted exactly;
- the gradient uses the same grids with the kernel derivative, plus a second grid weighted by 1 / Z<sub>j</sub>;
- large fields of view are processed in tiles; empty tiles are skipped.

The cost is O(N + grid cells) per evaluation, independent of the number of neighbours, so dense DNA-PAINT data
(thousands of neighbours per localization) is as cheap as sparse data. For sparse data an exact pair kernel (GPU
spatial hash, CuPy or PyTorch) is used instead; the cheaper backend is chosen automatically. Grid and exact evaluation
agree to ~10<sup>−5</sup> in the objective; the final drift is identical to < 10<sup>−6</sup> px.

The CRLB enters as one value per axis (the mean). Per-localization CRLBs changed the result by less than 1% in our
tests; they remain available in `method='v1'`.

## 3. Optimization

- **Start:** three initial estimates (intersection tracking as in AIM, RCC, zero drift) are each optimized at the
  coarsest level; the one with the highest DME score is continued. A single start can fail, e.g. AIM on sparse data
  with long binding events.
- **Coarse to fine:** kernel widths 4σ, 2σ, 1σ, with coarser time bins at the coarse levels; this smooths the
  objective and avoids local optima.
- **L-BFGS** preconditioned with the per-knot 2×2 (3×3 in 3D) block Hessian of the objective.

With a fixed `framesperbin`, the drift is a cubic Catmull-Rom spline with one knot per `framesperbin` frames and this
is the result (`dme_estimate(..., framesperbin=...)`).

## 4. Learned temporal prior (default)

A spline with a fixed knot spacing forces the user to trade noise (short bins) against bias (long bins). Without
`framesperbin`, DME v2 instead:

1. **Measures** short segments. Time is cut into segments of *L* frames. At the current drift estimate, the DME
   objective gives for every segment the gradient *g<sub>t</sub>* and curvature *H<sub>t</sub>* with respect to shifting
   that segment alone (all other localizations act as reference). This gives a Newton measurement of the segment's
   drift, z<sub>t</sub> = d<sub>t</sub> + (−H<sub>t</sub>)<sup>−1</sup> g<sub>t</sub> (limited to 1.5σ), with
   information J<sub>t</sub> = −N H<sub>t</sub>. Segments without a positive-definite curvature are marked invalid.
2. **Smooths** with a network, one axis at a time:
   - *input*, per segment *t*: the measurement z<sub>t</sub>, its standard deviation s<sub>t</sub> (from
     J<sub>t</sub><sup>−1</sup>) and a validity flag. Invalid segments get z interpolated from their neighbours and a
     30× larger s. The network sees scale-free features: the step z<sub>t</sub> − z<sub>t−1</sub> and log s<sub>t</sub>,
     both relative to the median s, the validity flag, and four robust estimates of the actual noise level (from second
     differences at lags 1, 2, 4 and 8 segments; these reveal a miscalibrated s);
   - *output*: the noise of each measurement in units of s<sub>t</sub>, *e<sub>t</sub>*. The drift of segment *t* is
     z<sub>t</sub> − e<sub>t</sub> s<sub>t</sub>, and the drift per frame is the linear interpolation between segment
     centres.

   The network is fully convolutional (20 dilated residual blocks, 64 channels, ~0.33M parameters, receptive field
   ~4000 segments), so it works for any number of segments, and it never sees the localizations themselves, so it does
   not depend on the sample or the field of view.
3. **Repeats** steps 1–2 at kernel widths 4σ, 2σ, 1σ, 1σ, 1σ, starting from the DME fixed-spline estimate.

The segment length *L* (6–192 frames, at least 30 localizations per segment and 24 segments) and the bin length of the
fixed-spline start (~20k localizations per bin, and 10× finer) are chosen by a **spatial cross-fit**: the localizations
are split into two halves by an 8×8 checkerboard (different binding sites, same drift); the drift estimated from one
half is scored by the DME objective of the other half, and vice versa. Too noisy and too smooth estimates both score
lower on the other half. The plain fixed-spline estimate is one of the candidates, so DME v2 falls back to it when the
prior does not help. The chosen candidate is then run on all localizations.

**Training.** The network is trained only on synthetic sequences (`training/train_temporal_prior.py`): drift from a
broad generator (polynomials, slow sinusoids, random walks, vibrations, steps, Gaussian-process drift, damped
oscillations, piecewise-linear) plus measurement noise with a random per-segment scale, 0.5–3× miscalibration, AR(1)
correlation up to 0.85 (multi-frame binding events make neighbouring segments correlated), outliers and missing
segments. No experimental data and none of the benchmark simulations are used for training.

**Why a generic, learned prior.** Any smoothing of noisy drift measurements is a prior. A smoothing spline is the
posterior mean under one particular Gaussian-process prior [1, 2], so choosing `framesperbin` is choosing how fast the
drift is assumed to change, and no single choice suits both slow thermal drift and a sudden step. Work on plug-and-play
priors and regularization by denoising has shown that the prior can be separated from the measurement model and
supplied by a generic denoiser [3, 4]. Here the physics stays in the DME measurements (z<sub>t</sub>,
J<sub>t</sub>) and the network only does the temporal denoising. A denoiser trained to minimize the squared error
estimates the posterior mean under the distribution it was trained on (Tweedie's formula [5]; a trained denoiser
contains a usable prior [6]). Trained on a deliberately broad drift generator, it therefore acts as a mixture of many
drift models and adapts its effective smoothing to each trace. Giving the network the per-segment noise level follows
noise-level-conditioned denoisers [7]. Training only on simulations amortizes Bayesian inference [8, 9], and
networks trained only on synthetic time series transfer to real ones [10]. Networks trained on simulated data are also
established in localization microscopy itself [11, 12]. The weak point is coverage: the prior is only as good as the
generator. Hence the broad generator with miscalibrated, correlated and missing measurements, and the cross-fit, which
keeps the plain spline as a candidate and so rejects the prior on held-out binding sites when it does not help.

**3D data.** The learned prior never builds a 3D grid: imaging is 2D, so in 3D the localizations are spread thinly
in z and a 3D grid would be mostly empty (billions of cells for a 200 × 200 px field of view). Instead, the x, y drift
is estimated on the x–y projection, and then the z drift on the x–z projection of the x, y-corrected localizations
(z rescaled so that its precision equals that of x); both use 2D grids. With fixed bins, 3D data uses the exact pair
sum or a 3D grid, whichever is cheaper.

## 5. Precision estimate

With `estimatePrecision=True` the estimation is repeated on the two halves of the data split along x or y; the RMS
difference between the two traces (`dme.precision`) indicates the precision.

## 6. Other implementations in this package

- `method='v1'`: the algorithm of the 2021 paper (exact pairwise objective with per-localization CRLBs, RCC start,
  optional coarse pass), with the neighbour search on the GPU (spatial hash) instead of a CPU KD-tree, and L-BFGS.
- `dme.legacy.dme_estimate`: the original C++/CUDA code (needs the compiled library).
- `dme.aim` and `dme.rcc_torch`: PyTorch implementations of AIM (Ma et al., Sci. Adv. 2024) and redundant
  cross-correlation (Wang et al. 2014), used as starting points and in the comparison.

## 7. Limitations

- The learned prior costs time: 3 s for 10k localizations, 20 s for 1M, about 3 minutes for 10M on an RTX 3090 (the
  cross-fit runs several candidates on both halves). With a fixed `framesperbin` it takes 0.2–3 s.
- It needs at least 24 segments; for very short recordings it falls back to the fixed spline.
- In 3D, the z drift comes from the x–z projection, where localizations at different y overlap; this is less
  precise than a full 3D evaluation would be for very dense samples.
- One CRLB per axis (see section 2).

## References

1. G. S. Kimeldorf, G. Wahba, "A correspondence between Bayesian estimation on stochastic processes and smoothing by
   splines", Ann. Math. Statist. 41, 495 (1970). [doi:10.1214/aoms/1177697089](https://doi.org/10.1214/aoms/1177697089)
2. G. Wahba, "Improper priors, spline smoothing and the problem of guarding against model errors in regression",
   J. R. Stat. Soc. B 40, 364 (1978). [doi:10.1111/j.2517-6161.1978.tb01050.x](https://doi.org/10.1111/j.2517-6161.1978.tb01050.x)
3. S. V. Venkatakrishnan, C. A. Bouman, B. Wohlberg, "Plug-and-play priors for model based reconstruction",
   IEEE GlobalSIP, 945 (2013). [doi:10.1109/GlobalSIP.2013.6737048](https://doi.org/10.1109/GlobalSIP.2013.6737048)
4. Y. Romano, M. Elad, P. Milanfar, "The little engine that could: regularization by denoising (RED)",
   SIAM J. Imaging Sci. 10, 1804 (2017). [doi:10.1137/16M1102884](https://doi.org/10.1137/16M1102884)
5. B. Efron, "Tweedie's formula and selection bias", J. Am. Stat. Assoc. 106, 1602 (2011).
   [doi:10.1198/jasa.2011.tm11181](https://doi.org/10.1198/jasa.2011.tm11181)
6. Z. Kadkhodaie, E. P. Simoncelli, "Stochastic solutions for linear inverse problems using the prior implicit in a
   denoiser", NeurIPS 34 (2021). [arXiv:2007.13640](https://arxiv.org/abs/2007.13640)
7. K. Zhang, W. Zuo, L. Zhang, "FFDNet: toward a fast and flexible solution for CNN-based image denoising",
   IEEE Trans. Image Process. 27, 4608 (2018). [doi:10.1109/TIP.2018.2839891](https://doi.org/10.1109/TIP.2018.2839891)
8. K. Cranmer, J. Brehmer, G. Louppe, "The frontier of simulation-based inference", PNAS 117, 30055 (2020).
   [doi:10.1073/pnas.1912789117](https://doi.org/10.1073/pnas.1912789117)
9. S. Müller, N. Hollmann, S. Pineda Arango, J. Grabocka, F. Hutter, "Transformers can do Bayesian inference",
   ICLR (2022). [arXiv:2112.10510](https://arxiv.org/abs/2112.10510)
10. S. Dooley, G. S. Khurana, C. Mohapatra, S. Naidu, C. White, "ForecastPFN: synthetically-trained zero-shot
    forecasting", NeurIPS (2023). [arXiv:2311.01933](https://arxiv.org/abs/2311.01933)
11. E. Nehme, L. E. Weiss, T. Michaeli, Y. Shechtman, "Deep-STORM: super-resolution single-molecule microscopy by deep
    learning", Optica 5, 458 (2018). [doi:10.1364/OPTICA.5.000458](https://doi.org/10.1364/OPTICA.5.000458)
12. A. Speiser et al., "Deep learning enables fast and dense single-molecule localization with high accuracy",
    Nat. Methods 18, 1082 (2021). [doi:10.1038/s41592-021-01236-x](https://doi.org/10.1038/s41592-021-01236-x)
