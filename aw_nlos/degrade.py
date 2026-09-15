"""Photon degradation: recreate the paper's low-SBR, photon-starved regime.

The public confocal captures are high-SNR lab data (SBR ~4-12, hundreds of signal PPP),
while AW-NLOS targets far weaker returns under strong ambient light. Two physically exact
operations bridge the gap:

  thinning   binomial subsampling of signal counts -- how a Poisson process responds to
             lower laser power or lower detection efficiency
  ambient    an added homogeneous Poisson background, matching the uniform arrival model
             of Eq. 3

Without this the block-aggregation and matched-filtering steps have nothing to do: the
cluster is already obvious in every single-pixel histogram.

Degrade only until the effect is visible, not further. Too little and the steps look like
no-ops; too much and the cluster vanishes entirely and window sizing becomes noise.
"""

import numpy as np

from . import metrics


def thin(cube, alpha, rng):
    """Binomial thinning: keep each detected photon with probability alpha."""
    if alpha >= 1.0:
        return cube.copy()
    counts = np.rint(cube).astype(np.int64)
    return rng.binomial(counts, alpha).astype(cube.dtype)


def add_ambient(cube, rate, rng, z_trim=0):
    """Add uniform Poisson background at `rate` counts per bin per pixel.

    Only past z_trim: the direct-component region is rejected before the algorithm
    ever sees it.
    """
    out = cube.copy()
    if rate <= 0:
        return out
    shape = (cube.shape[0], cube.shape[1], cube.shape[2] - z_trim)
    out[:, :, z_trim:] += rng.poisson(rate, shape).astype(cube.dtype)
    return out


def degrade(transient, target_sbr=None, target_ppp=None, z_trim=0, seed=0):
    """Produce a degraded copy at a requested SBR and/or signal PPP.

    target_ppp fixes the thinning factor; target_sbr then fixes the ambient rate from the
    resulting peak height. Either may be omitted.

    A scene's own clean SBR caps what target_sbr can do -- asking for an SBR above it adds
    no background at all.
    """
    rng = np.random.default_rng(seed)
    cube = transient.data

    alpha = 1.0
    if target_ppp is not None:
        current = metrics.measure_ppp(cube, z_trim)
        alpha = float(np.clip(target_ppp / max(current, 1e-12), 0.0, 1.0))
    thinned = thin(cube, alpha, rng)

    rate = 0.0
    if target_sbr is not None:
        peak = metrics.mean_profile(thinned, z_trim)[z_trim:].max()
        intrinsic = metrics.noise_floor(thinned, z_trim)
        rate = max(float(peak / target_sbr) - intrinsic, 0.0)
    noisy = add_ambient(thinned, rate, rng, z_trim)

    info = dict(alpha=alpha, ambient_rate=rate,
                **{k: v for k, v in metrics.summarize(noisy, z_trim).items()
                   if k != "label"})
    return transient.replace(noisy), info
