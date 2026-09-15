"""The instrument response function h(t), used as the matched-filter template.

h(t) is the combined temporal blur of the laser pulse width, SPAD jitter, TCSPC jitter
and circuit delay. The paper measures it by pre-calibration and reports FWHM = 300 ps.

It CANNOT be recovered from the O'Toole captures: those are pre-rectified so the direct
bounce starts at bin 0 with its rising edge already cut off, and what follows is
wall-scatter decay rather than the instrument response. So it is synthesized here, with
FWHM as an explicit parameter you should set to match your own system.
"""

import numpy as np

PAPER_FWHM_S = 300e-12
FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def gaussian_irf(fwhm_s, bin_resolution, n_sigma=3.0, dtype=np.float32):
    """Unit-sum Gaussian IRF sampled on the measurement's time grid.

    Returns (h, sigma_bins). Unit sum matters: it keeps the background level unchanged
    in the mean through the matched filter, so the noise floor stays comparable.
    """
    sigma = (fwhm_s / bin_resolution) * FWHM_TO_SIGMA
    if sigma <= 0:
        raise ValueError("fwhm_s must be positive and larger than one time bin")

    half = max(int(np.ceil(n_sigma * sigma)), 1)
    t = np.arange(-half, half + 1, dtype=np.float64)
    h = np.exp(-0.5 * (t / sigma) ** 2)
    h /= h.sum()
    return h.astype(dtype), float(sigma)
