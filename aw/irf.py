"""Instrument response function h(t).

The reference captures ship no IRF, and it cannot be recovered from them: the data is
pre-rectified so the direct-bounce peak starts at bin 0 with its rising edge already
cut off. The IRF is therefore synthesized as a Gaussian whose FWHM stands in for the
combined laser pulse width, SPAD jitter, TCSPC jitter and circuit delay.
"""

import numpy as np

PAPER_FWHM_S = 300e-12


def gaussian_irf(fwhm_s, bin_resolution, n_sigma=3.0, dtype=np.float32):
    """Unit-sum Gaussian IRF sampled on the measurement's time grid."""
    fwhm_bins = fwhm_s / bin_resolution
    sigma = fwhm_bins / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    half = max(int(np.ceil(n_sigma * sigma)), 1)
    t = np.arange(-half, half + 1, dtype=np.float64)
    h = np.exp(-0.5 * (t / sigma) ** 2)
    h /= h.sum()
    return h.astype(dtype), sigma


def fwhm_bins(fwhm_s, bin_resolution):
    return fwhm_s / bin_resolution
