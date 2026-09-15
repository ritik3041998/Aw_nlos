"""Steps (b) and (c): block aggregation and matched-filter window sizing.

  (b)  Eq. 4   s_{a,b}(t) = sum over the 4x4 block of Phi_{i,j}(t)
  (c)  Eq. 5   g_{a,b}(t) = s_{a,b}(t) (*) h(t)
       Eq. 6   T_{a,b}    = 2 * FWHM * sqrt(2 ln(max(g) / (eta * N_n)))
       Eq. 7   T_{i,j}    = T_{a,b}, broadcast back to every pixel in the block

This module stops there. It produces window WIDTHS, not windowed data -- placing and
applying the window is step (d).
"""

import numpy as np
from scipy.signal import fftconvolve

from . import irf as irf_mod

T_MIN_FACTOR = 2.0        # T_min = 2 x FWHM, the classical fixed window of prior work
T_MAX_S = 4050e-12        # T_max, the paper's full-pixel "global window" width


def block_sum(cube, block=4):
    """Eq. 4: sum non-overlapping block x block tiles of scan points.

    Signal adds coherently in time -- neighbouring wall points are centimetres apart
    while the object is metres away, so they share almost the same round-trip delay t_c.
    Uniform background does not. Summing 16 pixels scales the peak by 16 but the
    background's standard deviation only by 4, lifting the cluster out of the noise.

    The blocks are used ONLY to size the window. The window is later applied to the
    original full-resolution per-pixel data, so no spatial resolution is lost.
    """
    N, N2, M = cube.shape
    if N != N2:
        raise ValueError(f"expected a square scan grid, got {N}x{N2}")
    if N % block:
        raise ValueError(f"grid {N} is not divisible by block size {block}")
    nb = N // block
    return cube.reshape(nb, block, nb, block, M).sum(axis=(1, 3))


def matched_filter(blocks, h):
    """Eq. 5: g = s (*) h.

    Matched filtering is the SNR-optimal linear detector for a known waveform in white
    noise. The signal cluster IS the IRF shifted to t_c, and the background is flat, so
    correlating with h maximizes output SNR right at the cluster.

    Written as a convolution, per the paper. A strict matched filter correlates with the
    time-reversed template; for a near-symmetric measured IRF the difference is a
    sub-bin shift.
    """
    if blocks.ndim != 3:
        raise ValueError(f"expected (A, B, M) block cube, got {blocks.shape}")
    return fftconvolve(blocks, h[None, None, :], mode="same", axes=-1)


def window_widths(g, fwhm_s, bin_resolution, z_trim=0,
                  t_min_s=None, t_max_s=T_MAX_S):
    """Eq. 6: T = 2 * FWHM * sqrt(2 ln(max(g) / (eta * N_n))).

    Model the filtered peak as a Gaussian of amplitude max(g) and ask where it falls to
    the noise floor eta*N_n; the window is twice that offset. Substituting FWHM for sigma
    (as the paper writes it) widens the window ~2.35x, which is deliberate: the window
    must also span the object's own depth extent, since one wall point sees every object
    voxel at once.

    Behaviour: more ambient light raises the floor, shrinks the log, and NARROWS the
    window -- stricter in noisier conditions. A stronger peak widens it.

    IMPORTANT: the noise floor is estimated from g itself, not from the raw per-pixel
    rate. Block summing multiplies the background by block^2 and the filter reshapes it;
    comparing max(g) against an unscaled per-pixel rate inflates every window by ~16x.

    Returns (widths_s, peak, floor), each (A, B).
    """
    if t_min_s is None:
        t_min_s = T_MIN_FACTOR * fwhm_s
    if t_min_s > t_max_s:
        raise ValueError(f"t_min_s ({t_min_s}) exceeds t_max_s ({t_max_s})")

    searched = g[:, :, z_trim:]
    if searched.shape[2] == 0:
        raise ValueError("z_trim removes the entire histogram")

    peak = searched.max(axis=2)
    floor = np.median(searched, axis=2)

    ratio = np.where(floor > 0, peak / np.maximum(floor, 1e-12), 1.0)
    # ratio <= 1 means no cluster stands above background: fall back to T_min rather
    # than taking the square root of a negative number.
    ratio = np.maximum(ratio, 1.0 + 1e-12)

    widths = 2.0 * fwhm_s * np.sqrt(2.0 * np.log(ratio))
    return np.clip(widths, t_min_s, t_max_s), peak, floor


def broadcast_widths(widths, block, n):
    """Eq. 7: T_{i,j} = T_{a,b}, every pixel inheriting its block's width.

    Note the paper prints Eq. 7 as b = floor((i-1)/4)+1; that is a typo for (j-1).
    """
    full = np.repeat(np.repeat(widths, block, axis=0), block, axis=1)
    return full[:n, :n]


def compute_windows(transient, fwhm_s=irf_mod.PAPER_FWHM_S, block=4, z_trim=0,
                    t_min_s=None, t_max_s=T_MAX_S):
    """Run steps (b) and (c) end to end.

    Returns a dict with the block widths, the per-pixel widths (and their integer bin
    counts), plus the intermediates needed to plot each step.
    """
    cube = transient.data
    h, sigma = irf_mod.gaussian_irf(fwhm_s, transient.bin_resolution)

    blocks = block_sum(cube, block)                                     # (b) Eq. 4
    g = matched_filter(blocks, h)                                       # (c) Eq. 5
    widths, peak, floor = window_widths(                                # (c) Eq. 6
        g, fwhm_s, transient.bin_resolution, z_trim, t_min_s, t_max_s)
    per_pixel = broadcast_widths(widths, block, transient.N)            # (c) Eq. 7

    return dict(
        blocks=blocks,
        filtered=g,
        irf=h,
        irf_sigma_bins=sigma,
        block_widths_s=widths,
        block_widths_ps=widths * 1e12,
        block_peak=peak,
        block_floor=floor,
        pixel_widths_s=per_pixel,
        pixel_width_bins=np.maximum(
            np.rint(per_pixel / transient.bin_resolution).astype(int), 1),
        block=block,
        fwhm_s=fwhm_s,
        bin_resolution=transient.bin_resolution,
    )
