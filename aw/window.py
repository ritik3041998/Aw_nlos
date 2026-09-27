"""Adaptive windowing: steps (b), (c) and (d) of Fig. 3.

Block aggregation amplifies the sparse signal cluster (Eq. 4), matched filtering
against the IRF sizes the window (Eq. 5-6), and the window is then placed and applied
per pixel on the *raw* counts (Eq. 7-8).
"""

import numpy as np
from scipy.signal import fftconvolve

from . import irf as irf_mod

T_MIN_FACTOR = 2.0        # T_min = 2 x FWHM, the classical fixed window
T_MAX_S = 4050e-12        # paper's full-pixel global window


def block_sum(cube, block=4):
    """Eq. 4: sum non-overlapping block x block tiles of scan points.

    Signal adds coherently in time (neighbouring wall points share t_c) while uniform
    background does not, so the cluster rises out of the Poisson floor.
    """
    N, _, M = cube.shape
    if N % block:
        raise ValueError(f"grid {N} is not divisible by block size {block}")
    nb = N // block
    return cube.reshape(nb, block, nb, block, M).sum(axis=(1, 3))


def matched_filter(blocks, h):
    """Eq. 5: g = s (x) h. The IRF is the optimal template for the signal cluster."""
    return fftconvolve(blocks, h[None, None, :], mode="same", axes=-1)


def window_widths(g, fwhm_s, bin_resolution, z_trim=0,
                  t_min_s=None, t_max_s=T_MAX_S, floor_scale=1.0):
    """Eq. 6: T = 2 * FWHM * sqrt(2 ln(max(g) / (eta * N_n))).

    The noise floor is estimated from g itself rather than from the raw per-pixel rate:
    the matched filter and the block sum both change it, and comparing max(g) against
    an unscaled rate would inflate every window.
    """
    if t_min_s is None:
        t_min_s = T_MIN_FACTOR * fwhm_s

    searched = g[:, :, z_trim:]
    peak = searched.max(axis=2)
    floor = np.median(searched, axis=2) * floor_scale

    ratio = np.where(floor > 0, peak / np.maximum(floor, 1e-12), 1.0)
    # ratio <= 1 means no cluster stands above background; fall back to T_min.
    ratio = np.maximum(ratio, 1.0 + 1e-12)
    widths = 2.0 * fwhm_s * np.sqrt(2.0 * np.log(ratio))

    widths = np.clip(widths, t_min_s, t_max_s)
    return widths, peak, floor


def _place_and_mask(cube, w_bins, z_trim, out, starts):
    """Eq. 8 for one window length: argmax of the sliding sum, then zero outside."""
    N, _, M = cube.shape
    w = int(np.clip(w_bins, 1, M - z_trim))

    csum = np.zeros((N, N, M + 1), dtype=np.float64)
    np.cumsum(cube, axis=2, out=csum[:, :, 1:])
    # Sliding sum over every valid start position at or after z_trim.
    sliding = csum[:, :, z_trim + w:] - csum[:, :, z_trim:M - w + 1]
    t_m = sliding.argmax(axis=2) + z_trim

    idx = np.arange(M)[None, None, :]
    keep = (idx >= t_m[:, :, None]) & (idx < (t_m + w)[:, :, None])
    np.copyto(out, cube, where=keep)
    starts[:] = t_m
    return t_m


def apply_windows(cube, widths_s, bin_resolution, block=4, z_trim=0):
    """Eq. 7-8: broadcast block widths to pixels, then place and apply each window.

    Width is inherited from the block (it tracks noise level and object depth spread,
    which vary slowly), but the start time is found per pixel, since t_c changes with
    scan geometry.
    """
    N, _, M = cube.shape
    w_bins_blocks = np.maximum(np.rint(widths_s / bin_resolution).astype(int), 1)
    w_bins = np.repeat(np.repeat(w_bins_blocks, block, axis=0), block, axis=1)

    out = np.zeros_like(cube)
    starts = np.zeros((N, N), dtype=int)

    # Windows are constant within a block, so one pass per distinct length suffices.
    for w in np.unique(w_bins):
        sel = w_bins == w
        sub = np.zeros_like(cube)
        sub_starts = np.zeros((N, N), dtype=int)
        _place_and_mask(cube, w, z_trim, sub, sub_starts)
        out[sel] = sub[sel]
        starts[sel] = sub_starts[sel]

    return out, starts, w_bins


def adaptive_window(transient, fwhm_s=irf_mod.PAPER_FWHM_S, block=4, z_trim=0,
                    t_min_s=None, t_max_s=T_MAX_S):
    """Run steps (b) through (d). Returns the windowed transient and diagnostics."""
    cube = transient.data
    h, sigma = irf_mod.gaussian_irf(fwhm_s, transient.bin_resolution)

    s = block_sum(cube, block)                                    # (b) Eq. 4
    g = matched_filter(s, h)                                      # (c) Eq. 5
    widths, peak, floor = window_widths(                          # (c) Eq. 6
        g, fwhm_s, transient.bin_resolution, z_trim, t_min_s, t_max_s)
    windowed, starts, w_bins = apply_windows(                     # (d) Eq. 7-8
        cube, widths, transient.bin_resolution, block, z_trim)

    diag = dict(block_widths_s=widths, block_peak=peak, block_floor=floor,
                starts=starts, w_bins=w_bins, irf=h, irf_sigma_bins=sigma,
                blocks=s, filtered=g,
                width_ps=(widths * 1e12), duty=float((windowed > 0).mean()))
    return transient.replace(windowed), diag


def fixed_window(transient, fwhm_s=irf_mod.PAPER_FWHM_S, z_trim=0):
    """Baseline: one window of 2 x system jitter everywhere (Fig. 5(b))."""
    w = 2.0 * fwhm_s
    widths = np.full((transient.N, transient.N), w)
    out = np.zeros_like(transient.data)
    starts = np.zeros((transient.N, transient.N), dtype=int)
    _place_and_mask(transient.data, int(round(w / transient.bin_resolution)),
                    z_trim, out, starts)
    return transient.replace(out), dict(block_widths_s=widths, starts=starts)


def global_window(transient, width_s=T_MAX_S, z_trim=0):
    """Baseline: one window sized from the all-pixel summed histogram (Fig. 5(c))."""
    widths = np.full((transient.N, transient.N), width_s)
    out = np.zeros_like(transient.data)
    starts = np.zeros((transient.N, transient.N), dtype=int)
    _place_and_mask(transient.data, int(round(width_s / transient.bin_resolution)),
                    z_trim, out, starts)
    return transient.replace(out), dict(block_widths_s=widths, starts=starts)
