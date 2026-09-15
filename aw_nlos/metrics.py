"""Photon-statistics meters: SBR (Eq. 14) and signal PPP (Eq. 16)."""

import numpy as np


def mean_profile(cube, z_trim=0):
    """Per-bin mean photon count across all scan points."""
    profile = cube.mean(axis=(0, 1)).astype(np.float64)
    if z_trim:
        profile = profile.copy()
        profile[:z_trim] = 0
    return profile


def noise_floor(cube, z_trim=0):
    """Background rate N_n in counts per bin per pixel.

    Uses the median over searched bins. The object echo occupies a small fraction of the
    histogram, so the median is dominated by background and needs no hand-drawn mask.
    """
    profile = cube.mean(axis=(0, 1)).astype(np.float64)[z_trim:]
    return float(np.median(profile)) if profile.size else 0.0


def measure_sbr(cube, z_trim=0):
    """Eq. 14: SBR = peak height / background level, on the pixel-averaged histogram."""
    floor = noise_floor(cube, z_trim)
    if floor <= 0:
        return float("inf")
    return float(mean_profile(cube, z_trim)[z_trim:].max() / floor)


def measure_ppp(cube, z_trim=0):
    """Eq. 16: signal photons per pixel, total counts minus the background estimate."""
    n_bins = cube.shape[2] - z_trim
    total = float(cube[:, :, z_trim:].sum()) / (cube.shape[0] * cube.shape[1])
    return total - noise_floor(cube, z_trim) * n_bins


def peak_per_pixel(cube, z_trim=0):
    """Median per-pixel peak height, the N_s^PK of Eq. 14."""
    return float(np.median(cube[:, :, z_trim:].max(axis=2)))


def summarize(cube, z_trim=0, label=""):
    return dict(label=label,
                sbr=measure_sbr(cube, z_trim),
                ppp=measure_ppp(cube, z_trim),
                peak=peak_per_pixel(cube, z_trim),
                floor=noise_floor(cube, z_trim),
                total=float(cube[:, :, z_trim:].sum()))


def fmt(s):
    return (f"{s['label']:<22} SBR={s['sbr']:7.2f}  PPP={s['ppp']:10.4f}  "
            f"peak={s['peak']:7.2f}  floor={s['floor']:8.4f}")


def to_display(img, pct=99.5):
    """Percentile-clipped stretch for plotting, so a few hot values do not wash out."""
    a = np.asarray(img, dtype=np.float64)
    lo, hi = a.min(), np.percentile(a, pct)
    if hi - lo <= 0:
        hi = a.max()
    if hi - lo <= 0:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)
