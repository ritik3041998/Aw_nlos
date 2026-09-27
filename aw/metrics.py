"""Photon-statistics meters and reconstruction quality metrics."""

import numpy as np
from skimage.metrics import structural_similarity


def mean_profile(cube, z_trim=0):
    """Per-bin mean photon count across all scan points."""
    profile = cube.mean(axis=(0, 1)).astype(np.float64)
    if z_trim:
        profile = profile.copy()
        profile[:z_trim] = 0
    return profile


def noise_floor(cube, z_trim=0):
    """Background rate N_n, in counts per bin per pixel.

    Uses the median over searched bins: the object echo occupies a small fraction
    of the histogram, so the median is dominated by background and needs no mask.
    """
    profile = cube.mean(axis=(0, 1)).astype(np.float64)[z_trim:]
    if profile.size == 0:
        return 0.0
    return float(np.median(profile))


def measure_sbr(cube, z_trim=0):
    """SBR = peak height / background level (Eq. 14), on the pixel-averaged histogram."""
    profile = mean_profile(cube, z_trim)
    floor = noise_floor(cube, z_trim)
    if floor <= 0:
        return np.inf
    return float(profile[z_trim:].max() / floor)


def measure_ppp(cube, z_trim=0):
    """Signal photons per pixel (Eq. 16): total counts minus the background estimate."""
    n_bins = cube.shape[2] - z_trim
    total = float(cube[:, :, z_trim:].sum()) / (cube.shape[0] * cube.shape[1])
    return total - noise_floor(cube, z_trim) * n_bins


def peak_per_pixel(cube, z_trim=0):
    """Median per-pixel peak height, the N_s^PK of Eq. 14."""
    return float(np.median(cube[:, :, z_trim:].max(axis=2)))


def summarize(cube, z_trim=0, label=""):
    s = dict(label=label,
             sbr=measure_sbr(cube, z_trim),
             ppp=measure_ppp(cube, z_trim),
             peak=peak_per_pixel(cube, z_trim),
             floor=noise_floor(cube, z_trim),
             total=float(cube[:, :, z_trim:].sum()))
    return s


def fmt(s):
    return (f"{s['label']:<22} SBR={s['sbr']:7.2f}  PPP={s['ppp']:10.4f}  "
            f"peak={s['peak']:7.2f}  floor={s['floor']:8.4f}")


def normalize01(img):
    img = np.asarray(img, dtype=np.float64)
    lo, hi = img.min(), img.max()
    if hi - lo <= 0:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def to_display(img, pct=99.5):
    """Display-only stretch. Clips the top percentile so a few hot voxels do not
    wash out the image. SSIM always uses the unclipped normalize01."""
    a = np.asarray(img, dtype=np.float64)
    lo = a.min()
    hi = np.percentile(a, pct)
    if hi - lo <= 0:
        return normalize01(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def ssim(recon_img, gt_img):
    """SSIM between two front-view projections, each normalized to [0, 1]."""
    a, b = normalize01(recon_img), normalize01(gt_img)
    return float(structural_similarity(a, b, data_range=1.0))


def psnr(recon_img, gt_img):
    """PSNR in dB between two front-view projections, each normalized to [0, 1]."""
    a, b = normalize01(recon_img), normalize01(gt_img)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return np.inf
    return float(10.0 * np.log10(1.0 / mse))
