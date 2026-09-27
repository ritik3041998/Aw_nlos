"""Find which scenes hold a single temporal cluster (one object at one depth)."""

import numpy as np
from scipy.signal import find_peaks

from aw import irf as irf_mod, metrics, window
from aw.io_utils import SCENES, load_scene

Z_TRIM = 150

print(f"{'scene':<24} {'grid':>12} {'SBR':>6} {'PPP':>9}  clusters (ns @ prominence)")
print("-" * 92)

for name in SCENES:
    try:
        t = load_scene(name)
    except Exception as e:
        print(f"{name:<24} load failed: {e}")
        continue

    z_trim = Z_TRIM if t.M >= 512 else int(150 * t.M / 512)
    prof = t.data[:, :, z_trim:].sum(axis=(0, 1)).astype(float)
    if prof.max() <= 0:
        print(f"{name:<24} empty after trim")
        continue

    h, _ = irf_mod.gaussian_irf(irf_mod.PAPER_FWHM_S, t.bin_resolution)
    smooth = np.convolve(prof, h, mode="same")
    pk, props = find_peaks(smooth, prominence=0.15 * smooth.max(), distance=len(h))
    t_ns = (pk + z_trim) * t.bin_resolution * 1e9

    desc = ", ".join(f"{v:.2f}ns({p / smooth.max():.2f})"
                     for v, p in zip(t_ns, props["prominences"]))
    flag = "  <-- SINGLE" if len(pk) == 1 else ""
    print(f"{name:<24} {t.N}x{t.N}x{t.M:<6} {metrics.measure_sbr(t.data, z_trim):6.2f} "
          f"{metrics.measure_ppp(t.data, z_trim):9.1f}  {len(pk)}: {desc}{flag}")
