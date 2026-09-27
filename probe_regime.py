"""Find the photon regime where adaptive windowing begins to help."""

import numpy as np

from aw import degrade, lct, metrics, tv, window
from aw.io_utils import load_scene

Z_TRIM = 150
clean = load_scene("s_u")
gt, _, _ = lct.projections(lct.crop_for_display(lct.reconstruct(clean), clean))

print(f"{'PPP':>8} {'SBR':>7} | {'no-window':>10} {'AW':>10} {'AW+TV':>10}")
print("-" * 52)

for ppp in (5.0, 1.0, 0.2, 0.05):
    noisy, info = degrade.degrade(clean, target_sbr=2.12, target_ppp=ppp,
                                  z_trim=Z_TRIM, seed=1)
    sbr = metrics.measure_sbr(noisy.data, Z_TRIM)

    f0, _, _ = lct.projections(lct.crop_for_display(lct.reconstruct(noisy), noisy))
    aw, diag = window.adaptive_window(noisy, z_trim=Z_TRIM)
    f1, _, _ = lct.projections(lct.crop_for_display(lct.reconstruct(aw), aw))
    awtv = tv.complete(aw, mu_scale=0.15, z_trim=Z_TRIM)
    f2, _, _ = lct.projections(lct.crop_for_display(lct.reconstruct(awtv), awtv))

    print(f"{ppp:8.2f} {sbr:7.2f} | {metrics.ssim(f0, gt):10.4f} "
          f"{metrics.ssim(f1, gt):10.4f} {metrics.ssim(f2, gt):10.4f}"
          f"   [w={np.median(diag['width_ps']):.0f}ps duty={diag['duty']*100:.1f}%]")
