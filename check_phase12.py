"""Phase 1+2 checkpoint: degradation must break LCT, windowing must restore SBR."""

import numpy as np

from aw import degrade, metrics, window
from aw.io_utils import load_scene

Z_TRIM = 150  # 600 native bins / 4 after downsampling

clean = load_scene("s_u")
print(clean)
print(metrics.fmt(metrics.summarize(clean.data, Z_TRIM, "clean")))
print()

for sbr in (4.83, 2.92, 2.12):
    noisy, info = degrade.degrade(clean, target_sbr=sbr, z_trim=Z_TRIM, seed=1)
    print(metrics.fmt(metrics.summarize(noisy.data, Z_TRIM, f"degraded SBR~{sbr}")))
    print(f"    ambient rate = {info['ambient_rate']:.3f} counts/bin/pixel")

    win, diag = window.adaptive_window(noisy, z_trim=Z_TRIM)
    print(metrics.fmt(metrics.summarize(win.data, Z_TRIM, "  after AW")))
    w = diag["width_ps"]
    print(f"    window widths: min {w.min():.0f} ps  median {np.median(w):.0f} ps  "
          f"max {w.max():.0f} ps   duty {diag['duty'] * 100:.2f}%")
    print(f"    start spread: {diag['starts'].min()} .. {diag['starts'].max()} bins")
    print()
