"""Phase 0 checkpoint: the ported LCT must reconstruct a legible 'SU'."""

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aw import lct
from aw.io_utils import load_scene

t = load_scene("s_u")
print(t)
print(f"range = {t.range_m:.4f} m, slope = {t.width / t.range_m:.5f}")

t0 = time.time()
vol = lct.reconstruct(t, snr=0.8)
print(f"reconstructed {vol.shape} in {time.time() - t0:.1f} s")

vol = lct.crop_for_display(vol, t)
front, top, side = lct.projections(vol)
print(f"cropped {vol.shape}, front max {front.max():.4g}")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, img, name in zip(axes, (front, top, side), ("Front", "Top", "Side")):
    ax.imshow(img, cmap="gray")
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Phase 0 - LCT port, clean data_s_u")
fig.tight_layout()
fig.savefig("results/phase0_lct_s_u.png", dpi=130)
print("wrote results/phase0_lct_s_u.png")
