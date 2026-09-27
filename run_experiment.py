"""AW-NLOS ambient-light sweep: reproduce the comparison of Fig. 5 / Fig. 6.

For each ambient level, reconstruct the same measurement four ways -- no window,
fixed 2x-jitter window, global window, and adaptive windowing + TV -- and score each
front view against a ground truth built from the undegraded capture.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aw import degrade, lct, metrics, tv, window
from aw.io_utils import load_scene

Z_TRIM = 150


def front_view(transient, snr=None):
    """snr=None uses the scene's own Wiener parameter (diffuse scenes need a stronger prior)."""
    vol = lct.crop_for_display(lct.reconstruct(transient, snr=snr), transient)
    return lct.projections(vol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="mannequin")
    ap.add_argument("--sbr", type=float, nargs="+", default=[4.0, 3.0, 2.5])
    ap.add_argument("--ppp", type=float, default=60.0,
                    help="signal photons per pixel; None disables thinning entirely")
    ap.add_argument("--fwhm-ps", type=float, default=300.0)
    ap.add_argument("--mu", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/aw_nlos_sweep.png")
    args = ap.parse_args()

    fwhm_s = args.fwhm_ps * 1e-12
    clean = load_scene(args.scene)
    print(clean)

    gt_front, _, _ = front_view(clean)
    print(metrics.fmt(metrics.summarize(clean.data, Z_TRIM, "clean (GT)")))

    methods = ["No window", "Fixed 2x jitter", "Global window", "AW-NLOS"]
    rows, table = [], []

    for sbr in args.sbr:
        noisy, info = degrade.degrade(clean, target_sbr=sbr, target_ppp=args.ppp,
                                      z_trim=Z_TRIM, seed=args.seed)
        achieved = metrics.measure_sbr(noisy.data, Z_TRIM)
        print(f"\n--- target SBR {sbr} -> achieved {achieved:.2f} "
              f"(ambient {info['ambient_rate']:.2f}/bin) ---")

        variants = {
            "No window": noisy,
            "Fixed 2x jitter": window.fixed_window(noisy, fwhm_s, Z_TRIM)[0],
            "Global window": window.global_window(noisy, z_trim=Z_TRIM)[0],
        }
        aw, diag = window.adaptive_window(noisy, fwhm_s=fwhm_s, z_trim=Z_TRIM)
        variants["AW-NLOS"] = tv.complete(aw, mu_scale=args.mu, z_trim=Z_TRIM)
        print(f"    AW widths: median {np.median(diag['width_ps']):.0f} ps, "
              f"duty {diag['duty'] * 100:.1f}%")

        row, scores = [], []
        for name in methods:
            t0 = time.time()
            f, _, _ = front_view(variants[name])
            s = metrics.ssim(f, gt_front)
            row.append(f)
            scores.append(s)
            print(f"    {name:<18} SSIM={s:.4f}   ({time.time() - t0:.1f}s)")
        rows.append((achieved, row, scores))
        table.append(scores)

    n_rows, n_cols = len(rows), len(methods) + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.8 * n_rows),
                             squeeze=False)
    for r, (sbr_val, imgs, scores) in enumerate(rows):
        axes[r][0].imshow(metrics.to_display(gt_front), cmap="gray")
        axes[r][0].set_ylabel(f"SBR {sbr_val:.2f}", fontsize=11)
        if r == 0:
            axes[r][0].set_title("Ground truth", fontsize=10)
        for c, (img, name, s) in enumerate(zip(imgs, methods, scores), start=1):
            axes[r][c].imshow(metrics.to_display(img), cmap="gray")
            axes[r][c].set_xlabel(f"SSIM {s:.3f}", fontsize=9)
            if r == 0:
                axes[r][c].set_title(name, fontsize=10)
        for c in range(n_cols):
            axes[r][c].set_xticks([])
            axes[r][c].set_yticks([])

    fig.suptitle(f"AW-NLOS ambient sweep - scene '{args.scene}', front views", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")

    print("\nSSIM summary")
    print("SBR    " + "".join(f"{m:>18}" for m in methods))
    for (sbr_val, _, scores) in rows:
        print(f"{sbr_val:5.2f}  " + "".join(f"{s:>18.4f}" for s in scores))


if __name__ == "__main__":
    main()
