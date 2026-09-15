"""Combined visualizations for AW-NLOS steps (a)-(c).

  python visualize.py --scene mannequin --ppp 30 --sbr 3.5

Produces an overview sheet of all three steps and an SBR sweep showing that Eq. 6
really does narrow the window as ambient light rises.

For the individual per-step figures, use run_steps.py instead.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aw_nlos import degrade, metrics, plotting, window
from aw_nlos.io_utils import SCENES, load_mat, load_scene


def main():
    p = argparse.ArgumentParser(description="AW-NLOS steps (a)-(c) visualizations")
    p.add_argument("--scene", default="mannequin", help=f"one of {sorted(SCENES)}")
    p.add_argument("--mat", help="path to an arbitrary .mat cube instead of --scene")
    p.add_argument("--width", type=float, help="wall half-extent in m, required with --mat")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--ppp", type=float, default=30.0)
    p.add_argument("--sbr", type=float, default=3.5)
    p.add_argument("--block", type=int, default=4)
    p.add_argument("--fwhm-ps", type=float, default=300.0)
    p.add_argument("--z-trim", type=int, default=150)
    p.add_argument("--seed", type=int, default=1)
    # Keep targets below the scene's own clean SBR; above it no background is added
    # and the rows come out identical.
    p.add_argument("--sweep", type=float, nargs="+", default=[4.2, 3.8, 3.4, 3.0, 2.6],
                   help="SBR values for the adaptivity sweep")
    p.add_argument("--no-sweep", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fwhm_s = args.fwhm_ps * 1e-12
    Z = args.z_trim

    if args.mat:
        if args.width is None:
            p.error("--width is required with --mat")
        clean = load_mat(args.mat, args.width, downsample=2, z_trim=600)
    else:
        clean = load_scene(args.scene, data_dir=args.data_dir)
    print(clean)
    print("  " + metrics.fmt(metrics.summarize(clean.data, Z, "clean")))

    noisy, info = degrade.degrade(clean, target_sbr=args.sbr, target_ppp=args.ppp,
                                  z_trim=Z, seed=args.seed)
    print("  " + metrics.fmt(metrics.summarize(noisy.data, Z, "degraded")))

    result = window.compute_windows(noisy, fwhm_s=fwhm_s, block=args.block, z_trim=Z)
    w = result["block_widths_ps"]
    print(f"  widths: min {w.min():.0f}  median {np.median(w):.0f}  max {w.max():.0f} ps")

    fig = plotting.plot_overview(clean, noisy, result, Z, scene=clean.scene)
    fig.savefig(out / "overview_steps_abc.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out / 'overview_steps_abc.png'}")

    if not args.no_sweep:
        print("\n  SBR sweep (Eq. 6 adaptivity)")
        rows = []
        for target in sorted(args.sweep, reverse=True):
            n, _ = degrade.degrade(clean, target_sbr=target, target_ppp=args.ppp,
                                   z_trim=Z, seed=args.seed)
            achieved = metrics.measure_sbr(n.data, Z)
            r = window.compute_windows(n, fwhm_s=fwhm_s, block=args.block, z_trim=Z)
            widths = r["block_widths_ps"]
            rows.append((achieved, widths))
            clamped = int((widths <= 2 * args.fwhm_ps + 1e-6).sum())
            print(f"    SBR {achieved:5.2f}  median {np.median(widths):7.0f} ps   "
                  f"{clamped:3d}/{widths.size} clamped at T_min")

        fig = plotting.plot_width_vs_noise(rows, args.fwhm_ps)
        fig.savefig(out / "width_vs_sbr.png", dpi=130)
        plt.close(fig)
        print(f"  wrote {out / 'width_vs_sbr.png'}")


if __name__ == "__main__":
    main()
