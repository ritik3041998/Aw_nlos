"""Run AW-NLOS steps (a), (b) and (c) and save one figure per step.

  python run_steps.py --scene mannequin --ppp 30 --sbr 3.5

Outputs three figures plus window_widths.npz, which is the handoff to step (d).
For the combined overview sheet and the SBR sweep, use visualize.py.
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
    p = argparse.ArgumentParser(description="AW-NLOS steps (a)-(c)")
    p.add_argument("--scene", default="mannequin",
                   help=f"one of {sorted(SCENES)}, or use --mat")
    p.add_argument("--mat", help="path to an arbitrary .mat cube instead of --scene")
    p.add_argument("--width", type=float, help="wall half-extent in m, required with --mat")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--ppp", type=float, default=30.0,
                   help="target signal photons per pixel")
    p.add_argument("--sbr", type=float, default=3.5, help="target signal-to-background ratio")
    p.add_argument("--block", type=int, default=4, help="pixel block size (Eq. 4)")
    p.add_argument("--fwhm-ps", type=float, default=300.0, help="IRF FWHM in ps")
    p.add_argument("--z-trim", type=int, default=150,
                   help="time bins to ignore, after downsampling")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fwhm_s = args.fwhm_ps * 1e-12
    Z = args.z_trim

    # ---------------------------------------------------------------- step (a)
    if args.mat:
        if args.width is None:
            p.error("--width is required with --mat")
        clean = load_mat(args.mat, args.width, downsample=2, z_trim=600)
    else:
        clean = load_scene(args.scene, data_dir=args.data_dir)
    print(f"[a] {clean}")
    print("    " + metrics.fmt(metrics.summarize(clean.data, Z, "clean")))

    noisy, info = degrade.degrade(clean, target_sbr=args.sbr, target_ppp=args.ppp,
                                  z_trim=Z, seed=args.seed)
    print("    " + metrics.fmt(metrics.summarize(noisy.data, Z, "degraded")))
    print(f"    thinning alpha={info['alpha']:.3g}, "
          f"ambient={info['ambient_rate']:.3f} counts/bin/pixel")

    fig = plotting.plot_step_a(clean, noisy, Z)
    fig.savefig(out / "01_step_a_transient.png", dpi=130)
    plt.close(fig)
    print(f"    wrote {out / '01_step_a_transient.png'}")

    # ------------------------------------------------------------ steps (b)+(c)
    r = window.compute_windows(noisy, fwhm_s=fwhm_s, block=args.block, z_trim=Z)
    blocks, widths = r["blocks"], r["block_widths_ps"]
    ba, bb = plotting.brightest_block(blocks, Z)

    print(f"\n[b] {noisy.N}x{noisy.N} -> {blocks.shape[0]}x{blocks.shape[1]} blocks "
          f"of {args.block}x{args.block}; strongest block ({ba}, {bb})")
    print(f"    single pixel {noisy.data[args.block * ba, args.block * bb, Z:].sum():.0f} "
          f"counts -> block {blocks[ba, bb, Z:].sum():.0f} counts")

    fig = plotting.plot_step_b(noisy, r, Z)
    fig.savefig(out / "02_step_b_blocks.png", dpi=130)
    plt.close(fig)
    print(f"    wrote {out / '02_step_b_blocks.png'}")

    print(f"\n[c] IRF FWHM {args.fwhm_ps:.0f} ps = {fwhm_s / r['bin_resolution']:.1f} bins "
          f"(sigma {r['irf_sigma_bins']:.2f} bins)")
    print(f"    widths: min {widths.min():.0f}  median {np.median(widths):.0f}  "
          f"max {widths.max():.0f} ps")
    clamped = int((widths <= window.T_MIN_FACTOR * args.fwhm_ps + 1e-6).sum())
    print(f"    {clamped}/{widths.size} blocks clamped at T_min "
          f"(no cluster above background)")

    fig = plotting.plot_step_c(noisy, r, Z)
    fig.savefig(out / "03_step_c_window_widths.png", dpi=130)
    plt.close(fig)
    print(f"    wrote {out / '03_step_c_window_widths.png'}")

    npz = out / "window_widths.npz"
    np.savez_compressed(
        npz,
        block_widths_s=r["block_widths_s"],
        pixel_widths_s=r["pixel_widths_s"],
        pixel_width_bins=r["pixel_width_bins"],
        block_peak=r["block_peak"],
        block_floor=r["block_floor"],
        irf=r["irf"],
        bin_resolution=r["bin_resolution"],
        block=args.block,
        fwhm_s=fwhm_s,
        z_trim=Z,
        transient=noisy.data,
    )
    print(f"\n    wrote {npz}  (input for step (d))")


if __name__ == "__main__":
    main()
