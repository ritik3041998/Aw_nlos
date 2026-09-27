"""Run the full AW-NLOS pipeline on one transient capture.

The other drivers only accept `--scene`, a name from the table of bundled
captures in aw/io_utils.py. This one takes any .mat cube, so it is the entry
point for testing on your own data.

Two modes:

  default        degrade the capture synthetically, then reconstruct it four
                 ways: traditional LCT and AW-NLOS, each on the raw and the
                 degraded cube. Use this on a clean lab capture to see what
                 the method buys under simulated ambient light.

  --no-degrade   skip degradation and reconstruct the capture as given, two
                 ways: traditional LCT and AW-NLOS. Use this on a real
                 low-SBR capture that already carries its own ambient noise.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aw import degrade, lct, metrics, tv, window
from aw.io_utils import load_mat

VIEWS = ("Front", "Top", "Side")


def render(transient):
    return lct.projections(lct.crop_for_display(lct.reconstruct(transient), transient))


def apply_algorithm(transient, fwhm_s, mu, block, z_trim):
    aw, diag = window.adaptive_window(transient, fwhm_s=fwhm_s, block=block,
                                      z_trim=z_trim)
    return tv.complete(aw, mu_scale=mu, z_trim=z_trim), diag


def save(panels, title, stats, path):
    """One row of labelled panels per view."""
    n = len(panels)
    fig, ax = plt.subplots(3, n, figsize=(4.6 * n, 9.6), squeeze=False)
    for c, (views, label, score) in enumerate(panels):
        for r in range(3):
            ax[r][c].imshow(metrics.to_display(views[r]), cmap="gray")
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
            if c == 0:
                ax[r][c].set_ylabel(VIEWS[r], fontsize=12)
        ax[0][c].set_title(label, fontsize=12)
        tag = "reference" if score is None else f"SSIM {score[0]:.4f}   PSNR {score[1]:.1f} dB"
        ax[2][c].text(0.5, 0.015, tag, transform=ax[2][c].transAxes,
                      ha="center", va="bottom", fontsize=13, color="white",
                      fontweight="bold",
                      bbox=dict(facecolor="black", alpha=0.72, pad=5,
                                edgecolor="white", linewidth=0.6))
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.945, stats, ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.935], h_pad=2.0)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Run the AW-NLOS pipeline on a single .mat transient capture.")
    ap.add_argument("mat", help="path to the .mat capture")
    ap.add_argument("--width", type=float, default=None,
                    help="wall patch half-extent in metres; read from the file when "
                         "it carries a 'width' variable")
    ap.add_argument("--bin-ps", type=float, default=4.0,
                    help="NATIVE time bin resolution in ps, before downsampling")
    ap.add_argument("--downsample", type=int, default=2,
                    help="pairwise-sum the time axis this many times (4 ps -> 16 ps)")
    ap.add_argument("--z-trim", type=int, default=600,
                    help="native time bins of direct component to zero on load")
    ap.add_argument("--z-offset", type=int, default=0,
                    help="native bins to skip when cropping the volume for display; "
                         "raise it if the object sits deep in the reconstruction")
    ap.add_argument("--data-key", default=None,
                    help="variable holding the cube; auto-detected when the file "
                         "has exactly one 3D array")
    ap.add_argument("--diffuse", action="store_true",
                    help="diffuse target: use 1/r^4 falloff and a stronger prior")
    ap.add_argument("--snr", type=float, default=None,
                    help="Wiener parameter; defaults to 0.08 with --diffuse, else 0.8")

    ap.add_argument("--no-degrade", action="store_true",
                    help="reconstruct the capture as given, without adding noise")
    ap.add_argument("--ppp", type=float, default=60.0, help="target signal PPP")
    ap.add_argument("--sbr", type=float, default=4.0, help="target SBR")
    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--fwhm-ps", type=float, default=300.0,
                    help="IRF FWHM in ps -- set this to match your system")
    ap.add_argument("--mu", type=float, default=0.15, help="TV weight scale")
    ap.add_argument("--block", type=int, default=4, help="block size for Eq. 4")
    ap.add_argument("--out", default="results/single",
                    help="output directory")
    args = ap.parse_args()

    snr = args.snr if args.snr is not None else (0.08 if args.diffuse else 0.8)
    fwhm_s = args.fwhm_ps * 1e-12
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cap = load_mat(args.mat, width=args.width, bin_ps=args.bin_ps,
                   z_offset=args.z_offset, diffuse=args.diffuse, snr=snr,
                   data_key=args.data_key, downsample=args.downsample,
                   z_trim=args.z_trim)
    print(cap)

    # The bundled captures are 512 bins after downsampling and use z_trim 150 there;
    # scale that with the cube so short recordings are not over-trimmed.
    z_trim = max(1, int(round(150 * cap.M / 512)))
    stats_in = metrics.summarize(cap.data, z_trim)
    print(metrics.fmt({**stats_in, "label": "as loaded"}))

    if stats_in["ppp"] <= 0:
        print("\nWARNING: no signal measured above the background estimate. Check "
              "--z-trim (the direct component may still be present) and --bin-ps.")

    alg, diag = apply_algorithm(cap, fwhm_s, args.mu, args.block, z_trim)
    print(f"  AW window: median {np.median(diag['width_ps']):.0f} ps, "
          f"duty {diag['duty'] * 100:.1f}%")

    v_raw, v_alg = render(cap), render(alg)
    ref = v_raw[0]

    def score(v):
        return metrics.ssim(v[0], ref), metrics.psnr(v[0], ref)

    if args.no_degrade:
        panels = [(v_raw, "Traditional LCT", None),
                  (v_alg, "AW-NLOS + LCT", score(v_alg))]
        stats = (f"as given: SBR {stats_in['sbr']:.2f}, PPP {stats_in['ppp']:.1f}   |   "
                 f"window duty {diag['duty'] * 100:.1f}%")
        title = f"AW-NLOS on '{cap.scene}' (no synthetic degradation)"
    else:
        noisy, info = degrade.degrade(cap, target_sbr=args.sbr, target_ppp=args.ppp,
                                      z_trim=z_trim, seed=args.seed)
        noisy_alg, d_noisy = apply_algorithm(noisy, fwhm_s, args.mu, args.block, z_trim)
        v_noisy, v_nalg = render(noisy), render(noisy_alg)
        sn = metrics.summarize(noisy.data, z_trim)
        panels = [(v_raw, "Traditional LCT\nraw", None),
                  (v_alg, "AW-NLOS\nraw", score(v_alg)),
                  (v_noisy, "Traditional LCT\nnoisy", score(v_noisy)),
                  (v_nalg, "AW-NLOS\nnoisy", score(v_nalg))]
        stats = (f"raw: SBR {stats_in['sbr']:.1f}, PPP {stats_in['ppp']:.0f}   |   "
                 f"noisy: SBR {sn['sbr']:.2f}, PPP {sn['ppp']:.1f} "
                 f"(alpha {info['alpha']:.3f}, ambient {info['ambient_rate']:.3f}/bin)   |   "
                 f"window duty: raw {diag['duty'] * 100:.1f}%, "
                 f"noisy {d_noisy['duty'] * 100:.1f}%")
        title = f"Traditional LCT vs AW-NLOS, raw vs noisy - '{cap.scene}'"

    path = out / f"{cap.scene}.png"
    save(panels, title, stats, path)

    print()
    for _, label, s in panels:
        flat = label.replace("\n", " ")
        print(f"  {flat:<26} " +
              ("reference" if s is None else f"SSIM {s[0]:.4f}   PSNR {s[1]:.1f} dB"))
    print(f"\nwrote {path}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
