"""Clean vs noisy ablation: run AW-NLOS on the raw capture and on a degraded copy.

Four reconstructions per scene, all through the same LCT:

  1. Ground truth      the raw capture, no algorithm
  2. AW-NLOS on clean  the algorithm applied to the raw capture
  3. Degraded          the raw capture after thinning + ambient, no algorithm
  4. AW-NLOS on noisy  the algorithm applied to that degraded copy

Column 2 is the control the sweep figures never show: windowing and TV are lossy
operations, so this measures what they cost when there is no noise to remove. The
gap (4 - 3) is what the algorithm buys; the gap (1 - 2) is what it charges.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aw import degrade, lct, metrics, tv, window
from aw.io_utils import SCENES, load_scene

Z_TRIM = 150
VIEWS = ("Front", "Top", "Side")
COLS = ("Ground truth\n(clean, no algorithm)", "AW-NLOS on clean",
        "Degraded\n(no algorithm)", "AW-NLOS on degraded")


def render(transient):
    return lct.projections(lct.crop_for_display(lct.reconstruct(transient), transient))


def apply_algorithm(transient, fwhm_s, mu, block, z_trim):
    """Steps (b)-(e): adaptive windowing followed by TV transient completion."""
    aw, diag = window.adaptive_window(transient, fwhm_s=fwhm_s, block=block,
                                      z_trim=z_trim)
    return tv.complete(aw, mu_scale=mu, z_trim=z_trim), diag


def save_scene(views, scores, scene, stats, path):
    """One scene: 3 view rows x 4 condition columns."""
    fig, ax = plt.subplots(3, 4, figsize=(12.4, 9.6), squeeze=False)
    for c, (v, head, s) in enumerate(zip(views, COLS, scores)):
        title = head if s is None else f"{head}\nSSIM {s:.4f}"
        for r in range(3):
            ax[r][c].imshow(metrics.to_display(v[r]), cmap="gray")
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
            if c == 0:
                ax[r][c].set_ylabel(VIEWS[r], fontsize=12)
        ax[0][c].set_title(title, fontsize=10)
    fig.suptitle(f"AW-NLOS on clean vs degraded data - '{scene}'", fontsize=14)
    fig.text(0.5, 0.945, stats, ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="default: every scene in the table")
    ap.add_argument("--ppp", type=float, default=60.0)
    ap.add_argument("--sbr", type=float, default=4.0)
    ap.add_argument("--fwhm-ps", type=float, default=300.0)
    ap.add_argument("--mu", type=float, default=0.15)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/clean_vs_noisy")
    args = ap.parse_args()

    fwhm_s = args.fwhm_ps * 1e-12
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = args.scenes if args.scenes else list(SCENES)

    table = []
    for scene in scenes:
        t0 = time.time()
        try:
            clean = load_scene(scene)
        except Exception as e:
            print(f"{scene:<24} load failed: {e}")
            continue

        # Short cubes need the direct-component trim scaled down with them.
        z_trim = Z_TRIM if clean.M >= 512 else int(round(Z_TRIM * clean.M / 512))

        clean_alg, d_clean = apply_algorithm(clean, fwhm_s, args.mu, args.block, z_trim)
        noisy, info = degrade.degrade(clean, target_sbr=args.sbr, target_ppp=args.ppp,
                                      z_trim=z_trim, seed=args.seed)
        noisy_alg, d_noisy = apply_algorithm(noisy, fwhm_s, args.mu, args.block, z_trim)

        views = [render(t) for t in (clean, clean_alg, noisy, noisy_alg)]
        gt = views[0][0]
        scores = [None] + [metrics.ssim(v[0], gt) for v in views[1:]]

        cs = metrics.summarize(clean.data, z_trim)
        ns = metrics.summarize(noisy.data, z_trim)
        stats = (f"clean: SBR {cs['sbr']:.1f}, PPP {cs['ppp']:.0f}   |   "
                 f"degraded: SBR {ns['sbr']:.2f}, PPP {ns['ppp']:.1f} "
                 f"(alpha {info['alpha']:.3f}, ambient {info['ambient_rate']:.3f}/bin)   |   "
                 f"window duty: clean {d_clean['duty'] * 100:.1f}%, "
                 f"noisy {d_noisy['duty'] * 100:.1f}%")

        save_scene(views, scores, scene, stats, out / f"{scene}.png")
        table.append((scene, cs, ns, scores[1], scores[2], scores[3]))
        print(f"{scene:<24} clean+alg {scores[1]:.4f}   degraded {scores[2]:.4f}   "
              f"noisy+alg {scores[3]:.4f}   ({time.time() - t0:.0f}s)")

    if not table:
        return

    # ---- one summary sheet across every scene --------------------------------
    n = len(table)
    fig, a = plt.subplots(figsize=(max(9, 1.5 * n), 5.4))
    x = np.arange(n)
    w = 0.27
    a.bar(x - w, [r[3] for r in table], w, label="AW-NLOS on clean", color="#9aa0a6")
    a.bar(x, [r[4] for r in table], w, label="degraded, no algorithm", color="#d93025")
    a.bar(x + w, [r[5] for r in table], w, label="AW-NLOS on degraded", color="#1a73e8")
    a.set_xticks(x)
    a.set_xticklabels([r[0] for r in table], rotation=30, ha="right", fontsize=8)
    a.axhline(1.0, color="black", lw=1, ls="--", label="ground truth")
    a.set_ylabel("SSIM vs clean ground truth")
    a.set_ylim(0, 1.05)
    a.grid(axis="y", alpha=0.3)
    a.legend(fontsize=9)
    a.set_title(f"AW-NLOS on clean vs degraded data (PPP {args.ppp:g}, SBR {args.sbr:g})",
                fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)

    lines = ["scene,clean_sbr,clean_ppp,noisy_sbr,noisy_ppp,"
             "ssim_clean_alg,ssim_degraded,ssim_noisy_alg,gain"]
    for s, cs, ns, a1, a2, a3 in table:
        lines.append(f"{s},{cs['sbr']:.3f},{cs['ppp']:.2f},{ns['sbr']:.3f},"
                     f"{ns['ppp']:.2f},{a1:.4f},{a2:.4f},{a3:.4f},{a3 - a2:+.4f}")
    (out / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {out / 'summary.png'} and {out / 'summary.csv'}")
    print(f"\n{'scene':<24}{'clean+alg':>11}{'degraded':>11}{'noisy+alg':>11}{'gain':>10}")
    for s, _, _, a1, a2, a3 in table:
        print(f"{s:<24}{a1:>11.4f}{a2:>11.4f}{a3:>11.4f}{a3 - a2:>+10.4f}")


if __name__ == "__main__":
    main()
