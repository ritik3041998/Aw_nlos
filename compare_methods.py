"""Method x data matrix: traditional LCT vs AW-NLOS, each on raw and on noisy data.

                      |  raw capture        noisy capture
  --------------------+------------------------------------
  traditional LCT     |  reference          A
  AW-NLOS + LCT       |  B                  C

The cell the sweep figures report is C. This layout adds the two that make C
interpretable:

  drop_trad = 1 - SSIM(A)         how far noise alone pushes the traditional pipeline
  drop_aw   = SSIM(B) - SSIM(C)   how far the same noise pushes AW-NLOS

The smaller drop is the robustness claim, and it is a fairer statement than the raw
SSIM gap, because AW-NLOS starts from B < 1 rather than from the reference itself.
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
ROWS = ("Traditional LCT", "AW-NLOS + LCT")
COLS = ("Raw capture", "Noisy capture")


def render(transient):
    return lct.projections(lct.crop_for_display(lct.reconstruct(transient), transient))


def apply_algorithm(transient, fwhm_s, mu, block, z_trim):
    aw, diag = window.adaptive_window(transient, fwhm_s=fwhm_s, block=block,
                                      z_trim=z_trim)
    return tv.complete(aw, mu_scale=mu, z_trim=z_trim), diag


def save_matrix(cells, scene, stats, path):
    """2 method rows x 2 data columns, front views, one scene."""
    fig, ax = plt.subplots(2, 2, figsize=(9.2, 10.0), squeeze=False)
    for r in range(2):
        for c in range(2):
            img, s, p = cells[r][c]
            ax[r][c].imshow(metrics.to_display(img), cmap="gray")
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
            tag = "reference" if (r == 0 and c == 0) else f"SSIM {s:.4f}    PSNR {p:.1f} dB"
            # The score is drawn inside its own image. Put it in a title or an xlabel
            # and it lands in the gap between rows, where it reads as a caption for
            # the row above; a boxed overlay is unambiguous and survives the dark
            # backgrounds these reconstructions all have.
            ax[r][c].text(0.5, 0.015, tag, transform=ax[r][c].transAxes,
                          ha="center", va="bottom", fontsize=13, color="white",
                          fontweight="bold",
                          bbox=dict(facecolor="black", alpha=0.72, pad=5,
                                    edgecolor="white", linewidth=0.6))
            if r == 0:
                ax[r][c].set_title(COLS[c], fontsize=14)
            if c == 0:
                ax[r][c].set_ylabel(ROWS[r], fontsize=14)
    fig.suptitle(f"Traditional LCT vs AW-NLOS, raw vs noisy - '{scene}'", fontsize=14)
    fig.text(0.5, 0.945, stats, ha="center", fontsize=8.5, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.935], h_pad=2.0)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_views(views, scene, path):
    """The same four reconstructions, all three projections, for a closer look."""
    labels = [f"{ROWS[r]}\n{COLS[c]}" for r in range(2) for c in range(2)]
    fig, ax = plt.subplots(3, 4, figsize=(12.4, 9.6), squeeze=False)
    for c, (v, lab) in enumerate(zip(views, labels)):
        for r in range(3):
            ax[r][c].imshow(metrics.to_display(v[r]), cmap="gray")
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
            if c == 0:
                ax[r][c].set_ylabel(VIEWS[r], fontsize=12)
        ax[0][c].set_title(lab, fontsize=10)
    fig.suptitle(f"All projections - '{scene}'", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=None)
    ap.add_argument("--ppp", type=float, default=60.0)
    ap.add_argument("--sbr", type=float, default=4.0)
    ap.add_argument("--fwhm-ps", type=float, default=300.0)
    ap.add_argument("--mu", type=float, default=0.15)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/method_matrix")
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

        z_trim = Z_TRIM if clean.M >= 512 else int(round(Z_TRIM * clean.M / 512))

        noisy, info = degrade.degrade(clean, target_sbr=args.sbr, target_ppp=args.ppp,
                                      z_trim=z_trim, seed=args.seed)
        clean_aw, d_clean = apply_algorithm(clean, fwhm_s, args.mu, args.block, z_trim)
        noisy_aw, d_noisy = apply_algorithm(noisy, fwhm_s, args.mu, args.block, z_trim)

        v_tr, v_tn = render(clean), render(noisy)
        v_ar, v_an = render(clean_aw), render(noisy_aw)
        ref = v_tr[0]

        def score(v):
            return metrics.ssim(v[0], ref), metrics.psnr(v[0], ref)

        s_tn, s_ar, s_an = score(v_tn), score(v_ar), score(v_an)
        cells = [[(v_tr[0], 1.0, np.inf), (v_tn[0],) + s_tn],
                 [(v_ar[0],) + s_ar, (v_an[0],) + s_an]]

        cs = metrics.summarize(clean.data, z_trim)
        ns = metrics.summarize(noisy.data, z_trim)
        stats = (f"raw: SBR {cs['sbr']:.1f}, PPP {cs['ppp']:.0f}   |   "
                 f"noisy: SBR {ns['sbr']:.2f}, PPP {ns['ppp']:.1f} "
                 f"(alpha {info['alpha']:.3f}, ambient {info['ambient_rate']:.3f}/bin)   |   "
                 f"window duty: raw {d_clean['duty'] * 100:.1f}%, "
                 f"noisy {d_noisy['duty'] * 100:.1f}%")

        save_matrix(cells, scene, stats, out / f"{scene}.png")
        save_views([v_tr, v_tn, v_ar, v_an], scene, out / f"{scene}_all_views.png")

        drop_trad = 1.0 - s_tn[0]
        drop_aw = s_ar[0] - s_an[0]
        table.append((scene, s_tn[0], s_ar[0], s_an[0], drop_trad, drop_aw))
        print(f"{scene:<24} trad/noisy {s_tn[0]:.4f}   aw/raw {s_ar[0]:.4f}   "
              f"aw/noisy {s_an[0]:.4f}   drop: trad {drop_trad:.4f} vs aw {drop_aw:.4f}   "
              f"({time.time() - t0:.0f}s)")

    if not table:
        return

    # ---- robustness: how far each method falls when the noise is added -------
    n = len(table)
    fig, ax = plt.subplots(1, 2, figsize=(max(13, 1.7 * n), 5.2))
    x = np.arange(n)
    w = 0.38

    ax[0].bar(x - w / 2, [r[4] for r in table], w, label="Traditional LCT",
              color="#d93025")
    ax[0].bar(x + w / 2, [r[5] for r in table], w, label="AW-NLOS", color="#1a73e8")
    ax[0].set_ylabel("SSIM lost when noise is added")
    ax[0].set_title("Robustness: drop from raw to noisy (lower is better)", fontsize=11)

    ax[1].bar(x - w / 2, [r[1] for r in table], w, label="Traditional LCT",
              color="#d93025")
    ax[1].bar(x + w / 2, [r[3] for r in table], w, label="AW-NLOS", color="#1a73e8")
    ax[1].set_ylabel("SSIM vs raw traditional reference")
    ax[1].set_ylim(0, 1.0)
    ax[1].set_title("Absolute quality on the noisy capture", fontsize=11)

    for a in ax:
        a.set_xticks(x)
        a.set_xticklabels([r[0] for r in table], rotation=30, ha="right", fontsize=8)
        a.grid(axis="y", alpha=0.3)
        a.legend(fontsize=9)
    fig.suptitle(f"Traditional LCT vs AW-NLOS (PPP {args.ppp:g}, SBR {args.sbr:g})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "summary.png", dpi=130)
    plt.close(fig)

    lines = ["scene,ssim_trad_noisy,ssim_aw_raw,ssim_aw_noisy,drop_trad,drop_aw"]
    for s, a, b, c, dt, da in table:
        lines.append(f"{s},{a:.4f},{b:.4f},{c:.4f},{dt:.4f},{da:.4f}")
    (out / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {out / 'summary.png'} and {out / 'summary.csv'}")
    print(f"\n{'scene':<24}{'trad/noisy':>12}{'aw/raw':>10}{'aw/noisy':>10}"
          f"{'drop trad':>11}{'drop aw':>10}")
    for s, a, b, c, dt, da in table:
        print(f"{s:<24}{a:>12.4f}{b:>10.4f}{c:>10.4f}{dt:>11.4f}{da:>10.4f}")


if __name__ == "__main__":
    main()
