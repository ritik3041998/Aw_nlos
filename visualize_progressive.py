"""Progressive visualization: reconstruct after every pipeline step and score it.

Each stage applies one more operation to the transient, then runs the same LCT
reconstruction used for the ground truth, so the front/top/side views are directly
comparable and the SSIM shows what that one step bought.

Stages 2 and 3 are diagnostic. In the real algorithm, block aggregation and matched
filtering only *size the window* -- they are never applied to the data that gets
reconstructed. They are reconstructed here to show what each operation does on its own.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from aw import degrade, irf as irf_mod, lct, metrics, tv, window
from aw.io_utils import load_scene

Z_TRIM = 150
VIEWS = ("Front", "Top", "Side")


def upsample_blocks(blocks, block, shape):
    """Expand block sums back to full scan resolution, preserving count level."""
    up = np.repeat(np.repeat(blocks, block, axis=0), block, axis=1) / (block ** 2)
    return up[:shape[0], :shape[1], :].astype(np.float32)


def render(transient):
    """Reconstruct and return the three maximum-intensity projections."""
    return lct.projections(lct.crop_for_display(lct.reconstruct(transient), transient))


def save_stage(views, gt_views, title, subtitle, score, path):
    """One stage on its own, in the same layout as the ground truth."""
    wrapped = "\n".join(textwrap.wrap(subtitle, 110))
    n_lines = wrapped.count("\n") + 1
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6 + 0.18 * n_lines))
    for a, img, name in zip(ax, views, VIEWS):
        a.imshow(metrics.to_display(img), cmap="gray")
        a.set_title(name, fontsize=11)
        a.set_xticks([])
        a.set_yticks([])
    head = title if score is None else f"{title}     SSIM = {score:.4f}"
    top = 0.995 - 0.045 * n_lines
    fig.suptitle(head, fontsize=14, y=0.985)
    fig.text(0.5, top + 0.012, wrapped, ha="center", va="top",
             fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, top - 0.02])
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="mannequin")
    ap.add_argument("--ppp", type=float, default=60.0)
    ap.add_argument("--sbr", type=float, default=4.0)
    ap.add_argument("--fwhm-ps", type=float, default=300.0)
    ap.add_argument("--mu", type=float, default=0.15)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    out = Path("results/progressive") / args.scene
    out.mkdir(parents=True, exist_ok=True)

    fwhm_s = args.fwhm_ps * 1e-12
    clean = load_scene(args.scene)
    print(f"{clean}  snr={clean.snr}\n")

    # ---- stage 0: ground truth, from the undegraded capture -------------------
    gt = render(clean)
    save_stage(gt, gt, "Stage 0 - Ground truth",
               f"clean capture, {metrics.measure_ppp(clean.data, Z_TRIM):.0f} signal PPP, "
               f"SBR {metrics.measure_sbr(clean.data, Z_TRIM):.1f}",
               None, out / "stage_00_ground_truth.png")

    # ---- build each stage cumulatively ---------------------------------------
    noisy, info = degrade.degrade(clean, target_sbr=args.sbr, target_ppp=args.ppp,
                                  z_trim=Z_TRIM, seed=args.seed)
    st = metrics.summarize(noisy.data, Z_TRIM)

    h, _ = irf_mod.gaussian_irf(fwhm_s, clean.bin_resolution)
    blocks = window.block_sum(noisy.data, args.block)
    blocks_full = noisy.replace(upsample_blocks(blocks, args.block, noisy.data.shape))
    mf_full = noisy.replace(
        fftconvolve(noisy.data, h[None, None, :], mode="same", axes=-1).astype(np.float32))
    aw, diag = window.adaptive_window(noisy, fwhm_s=fwhm_s, block=args.block, z_trim=Z_TRIM)
    awtv = tv.complete(aw, mu_scale=args.mu, z_trim=Z_TRIM)

    # diag=True marks a side branch: the operation is applied to the data here only to
    # show what it does, never in the real pipeline.
    stages = [
        ("Stage 1 - Degraded input",
         f"after thinning + ambient: PPP {st['ppp']:.2f}, SBR {st['sbr']:.2f} "
         f"(what the detector actually sees)", noisy, "stage_01_degraded.png", False),
        (f"Stage 2 - (b) {args.block}x{args.block} block aggregation",
         "DIAGNOSTIC branch: signal adds coherently, noise does not -- but 4x spatial "
         "detail is lost. The pipeline uses blocks only to size the window.",
         blocks_full, "stage_02_block_aggregation.png", True),
        ("Stage 3 - (c) matched filtering with the IRF",
         f"DIAGNOSTIC branch: correlating with h(t), FWHM {args.fwhm_ps:.0f} ps. "
         "The pipeline filters only to locate the cluster.",
         mf_full, "stage_03_matched_filter.png", True),
        ("Stage 4 - (d) adaptive windowing",
         f"median window {np.median(diag['width_ps']):.0f} ps, "
         f"{diag['duty'] * 100:.1f}% of bins kept, "
         f"SBR {metrics.measure_sbr(aw.data, Z_TRIM):.1f}", aw, "stage_04_windowed.png",
         False),
        ("Stage 5 - (e) TV transient completion",
         f"isotropic spatial TV, mu = {args.mu} x peak -- repairs windowing holes",
         awtv, "stage_05_tv_completed.png", False),
    ]

    labels, scores, all_views, flags = ["Ground truth"], [None], [gt], [False]
    for title, subtitle, transient, fname, is_diag in stages:
        views = render(transient)
        score = metrics.ssim(views[0], gt[0])
        save_stage(views, gt, title, subtitle, score, out / fname)
        labels.append(title.split(" - ", 1)[1])
        scores.append(score)
        all_views.append(views)
        flags.append(is_diag)
        print(f"     -> SSIM {score:.4f}{'   [diagnostic]' if is_diag else ''}")

    # ---- the whole progression on one sheet ----------------------------------
    n = len(all_views)
    fig, ax = plt.subplots(3, n, figsize=(2.9 * n, 9.4), squeeze=False)
    for c, (views, label, score, is_diag) in enumerate(zip(all_views, labels, scores, flags)):
        head = label if score is None else f"{label}\nSSIM {score:.3f}"
        color = "darkorange" if is_diag else "black"
        for r in range(3):
            ax[r][c].imshow(metrics.to_display(views[r]), cmap="gray")
            ax[r][c].set_xticks([])
            ax[r][c].set_yticks([])
            if c == 0:
                ax[r][c].set_ylabel(VIEWS[r], fontsize=12)
            if is_diag:
                for sp in ax[r][c].spines.values():
                    sp.set_edgecolor("darkorange")
                    sp.set_linewidth(2.5)
        ax[0][c].set_title(head, fontsize=10, color=color)
    fig.suptitle(f"AW-NLOS progressive reconstruction - '{args.scene}', "
                 f"PPP {st['ppp']:.1f}, SBR {st['sbr']:.2f}", fontsize=14)
    fig.text(0.5, 0.945, "black = cumulative pipeline   |   "
             "orange = diagnostic side branch, not applied in the real algorithm",
             ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0, 1, 0.935])
    fig.savefig(out / "progression_grid.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out / 'progression_grid.png'}")

    # ---- SSIM progression ----------------------------------------------------
    idx = [i for i, s in enumerate(scores) if s is not None]
    fig, a = plt.subplots(figsize=(11, 4.8))
    vals = [scores[i] for i in idx]
    names = [labels[i] for i in idx]
    dflags = [flags[i] for i in idx]

    bars = a.bar(range(len(vals)), vals,
                 color=["#f9ab00" if d else "#1a73e8" for d in dflags],
                 hatch=["//" if d else "" for d in dflags], edgecolor="white")
    for b, v in zip(bars, vals):
        a.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
               ha="center", fontsize=10)

    # Overlay the cumulative chain only, so the real progression reads monotonically.
    cx = [i for i, d in enumerate(dflags) if not d]
    a.plot(cx, [vals[i] for i in cx], "o-", color="#1a73e8", lw=2, ms=7,
           label="cumulative pipeline")
    a.bar(0, 0, color="#f9ab00", hatch="//", edgecolor="white",
          label="diagnostic branch (not applied)")

    a.set_xticks(range(len(vals)))
    a.set_xticklabels([n.replace(" - ", "\n") for n in names], fontsize=8)
    a.set_ylabel("SSIM vs ground truth")
    a.set_ylim(0, max(vals) * 1.28)
    a.grid(axis="y", alpha=0.3)
    a.legend(fontsize=9, loc="upper left")
    a.set_title(f"SSIM after each step - '{args.scene}', PPP {st['ppp']:.1f}, "
                f"SBR {st['sbr']:.2f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "ssim_progression.png", dpi=130)
    plt.close(fig)
    print(f"  wrote {out / 'ssim_progression.png'}")

    print("\nSSIM progression")
    for name, s in zip(names, vals):
        print(f"  {name:<45} {s:.4f}")


if __name__ == "__main__":
    main()
