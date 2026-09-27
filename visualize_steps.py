"""Dump one figure per pipeline stage of Fig. 3, so every step can be inspected.

Runs the full AW-NLOS pipeline once at a chosen operating point and saves the
intermediate result of each step to results/steps/.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aw import degrade, irf as irf_mod, lct, metrics, tv, window
from aw.io_utils import load_scene

Z_TRIM = 150
OUT = Path("results/steps")  # replaced per scene in main()


def flat_xt(cube):
    """Flatten the cube along one spatial axis: an (x, t) map, as the paper plots."""
    return cube.sum(axis=1)


def show_xt(ax, cube, bin_res, title, vmax_pct=99.5):
    img = flat_xt(cube)
    t_ns = np.arange(cube.shape[2]) * bin_res * 1e9
    vmax = np.percentile(img, vmax_pct) or img.max() or 1
    ax.imshow(img, aspect="auto", cmap="inferno", vmin=0, vmax=vmax,
              extent=[t_ns[0], t_ns[-1], cube.shape[0], 0])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("scan x")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="s_u")
    ap.add_argument("--ppp", type=float, default=5.0)
    ap.add_argument("--sbr", type=float, default=2.12)
    ap.add_argument("--fwhm-ps", type=float, default=300.0)
    ap.add_argument("--mu", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    global OUT
    OUT = Path("results/steps") / args.scene

    fwhm_s = args.fwhm_ps * 1e-12
    clean = load_scene(args.scene)
    br = clean.bin_resolution
    t_ns = np.arange(clean.M) * br * 1e9
    print(clean)

    # ---------------------------------------------------------------- step (a)
    print("\n[a] raw transient acquisition")
    noisy, info = degrade.degrade(clean, target_sbr=args.sbr, target_ppp=args.ppp,
                                  z_trim=Z_TRIM, seed=args.seed)
    st_clean = metrics.summarize(clean.data, Z_TRIM, "clean")
    st_noisy = metrics.summarize(noisy.data, Z_TRIM, "degraded")
    print("   " + metrics.fmt(st_clean))
    print("   " + metrics.fmt(st_noisy))

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    show_xt(ax[0][0], clean.data, br, "Clean capture: flattened transient (x-t)")
    show_xt(ax[0][1], noisy.data, br,
            f"Degraded: PPP={st_noisy['ppp']:.2f}, SBR={st_noisy['sbr']:.2f}")
    ax[1][0].plot(t_ns, clean.data.mean(axis=(0, 1)), lw=0.8, label="clean")
    ax[1][0].plot(t_ns, noisy.data.mean(axis=(0, 1)), lw=0.8, label="degraded")
    ax[1][0].axvline(Z_TRIM * br * 1e9, color="k", ls=":", lw=1, label="z_trim")
    ax[1][0].set(xlabel="time (ns)", ylabel="mean counts / bin",
                 title="Pixel-averaged histogram")
    ax[1][0].legend(fontsize=8)
    px = tuple(int(v) for v in np.unravel_index(
        np.argmax(noisy.data[:, :, Z_TRIM:].sum(axis=2)), noisy.data.shape[:2]))
    ax[1][1].plot(t_ns, noisy.data[px[0], px[1]], lw=0.7, color="crimson")
    ax[1][1].set(xlabel="time (ns)", ylabel="counts",
                 title=f"Single-pixel transient at {px} - sparse, no visible cluster")
    fig.suptitle("Step (a) - acquire 3D transient photon data", fontsize=13)
    save(fig, "01_step_a_raw_transient.png")

    # ---------------------------------------------------------------- step (b)
    print("\n[b] pixel-block aggregation (Eq. 4)")
    blocks = window.block_sum(noisy.data, block=4)
    ba, bb = np.unravel_index(np.argmax(blocks[:, :, Z_TRIM:].sum(axis=2)),
                              blocks.shape[:2])
    print(f"   {noisy.N}x{noisy.N} -> {blocks.shape[0]}x{blocks.shape[1]} blocks; "
          f"strongest block {(ba, bb)}")
    single = noisy.data[4 * ba, 4 * bb]
    blk = blocks[ba, bb]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t_ns, single, lw=0.7, color="crimson")
    ax[0].set(xlabel="time (ns)", ylabel="counts",
              title=f"Single pixel ({4*ba},{4*bb}): {single[Z_TRIM:].sum():.0f} counts")
    ax[1].plot(t_ns, blk, lw=0.7, color="navy")
    ax[1].set(xlabel="time (ns)", ylabel="counts",
              title=f"4x4 block ({ba},{bb}): {blk[Z_TRIM:].sum():.0f} counts - cluster visible")
    im = ax[2].imshow(blocks[:, :, Z_TRIM:].sum(axis=2), cmap="viridis")
    ax[2].set_title("Total counts per block")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle("Step (b) - spatial correlation: signal adds coherently, noise does not",
                 fontsize=13)
    save(fig, "02_step_b_pixel_blocks.png")

    # ---------------------------------------------------------------- step (c)
    print("\n[c] matched filtering + window sizing (Eq. 5-6)")
    h, sigma = irf_mod.gaussian_irf(fwhm_s, br)
    g = window.matched_filter(blocks, h)
    widths, peak, floor = window.window_widths(g, fwhm_s, br, Z_TRIM)
    print(f"   IRF: FWHM={args.fwhm_ps:.0f} ps = {fwhm_s/br:.1f} bins, sigma={sigma:.2f} bins")
    print(f"   widths: min {widths.min()*1e12:.0f}  median {np.median(widths)*1e12:.0f}  "
          f"max {widths.max()*1e12:.0f} ps")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0][0].plot(np.arange(len(h)) * br * 1e12, h, "o-", ms=3)
    ax[0][0].set(xlabel="time (ps)", ylabel="weight",
                 title=f"IRF h(t), FWHM {args.fwhm_ps:.0f} ps - matched-filter template")
    ax[0][1].plot(t_ns, blk, lw=0.6, alpha=0.5, color="grey", label="block s(t)")
    ax[0][1].plot(t_ns, g[ba, bb], lw=1.2, color="navy", label="filtered g(t)")
    ax[0][1].axhline(floor[ba, bb], color="crimson", ls="--", lw=1,
                     label=f"noise floor {floor[ba,bb]:.2f}")
    ax[0][1].plot(t_ns[Z_TRIM:][g[ba, bb, Z_TRIM:].argmax()], peak[ba, bb], "v",
                  color="darkorange", ms=10, label=f"peak {peak[ba,bb]:.1f}")
    ax[0][1].set(xlabel="time (ns)", ylabel="counts",
                 title=f"Matched filter -> T = {widths[ba,bb]*1e12:.0f} ps")
    ax[0][1].legend(fontsize=8)
    im = ax[1][0].imshow(widths * 1e12, cmap="magma")
    ax[1][0].set_title("Window width $T_{a,b}$ per block (ps)")
    plt.colorbar(im, ax=ax[1][0], fraction=0.046)
    ax[1][1].hist(widths.ravel() * 1e12, bins=30, color="steelblue")
    ax[1][1].set(xlabel="window width (ps)", ylabel="blocks",
                 title="Width distribution (Eq. 6)")
    fig.suptitle("Step (c) - matched filtering determines the adaptive window width",
                 fontsize=13)
    save(fig, "03_step_c_matched_filter.png")

    # ---------------------------------------------------------------- step (d)
    print("\n[d] window placement and application (Eq. 7-8)")
    aw, diag = window.adaptive_window(noisy, fwhm_s=fwhm_s, z_trim=Z_TRIM)
    starts, w_bins = diag["starts"], diag["w_bins"]
    st_win = metrics.summarize(aw.data, Z_TRIM, "windowed")
    print("   " + metrics.fmt(st_win))
    print(f"   duty cycle {diag['duty']*100:.2f}% of bins retained")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    im = ax[0][0].imshow(starts * br * 1e9, cmap="turbo")
    # Meaningful only where a cluster exists; elsewhere argmax lands on noise.
    ax[0][0].set_title("Window start $t^m_{i,j}$ (ns) - depth where signal is present")
    plt.colorbar(im, ax=ax[0][0], fraction=0.046)
    im = ax[0][1].imshow(w_bins * br * 1e12, cmap="magma")
    ax[0][1].set_title("Window width per pixel (ps)")
    plt.colorbar(im, ax=ax[0][1], fraction=0.046)
    p = noisy.data[px[0], px[1]]
    ax[0][2].plot(t_ns, p, lw=0.7, color="grey", label="raw")
    s0, s1 = starts[px], starts[px] + w_bins[px]
    ax[0][2].axvspan(t_ns[s0], t_ns[min(s1, len(t_ns) - 1)], color="gold", alpha=0.35,
                     label="window")
    ax[0][2].plot(t_ns, aw.data[px[0], px[1]], lw=0.9, color="crimson", label="kept")
    ax[0][2].set(xlabel="time (ns)",
                 title=f"Pixel {px}: window applied to RAW counts, not the filtered signal")
    ax[0][2].legend(fontsize=8)
    show_xt(ax[1][0], noisy.data, br, f"Before: SBR {st_noisy['sbr']:.2f}")
    show_xt(ax[1][1], aw.data, br, f"After windowing: SBR {st_win['sbr']:.2f}")
    mask = np.zeros_like(aw.data, dtype=bool)
    idx = np.arange(aw.M)[None, None, :]
    np.copyto(mask, (idx >= starts[:, :, None]) & (idx < (starts + w_bins)[:, :, None]))
    ax[1][2].imshow(mask.sum(axis=1), aspect="auto", cmap="Greys_r",
                    extent=[t_ns[0], t_ns[-1], aw.N, 0])
    ax[1][2].set(xlabel="time (ns)", ylabel="scan x", title="Retained window mask")
    fig.suptitle("Step (d) - per-pixel windowing isolates signal at the data level",
                 fontsize=13)
    save(fig, "04_step_d_windowing.png")

    # ---------------------------------------------------------------- step (e)
    print("\n[e] TV transient completion (Eq. 9-10)")
    awtv = tv.complete(aw, mu_scale=args.mu, z_trim=Z_TRIM)
    peak_bin = int(np.argmax(aw.data.sum(axis=(0, 1))))
    print(f"   mu = {args.mu} x peak; showing time slice at bin {peak_bin}")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8))
    sl_a, sl_b = aw.data[:, :, peak_bin], awtv.data[:, :, peak_bin]
    vmax = max(sl_a.max(), sl_b.max()) or 1
    ax[0][0].imshow(sl_a, cmap="inferno", vmin=0, vmax=vmax)
    ax[0][0].set_title(f"Windowed, time slice {peak_bin} - holes and block edges")
    ax[0][1].imshow(sl_b, cmap="inferno", vmin=0, vmax=vmax)
    ax[0][1].set_title("After spatial TV - continuity restored")
    ax[0][2].plot(sl_a[:, sl_a.shape[1] // 2], lw=0.9, label="windowed")
    ax[0][2].plot(sl_b[:, sl_b.shape[1] // 2], lw=1.2, label="TV completed")
    ax[0][2].set(xlabel="scan x", ylabel="counts", title="Line profile through the slice")
    ax[0][2].legend(fontsize=8)
    show_xt(ax[1][0], aw.data, br, "Windowed transient")
    show_xt(ax[1][1], awtv.data, br, "TV-completed transient")
    ax[1][2].plot(t_ns, aw.data.mean(axis=(0, 1)), lw=0.8, label="windowed")
    ax[1][2].plot(t_ns, awtv.data.mean(axis=(0, 1)), lw=0.8, label="TV completed")
    ax[1][2].set(xlabel="time (ns)", ylabel="mean counts", title="Pixel-averaged histogram")
    ax[1][2].legend(fontsize=8)
    fig.suptitle("Step (e) - TV regularization fills the transient (spatial axes only)",
                 fontsize=13)
    save(fig, "05_step_e_tv_completion.png")

    # ---------------------------------------------------------------- step (f)
    print("\n[f] LCT reconstruction (Eq. 11-13)")
    gt = lct.projections(lct.crop_for_display(lct.reconstruct(clean), clean))
    stages = [("Degraded input", noisy), ("+ adaptive window", aw), ("+ TV completion", awtv)]
    recons = []
    for name, tr in stages:
        f, t_, s_ = lct.projections(lct.crop_for_display(lct.reconstruct(tr), tr))
        score = metrics.ssim(f, gt[0])
        recons.append((name, f, t_, s_, score))
        print(f"   {name:<20} SSIM={score:.4f}")

    fig, ax = plt.subplots(3, 4, figsize=(15, 10))
    for r, label in enumerate(["Front", "Top", "Side"]):
        ax[r][0].imshow(metrics.to_display(gt[r]), cmap="gray")
        ax[r][0].set_ylabel(label, fontsize=11)
        if r == 0:
            ax[r][0].set_title("Ground truth", fontsize=10)
        for c, (name, f, t_, s_, score) in enumerate(recons, start=1):
            ax[r][c].imshow(metrics.to_display((f, t_, s_)[r]), cmap="gray")
            if r == 0:
                ax[r][c].set_title(f"{name}\nSSIM {score:.3f}", fontsize=10)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Step (f) - reconstruct the hidden object (LCT + Wiener)", fontsize=13)
    save(fig, "06_step_f_reconstruction.png")

    # ------------------------------------------------------------- LCT internals
    print("\n[f*] LCT operator internals")
    psf = lct.define_psf(clean.N, clean.M, clean.width / clean.range_m)
    mtx, _ = lct.resampling_operator(clean.M)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].imshow(psf[:, :, psf.shape[2] // 2], aspect="auto", cmap="inferno")
    ax[0].set(title="Light-cone kernel g, (v, y) slice", xlabel="y", ylabel="v")
    ax[1].imshow(np.roll(psf, (-clean.N, -clean.N), axis=(1, 2))[:200, :, psf.shape[2] // 2],
                 aspect="auto", cmap="inferno")
    ax[1].set(title="Kernel near origin - the paraboloid", xlabel="y", ylabel="v")
    ax[2].spy(mtx[:64, :64], markersize=2)
    ax[2].set(title="Resampling operator $T_t$ (top-left 64x64)")
    fig.suptitle("Step (f) internals - the operators that make Eq. 11 a convolution",
                 fontsize=13)
    save(fig, "07_lct_internals.png")

    np.savez_compressed(OUT / "stage_data.npz",
                        widths=widths, starts=starts, w_bins=w_bins,
                        block_peak=peak, block_floor=floor, irf=h,
                        gt_front=gt[0], **{f"front_{i}": r[1] for i, r in enumerate(recons)})
    print(f"\n  wrote {OUT/'stage_data.npz'}")
    print("\nSSIM: " + "  ".join(f"{n}={s:.4f}" for n, _, _, _, s in recons))


if __name__ == "__main__":
    main()
