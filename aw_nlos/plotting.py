"""Reusable plots for AW-NLOS steps (a)-(c).

Every function takes matplotlib axes or returns a figure, so the same building blocks
serve both the per-step figures in run_steps.py and the combined sheets in visualize.py.
"""

import numpy as np

import matplotlib
if matplotlib.get_backend().lower() not in ("agg", "module://matplotlib_inline.backend_inline"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import metrics


# ---------------------------------------------------------------- primitives

def show_xt(ax, cube, bin_resolution, title="", pct=99.5, cmap="inferno"):
    """Flatten the cube along one spatial axis into an (x, t) map, as the paper plots."""
    img = cube.sum(axis=1)
    t_ns = np.arange(cube.shape[2]) * bin_resolution * 1e9
    vmax = np.percentile(img, pct) or img.max() or 1
    ax.imshow(img, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
              extent=[t_ns[0], t_ns[-1], cube.shape[0], 0])
    ax.set(title=title, xlabel="time (ns)", ylabel="scan x")
    return ax


def show_map(ax, arr, title="", cmap="magma", label=None):
    """A 2-D map with a colorbar, for width and count maps."""
    im = ax.imshow(arr, cmap=cmap)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046)
    if label:
        cb.set_label(label, fontsize=8)
    return im


def brightest_pixel(cube, z_trim=0):
    """Index of the scan point holding the most counts, for representative 1-D plots."""
    return tuple(int(v) for v in np.unravel_index(
        np.argmax(cube[:, :, z_trim:].sum(axis=2)), cube.shape[:2]))


def brightest_block(blocks, z_trim=0):
    return tuple(int(v) for v in np.unravel_index(
        np.argmax(blocks[:, :, z_trim:].sum(axis=2)), blocks.shape[:2]))


# ------------------------------------------------------------------- step (a)

def plot_step_a(clean, noisy, z_trim=0):
    """Clean vs degraded transient: x-t maps, averaged histogram, one raw pixel."""
    br = clean.bin_resolution
    t_ns = clean.time_axis_ns
    st = metrics.summarize(noisy.data, z_trim)

    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    show_xt(ax[0][0], clean.data, br, "Clean capture: flattened transient (x-t)")
    show_xt(ax[0][1], noisy.data, br,
            f"Degraded: PPP={st['ppp']:.1f}, SBR={st['sbr']:.2f}")

    ax[1][0].plot(t_ns, clean.data.mean(axis=(0, 1)), lw=0.8, label="clean")
    ax[1][0].plot(t_ns, noisy.data.mean(axis=(0, 1)), lw=0.8, label="degraded")
    ax[1][0].axvline(z_trim * br * 1e9, color="k", ls=":", lw=1, label="z_trim")
    ax[1][0].set(xlabel="time (ns)", ylabel="mean counts / bin",
                 title="Pixel-averaged histogram")
    ax[1][0].legend(fontsize=8)

    px = brightest_pixel(noisy.data, z_trim)
    ax[1][1].plot(t_ns, noisy.data[px], lw=0.7, color="crimson")
    ax[1][1].set(xlabel="time (ns)", ylabel="counts",
                 title=f"Single pixel {px}: sparse, no clear cluster")

    fig.suptitle("Step (a) - acquire the 3D transient cube", fontsize=13)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- step (b)

def plot_step_b(noisy, result, z_trim=0):
    """Single pixel vs its block, plus the per-block count map."""
    blocks, block = result["blocks"], result["block"]
    t_ns = noisy.time_axis_ns
    ba, bb = brightest_block(blocks, z_trim)
    single = noisy.data[block * ba, block * bb]
    blk = blocks[ba, bb]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(t_ns, single, lw=0.7, color="crimson")
    ax[0].set(xlabel="time (ns)", ylabel="counts",
              title=f"Single pixel: {single[z_trim:].sum():.0f} counts")
    ax[1].plot(t_ns, blk, lw=0.7, color="navy")
    ax[1].set(xlabel="time (ns)", ylabel="counts",
              title=f"{block}x{block} block: {blk[z_trim:].sum():.0f} counts, cluster visible")
    show_map(ax[2], blocks[:, :, z_trim:].sum(axis=2), "Total counts per block",
             cmap="viridis")

    fig.suptitle("Step (b) - block aggregation: signal adds coherently, noise does not",
                 fontsize=13)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- step (c)

def plot_step_c(noisy, result, z_trim=0):
    """IRF template, one block's matched filter, the width map and its distribution."""
    g, widths = result["filtered"], result["block_widths_ps"]
    h, br = result["irf"], result["bin_resolution"]
    fwhm_ps = result["fwhm_s"] * 1e12
    t_ns = noisy.time_axis_ns
    ba, bb = brightest_block(result["blocks"], z_trim)

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0][0].plot(np.arange(len(h)) * br * 1e12, h, "o-", ms=3)
    ax[0][0].set(xlabel="time (ps)", ylabel="weight",
                 title=f"IRF h(t), FWHM {fwhm_ps:.0f} ps - matched-filter template")

    ax[0][1].plot(t_ns, result["blocks"][ba, bb], lw=0.6, alpha=0.5, color="grey",
                  label="block s(t)")
    ax[0][1].plot(t_ns, g[ba, bb], lw=1.2, color="navy", label="filtered g(t)")
    ax[0][1].axhline(result["block_floor"][ba, bb], color="crimson", ls="--", lw=1,
                     label=f"noise floor {result['block_floor'][ba, bb]:.2f}")
    ax[0][1].plot(t_ns[z_trim:][g[ba, bb, z_trim:].argmax()],
                  result["block_peak"][ba, bb], "v", color="darkorange", ms=10,
                  label=f"peak {result['block_peak'][ba, bb]:.1f}")
    ax[0][1].set(xlabel="time (ns)", ylabel="counts",
                 title=f"Matched filter -> T = {widths[ba, bb]:.0f} ps")
    ax[0][1].legend(fontsize=8)

    show_map(ax[1][0], widths, "Window width $T_{a,b}$ per block (ps)", label="ps")
    ax[1][1].hist(widths.ravel(), bins=30, color="steelblue")
    ax[1][1].set(xlabel="window width (ps)", ylabel="blocks",
                 title="Width distribution (Eq. 6)")

    fig.suptitle("Step (c) - matched filtering sets the adaptive window width",
                 fontsize=13)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------- overview

def plot_overview(clean, noisy, result, z_trim=0, scene=""):
    """All three steps on one sheet, left to right."""
    blocks, g = result["blocks"], result["filtered"]
    block, br = result["block"], result["bin_resolution"]
    widths = result["block_widths_ps"]
    t_ns = noisy.time_axis_ns
    st = metrics.summarize(noisy.data, z_trim)
    ba, bb = brightest_block(blocks, z_trim)

    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))

    show_xt(ax[0][0], noisy.data, br,
            f"(a) degraded transient - PPP {st['ppp']:.1f}, SBR {st['sbr']:.2f}")
    ax[1][0].plot(t_ns, noisy.data[block * ba, block * bb], lw=0.7, color="crimson")
    ax[1][0].set(xlabel="time (ns)", ylabel="counts",
                 title="(a) one raw pixel - no usable cluster")

    ax[0][1].plot(t_ns, blocks[ba, bb], lw=0.7, color="navy")
    ax[0][1].set(xlabel="time (ns)", ylabel="counts",
                 title=f"(b) {block}x{block} block - cluster emerges")
    show_map(ax[1][1], blocks[:, :, z_trim:].sum(axis=2), "(b) counts per block",
             cmap="viridis")

    ax[0][2].plot(t_ns, blocks[ba, bb], lw=0.5, alpha=0.4, color="grey", label="s(t)")
    ax[0][2].plot(t_ns, g[ba, bb], lw=1.2, color="darkgreen", label="g(t) = s ⊗ h")
    ax[0][2].axhline(result["block_floor"][ba, bb], color="crimson", ls="--", lw=1,
                     label="noise floor")
    ax[0][2].set(xlabel="time (ns)", ylabel="counts",
                 title=f"(c) matched filter -> T = {widths[ba, bb]:.0f} ps")
    ax[0][2].legend(fontsize=8)
    show_map(ax[1][2], widths, "(c) window width per block (ps)", label="ps")

    fig.suptitle(f"AW-NLOS steps (a) -> (b) -> (c)"
                 + (f" - scene '{scene}'" if scene else ""), fontsize=14)
    fig.tight_layout()
    return fig


def plot_width_vs_noise(rows, fwhm_ps=300.0):
    """Median window width against SBR, showing the adaptivity of Eq. 6.

    rows: list of (sbr, widths_ps_array).
    """
    sbrs = [r[0] for r in rows]
    med = [float(np.median(r[1])) for r in rows]
    lo = [float(np.percentile(r[1], 10)) for r in rows]
    hi = [float(np.percentile(r[1], 90)) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.fill_between(sbrs, lo, hi, alpha=0.2, color="steelblue", label="10-90th percentile")
    ax.plot(sbrs, med, "o-", color="steelblue", lw=2, label="median width")
    ax.axhline(2 * fwhm_ps, color="crimson", ls="--", lw=1,
               label=f"$T_{{min}}$ = 2 x FWHM = {2 * fwhm_ps:.0f} ps")
    ax.set(xlabel="SBR (higher = less ambient light)", ylabel="window width (ps)",
           title="Eq. 6 adaptivity: more ambient light gives a stricter, narrower window")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig
