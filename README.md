# AW-NLOS — Python reimplementation

A Python implementation of **adaptive windowing non-line-of-sight (AW-NLOS) imaging**:

> Jinye Miao, Fuyao Cai, Taotao Qin, Lianfa Bai, Enlai Guo, Yingjie Shi, Jing Han.
> *"Adaptive windowing for photon-efficient non-line-of-sight imaging under high ambient light."*
> **Optics Express** 33(21), 44522 (2025). [doi:10.1364/OE.575419](https://doi.org/10.1364/OE.575419)

The repository holds **two packages**, both on the paper's Fig. 3 pipeline.

```
                                       aw_nlos/   aw/
(a) acquire 3D transient cube             yes     yes
(b) 4x4 pixel-block aggregation           yes     yes
(c) matched filtering -> window width     yes     yes
(d) apply window in the time domain        -      yes
(e) TV transient completion                -      yes
(f) LCT reconstruction (Wiener)            -      yes
```

- **`aw_nlos/`** — steps (a)–(c) only, documented in depth below. Stops at the window width and writes `window_widths.npz`. Read this first to understand how the window is sized.
- **`aw/`** — the full pipeline through reconstruction, plus the comparison experiments and every figure in `results/`. See **The full pipeline** immediately below.

Neither package imports the other; they are independent and can be read separately.

---

## The full pipeline — `aw/`

`aw/` carries steps (a)–(c) through to a reconstructed volume: the window is applied to the raw counts (d), the holes are repaired with spatial TV (e), and the result is inverted with a Python port of the O'Toole Light Cone Transform (f).

```
aw/io_utils.py   (a) loading, downsampling, direct-component removal
aw/degrade.py        binomial thinning + Poisson ambient injection
aw/irf.py            synthesized Gaussian IRF
aw/window.py     (b) Eq. 4, (c) Eqs. 5-7, (d) Eq. 8
aw/tv.py         (e) Eqs. 9-10, isotropic spatial TV
aw/lct.py        (f) Eqs. 11-13, LCT + Wiener deconvolution
aw/metrics.py        SBR, PPP, SSIM, PSNR, display scaling
```

### Running it on your own capture

`run_single.py` is the entry point for a single `.mat` file. Every other driver
takes `--scene`, a name from the bundled table, so this is the one to use for
data of your own.

```bash
python run_single.py path/to/capture.mat --width 0.35
```

`--width` is the half-extent of the scanned wall patch in metres, and is read
from the file when it carries a `width` variable. The cube is auto-detected when
the file holds exactly one 3D array; otherwise pass `--data-key`.

Two modes:

```bash
# clean lab capture: degrade it synthetically, reconstruct four ways
python run_single.py capture.mat --width 0.35 --ppp 60 --sbr 4.0

# real low-SBR capture that already carries ambient noise: no degradation
python run_single.py capture.mat --width 0.35 --no-degrade
```

The first writes a 2x2 matrix (traditional LCT and AW-NLOS, each on the raw and
the degraded cube); the second writes traditional LCT against AW-NLOS on the
capture as given. Both land in `results/single/<filename>.png`.

**What your capture must look like.** A `(N, N, M)` array of photon counts:
two wall-scan axes, square, and one TCSPC time axis, pre-rectified so the direct
component starts at the first time bin.

**Flags that usually need setting for a new rig:**

| Flag | Default | Set it when |
|---|---|---|
| `--width` | from file | The file carries no `width` variable |
| `--bin-ps` | `4.0` | Your TCSPC bin is not 4 ps *before* downsampling |
| `--fwhm-ps` | `300` | Your measured system jitter differs |
| `--z-trim` | `600` | The direct component occupies more or fewer native bins |
| `--z-offset` | `0` | The object sits deep and the crop cuts it off |
| `--diffuse` | off | The target is diffuse, not retroreflective |
| `--data-key` | auto | The file holds more than one 3D array |

If the run prints `WARNING: no signal measured above the background estimate`,
raise `--z-trim` first: the direct wall bounce is probably still in the cube and
is being taken for signal.

### The four experiments

```bash
python visualize_progressive.py --scene mannequin      # SSIM after each step
python run_experiment.py        --scene mannequin      # vs 3 gating baselines
python compare_clean_vs_noisy.py                       # what the algorithm costs
python compare_methods.py                              # method x data matrix
```

All four default to **PPP 60, SBR 4.0** and write into `results/`.

**`visualize_progressive.py`** reconstructs after every step and scores it. On `mannequin` the cumulative chain runs 0.588 (degraded) -> 0.607 (windowed) -> **0.736** (TV completed).

**`run_experiment.py`** compares AW-NLOS against no window, a fixed 2x-jitter window and a global window, at three ambient levels. AW-NLOS wins at every level and stays nearly flat while the baselines fall:

| SBR | No window | Fixed 2x | Global | **AW-NLOS** |
|---|---|---|---|---|
| 4.04 | 0.588 | 0.624 | 0.584 | **0.736** |
| 3.32 | 0.572 | 0.624 | 0.570 | **0.734** |
| 2.94 | 0.528 | 0.616 | 0.538 | **0.733** |

**`compare_clean_vs_noisy.py`** adds the control the sweep never shows: the algorithm run on *undegraded* data. Windowing and TV are lossy operations, so this measures what they cost when there is no noise to remove — 0.05 to 0.34 SSIM depending on the scene. That cost is also the ceiling for the noisy case.

**`compare_methods.py`** lays the same four reconstructions out as a method x data matrix and reports both SSIM and PSNR:

|  | Raw capture | Noisy capture |
|---|---|---|
| Traditional LCT | reference | A |
| AW-NLOS + LCT | B | C |

From it, the robustness measure — how far each method falls when the same noise is added:

| scene | traditional drop | AW-NLOS drop |
|---|---|---|
| s_u | 0.225 | **0.012** |
| mannequin | 0.412 | **0.016** |
| diffuse_s | 0.497 | **0.018** |
| dot_chart_40cm | 0.465 | **0.122** |
| dot_chart_65cm | 0.572 | **0.139** |
| outdoor_s | 0.467 | **0.143** |
| resolution_chart_40cm | 0.614 | **0.169** |
| exit_sign | 0.641 | **0.214** |
| resolution_chart_65cm | 0.731 | **0.240** |

**AW-NLOS loses less on all nine scenes, by 2.7x to 28x.** This is a fairer statement than the absolute SSIM gap, because AW-NLOS starts from B < 1 rather than from the reference itself.

### What these results do not show

- **The noise is milder than the paper's regime.** These runs sit near 60 signal PPP; the paper targets 0.02–2 PPP. This is an easier problem.
- **One scene regresses.** On `s_u` the clean SBR is 9.9, so windowing and TV cost more than the noise they remove: 0.717 against 0.775 for no windowing at all. The method is a low-SBR tool, not a general improvement.
- **Achieved SBR overshoots the target by about 25%.** The ambient rate is solved from the pre-ambient peak, but adding background raises the measured peak too.
- **The ground truth is itself a reconstruction,** not a CAD model — its own SBR is only 4 to 10.
- **Window duty differs between conditions.** The window sizes itself from the measured noise floor, so the clean and noisy runs do not share an identical window.

### Scene choice matters for the ground truth

`diffuse_s` reconstructs as a blurry blob at any Wiener parameter — that is intrinsic to the LCT on a diffuse target, not a tuning problem (see `results/gt_quality.png`). For a legible ground truth use `mannequin`, `resolution_chart_40cm`, `dot_chart_40cm` or `exit_sign`.

---

## The idea in one paragraph

Signal photons and background photons obey **different probability laws in time**. A signal photon has undergone three bounces, so its arrival time clusters around the true time-of-flight `t_c`, blurred by the instrument response `h(t)` (Eq. 2). Ambient light and dark counts are a homogeneous Poisson process, uniform over the whole repetition period (Eq. 3). So: find the cluster, keep only that slice of the histogram, discard the rest. The hard part — and what these three steps do — is **finding the cluster when a single pixel holds almost no photons**.

---

## What each step does

### Step (a) — Acquire the transient cube

Loads a confocal capture into a `(N, N, M)` array: two wall-scan axes, one TCSPC time axis, values in photon counts.

Two things happen on load:
- **Temporal downsampling** — native 4 ps bins are pairwise-summed to 16 ps. Summing preserves total counts, so Poisson statistics survive.
- **Direct-component removal** — the first bounce off the relay wall holds ~90% of all counts and would swamp everything, so the first `z_trim` bins are zeroed. (The paper's own rig rejects it optically with a 2 cm Tx/Rx offset instead.)

### Step (b) — Block aggregation (Eq. 4)

```
s_{a,b}(t) = Σ_{i=4a-3}^{4a} Σ_{j=4b-3}^{4b} Φ_{i,j}(t)
```

Non-overlapping 4×4 tiles, so a 64×64 grid becomes 16×16 blocks.

**Why it works.** Neighbouring wall points are centimetres apart while the object is metres away, so they share almost the same delay `t_c`: signal adds **coherently in time**. Uniform background does not. Summing 16 pixels scales the peak by 16 but the background's standard deviation only by √16 = 4, lifting the cluster out of the Poisson floor.

> **The blocks only size the window.** The window is later applied to the *original full-resolution* per-pixel data. No spatial resolution is lost. Reconstructing from block-summed data instead would throw away 4× resolution and miss the point of the method.

### Step (c) — Matched filtering and window width (Eqs. 5–7)

```
g_{a,b}(t) = s_{a,b}(t) ⊗ h(t)                              (Eq. 5)

T_{a,b} = 2 · FWHM · √( 2 · ln( max(g) / (η·N_n) ) )        (Eq. 6)

T_{i,j} = T_{a,b}                                            (Eq. 7)
```

Matched filtering is the SNR-optimal linear detector for a known waveform in white noise. The signal cluster *is* the IRF shifted to `t_c`, and the background is flat, so correlating with `h` maximizes output SNR exactly at the cluster.

Eq. 6 then models the filtered peak as a Gaussian and asks where it falls to the noise floor. **This is the adaptive part:** more ambient light raises `η·N_n`, shrinks the logarithm, and *narrows* the window — stricter in noisier conditions. A stronger peak widens it.

Widths are clamped to `[T_min, T_max]`, with `T_min = 2×FWHM` (the classical fixed window of prior work) and `T_max = 4050 ps` (the paper's "global window").

---

## Install

Python 3.9+.

```bash
pip install -r requirements.txt     # numpy, scipy, matplotlib, scikit-image
```

No MATLAB required. No GPU required. A full run takes a few seconds.

---

## The data

**The captures are included in `data/` — you can clone and run immediately.**

They come from **O'Toole, Lindell & Wetzstein, "Confocal non-line-of-sight imaging based on the light-cone transform", *Nature* 555, 338–341 (2018)**, released by the Stanford Computational Imaging Lab for research use. Nine scenes, ~31 MB total:

```
data/
├── data_mannequin.mat              ← default; cleanest ground truth
├── data_s_u.mat                    two objects, 2.6 ns apart
├── data_diffuse_s.mat              diffuse target
├── data_outdoor_s.mat              32×32, crisp "S"
├── data_exit_sign.mat
├── data_resolution_chart_{40,65}cm.mat
└── data_dot_chart_{40,65}cm.mat
```

Each file holds `rect_data` of shape `(N, N, M)` in `uint16` photon counts, plus `width`, the half-extent of the scanned wall patch in metres. They are **MATLAB v7**, so `scipy.io.loadmat` reads them directly — no `h5py` needed.

Scenes are registered in `aw_nlos/io_utils.py:SCENES`. For any other capture, use `--mat` and `--width` directly.

---

## Usage

```bash
python run_steps.py --scene mannequin --ppp 30 --sbr 3.5
```

Arbitrary file:

```bash
python run_steps.py --mat path/to/my_capture.mat --width 0.35
```

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--scene` | `mannequin` | Scene name from the table in `io_utils.py` |
| `--mat` / `--width` | — | Use any `.mat` cube instead; `width` is the wall half-extent in m |
| `--ppp` | `30` | Target signal photons per pixel after thinning |
| `--sbr` | `3.5` | Target signal-to-background ratio |
| `--block` | `4` | Block size for Eq. 4 |
| `--fwhm-ps` | `300` | IRF FWHM in ps — **set this to match your system** |
| `--z-trim` | `150` | Time bins to ignore (post-downsampling) |
| `--seed` | `1` | RNG seed for reproducible degradation |

### Visualization

`run_steps.py` writes one figure per step. `visualize.py` adds two combined views:

```bash
python visualize.py --scene mannequin --ppp 30 --sbr 3.5
```

- **`overview_steps_abc.png`** — all three steps on one sheet, left to right: degraded transient and a single raw pixel → the block histogram and per-block counts → the matched filter and the window-width map.
- **`width_vs_sbr.png`** — an SBR sweep demonstrating the adaptivity of Eq. 6. Median width falls as ambient light rises, and the count of blocks clamped at `T_min` climbs:

  ```
  SBR 4.24   median 1020 ps    2/256 clamped
  SBR 3.92   median  979 ps    4/256 clamped
  SBR 3.67   median  953 ps    8/256 clamped
  SBR 3.27   median  921 ps    9/256 clamped
  SBR 2.99   median  878 ps   12/256 clamped
  ```

Add `--no-sweep` to skip the sweep. All plotting lives in `aw_nlos/plotting.py`, so you can reuse the pieces (`plot_step_a/b/c`, `plot_overview`, `show_xt`, `show_map`) in a notebook.

### Outputs

```
results/
├── 01_step_a_transient.png       clean vs degraded, x–t maps and histograms
├── 02_step_b_blocks.png          single pixel vs block, per-block counts
├── 03_step_c_window_widths.png   IRF, matched filter, width map, distribution
├── overview_steps_abc.png        all three steps on one sheet
├── width_vs_sbr.png              Eq. 6 adaptivity sweep
└── window_widths.npz             ← the handoff to step (d)
```

`window_widths.npz` contains `block_widths_s`, `pixel_widths_s`, `pixel_width_bins`, `block_peak`, `block_floor`, `irf`, `bin_resolution`, `block`, `fwhm_s`, `z_trim`, and the degraded `transient` itself.

### Using it as a library

```python
from aw_nlos.io_utils import load_scene
from aw_nlos.window import compute_windows

t = load_scene("mannequin")
r = compute_windows(t, fwhm_s=300e-12, block=4, z_trim=150)

r["pixel_widths_s"]     # (N, N)  window width per scan point, seconds
r["pixel_width_bins"]   # (N, N)  same, in integer time bins
r["block_widths_ps"]    # (N/4, N/4) per block, picoseconds
```

---

## Why degradation is included

The public captures are high-SNR lab data — `data_mannequin` measures **SBR 4.4 at ~620 signal PPP**. AW-NLOS targets far weaker returns. On the raw captures the cluster is already obvious in every single-pixel histogram, so steps (b) and (c) look like no-ops and you cannot tell whether your implementation is correct.

`aw_nlos/degrade.py` therefore reproduces the paper's regime with two physically exact operations:

- **Binomial thinning** — how a Poisson process responds to lower laser power or lower detection efficiency. Sets signal PPP.
- **Additive uniform Poisson background** — matching the uniform arrival model of Eq. 3. Sets SBR.

> **Degrade only until the effect is visible, not further.** The useful window is narrow: too little degradation and the steps do nothing; too much and the cluster vanishes entirely, every block clamps to `T_min`, and the widths are noise. For `mannequin`, `--ppp 30 --sbr 3.5` is a good operating point. Watch the `clamped at T_min` count in the output — if most blocks are clamped, you have over-degraded.

A scene's own clean SBR caps what `--sbr` can do; asking for an SBR above it adds no background at all.

---

## Implementation notes

Three things that are easy to get wrong:

**1. The noise floor must be measured through the matched filter.** Eq. 6 compares `max(g)` against `η·N_n`, and `g = s ⊗ h`. Block summing multiplies the background by `block²` and the filter reshapes it. Estimating the floor from the *raw per-pixel* rate inflates every window by ~16×. This code takes the median of `g` itself over searched bins — robust, and automatically correct.

**2. The IRF cannot be recovered from these captures.** They are pre-rectified so the direct-bounce peak starts at bin 0 with its rising edge cut off; what follows is wall-scatter decay, not the instrument response. `aw_nlos/irf.py` synthesizes a Gaussian instead, with FWHM as an explicit parameter. Set `--fwhm-ps` to your own measured system jitter (laser pulse ⊗ SPAD jitter ⊗ TCSPC jitter ⊗ cable delay). The paper's rig measures 300 ps.

**3. Guard the square root.** When `max(g) ≤ η·N_n` the logarithm goes negative and Eq. 6 returns NaN. That case means "no cluster stands above background", so it falls back to `T_min`.

Minor: the paper prints Eq. 7 as `b = floor((i−1)/4)+1`. That is a typo for `(j−1)`.

---

## Continuing to steps (d)–(f)

With `pixel_width_bins` in hand:

**(d) Apply the window (Eq. 8).** For each pixel, slide a window of its own width over the **raw** histogram and take the position of maximum enclosed counts, then zero everything outside.

```
t^m_{i,j} = argmax_t ∫_t^{t+T} Φ_{i,j}(t')dt'
```

Width comes from the block, but the **start time must be found per pixel** — `t_c` changes point to point with scan geometry. Use cumulative sums: `S(t) = C(t+W) − C(t)`, one pass, O(M).

> Window the **raw counts**, never the filtered signal `g`. The matched filter is only a detection statistic; windowing `g` corrupts the photometry that feeds reconstruction.

**(e) TV completion (Eqs. 9–10).** Hard zeroing leaves block-edge discontinuities that a deconvolution will ring on. Repair with isotropic TV over the **spatial axes only** — never along `t`, since the time axis carries the depth code.

**(f) Reconstruction (Eqs. 11–13).** Standard Light Cone Transform: warp `t → v = (ct/2)²` and `z → u = z²` to turn the shift-variant hypercone integral into a 3D convolution, then invert with a Wiener filter. Reference MATLAB is released with the O'Toole paper.

---

## Known limitation

Step (d) takes an **argmax**, so only the single most prominent cluster survives. Two objects separated by more than roughly 50 cm in depth will lose the weaker one — the paper's own Discussion (iii). `data_mannequin` is safe: its two temporal clusters are 1.24 ns apart (≈19 cm), so one window covers the whole body and it behaves as a single extended object. `data_s_u`, with its letters 2.6 ns apart, does not.

The paper suggests an iterative fix: reconstruct the dominant object, subtract its predicted counts, then re-run matched filtering on the residual.

---

## Layout

```
.
├── README.md
├── requirements.txt
│
├── run_steps.py                    (a)->(c) driver, one figure per step + npz
├── visualize.py                    overview sheet + Eq. 6 adaptivity sweep
├── aw_nlos/                        steps (a)-(c) package
│   ├── io_utils.py                 (a) loading, downsampling, trim
│   ├── degrade.py                  thinning + ambient injection
│   ├── irf.py                      synthesized Gaussian IRF
│   ├── window.py                   (b) Eq. 4, (c) Eqs. 5-7     <- the core
│   ├── plotting.py                 reusable figure building blocks
│   └── metrics.py                  SBR (Eq. 14), PPP (Eq. 16)
│
├── aw/                             full pipeline (a)-(f)
│   ├── io_utils.py  degrade.py  irf.py  metrics.py
│   ├── window.py                   (b) (c) (d)
│   ├── tv.py                       (e) spatial TV completion
│   └── lct.py                      (f) LCT + Wiener
├── visualize_progressive.py        SSIM after each step
├── visualize_steps.py              per-step diagnostic sheets
├── run_experiment.py               AW-NLOS vs 3 gating baselines
├── compare_clean_vs_noisy.py       what the algorithm costs on clean data
├── compare_methods.py              method x data matrix, SSIM + PSNR
├── run_single.py                   one .mat file  <- your own data
├── check_phase0.py  check_phase12.py  probe_regime.py  probe_scenes.py
│
├── data/                           .mat captures (included)
└── results/
    ├── 01_..03_*.png  overview_steps_abc.png  width_vs_sbr.png    (a)-(c)
    ├── progressive/<scene>/        step-by-step reconstructions
    ├── steps/<scene>/              per-step diagnostic sheets
    ├── clean_vs_noisy/             algorithm on clean vs degraded data
    ├── method_matrix/              traditional vs AW-NLOS, raw vs noisy
    ├── single/                     run_single.py output
    └── aw_nlos_sweep_<scene>.png   ambient sweeps
```

## License

Code released for research use. The AW-NLOS method is due to Miao et al. (2025); the datasets and the LCT reconstruction referenced above are due to O'Toole, Lindell & Wetzstein (2018). Please cite both.
