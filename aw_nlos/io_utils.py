"""Step (a): load a confocal NLOS transient cube and prepare it for windowing."""

from pathlib import Path

import numpy as np
import scipy.io as sio

NATIVE_BIN_RESOLUTION = 4e-12   # SPAD native bin width, seconds
C = 3e8                         # speed of light, m/s

# Scene table for the O'Toole et al. confocal NLOS release. `diffuse` and `snr` are
# only needed once you add reconstruction (step f); they are carried for completeness.
SCENES = {
    "mannequin":             dict(file="data_mannequin.mat",             diffuse=False, snr=0.8),
    "s_u":                   dict(file="data_s_u.mat",                   diffuse=False, snr=0.8),
    "exit_sign":             dict(file="data_exit_sign.mat",             diffuse=False, snr=0.8),
    "diffuse_s":             dict(file="data_diffuse_s.mat",             diffuse=True,  snr=0.08),
    "outdoor_s":             dict(file="data_outdoor_s.mat",             diffuse=False, snr=0.8),
    "resolution_chart_40cm": dict(file="data_resolution_chart_40cm.mat", diffuse=False, snr=0.8),
    "resolution_chart_65cm": dict(file="data_resolution_chart_65cm.mat", diffuse=False, snr=0.8),
    "dot_chart_40cm":        dict(file="data_dot_chart_40cm.mat",        diffuse=False, snr=0.8),
    "dot_chart_65cm":        dict(file="data_dot_chart_65cm.mat",        diffuse=False, snr=0.8),
}


class Transient:
    """A confocal transient cube of shape (N, N, M).

    N x N are the wall scan points, M the TCSPC time bins, values are photon counts.
    """

    def __init__(self, data, width, bin_resolution, diffuse=False, snr=0.8, scene=""):
        self.data = data
        self.width = float(width)
        self.bin_resolution = float(bin_resolution)
        self.diffuse = bool(diffuse)
        self.snr = float(snr)
        self.scene = scene

    @property
    def N(self):
        return self.data.shape[0]

    @property
    def M(self):
        return self.data.shape[2]

    @property
    def range_m(self):
        return self.M * C * self.bin_resolution

    @property
    def time_axis_ns(self):
        return np.arange(self.M) * self.bin_resolution * 1e9

    def replace(self, data):
        return Transient(data, self.width, self.bin_resolution, self.diffuse,
                         self.snr, self.scene)

    def __repr__(self):
        return (f"Transient(scene={self.scene!r}, {self.N}x{self.N}x{self.M}, "
                f"bin={self.bin_resolution * 1e12:.0f}ps, width={self.width})")


def load_scene(scene, data_dir="data", downsample=2, z_trim=600, dtype=np.float32):
    """Load a scene by name, downsample in time, and zero the direct component.

    downsample=2 takes native 4 ps bins to 16 ps by pairwise summing, which preserves
    total photon counts and therefore Poisson statistics.

    z_trim is given in NATIVE bins and is rescaled alongside the data. The first bounce
    off the relay wall carries ~90% of all counts and would otherwise swamp everything;
    the paper's own system rejects it optically via a 2 cm Tx/Rx offset instead.
    """
    if scene not in SCENES:
        raise KeyError(f"unknown scene {scene!r}; known: {sorted(SCENES)}")
    meta = SCENES[scene]

    path = Path(data_dir) / meta["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Place the .mat captures in '{data_dir}/' "
            f"(see README, 'Getting the data').")

    mat = sio.loadmat(path)
    data = np.ascontiguousarray(mat["rect_data"]).astype(dtype)
    width = float(mat["width"].squeeze())

    bin_resolution = NATIVE_BIN_RESOLUTION
    for _ in range(downsample):
        data = data[:, :, 0::2] + data[:, :, 1::2]
        bin_resolution *= 2
        z_trim = int(round(z_trim / 2))

    if z_trim > 0:
        data[:, :, :z_trim] = 0

    return Transient(data, width, bin_resolution, meta["diffuse"], meta["snr"], scene)


def load_mat(path, width, bin_resolution=NATIVE_BIN_RESOLUTION, key="rect_data",
             downsample=0, z_trim=0, dtype=np.float32):
    """Load an arbitrary .mat transient cube, for data outside the scene table.

    The array under `key` must be (N, N, M). `width` is the half-extent of the scanned
    wall patch in metres.
    """
    mat = sio.loadmat(path)
    if key not in mat:
        raise KeyError(f"{path} has no variable {key!r}; found "
                       f"{[k for k in mat if not k.startswith('__')]}")
    data = np.ascontiguousarray(mat[key]).astype(dtype)
    if data.ndim != 3 or data.shape[0] != data.shape[1]:
        raise ValueError(f"expected a square (N, N, M) cube, got {data.shape}")

    for _ in range(downsample):
        data = data[:, :, 0::2] + data[:, :, 1::2]
        bin_resolution *= 2
        z_trim = int(round(z_trim / 2))
    if z_trim > 0:
        data[:, :, :z_trim] = 0

    return Transient(data, width, bin_resolution, scene=Path(path).stem)
