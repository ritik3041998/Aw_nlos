"""Loading and preprocessing of confocal NLOS transient measurements."""

from pathlib import Path

import numpy as np
import scipy.io as sio

_ROOT = Path(__file__).resolve().parent.parent

# The captures live in data/mat_files/ in the development tree, but this repository
# ships them flat in data/. Prefer the nested layout when it is present.
DATA_DIR = _ROOT / "data" / "mat_files"
if not DATA_DIR.is_dir():
    DATA_DIR = _ROOT / "data"

NATIVE_BIN_RESOLUTION = 4e-12
C = 3e8

# z_offset controls where the reconstructed volume is cropped for display; taken
# from the scene table in cnlos_reconstruction.m.
# snr is the Wiener regularization parameter (alpha in Eq. 13); diffuse scenes need a
# stronger prior, matching the reference reconstruction.
SCENES = {
    "s_u": dict(file="data_s_u.mat", z_offset=800, diffuse=False, snr=0.8),
    "mannequin": dict(file="data_mannequin.mat", z_offset=300, diffuse=False, snr=0.8),
    "exit_sign": dict(file="data_exit_sign.mat", z_offset=600, diffuse=False, snr=0.8),
    "diffuse_s": dict(file="data_diffuse_s.mat", z_offset=100, diffuse=True, snr=0.08),
    "outdoor_s": dict(file="data_outdoor_s.mat", z_offset=700, diffuse=False, snr=0.8),
    "resolution_chart_40cm": dict(file="data_resolution_chart_40cm.mat", z_offset=350, diffuse=False, snr=0.8),
    "resolution_chart_65cm": dict(file="data_resolution_chart_65cm.mat", z_offset=700, diffuse=False, snr=0.8),
    "dot_chart_40cm": dict(file="data_dot_chart_40cm.mat", z_offset=350, diffuse=False, snr=0.8),
    "dot_chart_65cm": dict(file="data_dot_chart_65cm.mat", z_offset=700, diffuse=False, snr=0.8),
}


class Transient:
    """A confocal transient cube with the metadata the pipeline needs.

    data has shape (N, N, M): two wall-scan axes and one time axis, in photon counts.
    """

    def __init__(self, data, width, bin_resolution, z_offset, diffuse, scene="", snr=0.8):
        self.data = data
        self.width = float(width)
        self.bin_resolution = float(bin_resolution)
        self.z_offset = int(z_offset)
        self.diffuse = bool(diffuse)
        self.scene = scene
        self.snr = float(snr)

    @property
    def N(self):
        return self.data.shape[0]

    @property
    def M(self):
        return self.data.shape[2]

    @property
    def range_m(self):
        # Invariant under the pairwise-sum downsampling below, since M halves as
        # bin_resolution doubles.
        return self.M * C * self.bin_resolution

    def replace(self, data):
        return Transient(data, self.width, self.bin_resolution, self.z_offset,
                         self.diffuse, self.scene, self.snr)

    def __repr__(self):
        return (f"Transient(scene={self.scene!r}, {self.N}x{self.N}x{self.M}, "
                f"bin={self.bin_resolution * 1e12:.0f}ps, width={self.width}, "
                f"diffuse={self.diffuse})")


def load_scene(scene, downsample=2, z_trim=600, dtype=np.float32):
    """Load a scene, downsample in time, and zero the direct (first-bounce) component.

    downsample=2 takes the native 4 ps bins to 16 ps, matching the reference
    reconstruction. z_trim is given in native bins and is scaled along with the data.
    """
    if scene not in SCENES:
        raise KeyError(f"unknown scene {scene!r}; known: {sorted(SCENES)}")
    meta = SCENES[scene]

    mat = sio.loadmat(DATA_DIR / meta["file"])
    data = np.ascontiguousarray(mat["rect_data"]).astype(dtype)
    width = float(mat["width"].squeeze())

    bin_resolution = NATIVE_BIN_RESOLUTION
    if scene == "bunny":
        bin_resolution = 8e-12

    z_offset = meta["z_offset"]
    for _ in range(downsample):
        # Pairwise sum preserves total photon counts, so Poisson statistics survive.
        data = data[:, :, 0::2] + data[:, :, 1::2]
        bin_resolution *= 2
        z_trim = int(round(z_trim / 2))
        z_offset = int(round(z_offset / 2))

    if z_trim > 0:
        data[:, :, :z_trim] = 0

    return Transient(data, width, bin_resolution, z_offset, meta["diffuse"], scene,
                     meta["snr"])


def load_mat(path, width=None, bin_ps=4.0, z_offset=0, diffuse=False, snr=0.8,
             data_key=None, width_key="width", downsample=2, z_trim=600,
             dtype=np.float32):
    """Load an arbitrary confocal .mat capture, not one of the registered scenes.

    The cube must be (N, N, M): two wall-scan axes and one time axis, in photon
    counts, pre-rectified so the direct component starts at the first time bin.

    data_key picks the variable holding the cube; when omitted, the only 3D array
    in the file is used. width is the half-extent of the scanned wall patch in
    metres, read from the file when it carries one. bin_ps is the NATIVE bin
    resolution before downsampling.
    """
    mat = sio.loadmat(path)

    if data_key is None:
        cubes = [k for k, v in mat.items()
                 if not k.startswith("__") and getattr(v, "ndim", 0) == 3]
        if len(cubes) != 1:
            raise KeyError(
                f"pass data_key: expected exactly one 3D array in {path}, found "
                f"{cubes or 'none'}")
        data_key = cubes[0]
    if data_key not in mat:
        raise KeyError(f"{data_key!r} not in {path}; keys: "
                       f"{[k for k in mat if not k.startswith('__')]}")

    data = np.ascontiguousarray(mat[data_key]).astype(dtype)
    if data.ndim != 3:
        raise ValueError(f"{data_key!r} has shape {data.shape}, expected 3 axes")
    if data.shape[0] != data.shape[1]:
        raise ValueError(f"expected a square scan grid, got {data.shape[:2]}")

    if width is None:
        if width_key not in mat:
            raise KeyError(f"no {width_key!r} in {path}: pass width explicitly "
                           "(the wall patch half-extent, in metres)")
        width = float(np.asarray(mat[width_key]).squeeze())

    bin_resolution = float(bin_ps) * 1e-12
    for _ in range(downsample):
        # Pairwise sum preserves total photon counts, so Poisson statistics survive.
        if data.shape[2] % 2:
            data = data[:, :, :-1]
        data = data[:, :, 0::2] + data[:, :, 1::2]
        bin_resolution *= 2
        z_trim = int(round(z_trim / 2))
        z_offset = int(round(z_offset / 2))

    if z_trim > 0:
        data[:, :, :z_trim] = 0

    name = Path(path).stem
    return Transient(data, width, bin_resolution, z_offset, diffuse, name, snr)


def signal_support(cube, z_trim_bins=0):
    """Index of the first and last time bin carrying appreciable signal.

    Used to pick a background-only region for noise estimation and to report where
    the object echo actually lives.
    """
    profile = cube.sum(axis=(0, 1))
    profile[:z_trim_bins] = 0
    if profile.max() <= 0:
        return 0, cube.shape[2] - 1
    above = np.where(profile > 0.05 * profile.max())[0]
    return int(above[0]), int(above[-1])
