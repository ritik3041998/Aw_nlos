"""Light Cone Transform reconstruction (Eq. 11-13).

Python port of cnlos_reconstruction.m from O'Toole, Lindell and Wetzstein,
"Confocal Non-Line-of-Sight Imaging Based on the Light Cone Transform".

The change of variables z = sqrt(u), v = (t_c/2)^2 turns the shift-variant
hypercone integral of Eq. 11 into the shift-invariant 3D convolution
T_t U = g * T_z rho, which then inverts as a Wiener filter.
"""

import numpy as np
import scipy.sparse as sp
from scipy.fft import fftn, ifftn

C = 3e8


def define_psf(U, V, slope):
    """The light-cone blur kernel g, as a paraboloid surface in (x, y, v)."""
    x = np.linspace(-1, 1, 2 * U)
    y = np.linspace(-1, 1, 2 * U)
    z = np.linspace(0, 2, 2 * V)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing="ij")

    psf = np.abs(((4.0 * slope) ** 2) * (grid_x ** 2 + grid_y ** 2) - grid_z)
    # Keep, for each (y, x), the single v bin closest to the paraboloid.
    psf = (psf == psf.min(axis=0, keepdims=True)).astype(np.float32)
    psf /= psf[:, U - 1, U - 1].sum()
    psf /= np.linalg.norm(psf.ravel())
    # Move the kernel origin to index 0 so the FFT convolution is centred.
    psf = np.roll(psf, (U, U), axis=(1, 2))
    return psf


def resampling_operator(M):
    """Sparse operators for the t -> v and z -> u axis warps (T_t and T_z)."""
    n = M * M
    x = np.arange(1, n + 1, dtype=np.float64)
    rows = np.arange(n)
    cols = np.ceil(np.sqrt(x)).astype(np.int64) - 1
    mtx = sp.csr_matrix((1.0 / np.sqrt(x), (rows, cols)), shape=(n, M))

    # Decimate M^2 rows back down to M by repeated pairwise averaging.
    for _ in range(int(round(np.log2(M)))):
        mtx = 0.5 * (mtx[0::2, :] + mtx[1::2, :])

    mtx = mtx.tocsr().astype(np.float32)
    return mtx, mtx.T.tocsr()


def reconstruct(transient, snr=None, backprojection=False):
    """Invert the light cone transform. Returns the albedo volume as (M, N, N)."""
    if snr is None:
        snr = getattr(transient, "snr", 0.8)
    data = transient.data
    N, M = transient.N, transient.M
    slope = transient.width / transient.range_m

    psf = define_psf(N, M, slope)
    fpsf = fftn(psf)
    if backprojection:
        invpsf = np.conj(fpsf)
    else:
        invpsf = np.conj(fpsf) / (np.abs(fpsf) ** 2 + 1.0 / snr)
    del psf, fpsf

    mtx, mtxi = resampling_operator(M)

    # (N, N, M) -> (M, N, N): time first, matching the MATLAB permute([3 2 1]).
    vol = np.transpose(data, (2, 1, 0)).astype(np.float32)

    # Step 1: undo radiometric falloff. 1/r^4 for diffuse, 1/r^2 retroreflective.
    grid_z = np.linspace(0, 1, M, dtype=np.float32)[:, None, None]
    vol = vol * (grid_z ** 4 if transient.diffuse else grid_z ** 2)

    # Step 2: warp the time axis onto v and zero-pad to avoid circular wraparound.
    tdata = np.zeros((2 * M, 2 * N, 2 * N), dtype=np.float32)
    tdata[:M, :N, :N] = (mtx @ vol.reshape(M, -1)).reshape(M, N, N)

    # Step 3: Wiener deconvolution against the light-cone kernel.
    tvol = ifftn(fftn(tdata) * invpsf)
    tvol = tvol[:M, :N, :N]
    del tdata, invpsf

    # Step 4: warp back to metric depth and clamp.
    vol = (mtxi @ np.real(tvol).reshape(M, -1)).reshape(M, N, N)
    return np.maximum(vol, 0.0)


def crop_for_display(vol, transient):
    """Crop the depth range the object actually occupies, as the reference does."""
    M = vol.shape[0]
    ind = int(round(M * 2 * transient.width / (transient.range_m / 2)))
    z0 = transient.z_offset
    z1 = min(ind + z0, M)
    vol = vol[:, :, ::-1]
    return vol[z0:z1]


def projections(vol):
    """Maximum-intensity projections: (front, top, side)."""
    return (vol.max(axis=0), vol.max(axis=1), vol.max(axis=2).T)
