"""Step (e): TV-regularized transient completion (Eq. 9-10).

Hard windowing zeroes everything outside the window, which leaves block-edge
discontinuities and holes in the (i, j) plane. The LCT that follows is a
deconvolution and rings on such steps, so the windowed transient is repaired with
isotropic TV taken over the spatial axes only.

Gradients are never taken along t: the time axis carries the depth code, and
smoothing it would blur exactly the quantity being measured.
"""

import numpy as np


def _grad(u):
    """Forward differences along the two spatial axes, Neumann boundary."""
    gi = np.zeros_like(u)
    gj = np.zeros_like(u)
    gi[:-1] = u[1:] - u[:-1]
    gj[:, :-1] = u[:, 1:] - u[:, :-1]
    return gi, gj


def _div(pi, pj):
    """Adjoint (negative divergence) of _grad."""
    d = np.zeros_like(pi)
    d[0] += pi[0]
    d[1:-1] += pi[1:-1] - pi[:-2]
    d[-1] += -pi[-2]

    d[:, 0] += pj[:, 0]
    d[:, 1:-1] += pj[:, 1:-1] - pj[:, :-2]
    d[:, -1] += -pj[:, -2]
    return d


def tv_complete(cube, mu, n_iter=100, tau=0.25):
    """Solve min_U ||U - cube||^2 + mu * ||U||_TV over the spatial dimensions.

    Chambolle's dual projection, applied to every time slice at once.
    """
    if mu <= 0:
        return cube.copy()

    f = cube.astype(np.float32)
    lam = np.float32(mu / 2.0)  # ROF convention is (1/2)||u-f||^2 + lam*TV

    pi = np.zeros_like(f)
    pj = np.zeros_like(f)

    for _ in range(n_iter):
        gi, gj = _grad(_div(pi, pj) - f / lam)
        norm = np.sqrt(gi * gi + gj * gj)
        denom = 1.0 + tau * norm
        pi = (pi + tau * gi) / denom
        pj = (pj + tau * gj) / denom

    out = f - lam * _div(pi, pj)
    return np.maximum(out, 0.0).astype(cube.dtype)


def complete(transient, mu_scale=0.15, n_iter=60, z_trim=0):
    """Apply TV completion with mu set relative to the data's own peak amplitude."""
    cube = transient.data
    peak = float(cube[:, :, z_trim:].max())
    if peak <= 0:
        return transient.replace(cube.copy())
    out = tv_complete(cube, mu=mu_scale * peak, n_iter=n_iter)
    if z_trim:
        out[:, :, :z_trim] = 0
    return transient.replace(out)
