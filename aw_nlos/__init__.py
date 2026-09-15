"""AW-NLOS steps (a)-(c): transient loading, block aggregation, window sizing.

Implements the first three stages of Miao et al., "Adaptive windowing for
photon-efficient non-line-of-sight imaging under high ambient light",
Optics Express 33(21):44522, 2025.
"""

from . import degrade, io_utils, irf, metrics, window

__all__ = ["degrade", "io_utils", "irf", "metrics", "window"]
__version__ = "0.1.0"
