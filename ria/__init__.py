"""
RIA — Radio Interferometric Analysis
=====================================

Package for Bayesian reconstruction of air-shower radio
footprints.  Provides the full forward model (LDF + wavefront timing),
atmospheric models, coordinate transforms, and a high-level inference
wrapper around NIFTy's ``optimize_kl``.

Quick start
-----------
>>> from ria import FootprintModel, Atmosphere, reconstruct, config
>>> config.N_VI_ITERATIONS = 6   # adjust before calling reconstruct()
>>> results = reconstruct(positions, fluences, times, noise,
...                       mean_zenith=zen, mean_azimuth=az)
"""

from .forward_model import FootprintModel
from .optimize import reconstruct
from .atmosphere import Atmosphere
from .coordinates import CoordinateTransform
from . import config
from . import ldf
from . import timing
from . import plotting

__all__ = [
    "FootprintModel",
    "reconstruct",
    "Atmosphere",
    "CoordinateTransform",
    "config",
    "ldf",
    "timing",
    "plotting",
]
