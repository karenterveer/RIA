"""
Configuration parameters for the RIA reconstruction package.

All tunable constants are centralised here so they can be inspected and
overridden before a reconstruction run.  Values are grouped by category
and each constant is documented inline.

Usage
-----
>>> from ria import config
>>> config.N_VI_ITERATIONS = 12          # override before calling reconstruct()
>>> config.TIMING_CF = False             # disable timing correlated field
"""

import numpy as np

# ===========================================================================
#  Physical constants
# ===========================================================================

C_LIGHT = 299792458.0
"""Speed of light in vacuum [m/s]."""

N_SEA_LEVEL = 2.8e-4
"""Excess refractive index at sea level (n - 1)."""

SCALE_HEIGHT = 8400.0
"""Atmospheric scale height for the refractive-index profile [m]."""


# ===========================================================================
#  Default magnetic field vectors  (Gauss, geographic CS: East, North, Up)
# ===========================================================================

MAGNETIC_FIELD_VECTORS = {
    "auger":      np.array([0.00871198,  0.19693423,  0.1413841]),
    "mooresbay":  np.array([0.058457,   -0.09042,     0.61439]),
    "summit":     np.array([-0.037467,   0.075575,   -0.539887]),
    "southpole":  np.array([-0.14390398, 0.08590658,  0.52081228]),
    "lofar":      np.array([0.004675,    0.186270,   -0.456412]),
}
"""Geomagnetic field vectors per observatory site [Gauss]."""

DEFAULT_MAGNETIC_FIELD_VECTOR = MAGNETIC_FIELD_VECTORS["lofar"]
"""Default magnetic field vector used when none is specified."""


# ===========================================================================
#  Grid generation (correlated-field grids)
#  finer grids cost runtime, but give better spatial resolution
# ===========================================================================

GRID_PAD = 150.0
"""Padding added to each side of the antenna footprint [m]."""

TARGET_RESOLUTION = 5.0
"""Target spatial resolution of the correlated-field grid [m/pixel]."""

MAX_GRID_DIM = 400
"""Maximum allowed grid dimension (pixels per side)."""

MIN_GRID_DIM = 64
"""Minimum allowed grid dimension (pixels per side)."""


# ===========================================================================
#  Correlated-field (CF) hyper-parameters - fluence systematics
# ===========================================================================

SYST_CF_ZM = dict(offset_mean=0.01, offset_std=(1e-3, 1e-4))
"""Zero-mode prior for the fluence systematic correlated field."""

SYST_TARGET_SIGMA = 0.01
"""Target fluctuation amplitude for fluence systematics."""

SYST_CF_FL = dict(
    fluctuations=(SYST_TARGET_SIGMA, SYST_TARGET_SIGMA * 0.5),
    loglogavgslope=(-4.5, 0.5),
)
"""Fluctuation prior for the fluence systematic correlated field."""

SYST_MULT_MIN = 0.8
"""Lower bound for the systematic multiplicative factor."""

SYST_MULT_MAX = 1.2
"""Upper bound for the systematic multiplicative factor."""


# ===========================================================================
#  Correlated-field hyper-parameters - timing
# ===========================================================================

TIMING_CF_ZM = dict(offset_mean=0.0, offset_std=(1e-10, 1e-11))
"""Zero-mode prior for the timing correlated field."""

TIMING_CLIP_NS = 10.0
"""Clip bound for timing CF corrections [ns]."""


# ===========================================================================
#  Timing quality control
# ===========================================================================

OUTLIER_NEIGHBORS_K = 10
"""Number of nearest neighbours used for per-station outlier detection."""

OUTLIER_STD_FACTOR = 2.5
"""Factor of the station timing RMS beyond which a point is flagged."""

MIN_ABSOLUTE_NEIGHBORS = 8
"""Minimum cluster size required to attempt outlier detection."""

MIN_TIMING_POINTS = 49
"""Minimum number of timing data points required for reconstruction."""

TIMING_UNCERT_THRESHOLD = 15e-9
"""Maximum acceptable timing residual standard deviation [s]."""

LOCAL_FIT_NEIGHBORS_K = 20
"""Cluster size for local per-antenna timing uncertainty estimation."""


# ===========================================================================
#  Optimiser (NIFTy optimize_kl) defaults
# ===========================================================================

N_VI_ITERATIONS = 8
"""Number of variational-inference iterations."""

N_SAMPLES = 60
"""Number of posterior samples drawn in the final iteration."""

SAMPLING_MODE = "nonlinear_sample"
"""NIFTy sample mode used after initial iterations.
    nonlinear_sample is geoVI, slower but performs better
    linear_sample (MGVI) is faster, use for testing.
    """

RESAMPLING_MODE = "nonlinear_resample"
"""NIFTy sample mode used for the first few iterations.
    nonlinear-resample is geoVI, slower but performs better
    linear_resample (MGVI) is faster, use for testing.
    """

UPDATE_MODE = "nonlinear_update"
"""NIFTy sample mode used in the final iterations."""


# ===========================================================================
#  Default prior ranges
# ===========================================================================

DEFAULT_PARAMS_ERAD = {"mean": np.log(1e7), "std": 3.0}
"""Log-normal prior for the radiation energy [eV]."""

DEFAULT_PARAMS_XMAX = {"a_min": 420.0, "a_max": 1020.0}
"""Uniform prior bounds for X_max [g/cm^2]."""

DEFAULT_PARAMS_T0 = {"mean": 0.0, "std": 200e-9}
"""Normal prior for the reference arrival time t_0 [s]."""

# ===========================================================================
#  Noise models and parameters
# ===========================================================================
# In this reconstruction framework, we explicitly model two components of noise:
#
# 1. Noise floor (`noise_floor_mean`): 
#    This should always be the mean of the fluence noise calculated from non-overlapping noise
#    windows (the same length as the window from which the fluence value is calculated).
#    It is a free parameter inferred during the reconstruction using the
#    `DEFAULT_PARAMS_NOISE_FLOOR_MEAN` prior defined below.
#
# 2. Noise standard deviation (`noise_std`): The absolute standard deviation 
#    of the fluence noise. This value dictates the uncertainty of the fluence 
#    measurements and directly populates the noise covariance matrix in the 
#    Gaussian likelihood. Sensible values typically correspond to the RMS standard 
#    deviation of the noise window in each antenna trace.

DEFAULT_PARAMS_NOISE_FLOOR_MEAN = {"mean": 1.0, "std": 0.1}
"""Log-normal prior for the noise floor (mean of the fluence noise).
Sensible values depend on the mean noise in the data.
"""

DEFAULT_PARAMS_GAMMA = {"mean": 1.465, "std": 0.292}
"""Normal prior for the wavefront shape parameter gamma."""

DEFAULT_PARAMS_CONST_RHO_RES = {"mean": 0.0, "std": 1513.2}
"""Normal prior for the residual constant in the Cherenkov-angle polynomial."""

DEFAULT_PARAMS_XMAX_TIMING_OFFSET = {"mean": 0.0, "std": 10.0}
"""Normal prior for the X_max offset between fluence and timing [g/cm^2]."""

DEFAULT_PARAMS_CORE_TIMING_OFFSET = {"mean": 0.0, "std": 2.5}
"""Normal prior for the core-position offset between fluence and timing [m]."""

RELATIVE_SYSTEMATIC_ERROR = 0.1
"""Relative systematic uncertainty added in quadrature to fluence data.
    0.1 = 10% systematic uncertainty. Can be detector absolute calibration, ...
    """

TIMING_UNCERTAINTY_S = 1.5e-9
"""Default timing uncertainty used for GP fluctuation amplitude [s]."""

WAVEFRONT_B_OFFSET_S = -3e-9
"""Constant offset in the hyperbolic wavefront model [s]."""


# ===========================================================================
#  Feature toggles
# ===========================================================================

FLUENCE_CF = False
"""Enable the fluence systematic correlated field."""

TIMING_CF = True
"""Enable the timing correlated field."""

PARTICLE_CF = True
"""Enable the particle systematic correlated field."""


# ===========================================================================
#  Particle model constants - experimental!
# ===========================================================================

LAMBDA_ATT_G_CM2 = 220.0
"""Atmospheric attenuation length for particles [g/cm²]."""

PAR_A_INTERCEPT = 8.15
"""Intercept of log10(N_max) vs log10(E_cr) relation."""

PAR_B_SLOPE = 1.03
"""Slope of log10(N_max) vs log10(E_cr) relation."""

DETECTOR_AREA_M2 = 0.96
"""Effective area of a single particle detector [m²]."""

X_GROUND_GPCM2_FALLBACK = 1024.0
"""Fallback vertical atmospheric depth at ground level [g/cm²]."""

MOLIERE_RADIUS_M = 25.0
"""Effective Molière radius for particle LDF [m]."""

RHO_SEA_LEVEL = 1.225e-3
"""Atmospheric density at sea level [g/cm³]."""

SCALE_HEIGHT = 8400.0
"""Atmospheric scale height [m]."""

N_SEA_LEVEL = 2.8e-4
"""Refractivity at sea level."""

X0_AIR_G_CM2 = 36.62
"""Radiation length of air [g/cm²]."""

E_SCALE_GEV = 0.0212
"""Energy scale for Molière radius calculation [GeV]."""

E_CRIT_GEV = 0.081
"""Critical energy of air [GeV]."""

POISSON_GAUSS_THRESHOLD = 20
"""Counts threshold: Poisson below, Gaussian above."""

NOISE_CALIB_SIGMA = 0.03
"""Calibration fractional uncertainty for particle Gaussian likelihood."""

PARTICLE_GAUSS_SYST_SIGMA = 0.30
"""Systematic fractional uncertainty for particle Gaussian likelihood."""

NOISE_BACKGROUND_COUNTS = 1.0
"""Background counts floor for particle Gaussian likelihood."""

# Particle correlated-field settings
PARTICLE_CF_ZM = {"offset_mean": 0.0, "offset_std": (0.1, 0.05)}
"""Zero-mode settings for the particle correction field."""

PARTICLE_CF_FL = {"fluctuations": (0.3, 0.1), "loglogavgslope": (-4.0, 1.0)}
"""Fluctuation settings for the particle correction field."""

PARTICLE_LOG_MIN = -0.693  # log(0.5)
"""Minimum log-multiplier for particle correction field."""

PARTICLE_LOG_MAX = 0.693   # log(2.0)
"""Maximum log-multiplier for particle correction field."""


# ===========================================================================
#  Default particle priors
# ===========================================================================

DEFAULT_PARAMS_ERAD_FACTOR = {"mean": 15.8, "std": 7.4}
"""LogNormal prior for Erad-to-Ecr conversion factor.
    15.8 is from https://arxiv.org/abs/1605.02564, 
    for LOFAR simulations we found 9.57+-0.9.
    This needs to be tuned for the specific detector setup.
    """

DEFAULT_PARAMS_XMAX_PARTICLE = {"a_min": 400, "a_max": 1100}
"""Uniform prior for particle Xmax [g/cm²]."""

DEFAULT_PARAMS_THETA_PARTICLE = {"a_min": 0.0, "a_max": np.pi / 2.0}
"""Uniform prior for particle zenith [rad]."""

DEFAULT_PARAMS_PHI_PARTICLE = {"a_min": 0.0, "a_max": 2.0 * np.pi}
"""Uniform prior for particle azimuth [rad]."""

DEFAULT_PARAMS_LOG_SCALE_PARTICLE = {"mean": 0.0, "std": 0.5}
"""Normal prior for particle log-scale factor."""

DEFAULT_PARAMS_PARTICLE_BG = {"mean": -1.0, "std": 1.0}
"""Normal prior for particle background (log10 counts)."""

