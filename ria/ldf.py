"""
Lateral distribution functions (LDF) for air-shower radio emission.

Implements the geomagnetic and charge-excess LDF components using
calibrated B-spline parametrisations.  All functions are JAX-compatible
and suitable for JIT compilation and automatic differentiation.

This module is a JAX-differentiable re-implementation of the LDF model
from the `geoceLDF <https://github.com/cg-laser/geoceLDF>`_ package.

The module loads pre-fitted spline data from pickle files at import time.

Functions
---------
LDF
    Combined total-fluence LDF (geomagnetic + charge-excess).
LDF_geo_ce
    Combined LDF accepting separate E_geo and E_ce energies.
evaluate_bspline
    Vectorised B-spline evaluation (low-level).
"""

from __future__ import annotations

import os
import pickle
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax, vmap
from jax.typing import ArrayLike

from .atmosphere import Atmosphere
from .coordinates import spherical_to_cartesian, get_angle


# ===================================================================
#  B-spline evaluation (fully JAX-compatible under vmap + jit)
# ===================================================================

def _evaluate_bspline_scalar(xx, t, c, k, i):
    """De Boor evaluation of a single B-spline basis at point *xx*.

    Parameters
    ----------
    xx : scalar
        Evaluation point.
    t : 1-D array
        Padded knot vector.
    c : 1-D array
        Spline coefficients.
    k : int
        Spline degree (static).
    i : int
        Knot-span index.
    """
    start = i - k
    d = lax.dynamic_slice(c, (start,), (k + 1,))

    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left_idx = i - k + j
            right_idx = i + j + 1 - r
            denom = t[right_idx] - t[left_idx]
            alpha = jnp.where(denom > 1e-8, (xx - t[left_idx]) / denom, 0.0)
            new_val = (1.0 - alpha) * d[j - 1] + alpha * d[j]
            d = d.at[j].set(new_val)

    return d[k]


def evaluate_bspline(
    x: ArrayLike, t: jnp.ndarray, c: jnp.ndarray, k: int
) -> jnp.ndarray:
    """Evaluate a univariate B-spline at point(s) *x*.

    Vectorised over *x*; the degree *k* must be static.

    Parameters
    ----------
    x : array_like
        Evaluation point(s).
    t : array_like
        Padded knot vector.
    c : array_like
        Spline coefficients.
    k : int
        Spline degree.

    Returns
    -------
    jnp.ndarray
        Spline values at each point in *x*.
    """
    x = jnp.atleast_1d(jnp.asarray(x))
    t = jnp.asarray(t)
    c = jnp.asarray(c)
    n = c.shape[0]

    x = jnp.clip(x, t[k], t[-(k + 1)])

    def eval_single(xx):
        i = jnp.searchsorted(t, xx, side="right") - 1
        i = jnp.clip(i, k, n - 1)
        return _evaluate_bspline_scalar(xx, t, c, k, i)

    return vmap(eval_single)(x)


# ===================================================================
#  Load pre-fitted spline data from pickle files
# ===================================================================

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_spline_data: dict[str, tuple] = {}


def _pad_and_store(key, t, c, k):
    """Pad knots and store as JAX arrays."""
    t_padded = np.append(
        np.append(np.ones(k) * t[0], t), np.ones(k) * t[-1]
    )
    _spline_data[key] = (jnp.array(t_padded), jnp.array(c), k)


def _load_pickle(filename):
    """Read a pickle file with latin-1 encoding."""
    with open(os.path.join(_DATA_DIR, filename), "rb") as f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        return u.load()


# --- geo_rcut_b_splines ---
_rcut_b = _load_pickle("geo_rcut_b_splines.pickle")
_pad_and_store("rcut_geo", *_rcut_b[0])
_pad_and_store("b_geo", *_rcut_b[1])

# --- geo_sigmaR_spl ---
_sigmaR = _load_pickle("geo_sigmaR_spl.pickle")
for _suffix in ("geo_R_0m", "geo_R_1564m", "geo_sigma_0m", "geo_sigma_1564m"):
    _pad_and_store(_suffix, *_sigmaR[_suffix])

# --- ce_sigma_spl ---
_ce_sigma = _load_pickle("ce_sigma_spl.pickle")
for _suffix in ("ce_sigma_0m", "ce_sigma_1564m"):
    _pad_and_store(_suffix, *_ce_sigma[_suffix])

# --- Ecorr ---
_ecorr = _load_pickle("Ecorr.pickle")
for _suffix in (
    "geo_Ecorr_1564m", "geo_Ecorr_0m", "ce_Ecorr_1564m", "ce_Ecorr_0m"
):
    _pad_and_store(_suffix, *_ecorr[_suffix])

# Clean up module-level temporaries
del _rcut_b, _sigmaR, _ce_sigma, _ecorr, _suffix


def _get_spline_tck(key_base: str, obsheight: float = 0) -> tuple:
    """Retrieve ``(t, c, k)`` for a named spline.

    Parameters
    ----------
    key_base : str
        Base key, e.g. ``'geo_R'``.
    obsheight : float
        Observation height label (``0`` or ``1564``).
    """
    key = f"{key_base}_{int(obsheight)}m"
    if key in _spline_data:
        return _spline_data[key]
    if key_base in _spline_data:
        return _spline_data[key_base]
    raise ValueError(
        f"Spline data not found for key '{key}' (base='{key_base}', "
        f"height={obsheight})"
    )


# ===================================================================
#  Physics helper functions
# ===================================================================

def get_lorentz_force_vector(
    zenith: ArrayLike,
    azimuth: ArrayLike,
    magnetic_field_vector: ArrayLike,
) -> jnp.ndarray:
    """Lorentz force direction (v x B) as a Cartesian 3-vector.

    Parameters
    ----------
    zenith, azimuth : array_like
        Shower direction [rad].
    magnetic_field_vector : array_like
        Geomagnetic field vector.
    """
    magnetic_field_vector = jnp.asarray(magnetic_field_vector)
    axis = spherical_to_cartesian(zenith, azimuth)
    norm = jnp.linalg.norm(magnetic_field_vector, axis=-1, keepdims=True)
    b_hat = jnp.where(norm > 1e-9, magnetic_field_vector / norm, 0.0)
    return jnp.cross(axis, b_hat)


def get_sine_angle_to_lorentz_force(
    zenith: ArrayLike,
    azimuth: ArrayLike,
    magnetic_field_vector: ArrayLike,
) -> ArrayLike:
    """Sine of the angle between shower axis and Lorentz force.

    Parameters
    ----------
    zenith, azimuth : array_like
        Shower direction [rad].
    magnetic_field_vector : array_like
        Geomagnetic field vector.
    """
    lf = get_lorentz_force_vector(zenith, azimuth, magnetic_field_vector)
    return jnp.linalg.norm(lf, axis=-1)


def get_a(
    rho: ArrayLike,
    magnetic_field_strength: ArrayLike = 0.243,
    magnetic_field_vector=None,
) -> ArrayLike:
    """Relative charge-excess fraction *a*.

    Parameters
    ----------
    rho : array_like
        Air density at X_max [g/m^3].
    magnetic_field_strength : float, optional
        Magnitude of the geomagnetic field [Gauss].
    magnetic_field_vector : array_like or None
        If given, overrides *magnetic_field_strength*.
    """
    rho = jnp.asarray(rho)
    if magnetic_field_vector is not None:
        magnetic_field_strength = np.linalg.norm(magnetic_field_vector)
    else:
        magnetic_field_strength = jnp.asarray(magnetic_field_strength)
    avg_density = 648.18353008270035
    a_calc = -0.23604683 + 0.43426141 * jnp.exp(
        1.11141046e-3 * (rho - avg_density)
    )
    norm_factor = jnp.where(
        magnetic_field_strength > 1e-9,
        (magnetic_field_strength / 0.243) ** 0.9,
        1.0,
    )
    return a_calc / norm_factor


# ===================================================================
#  Charge-excess helpers
# ===================================================================

def get_k_ce(dxmax: ArrayLike) -> ArrayLike:
    """CE shape parameter *k* from distance to X_max."""
    dxmax = jnp.asarray(dxmax)
    a, b, c, d = 5.80505613e2, -1.76588481e0, 3.12029983e0, 3.73038601e-3
    res = b + (c - b) / (1.0 + jnp.exp(-d * (dxmax - a)))
    return jnp.maximum(res, 0.0)


def get_b_ce(k: ArrayLike, dxmax: ArrayLike) -> ArrayLike:
    """CE exponent parameter *b*."""
    k = jnp.asarray(k)
    dxmax = jnp.asarray(dxmax)
    return jnp.where(
        k < 1e-5,
        146.92691815 - 0.25112664 * dxmax,
        55.55667917 + 0.32392104 * dxmax,
    )


def get_rcut_ce(k: ArrayLike, dxmax: ArrayLike) -> ArrayLike:
    """CE exponent knee radius *rcut*."""
    k = jnp.asarray(k)
    dxmax = jnp.asarray(dxmax)
    b = get_b_ce(k, dxmax)
    p0, p1, p2 = 2.90571462e1, 1.97413284e-1, 1.80588511e-3
    sqrt_term = (4.0 * b - 4.0 * p0) * p2 + p1 ** 2
    sqrt_val = jnp.sqrt(jnp.maximum(0.0, sqrt_term))
    rcut_calc = jnp.where(
        jnp.abs(p2) > 1e-9, 0.5 * (-p1 + sqrt_val) / p2, 0.0
    )
    return jnp.where(k < 1e-5, jnp.zeros_like(dxmax), rcut_calc)


# ===================================================================
#  Geomagnetic spline helpers
# ===================================================================

def _get_b_geo_spl(dxmax: ArrayLike) -> ArrayLike:
    """Geo exponent parameter *b* from spline."""
    t, c, k = _spline_data["b_geo"]
    return evaluate_bspline(dxmax, t, c, k)


def _get_rcut_geo_spl(dxmax: ArrayLike) -> ArrayLike:
    """Geo knee radius *rcut* from spline."""
    t, c, k = _spline_data["rcut_geo"]
    return evaluate_bspline(dxmax, t, c, k)


# ===================================================================
#  Exponent parametrisation
# ===================================================================

def get_p(
    r: ArrayLike, rcut: ArrayLike, b_param: ArrayLike
) -> ArrayLike:
    """Radial exponent *p(r)* for the generalised-Gaussian LDF.

    Parameters
    ----------
    r : array_like
        Radial distance [m].
    rcut : array_like
        Knee radius [m].
    b_param : array_like
        Exponent parameter (will be multiplied by 1e-3 internally).
    """
    r = jnp.abs(jnp.asarray(r))
    rcut = jnp.maximum(1.0, jnp.abs(jnp.asarray(rcut)))
    b_param = 1e-3 * jnp.asarray(b_param)

    p_geo_base = jnp.power(rcut, b_param)
    p_geo = 2.0 * jnp.nan_to_num(p_geo_base, nan=1.0, posinf=1.0, neginf=1.0)

    r_pow = jnp.power(jnp.maximum(r, 1e-9), -b_param)
    r_pow = jnp.nan_to_num(r_pow, nan=0.0, posinf=0.0, neginf=0.0)

    return jnp.where(r <= rcut, 2.0, p_geo * r_pow)


# ===================================================================
#  LDF component functions
# ===================================================================

def _ldf_vB_parts(
    r: ArrayLike, sigma: ArrayLike, R_val: ArrayLike, p: ArrayLike = 2.0
) -> ArrayLike:
    """Gaussian-ring base function for the geomagnetic LDF."""
    r = jnp.asarray(r)
    sigma = jnp.maximum(jnp.asarray(sigma), 1e-9)
    R_val = jnp.asarray(R_val)
    p = jnp.asarray(p)

    base = jnp.abs(r - R_val) / (jnp.sqrt(2.0) * sigma)
    exponent = jnp.power(base, p)
    exponent = jnp.nan_to_num(exponent, nan=jnp.inf, posinf=jnp.inf)
    return jnp.exp(-exponent)


def ldf_vB(
    x: ArrayLike,
    y: ArrayLike,
    sigma: ArrayLike,
    R_val: ArrayLike,
    E: ArrayLike,
    p: ArrayLike = 2.0,
) -> ArrayLike:
    """Geomagnetic LDF component (ring-Gaussian shape).

    Parameters
    ----------
    x, y : array_like
        Shower-plane coordinates [m].
    sigma : array_like
        Ring width [m].
    R_val : array_like
        Ring radius [m].
    E : array_like
        Energy normalisation.
    p : array_like
        Exponent.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    sigma = jnp.maximum(jnp.asarray(sigma), 1e-12)
    R_val = jnp.asarray(R_val)
    E = jnp.asarray(E)
    p = jnp.asarray(p)

    r = jnp.sqrt(x ** 2 + y ** 2)

    sqrt2 = jnp.sqrt(2.0)
    sqrtpi = jnp.sqrt(jnp.pi)

    # R < 0 branch
    arg_erfc = -R_val * sqrt2 / (2.0 * sigma)
    exp1 = jnp.exp(-R_val ** 2 / (2.0 * sigma ** 2))
    norm1_denom = (
        jax.scipy.special.erfc(arg_erfc) * sqrtpi * R_val
        + sqrt2 * sigma * exp1
    )
    norm1 = jnp.maximum(jnp.abs(sigma * jnp.pi * sqrt2 * norm1_denom), 1e-12)
    val1 = (E / norm1) * _ldf_vB_parts(r, sigma, R_val, p)

    # R >= 0 branch
    arg_erf = 0.5 * R_val * sqrt2 / sigma
    exp2 = jnp.exp(0.5 * R_val ** 2 / sigma ** 2)
    denom_inner = (
        jax.scipy.special.erf(arg_erf) * sqrtpi * sqrt2 * exp2 * R_val
        + 2.0 * sigma
    ) * jnp.pi
    denom_inner = jnp.where(
        jnp.abs(denom_inner) > 1e-12, denom_inner, 1e-12
    )
    nf2 = 0.5 * exp2 / (sigma * denom_inner)
    nf2 = jnp.nan_to_num(nf2, nan=0.0, posinf=0.0, neginf=0.0)

    val2 = E * nf2 * (
        _ldf_vB_parts(r, sigma, R_val, p)
        + _ldf_vB_parts(r, sigma, -R_val, p)
    )

    result = jnp.where(R_val < 0, val1, val2)
    return jnp.maximum(0.0, result)


def _ldf_ce_base(
    xx: ArrayLike,
    E: ArrayLike,
    sigma: ArrayLike,
    k: ArrayLike,
    rcut: ArrayLike,
    b_param: ArrayLike,
    p: ArrayLike = None,
) -> ArrayLike:
    """Charge-excess LDF component (generalised-Gaussian shape).

    Parameters
    ----------
    xx : array_like
        Radial distance [m].
    E : array_like
        Energy normalisation.
    sigma : array_like
        Width parameter [m].
    k : array_like
        Shape parameter.
    rcut : array_like
        Knee radius [m].
    b_param : array_like
        Exponent parameter.
    p : array_like or None
        Explicit exponent (computed from *rcut*/*b_param* if ``None``).
    """
    xx = jnp.asarray(xx)
    E = jnp.asarray(E)
    sigma = jnp.maximum(jnp.asarray(sigma), 1e-12)
    k = jnp.asarray(k)
    rcut = jnp.asarray(rcut)
    b_param = jnp.asarray(b_param)

    r = jnp.abs(xx)

    if p is None:
        p = get_p(r, rcut, b_param)
    else:
        p = jnp.asarray(p)

    # Normalisation
    gamma_arg = 0.5 * k + 1.0
    gamma_val = jax.scipy.special.gamma(jnp.maximum(gamma_arg, 1e-6))

    k1 = k + 1.0
    sigma_k2 = jnp.power(sigma, k1 + 1.0)
    term_pow = jnp.power(jnp.maximum(2.0 * k1, 1e-9), -0.5 * k)
    pow_2k = jnp.power(2.0, k)

    norm_num = k1 / (pow_2k * term_pow)
    norm_den = jnp.maximum(jnp.abs(sigma_k2 * 2.0 * jnp.pi * gamma_val), 1e-12)
    norm = jnp.nan_to_num(norm_num / norm_den, nan=0.0, posinf=0.0, neginf=0.0)

    # Exponential
    exp_den_factor = p / jnp.maximum(k1, 1e-9)
    sig_p = jnp.power(sigma, p)
    exp_den = jnp.maximum(jnp.abs(exp_den_factor * sig_p), 1e-12)
    exp_term = jnp.exp(-jnp.power(r, p) / exp_den)

    # r^k
    r_k = jnp.power(r, k)
    r_k = jnp.where(r < 1e-12, 0.0, r_k)
    r_k = jnp.nan_to_num(r_k, nan=0.0)

    fluence = norm * E * r_k * exp_term
    result = jnp.where(k < 0.0, jnp.nan, fluence)
    return jnp.where(jnp.isnan(result), jnp.nan, jnp.maximum(0.0, result))


# ===================================================================
#  LDF functions using splines
# ===================================================================

def _ldf_geo_dxmax(
    x: ArrayLike,
    y: ArrayLike,
    dxmax: ArrayLike,
    E: ArrayLike,
    obsheight: float = 0,
) -> ArrayLike:
    """Geomagnetic LDF using dXmax parametrisation."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    dxmax = jnp.asarray(dxmax)
    E = jnp.asarray(E)

    t_R, c_R, k_R = _get_spline_tck("geo_R", obsheight)
    R = evaluate_bspline(dxmax, t_R, c_R, k_R)

    t_s, c_s, k_s = _get_spline_tck("geo_sigma", obsheight)
    sigma = evaluate_bspline(dxmax, t_s, c_s, k_s)

    t_e, c_e, k_e = _get_spline_tck("geo_Ecorr", obsheight)
    Ecorr = evaluate_bspline(dxmax, t_e, c_e, k_e)
    Ecorr = jnp.where(jnp.abs(Ecorr) < 1e-9, 1.0, Ecorr)

    rcut = _get_rcut_geo_spl(dxmax)
    b = _get_b_geo_spl(dxmax)

    r = jnp.sqrt(x ** 2 + y ** 2)
    p = get_p(r, rcut, b)

    return ldf_vB(x, y, sigma, R, E, p) / Ecorr


def _ldf_ce_dxmax(
    x: ArrayLike,
    y: ArrayLike,
    dxmax: ArrayLike,
    E: ArrayLike,
    obsheight: float = 0,
) -> ArrayLike:
    """Charge-excess LDF using dXmax parametrisation."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    dxmax = jnp.asarray(dxmax)
    E = jnp.asarray(E)

    t_s, c_s, k_s = _get_spline_tck("ce_sigma", obsheight)
    sigma = evaluate_bspline(dxmax, t_s, c_s, k_s)

    t_e, c_e, k_e = _get_spline_tck("ce_Ecorr", obsheight)
    Ecorr = evaluate_bspline(dxmax, t_e, c_e, k_e)
    Ecorr = jnp.where(jnp.abs(Ecorr) < 1e-9, 1.0, Ecorr)

    k = get_k_ce(dxmax)
    rcut = get_rcut_ce(k, dxmax)
    b = get_b_ce(k, dxmax)

    r = jnp.sqrt(x ** 2 + y ** 2)
    return _ldf_ce_base(r, E=E, sigma=sigma, k=k, rcut=rcut, b_param=b) / Ecorr


# ===================================================================
#  Top-level LDF functions
# ===================================================================

@partial(jax.jit, static_argnames=("obsheight",))
def _ldf_core(
    x, y, Erad, xmax, zenith, azimuth, core,
    obsheight, magnetic_field_vector, Xatm_obs_gm2, rho_xmax,
):
    """JIT-compiled core of the combined LDF calculation."""
    slant_depth = Xatm_obs_gm2 / jnp.cos(zenith)
    dxmax = (slant_depth - xmax * 1e4) * 1e-4

    x_in = jnp.asarray(x)
    y_in = jnp.asarray(y)
    Erad_in = jnp.asarray(Erad)
    dxmax_in = jnp.asarray(dxmax)
    zenith_in = jnp.asarray(zenith)
    azimuth_in = jnp.asarray(azimuth)
    core_in = jnp.asarray(core)
    B = jnp.asarray(magnetic_field_vector)

    B_strength = jnp.maximum(jnp.linalg.norm(B), 1e-9)
    a = get_a(rho_xmax, B_strength)
    sin_alpha = get_sine_angle_to_lorentz_force(zenith_in, azimuth_in, B)
    sin_alpha_safe = jnp.maximum(sin_alpha, 1e-9)
    a_over_sin2 = jnp.power(a / sin_alpha_safe, 2)
    E_geo = jnp.where(sin_alpha > 1e-9, Erad_in / (1.0 + a_over_sin2), 0.0)
    E_ce = Erad_in - E_geo

    x2 = x_in - core_in[0]
    y2 = y_in - core_in[1]

    f_ce = _ldf_ce_dxmax(x2, y2, dxmax_in, E_ce, obsheight=obsheight)
    f_geo = _ldf_geo_dxmax(x2, y2, dxmax_in, E_geo, obsheight=obsheight)

    az = jnp.arctan2(y2, x2)
    sqrt_fgeo = jnp.sqrt(jnp.maximum(f_geo, 0.0))
    sqrt_fce = jnp.sqrt(jnp.maximum(f_ce, 0.0))
    f_vB = jnp.power(sqrt_fgeo + sqrt_fce * jnp.cos(az), 2)
    f_vvB = f_ce * jnp.power(jnp.sin(az), 2)
    f = f_vB + f_vvB

    return f, f_vB, f_vvB, f_geo, f_ce


def LDF(
    x: ArrayLike,
    y: ArrayLike,
    Erad: ArrayLike,
    xmax: ArrayLike,
    zenith: ArrayLike,
    azimuth: ArrayLike,
    core: ArrayLike = None,
    obsheight: float = 0.0,
    magnetic_field_vector: ArrayLike = None,
    atmosphere_path: str | None = None,
) -> tuple:
    """Combined LDF calculation (geomagnetic + charge-excess).

    Parameters
    ----------
    x, y : array_like
        Shower-plane antenna positions [m].
    Erad : array_like
        Radiation energy [eV].
    xmax : array_like
        Atmospheric depth of the shower maximum [g/cm^2].
    zenith, azimuth : array_like
        Shower direction [rad].
    core : array_like or None
        Core position ``(x, y)`` [m].  Defaults to ``(0, 0)``.
    obsheight : float
        Observer altitude [m].
    magnetic_field_vector : array_like or None
        Geomagnetic field; defaults to LOFAR.
    atmosphere_path : str or None
        Path to a GDAS data file; uses built-in model if ``None``.

    Returns
    -------
    tuple of jnp.ndarray
        ``(f_total, f_vB, f_vvB, f_geo, f_ce)``
    """
    if atmosphere_path is None:
        atmosphere_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "example_data", "ATMOSPHERE_EXAMPLE.DAT")
        )
    atmosphere = Atmosphere(gdas_file=atmosphere_path)

    Xatm = atmosphere.get_atmosphere(obsheight)
    rho = atmosphere.get_density(xmax, zenith)

    if core is None:
        core = jnp.array([0.0, 0.0], dtype=jnp.float64)
    if magnetic_field_vector is None:
        magnetic_field_vector = jnp.array(
            [0.004675, 0.186270, -0.456412], dtype=jnp.float64
        )

    return _ldf_core(
        x, y, Erad, xmax, zenith, azimuth, core,
        obsheight, magnetic_field_vector, Xatm, rho,
    )


@partial(jax.jit, static_argnames=("obsheight",))
def LDF_geo_ce(
    x: ArrayLike,
    y: ArrayLike,
    Egeo: ArrayLike,
    Ece: ArrayLike,
    dxmax: ArrayLike,
    core: ArrayLike = None,
    obsheight: float = 0.0,
) -> tuple:
    """Combined LDF accepting separate geomagnetic and CE energies.

    Parameters
    ----------
    x, y : array_like
        Shower-plane antenna positions [m].
    Egeo, Ece : array_like
        Geomagnetic and charge-excess energies.
    dxmax : array_like
        Distance to X_max in slant depth [g/cm^2].
    core : array_like or None
        Core position ``(x, y)`` [m].
    obsheight : float
        Observer altitude [m].

    Returns
    -------
    tuple of jnp.ndarray
        ``(f_total, f_vB, f_vvB, f_geo, f_ce)``
    """
    if core is None:
        core = jnp.array([0.0, 0.0], dtype=jnp.float64)
    core = jnp.asarray(core)

    x2 = jnp.asarray(x) - core[0]
    y2 = jnp.asarray(y) - core[1]

    f_ce = _ldf_ce_dxmax(x2, y2, jnp.asarray(dxmax), jnp.asarray(Ece), obsheight=obsheight)
    f_geo = _ldf_geo_dxmax(x2, y2, jnp.asarray(dxmax), jnp.asarray(Egeo), obsheight=obsheight)

    az = jnp.arctan2(y2, x2)
    sqrt_fgeo = jnp.sqrt(jnp.maximum(f_geo, 0.0))
    sqrt_fce = jnp.sqrt(jnp.maximum(f_ce, 0.0))
    f_vB = jnp.power(sqrt_fgeo + sqrt_fce * jnp.cos(az), 2)
    f_vvB = f_ce * jnp.power(jnp.sin(az), 2)
    f_total = f_vB + f_vvB

    return f_total, f_vB, f_vvB, f_geo, f_ce
