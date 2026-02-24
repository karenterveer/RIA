"""
Bayesian reconstruction wrapper for air-shower radio data.

Provides a high-level ``reconstruct()`` function that assembles the
:class:`~ria.forward_model.FootprintModel`, builds a Gaussian
likelihood, runs NIFTy's ``optimize_kl``, and returns a structured
results dictionary.

Functions
---------
reconstruct
    Run a full Bayesian reconstruction from prepared data arrays.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import nifty8.re as jft

from jax import random, vmap

from .forward_model import FootprintModel
from . import timing as _timing
from . import config

logger = logging.getLogger(__name__)


class _DiagCovInv:
    """Pickleable diagonal inverse covariance operator.

    ``jft.Gaussian`` needs a callable ``noise_cov_inv`` that can be
    pickled by ``optimize_kl``.  A plain ``lambda`` capturing local
    arrays is not pickleable, so we use this small class instead.
    """

    def __init__(self, inv_var):
        self._inv_var = inv_var

    def __call__(self, tangents):
        return self._inv_var * tangents




def reconstruct(
    positions: np.ndarray,
    fluences: np.ndarray,
    times: np.ndarray,
    noise_std: np.ndarray | float,
    *,
    noise_floor_mean: float | None = None,
    # --- shower priors ---
    mean_zenith: float,
    mean_azimuth: float,
    mean_core_x: float = 0.0,
    mean_core_y: float = 0.0,
    std_angle_rad: float = np.radians(3.0),
    std_core_m: float = 30.0,
    # --- timing per antenna ---
    timing_uncertainties: np.ndarray | None = None,
    timing_mask: np.ndarray | None = None,
    station_ids: np.ndarray | None = None,
    # --- optional timing QC ---
    enable_timing_qc: bool = True,
    t0_initial_guess: float | None = None,
    # --- particle data (optional) ---
    particle_positions: np.ndarray | None = None,
    particle_counts: np.ndarray | None = None,
    # --- atmosphere ---
    atmosphere_path: str | None = None,
    magnetic_field_vector: np.ndarray | None = None,
    # --- optimiser ---
    n_iterations: int | None = None,
    n_samples: int | None = None,
    seed: int = 42,
    # --- extra model kwargs ---
    model_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a full Bayesian reconstruction.

    Parameters
    ----------
    positions : ndarray, shape ``(2 or 3, N)``
        Ground-plane antenna positions [m].  Only the first two rows
        (x, y) are used.
    fluences : ndarray, shape ``(N,)``
        Measured energy fluence at each antenna [eV/m^2].
    times : ndarray, shape ``(N,)``
        Measured arrival times at each antenna [s].
    noise_std : float or ndarray, shape ``(N,)``
        Fluence noise standard deviation per antenna [eV/m^2]. Populates 
        the noise covariance matrix.
    mean_zenith, mean_azimuth : float
        Prior mean for the shower direction [rad].
    mean_core_x, mean_core_y : float
        Prior mean for the core position [m].
    std_angle_rad : float
        Half-width of the uniform direction prior [rad].
    std_core_m : float
        Half-width of the uniform core-position prior [m].
    timing_uncertainties : ndarray or None
        Per-antenna timing uncertainty [s].  If ``None`` and
        *enable_timing_qc* is ``True``, uncertainties are estimated
        from the data (see :func:`ria.timing.get_local_timing_uncertainties`).
    timing_mask : ndarray of bool or None
        Mask indicating which antennas have reliable timing (``True`` =
        reliable).  If ``None``, all are assumed reliable.
    station_ids : ndarray or None
        Station labels for each antenna (needed for timing QC).
    enable_timing_qc : bool
        Run timing outlier removal before reconstruction.
    t0_initial_guess : float or None
        Initial guess for t_0 [s].  Defaults to the median of *times*.
    atmosphere_path : str or None
        Path to a GDAS atmosphere file.
    magnetic_field_vector : ndarray or None
        Geomagnetic field vector [Gauss].
    n_iterations : int or None
        Number of VI iterations.  Default ``config.N_VI_ITERATIONS``.
    n_samples : int or None
        Number of posterior samples.  Default ``config.N_SAMPLES``.
    seed : int
        PRNG seed.
    model_kwargs : dict or None
        Additional keyword arguments forwarded to
        :class:`~ria.forward_model.FootprintModel`.
    particle_positions : ndarray, shape ``(2 or 3, M)`` or None
        Particle detector positions [m].  If provided together with
        *particle_counts*, a combined radio + particle likelihood is
        built.
    noise_floor_mean : float or None
        Prior mean for the noise floor (the mean of the fluence noise). 
        Conceptually acts as the background noise level. If ``None``, defaults 
        to the median of the `noise_std` array.
    particle_counts : ndarray, shape ``(M,)`` or None
        Observed particle counts at each detector.

    Returns
    -------
    dict
        Structured results with keys:

        - ``"zenith"``, ``"azimuth"``, ``"core_x"``, ``"core_y"``,
          ``"erad"``, ``"xmax"``, ``"t0"`` : tuples ``(mean, std)``
        - ``"xmax_timing_offset"`` : tuple ``(mean, std)``
        - ``"xmax_combined"`` : tuple ``(mean, std)``
        - ``"energy_correction_factor"`` : tuple ``(mean, std)``
        - ``"xmax_shape_factor"`` : tuple ``(mean, std)``
        - ``"chi_sq"`` : reduced chi-squared
        - ``"samples"`` : list of posterior sample dicts
        - ``"model"`` : the ``FootprintModel`` instance
        - ``"timing_mask"`` : boolean mask after QC
    """
    if n_iterations is None:
        n_iterations = config.N_VI_ITERATIONS
    if n_samples is None:
        n_samples = config.N_SAMPLES

    x = np.asarray(positions[0], dtype=np.float64)
    y = np.asarray(positions[1], dtype=np.float64)
    fluences = np.asarray(fluences, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n_antennas = len(x)

    noise_std = np.broadcast_to(
        np.asarray(noise_std, dtype=np.float64), (n_antennas,)
    ).copy()

    import os
    if atmosphere_path is None:
        atmosphere_path = os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "example_data", "ATMOSPHERE_EXAMPLE.DAT")
        )

    if t0_initial_guess is None:
        t0_initial_guess = float(np.median(times))

    if timing_mask is None:
        timing_mask = np.ones(n_antennas, dtype=bool)
    else:
        timing_mask = np.asarray(timing_mask, dtype=bool).copy()

    # ------------------------------------------------------------------
    #  Optional timing quality control
    # ------------------------------------------------------------------
    if enable_timing_qc and station_ids is not None:
        pos_2d = np.stack([x, y])
        keep = _timing.detect_timing_outliers(
            pos_2d, times, station_ids,
        )
        timing_mask = timing_mask & keep

        timing_mask, final_std, *_ = _timing.iterative_timing_pruning(
            pos_2d, times, timing_mask,
        )
        logger.info(
            "After QC: %d timing points retained (std %.2f ns).",
            int(np.sum(timing_mask)), final_std * 1e9,
        )

    # ------------------------------------------------------------------
    #  Timing uncertainties
    # ------------------------------------------------------------------
    if timing_uncertainties is None:
        pos_for_unc = np.stack([x, y])
        timing_uncertainties = _timing.get_local_timing_uncertainties(
            pos_for_unc, times,
        )
    else:
        timing_uncertainties = np.asarray(timing_uncertainties, dtype=np.float64)

    inv_var_timing = 1.0 / (timing_uncertainties ** 2 + 1e-30)

    # For demoted points: inflate uncertainty
    demoted_std = 2.0 * np.abs(times - t0_initial_guess)
    demoted_std = np.maximum(demoted_std, 10e-9)
    inv_var_demoted = 1.0 / (demoted_std ** 2)

    # ------------------------------------------------------------------
    #  Estimate noise mean for model
    # ------------------------------------------------------------------
    if noise_floor_mean is None:
        noise_floor_mean = float(np.median(noise_std))
    noise_floor_mean = max(noise_floor_mean, 1e-6)

    # ------------------------------------------------------------------
    #  Build the forward model
    # ------------------------------------------------------------------
    kwargs = dict(
        params_phi={
            "a_min": mean_azimuth - std_angle_rad,
            "a_max": mean_azimuth + std_angle_rad,
        },
        params_theta={
            "a_min": mean_zenith - std_angle_rad,
            "a_max": mean_zenith + std_angle_rad,
        },
        params_X={
            "a_min": mean_core_x - std_core_m,
            "a_max": mean_core_x + std_core_m,
        },
        params_Y={
            "a_min": mean_core_y - std_core_m,
            "a_max": mean_core_y + std_core_m,
        },
        params_noise_floor_mean={"mean": noise_floor_mean, "std": noise_floor_mean / 10.0},
        atmosphere_path=atmosphere_path,
        params_t0={"mean": t0_initial_guess, "std": 200e-9},
        timing_std_s=config.TIMING_UNCERTAINTY_S,
        enable_syst_cf=config.FLUENCE_CF,
        enable_timing_cf=config.TIMING_CF,
    )
    if magnetic_field_vector is not None:
        kwargs["magnetic_field_vector"] = np.asarray(magnetic_field_vector)

    # Forward particle detector positions if available
    if (particle_positions is not None and particle_counts is not None
            and len(particle_counts) > 0):
        p_x = np.asarray(particle_positions[0], dtype=np.float64)
        p_y = np.asarray(particle_positions[1], dtype=np.float64)
        p_z = (
            np.asarray(particle_positions[2], dtype=np.float64)
            if particle_positions.shape[0] > 2
            else np.zeros_like(p_x)
        )
        kwargs["x_particle"] = p_x
        kwargs["y_particle"] = p_y
        kwargs["z_particle"] = p_z
        kwargs["enable_particle_cf"] = True

    if model_kwargs is not None:
        kwargs.update(model_kwargs)

    model = FootprintModel(x, y, **kwargs)

    # ------------------------------------------------------------------
    #  Determine if particle data is available
    # ------------------------------------------------------------------
    has_particles = (
        particle_positions is not None
        and particle_counts is not None
        and len(particle_counts) > 0
    )

    # ------------------------------------------------------------------
    #  Construct the likelihood
    # ------------------------------------------------------------------
    data = jnp.stack([jnp.array(fluences), jnp.array(times)])

    rel_sys = config.RELATIVE_SYSTEMATIC_ERROR
    fluence_var = noise_std ** 2 + (fluences * rel_sys) ** 2
    inv_var_fluence = jnp.array(1.0 / fluence_var)

    use_timing = jnp.array(timing_mask, dtype=jnp.float64)
    total_inv_var_time = jnp.where(
        use_timing.astype(bool),
        jnp.array(inv_var_timing),
        jnp.array(inv_var_demoted),
    )

    noise_cov_inv = jnp.stack([inv_var_fluence, total_inv_var_time])

    if has_particles:
        # --- Combined radio + particle likelihood ---
        # Following the pattern from combined_reco.py
        class _RadioOp(jft.Model):
            def __init__(self, m):
                self._m = m
                super().__init__(init=m.init)
            def __call__(self, x):
                return self._m(x)["radio"]

        lh_radio = jft.Gaussian(
            data=data,
            noise_cov_inv=_DiagCovInv(noise_cov_inv),
        ).amend(_RadioOp(model))

        # Hybrid Poisson/Gaussian particle likelihood
        p_obs_arr = jnp.array(particle_counts)
        threshold = config.POISSON_GAUSS_THRESHOLD
        poisson_mask = p_obs_arr <= threshold
        gauss_mask = ~poisson_mask
        n_poisson = int(jnp.sum(poisson_mask))
        n_gauss = int(jnp.sum(gauss_mask))

        logger.info(
            f"Particle likelihood: {n_poisson} Poisson (counts<={threshold}), "
            f"{n_gauss} Gaussian (counts>{threshold})"
        )

        class _PartOpPoisson(jft.Model):
            def __init__(self, m, mask):
                self._m = m
                self._mask = mask
                super().__init__(init=m.init)
            def __call__(self, x):
                return self._m(x)["particle"][self._mask]

        class _PartOpGauss(jft.Model):
            def __init__(self, m, mask):
                self._m = m
                self._mask = mask
                super().__init__(init=m.init)
            def __call__(self, x):
                return self._m(x)["particle"][self._mask]

        lh_p_components = []

        if n_poisson > 0:
            p_obs_poisson = p_obs_arr[poisson_mask].astype(jnp.int32)
            lh_p_poisson = jft.Poissonian(
                data=p_obs_poisson,
            ).amend(_PartOpPoisson(model, poisson_mask))
            lh_p_components.append(lh_p_poisson)

        if n_gauss > 0:
            p_obs_gauss = p_obs_arr[gauss_mask].astype(jnp.float64)
            poisson_var = p_obs_gauss
            calib_var = (p_obs_gauss * config.NOISE_CALIB_SIGMA) ** 2
            syst_var = (p_obs_gauss * config.PARTICLE_GAUSS_SYST_SIGMA) ** 2
            background_var = config.NOISE_BACKGROUND_COUNTS ** 2
            gauss_var = jnp.maximum(
                poisson_var + calib_var + syst_var + background_var, 1.0,
            )
            lh_p_gauss = jft.Gaussian(
                data=p_obs_gauss,
                noise_cov_inv=_DiagCovInv(1.0 / gauss_var),
            ).amend(_PartOpGauss(model, gauss_mask))
            lh_p_components.append(lh_p_gauss)

        # Combine particle components
        if len(lh_p_components) == 2:
            lh_p_components[0]._domain = jft.Vector(lh_p_components[0]._domain)
            lh_p_components[1]._domain = jft.Vector(lh_p_components[1]._domain)
            lh_p = lh_p_components[0] + lh_p_components[1]
        elif len(lh_p_components) == 1:
            lh_p = lh_p_components[0]
            lh_p._domain = jft.Vector(lh_p._domain)
        else:
            lh_p = None

        # Combine radio + particle
        lh_radio._domain = jft.Vector(lh_radio._domain)
        if lh_p is not None:
            lh = lh_radio + lh_p
        else:
            lh = lh_radio
    else:
        # --- Radio-only likelihood ---
        lh = jft.Gaussian(
            data=data,
            noise_cov_inv=_DiagCovInv(noise_cov_inv),
        ).amend(model)

    # ------------------------------------------------------------------
    #  Run optimize_kl
    # ------------------------------------------------------------------
    key = random.PRNGKey(seed)
    key, k_i, k_o = random.split(key, 3)
    delta = 1e-7

    samples, _ = jft.optimize_kl(
        lh,
        jft.Vector(lh.init(k_i)),
        n_total_iterations=n_iterations,
        n_samples=lambda i: n_samples // 2 if i < n_iterations // 2 else n_samples,
        key=k_o,
        draw_linear_kwargs=dict(
            cg_name="SL",
            cg_kwargs=dict(
                absdelta=delta * jft.size(lh.domain) / 10.0, maxiter=800,
            ),
        ),
        nonlinearly_update_kwargs=dict(
            minimize_kwargs=dict(
                name="SN", xtol=delta,
                cg_kwargs=dict(name=None), maxiter=15,
            ),
        ),
        kl_kwargs=dict(
            minimize_kwargs=dict(
                name="M", xtol=delta,
                cg_kwargs=dict(name=None), maxiter=25,
            ),
        ),
        sample_mode=lambda i: (
            config.RESAMPLING_MODE if i < 3
            else config.SAMPLING_MODE if i < 7
            else config.UPDATE_MODE
        ),
        kl_map=vmap,
    )

    # ------------------------------------------------------------------
    #  Extract posterior statistics
    # ------------------------------------------------------------------
    samples_list = list(samples)

    param_methods = {
        "zenith": lambda s: model.zen_and_az(s)[0],
        "azimuth": lambda s: model.zen_and_az(s)[1],
        "core_x": lambda s: model.core(s)[0],
        "core_y": lambda s: model.core(s)[1],
        "erad": lambda s: model.Erad(s),
        "xmax": lambda s: model.X_max(s),
        "t0": lambda s: model.t0(s),
        "xmax_timing_offset": lambda s: model.xmax_timing_offset(s),
        "core_x_timing_offset": lambda s: model.core_x_timing_offset(s),
        "core_y_timing_offset": lambda s: model.core_y_timing_offset(s),
    }

    reco = {}
    for name, func in param_methods.items():
        vals = tuple(func(s) for s in samples_list)
        reco[name] = jft.mean_and_std(vals, correct_bias=True)

    # Combined X_max
    xmax_combined = np.array([
        model.X_max(s) + 0.5 * model.xmax_timing_offset(s)
        for s in samples_list
    ])
    reco["xmax_combined"] = (float(np.mean(xmax_combined)), float(np.std(xmax_combined)))

    # Chi-squared
    try:
        sanity = jft.minisanity(
            samples, lh.normalized_residual, map=vmap,
        )
        # Combined likelihood returns a Vector tree; radio-only returns a tuple
        s0 = sanity[0]
        if hasattr(s0, 'reduced_chisq'):
            reco["chi_sq"] = s0.reduced_chisq
        else:
            # Vector of per-component results — extract each
            import jax
            leaves = jax.tree.leaves(s0)
            reco["chi_sq"] = np.array([
                l.reduced_chisq for l in leaves if hasattr(l, 'reduced_chisq')
            ])
    except Exception as e:
        logger.warning(f"Could not compute chi_sq: {e}")
        reco["chi_sq"] = np.nan

    # Correction factors
    e_corr = [model.get_energy_correction_factor(s) for s in samples_list]
    reco["energy_correction_factor"] = jft.mean_and_std(
        tuple(e_corr), correct_bias=True,
    )

    try:
        xmax_shape = [model.get_xmax_correction_factor(s) for s in samples_list]
        reco["xmax_shape_factor"] = jft.mean_and_std(
            tuple(xmax_shape), correct_bias=True,
        )
    except AttributeError:
        reco["xmax_shape_factor"] = (np.nan, np.nan)

    reco["samples"] = samples_list
    reco["model"] = model
    reco["timing_mask"] = timing_mask
    reco["has_particles"] = has_particles

    # Particle-specific posteriors
    if has_particles:
        particle_methods = {
            "ecr": lambda s: model._calculate_Ecr_eV(s),
            "erad_factor": lambda s: model.Erad_factor(s),
            "xmax_particle": lambda s: model.X_max_particle(s),
            "zenith_particle": lambda s: model.theta_particle(s),
            "azimuth_particle": lambda s: model.phi_particle(s),
            "log_scale_particle": lambda s: model.log_scale_particle(s),
            "particle_bg": lambda s: model.particle_bg(s),
        }
        for name, func in particle_methods.items():
            vals = tuple(func(s) for s in samples_list)
            reco[name] = jft.mean_and_std(vals, correct_bias=True)

    return reco
