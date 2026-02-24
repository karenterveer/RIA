"""
Forward model for air-shower radio footprint reconstruction.

Combines the lateral distribution function (LDF) and hyperbolic
wavefront timing model into a single differentiable model that maps
latent shower parameters to predicted fluence and arrival-time arrays.

The ``FootprintModel`` class extends ``nifty8.re.Model`` and can be used
directly with NIFTy's ``optimize_kl`` for Bayesian inference.

Classes
-------
FootprintModel
    Full forward model (LDF + wavefront timing).
"""

from __future__ import annotations

import os

import numpy as np
import numpy.typing as npt
import jax
import jax.numpy as jnp
import nifty8.re as jft

from jax import vmap
from jax.scipy.interpolate import RegularGridInterpolator
from jax.scipy.special import gammaln

from .atmosphere import Atmosphere
from .coordinates import CoordinateTransform
from . import ldf as _ldf
from . import config


# -------------------------------------------------------------------
#  Utility
# -------------------------------------------------------------------

def soft_clip(x, a_min, a_max, sharpness=20.0):
    """Sigmoid-based smooth clipping to avoid gradient discontinuities.

    Parameters
    ----------
    x : array_like
        Input values.
    a_min, a_max : float
        Target range for the clipped output.
    sharpness : float, optional
        Steepness of the sigmoid transition.
    """
    mid = (a_min + a_max) / 2.0
    half_range = (a_max - a_min) / 2.0
    return mid + half_range * jnp.tanh(
        sharpness * (x - mid) / half_range / 2.0
    )


# -------------------------------------------------------------------
#  LDF adapter  (fluences in shower-plane coordinates)
# -------------------------------------------------------------------

def _fluence_from_ldf(
    E, X_max, azimuth, zenith, x_pos, y_pos, magnetic_field_vector, atmosphere_path,
):
    """Evaluate the LDF at a single shower-plane point.

    Returns ``(fluence, f_vB, f_vvB, f_geo, f_ce)``.
    """
    return _ldf.LDF(
        x_pos, y_pos, E, X_max, zenith, azimuth,
        core=jnp.array([0.0, 0.0]),
        magnetic_field_vector=magnetic_field_vector,
        atmosphere_path=atmosphere_path,
    )


# ===================================================================
#  FootprintModel
# ===================================================================

class FootprintModel(jft.Model):
    """Forward model combining LDF and wavefront timing.

    Maps latent standard-normal parameters to predicted fluence and
    arrival-time arrays at the antenna positions ``(x, y)``.

    Parameters
    ----------
    x, y : ndarray
        Ground-plane antenna positions [m].
    magnetic_field_vector : ndarray, optional
        Geomagnetic field vector ``(Bx, By, Bz)`` [Gauss].
    params_Erad : dict
        Log-normal prior parameters ``{"mean", "std"}`` for radiation energy.
    params_phi : dict
        Uniform prior ``{"a_min", "a_max"}`` for azimuth [rad].
    params_theta : dict
        Uniform prior ``{"a_min", "a_max"}`` for zenith [rad].
    params_X_max : dict
        Uniform prior ``{"a_min", "a_max"}`` for X_max [g/cm^2].
    params_X, params_Y : dict
        Uniform prior ``{"a_min", "a_max"}`` for core position [m].
    params_t0 : dict
        Normal prior ``{"mean", "std"}`` for reference time t_0 [s].
    params_noise_floor_mean : dict
        Log-normal prior for noise-floor fluence.
    params_gamma : dict
        Normal prior for wavefront shape parameter gamma.
    params_const_rho_res : dict
        Normal prior for wavefront C polynomial residual.
    params_xmax_timing_offset : dict
        Normal prior for X_max offset between fluence and timing [g/cm^2].
    atmosphere_path : str or None
        Path to GDAS atmosphere file; built-in model if ``None``.
    prefix : str
        Prefix added to all NIFTy parameter names.
    timing_std_s : float
        Amplitude of timing correlated-field fluctuations [s].
    enable_syst_cf : bool
        Enable fluence systematic correlated field.
    enable_timing_cf : bool
        Enable timing correlated field.
    grid_settings : dict or None
        Explicit grid ``{"min_x", "min_y", "extent", "dims"}``;
        computed from antenna positions if ``None``.
    """

    def __init__(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        magnetic_field_vector=None,
        params_Erad: dict | None = None,
        params_phi: dict = None,
        params_theta: dict = None,
        params_X_max: dict | None = None,
        params_X: dict = None,
        params_Y: dict = None,
        params_t0: dict | None = None,
        params_noise_floor_mean: dict | None = None,
        params_gamma: dict | None = None,
        params_const_rho_res: dict | None = None,
        params_xmax_timing_offset: dict | None = None,
        params_core_x_timing_offset: dict | None = None,
        params_core_y_timing_offset: dict | None = None,
        atmosphere_path: str | None = None,
        prefix: str = "",
        timing_std_s: float = 1.5e-9,
        enable_syst_cf: bool | None = None,
        enable_timing_cf: bool | None = None,
        grid_settings: dict | None = None,
        # --- Particle model (optional) ---
        x_particle: npt.NDArray[np.float64] | None = None,
        y_particle: npt.NDArray[np.float64] | None = None,
        z_particle: npt.NDArray[np.float64] | None = None,
        params_Erad_factor: dict | None = None,
        params_X_max_particle: dict | None = None,
        params_theta_particle: dict | None = None,
        params_phi_particle: dict | None = None,
        params_log_scale_particle: dict | None = None,
        params_particle_bg: dict | None = None,
        enable_particle_cf: bool | None = None,
    ):
        # ----- defaults from config -----
        if magnetic_field_vector is None:
            magnetic_field_vector = config.DEFAULT_MAGNETIC_FIELD_VECTOR
        if params_Erad is None:
            params_Erad = dict(config.DEFAULT_PARAMS_ERAD)
        if params_phi is None:
            params_phi = {"a_min": 0.0, "a_max": 2.0 * np.pi}
        if params_theta is None:
            params_theta = {"a_min": 0.0, "a_max": np.pi / 2.0}
        if params_X_max is None:
            params_X_max = dict(config.DEFAULT_PARAMS_XMAX)
        if params_X is None:
            params_X = {"a_min": -500.0, "a_max": 500.0}
        if params_Y is None:
            params_Y = {"a_min": -500.0, "a_max": 500.0}
        if params_t0 is None:
            params_t0 = dict(config.DEFAULT_PARAMS_T0)
        if params_noise_floor_mean is None:
            params_noise_floor_mean = dict(config.DEFAULT_PARAMS_NOISE_FLOOR_MEAN)
        if params_gamma is None:
            params_gamma = dict(config.DEFAULT_PARAMS_GAMMA)
        if params_const_rho_res is None:
            params_const_rho_res = dict(config.DEFAULT_PARAMS_CONST_RHO_RES)
        if params_xmax_timing_offset is None:
            params_xmax_timing_offset = dict(config.DEFAULT_PARAMS_XMAX_TIMING_OFFSET)
        if params_core_x_timing_offset is None:
            params_core_x_timing_offset = dict(config.DEFAULT_PARAMS_CORE_TIMING_OFFSET)
        if params_core_y_timing_offset is None:
            params_core_y_timing_offset = dict(config.DEFAULT_PARAMS_CORE_TIMING_OFFSET)
        if enable_syst_cf is None:
            enable_syst_cf = config.FLUENCE_CF
        if enable_timing_cf is None:
            enable_timing_cf = config.TIMING_CF

        # ----- fixed attributes -----
        self.magnetic_field_vector = np.asarray(magnetic_field_vector)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        if atmosphere_path is None:
            atmosphere_path = os.path.normpath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "example_data", "ATMOSPHERE_EXAMPLE.DAT")
            )
        self.atmosphere_path = atmosphere_path
        self.speedoflight = config.C_LIGHT
        self.b_offset_s = config.WAVEFRONT_B_OFFSET_S

        self.enable_syst_cf = enable_syst_cf
        self.enable_timing_cf = enable_timing_cf

        # ----- particle detector positions (optional) -----
        self.has_particles = x_particle is not None and len(x_particle) > 0
        if self.has_particles:
            self.x_particle = jnp.array(x_particle, dtype=jnp.float64)
            self.y_particle = jnp.array(y_particle, dtype=jnp.float64)
            self.z_particle = (
                jnp.array(z_particle, dtype=jnp.float64)
                if z_particle is not None
                else jnp.zeros_like(self.x_particle)
            )
            self.lora_Aeff = config.DETECTOR_AREA_M2
        else:
            self.x_particle = None
            self.y_particle = None
            self.z_particle = None

        if enable_particle_cf is None:
            enable_particle_cf = config.PARTICLE_CF
        self.enable_particle_cf = enable_particle_cf and self.has_particles

        # Atmosphere object for Cherenkov-angle calculation
        self.atmosphere = None
        if self.atmosphere_path and os.path.exists(self.atmosphere_path):
            self.atmosphere = Atmosphere(gdas_file=self.atmosphere_path)

        # ----- priors -----
        self.log_Erad_prior = jft.prior.LogNormalPrior(
            **params_Erad, shape=(1,), name=prefix + "log_Erad",
        )
        self.phi = jft.prior.UniformPrior(
            **params_phi, shape=(1,), name=prefix + "phi",
        )
        self.theta = jft.prior.UniformPrior(
            **params_theta, shape=(1,), name=prefix + "theta",
        )
        self.X_max = jft.prior.UniformPrior(
            **params_X_max, shape=(1,), name=prefix + "X_max",
        )
        self.X_core = jft.prior.UniformPrior(
            **params_X, shape=(1,), name=prefix + "X",
        )
        self.Y_core = jft.prior.UniformPrior(
            **params_Y, shape=(1,), name=prefix + "Y",
        )
        self.t0 = jft.prior.NormalPrior(
            **params_t0, shape=(1,), name=prefix + "t0",
        )
        self.noise_floor_mean = jft.prior.LogNormalPrior(
            **params_noise_floor_mean, shape=(1,), name=prefix + "noise_floor_mean",
        )
        self.gamma_prior = jft.prior.NormalPrior(
            **params_gamma, shape=(1,), name=prefix + "gamma",
        )
        self.const_rho_residual = jft.prior.NormalPrior(
            **params_const_rho_res, shape=(1,), name=prefix + "const_rho_res",
        )
        self.xmax_timing_offset = jft.prior.NormalPrior(
            **params_xmax_timing_offset, shape=(1,),
            name=prefix + "xmax_timing_offset",
        )

        self.core_x_timing_offset = jft.prior.NormalPrior(
            **params_core_x_timing_offset, shape=(1,),
            name=prefix + "core_x_timing_offset",
        )
        self.core_y_timing_offset = jft.prior.NormalPrior(
            **params_core_y_timing_offset, shape=(1,),
            name=prefix + "core_y_timing_offset",
        )

        # ----- particle priors (only if particle data provided) -----
        if self.has_particles:
            if params_Erad_factor is None:
                params_Erad_factor = dict(config.DEFAULT_PARAMS_ERAD_FACTOR)
            if params_X_max_particle is None:
                params_X_max_particle = dict(config.DEFAULT_PARAMS_XMAX_PARTICLE)
            if params_theta_particle is None:
                params_theta_particle = dict(config.DEFAULT_PARAMS_THETA_PARTICLE)
            if params_phi_particle is None:
                params_phi_particle = dict(config.DEFAULT_PARAMS_PHI_PARTICLE)
            if params_log_scale_particle is None:
                params_log_scale_particle = dict(config.DEFAULT_PARAMS_LOG_SCALE_PARTICLE)
            if params_particle_bg is None:
                params_particle_bg = dict(config.DEFAULT_PARAMS_PARTICLE_BG)

            self.Erad_factor = jft.prior.LogNormalPrior(
                **params_Erad_factor, shape=(1,), name=prefix + "eradfactor",
            )
            self.X_max_particle = jft.prior.UniformPrior(
                **params_X_max_particle, shape=(1,), name=prefix + "X_max_p",
            )
            self.theta_particle = jft.prior.UniformPrior(
                **params_theta_particle, shape=(1,), name=prefix + "theta_p",
            )
            self.phi_particle = jft.prior.UniformPrior(
                **params_phi_particle, shape=(1,), name=prefix + "phi_p",
            )
            self.log_scale_particle = jft.prior.NormalPrior(
                **params_log_scale_particle, shape=(1,), name=prefix + "log_scale_part",
            )
            self.particle_bg = jft.prior.NormalPrior(
                **params_particle_bg, shape=(1,), name=prefix + "part_bg",
            )

        # ----- grid for correlated fields -----
        if grid_settings is not None:
            self.min_x = grid_settings["min_x"]
            self.min_y = grid_settings["min_y"]
            self.extent = grid_settings["extent"]
            self.dims = grid_settings["dims"]
            self.distances = self.extent / self.dims[0]
        else:
            all_x = np.asarray(x)
            all_y = np.asarray(y)
            if self.has_particles:
                all_x = np.concatenate([all_x, np.asarray(x_particle)])
                all_y = np.concatenate([all_y, np.asarray(y_particle)])
            min_x_d, max_x_d = float(np.min(all_x)), float(np.max(all_x))
            min_y_d, max_y_d = float(np.min(all_y)), float(np.max(all_y))
            center_x = (min_x_d + max_x_d) / 2.0
            center_y = (min_y_d + max_y_d) / 2.0
            max_span = max(max_x_d - min_x_d, max_y_d - min_y_d)
            self.extent = max_span + 2.0 * config.GRID_PAD
            self.min_x = center_x - self.extent / 2.0
            self.min_y = center_y - self.extent / 2.0

            calc_dim = int(np.ceil(self.extent / config.TARGET_RESOLUTION))
            calc_dim = max(config.MIN_GRID_DIM, min(calc_dim, config.MAX_GRID_DIM))
            if calc_dim % 2 != 0:
                calc_dim += 1
            self.dims = (calc_dim, calc_dim)
            self.distances = self.extent / self.dims[0]

        # Systematics limits
        self.sys_mult_min = config.SYST_MULT_MIN
        self.sys_mult_max = config.SYST_MULT_MAX
        self._sys_log_min = jnp.log(self.sys_mult_min)
        self._sys_log_max = jnp.log(self.sys_mult_max)

        # ----- correlated fields -----
        cf_initializers = []

        if self.enable_syst_cf:
            cfm_fluence = jft.CorrelatedFieldMaker(prefix + "syst_cf")
            cfm_fluence.set_amplitude_total_offset(**config.SYST_CF_ZM)
            cfm_fluence.add_fluctuations(
                self.dims, distances=self.distances,
                **config.SYST_CF_FL, prefix="ax1",
                non_parametric_kind="power",
            )
            self.syst_cf_op = cfm_fluence.finalize()
            self.syst_cf_raw = self.syst_cf_op
            cf_initializers.append(self.syst_cf_op.init)
        else:
            zero_grid = jnp.zeros(self.dims)
            self.syst_cf_op = lambda _x: zero_grid
            self.syst_cf_raw = self.syst_cf_op

        if self.enable_timing_cf:
            timing_fl = dict(
                fluctuations=(timing_std_s, timing_std_s),
                loglogavgslope=(-1.0, 0.5),
            )
            cfm_timing = jft.CorrelatedFieldMaker(prefix + "timing_cf_2")
            cfm_timing.set_amplitude_total_offset(**config.TIMING_CF_ZM)
            cfm_timing.add_fluctuations(
                self.dims, distances=self.distances, **timing_fl,
                prefix="ax1_time_2", non_parametric_kind="power",
            )
            self.timing_cf_op_2 = cfm_timing.finalize()
            cf_initializers.append(self.timing_cf_op_2.init)
        else:
            zero_grid = jnp.zeros(self.dims)
            self.timing_cf_op_2 = lambda _x: zero_grid

        if self.enable_particle_cf:
            cfm_p = jft.CorrelatedFieldMaker(prefix + "particle_cf")
            cfm_p.set_amplitude_total_offset(**config.PARTICLE_CF_ZM)
            cfm_p.add_fluctuations(
                self.dims, distances=self.distances,
                **config.PARTICLE_CF_FL,
                prefix="ax1_part", non_parametric_kind="power",
            )
            self.particle_cf_op = cfm_p.finalize()
            cf_initializers.append(self.particle_cf_op.init)
        else:
            zero_grid_p = jnp.zeros(self.dims)
            self.particle_cf_op = lambda _x: zero_grid_p

        self.syst_cf = self._get_cherenkov_ring_cf_log

        # ----- combine initialisers -----
        init = (
            self.log_Erad_prior.init | self.phi.init | self.theta.init
            | self.X_max.init | self.X_core.init | self.Y_core.init
            | self.t0.init | self.noise_floor_mean.init | self.gamma_prior.init
            | self.const_rho_residual.init | self.xmax_timing_offset.init
            | self.core_x_timing_offset.init | self.core_y_timing_offset.init
        )
        if self.has_particles:
            init = (
                init | self.Erad_factor.init
                | self.X_max_particle.init | self.theta_particle.init
                | self.phi_particle.init | self.log_scale_particle.init
                | self.particle_bg.init
            )
        for cf_init in cf_initializers:
            init = init | cf_init
        super().__init__(init=init)

    # ------------------------------------------------------------------
    #  Parameter accessors
    # ------------------------------------------------------------------

    @property
    def ops(self):
        """Ordered tuple of prior operators for the LDF call."""
        return (
            self.Erad, self.X_max, self.phi, self.theta,
            self.X_core, self.Y_core,
        )

    def Erad(self, x):
        """Radiation energy [eV]."""
        return jnp.exp(self.log_Erad_prior(x))

    def core(self, x):
        """Return the shower-core position (x_core, y_core)."""
        return self.X_core(x), self.Y_core(x)

    def core_for_timing(self, x):
        """Core position with timing-specific offset.

        This allows a small tension between the fluence and timing
        descriptions of the core location.
        """
        return (
            self.X_core(x) + self.core_x_timing_offset(x),
            self.Y_core(x) + self.core_y_timing_offset(x),
        )

    def zen_and_az(self, x):
        """Return (zenith, azimuth)."""
        return self.theta(x), self.phi(x)

    def xmax_timing_offset(self, x):
        """X_max offset between fluence and timing [g/cm^2]."""
        return self.xmax_timing_offset_(x)

    def core_x_timing_offset(self, x):
        """Core X offset between fluence and timing [m]."""
        return self.core_x_timing_offset_(x)

    def core_y_timing_offset(self, x):
        """Core Y offset between fluence and timing [m]."""
        return self.core_y_timing_offset_(x)

    def X_max_combined(self, x):
        """Combined X_max (fluence Xmax + half the timing offset).

        This provides a single best-estimate of X_max that accounts for
        the tension allowed by the timing offset prior.
        """
        return self.X_max(x) + 0.5 * self.xmax_timing_offset_(x)

    # ------------------------------------------------------------------
    #  Cherenkov angle via dXmax polynomial
    # ------------------------------------------------------------------

    def calculate_rho(self, x, zenith, xmax):
        """Cherenkov angle from a dXmax-based polynomial parametrisation.

        Uses the calibrated order-3 polynomial from LOFAR wavefront
        calibration plus the timing-specific X_max offset and a residual
        normal prior.

        Parameters
        ----------
        x : dict
            Latent-parameter dict (NIFTy convention).
        zenith : array_like
            Zenith angle [rad].
        xmax : array_like
            X_max [g/cm^2].

        Returns
        -------
        jnp.ndarray
            Cherenkov angle *rho* [rad].
        """
        obs_level_m = 0.0
        if self.atmosphere is not None:
            x_atm_obs_gm2 = self.atmosphere.get_atmosphere(obs_level_m)
        else:
            x_atm_obs_gm2 = 10338000.0  # sea level g/m^2 (US standard model 17)

        slant_depth_obs_gm2 = x_atm_obs_gm2 / jnp.cos(zenith)

        xmax_for_timing = xmax + self.xmax_timing_offset(x)
        # slant_depth_obs_gm2 is already slant depth in g/m²
        # Convert to g/cm² (÷ 10000) then subtract xmax (in g/cm²)
        d_xmax = slant_depth_obs_gm2 * 1e-4 - xmax_for_timing

        # Order-3 polynomial for the normalisation constant
        const = (
            53377.0
            + (-130.15) * d_xmax
            + 0.187281 * d_xmax ** 2
            + (-0.0000895496) * d_xmax ** 3
        )
        const += self.const_rho_residual(x)

        gamma = self.gamma_prior(x)
        return (xmax_for_timing / const) * (jnp.cos(zenith) ** gamma)

    # ------------------------------------------------------------------
    #  LDF — fluence at antenna positions
    # ------------------------------------------------------------------

    def fluence(self, E, X_max, azimuth, zenith, x_core, y_core, x_pos, y_pos):
        """Evaluate the LDF at a single shower-plane point.

        Parameters
        ----------
        E : float
            Radiation energy.
        X_max : float
            Atmospheric depth of shower max [g/cm^2].
        azimuth, zenith : float
            Shower direction [rad].
        x_core, y_core : float
            Core position (applied as zero — subtraction done upstream).
        x_pos, y_pos : float
            Antenna position in shower-plane coordinates.
        """
        return _ldf.LDF(
            x_pos, y_pos, E, X_max, zenith, azimuth,
            core=jnp.array([0.0, 0.0]),
            magnetic_field_vector=self.magnetic_field_vector,
            atmosphere_path=self.atmosphere_path,
        )

    # ------------------------------------------------------------------
    #  Wavefront timing model
    # ------------------------------------------------------------------

    def get_arrival_time_differences(self, x):
        """Predicted arrival-time differences at all antennas.

        Uses a hyperbolic wavefront parameterised by the Cherenkov angle
        *rho*, offset parameter *b*, and reference time *t0*.

        Parameters
        ----------
        x : dict
            Latent-parameter dict.
        """
        Xmax = self.X_max(x)
        zenith = self.theta(x)
        azimuth = self.phi(x)
        x_core, y_core = self.core_for_timing(x)
        t0_val = self.t0(x)

        rho = self.calculate_rho(x, zenith, Xmax)

        ct = CoordinateTransform(
            zenith, azimuth,
            magnetic_field_vector=self.magnetic_field_vector,
        )
        pos_ground = jnp.array([
            self.x, self.y, jnp.zeros_like(self.x)
        ])
        pos_sp = ct.transform_to_vxB_vxvxB(
            pos_ground,
            core=jnp.array([
                x_core.squeeze(), y_core.squeeze(),
                jnp.zeros_like(x_core.squeeze()),
            ]),
        )
        d = jnp.sqrt(pos_sp[0] ** 2 + pos_sp[1] ** 2)
        z_s = pos_sp[2]

        term1 = (d * jnp.sin(rho)) ** 2
        term2 = (self.speedoflight * self.b_offset_s) ** 2
        tau_geo = (1.0 / self.speedoflight) * (
            jnp.sqrt(term1 + term2)
            + z_s * jnp.cos(rho)
            + self.speedoflight * self.b_offset_s
        )
        return t0_val + tau_geo

    # ------------------------------------------------------------------
    #  Signal predictions without correlated fields
    # ------------------------------------------------------------------

    def get_signal_fluence_without_cf(self, x):
        """Predicted fluence at every antenna without systematic CF.

        Parameters
        ----------
        x : dict
            Latent-parameter dict.
        """
        ct = CoordinateTransform(
            self.theta(x), self.phi(x),
            magnetic_field_vector=self.magnetic_field_vector,
        )
        pos_array = jnp.array([
            self.x, self.y, jnp.zeros_like(self.x)
        ])
        vxvxB = ct.transform_to_vxB_vxvxB(
            pos_array,
            core=jnp.array([
                self.X_core(x).squeeze(), self.Y_core(x).squeeze(),
                jnp.zeros_like(self.X_core(x).squeeze()),
            ]),
        )

        def _single_fluence(x_pos, y_pos):
            return self.fluence(
                *(oo(x) for oo in self.ops), x_pos, y_pos
            )[0]

        return vmap(_single_fluence)(vxvxB[0, :], vxvxB[1, :]).squeeze()

    def get_signal_timing_without_cf(self, x):
        """Predicted arrival times at every antenna without CF.

        Parameters
        ----------
        x : dict
            Latent-parameter dict.
        """
        return self.get_arrival_time_differences(x).squeeze()

    # ------------------------------------------------------------------
    #  Correction factors (for posterior diagnostics)
    # ------------------------------------------------------------------

    def get_energy_correction_factor(self, x):
        """Ratio of base-model total fluence to CF-corrected total fluence.

        Used to assess how much the systematic CF shifts the energy scale.

        Parameters
        ----------
        x : dict
            Latent-parameter dict.
        """
        flu_base = self.get_signal_fluence_without_cf(x)
        syst_log = self._get_cherenkov_ring_cf_log(x)

        xi = jnp.linspace(0, self.dims[0] - 1, self.dims[0])
        yi = jnp.linspace(0, self.dims[1] - 1, self.dims[1])
        points = jnp.stack([
            (self.x - self.min_x) / self.distances,
            (self.y - self.min_y) / self.distances,
        ], axis=-1)
        interp = RegularGridInterpolator(
            (xi, yi), syst_log, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        total_mult = jnp.exp(interp(points))
        flu_cf = flu_base * total_mult
        return jnp.sum(flu_base) / (jnp.sum(flu_cf) + 1e-9)


    # ------------------------------------------------------------------
    #  Correlated-field helpers (private)
    # ------------------------------------------------------------------

    def _get_cherenkov_ring_cf_log(self, x):
        """Systematic CF in log-fluence space, clipped to the allowed range."""
        if not getattr(self, "enable_syst_cf", True):
            return jnp.zeros(self.dims)
        return soft_clip(
            self.syst_cf_raw(x), self._sys_log_min, self._sys_log_max,
        )

    # ------------------------------------------------------------------
    #  Forward call
    # ------------------------------------------------------------------

    def __call__(self, x):
        """Evaluate the full forward model.

        Without particle data, returns a ``(2, N_antennas)`` array where
        row 0 is predicted fluence and row 1 is predicted arrival time.

        With particle data, returns a dict::

            {"radio": (2, N_radio), "particle": (N_particle,)}

        Parameters
        ----------
        x : dict
            Latent standard-normal parameter dict.
        """
        radio = self.get_radio_prediction(x)

        if not self.has_particles:
            return radio

        particle = self.get_particle_prediction(x)
        return {"radio": radio, "particle": particle}

    def get_radio_prediction(self, x):
        """Compute radio fluence and timing predictions.

        Returns
        -------
        jnp.ndarray, shape ``(2, N_radio)``
            Row 0: fluence [eV/m²], Row 1: arrival time [s].
        """
        ct = CoordinateTransform(
            self.theta(x), self.phi(x),
            magnetic_field_vector=self.magnetic_field_vector,
        )
        pos_array = jnp.array([
            self.x, self.y, jnp.zeros_like(self.x)
        ])
        vxvxB_pos = ct.transform_to_vxB_vxvxB(
            pos_array,
            core=jnp.array([
                self.X_core(x).squeeze(), self.Y_core(x).squeeze(),
                jnp.zeros_like(self.X_core(x).squeeze()),
            ]),
        )

        # Systematic fluence correction
        cf_grid_log = self.syst_cf(x)
        xi = jnp.linspace(0, self.dims[0] - 1, self.dims[0])
        yi = jnp.linspace(0, self.dims[1] - 1, self.dims[1])
        points = jnp.stack([
            (self.x - self.min_x) / self.distances,
            (self.y - self.min_y) / self.distances,
        ], axis=-1)
        interp_syst = RegularGridInterpolator(
            (xi, yi), cf_grid_log, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        total_multiplier = jnp.exp(interp_syst(points))

        # Timing correction
        timing_grid = self.timing_cf_op_2(x)
        interp_timing = RegularGridInterpolator(
            (xi, yi), timing_grid, method="linear",
            bounds_error=False, fill_value=0.0,
        )
        clipped_timing = soft_clip(
            interp_timing(points),
            -config.TIMING_CLIP_NS * 1e-9,
            config.TIMING_CLIP_NS * 1e-9,
        )

        arrival_times = self.get_arrival_time_differences(x) + clipped_timing

        # Fluence
        def _single_fluence(x_pos, y_pos, mult_val):
            return (
                self.fluence(*(oo(x) for oo in self.ops), x_pos, y_pos)[0]
                * mult_val
                + self.noise_floor_mean(x)
            )

        fluence_values = vmap(_single_fluence)(
            vxvxB_pos[0, :], vxvxB_pos[1, :], total_multiplier,
        )

        return jnp.stack([
            fluence_values.squeeze(), arrival_times.squeeze()
        ])

    # ------------------------------------------------------------------
    #  Particle model
    # ------------------------------------------------------------------

    def _calculate_Ecr_eV(self, x):
        """Cosmic-ray energy from radiated energy via the Glaser relation.

        Parameters
        ----------
        x : dict
            Latent-parameter dict.

        Returns
        -------
        jnp.ndarray
            Cosmic-ray energy [eV].
        """
        Erad_lin = self.Erad(x)
        Erad_factor = self.Erad_factor(x)
        zen, az = self.theta(x), self.phi(x)

        B_vect = jnp.array(self.magnetic_field_vector, dtype=jnp.float64)
        B_mag = jnp.linalg.norm(B_vect)
        B_gauss = jnp.where(B_mag < 0.01, B_mag * 1e4, B_mag)

        s_vec = jnp.stack([
            jnp.sin(zen) * jnp.cos(az),
            jnp.sin(zen) * jnp.sin(az),
            jnp.cos(zen),
        ], axis=-1)
        cross_prod = jnp.cross(s_vec.squeeze(), B_vect / B_mag)
        sin_alpha = jnp.clip(jnp.linalg.norm(cross_prod), 0.05, 1.0)

        return (
            1e18 * (0.24 / B_gauss) * (1.0 / sin_alpha)
            * jnp.sqrt(jnp.maximum(Erad_lin, 1e-1) * 1e-6 / Erad_factor)
        )

    def _get_ground_depth_gcm2(self):
        """Atmospheric depth at ground level [g/cm²]."""
        if self.atmosphere is not None and self.z_particle is not None:
            z_avg = jnp.mean(self.z_particle)
            X_ground_gm2 = self.atmosphere.get_atmosphere(z_avg)
            return X_ground_gm2 / 10000.0
        return config.X_GROUND_GPCM2_FALLBACK

    def _interpolate_grid(self, x_coords, y_coords, grid):
        """Interpolate a correction-field grid at given coordinates."""
        xi = jnp.linspace(0, self.dims[0] - 1, self.dims[0])
        yi = jnp.linspace(0, self.dims[1] - 1, self.dims[1])
        px = (x_coords - self.min_x) / self.distances
        py = (y_coords - self.min_y) / self.distances
        points = jnp.stack([px, py], axis=-1)
        return RegularGridInterpolator(
            (xi, yi), grid, bounds_error=False, fill_value=0.0,
        )(points)

    def get_particle_prediction(self, x):
        """Predicted particle counts at detector positions.

        Parameters
        ----------
        x : dict
            Latent-parameter dict.

        Returns
        -------
        jnp.ndarray, shape ``(N_particle,)``
            Expected counts at each particle detector.
        """
        return self.get_particle_map_prediction(
            x, self.x_particle, self.y_particle, self.z_particle,
            is_detector=True,
        )

    def get_particle_map_prediction(
        self, x, x_coords, y_coords, z_coords, is_detector=False,
    ):
        """Predicted particle density or counts at arbitrary positions.

        Parameters
        ----------
        x : dict
            Latent-parameter dict.
        x_coords, y_coords, z_coords : array_like
            Positions [m].
        is_detector : bool
            If True, returns expected counts (density × area × cos θ).
            If False, returns raw density.

        Returns
        -------
        jnp.ndarray
            Particle counts or density.
        """
        # Cosmic-ray energy from shared radio fit
        Ecr_eV = self._calculate_Ecr_eV(x)

        # N_max from energy
        log10_N = (
            jnp.log10(jnp.maximum(Ecr_eV, 1e14)) - config.PAR_A_INTERCEPT
        ) / config.PAR_B_SLOPE
        ln_N_max = log10_N * jnp.log(10.0)

        # Particle-specific direction and Xmax
        zen = self.theta_particle(x)
        az = self.phi_particle(x)
        Xmax_p = self.X_max_particle(x)
        xc, yc = self.X_core(x), self.Y_core(x)

        # Shower age
        X_ground = self._get_ground_depth_gcm2()
        cos_theta = jnp.clip(jnp.cos(zen), 0.3, 1.0)
        X_slant = X_ground / cos_theta
        s_long = 3.0 * X_slant / (X_slant + 2.0 * Xmax_p + 1e-6)
        s = jnp.clip(s_long - 0.15, 0.7, 1.7)

        # Atmospheric attenuation
        ln_attenuation = (Xmax_p - X_slant) / config.LAMBDA_ATT_G_CM2
        ln_N_ground = ln_N_max + ln_attenuation

        # Transform to shower plane
        ct = CoordinateTransform(
            zen, az, magnetic_field_vector=self.magnetic_field_vector,
        )
        pos_target = jnp.stack([x_coords, y_coords, z_coords], axis=0)
        pos_sp = ct.transform_to_vxB_vxvxB(
            pos_target,
            core=jnp.array([
                xc.squeeze(), yc.squeeze(), jnp.zeros_like(xc.squeeze()),
            ]),
        )

        r_plane = jnp.sqrt(pos_sp[0] ** 2 + pos_sp[1] ** 2 + 1e-12)
        r_plane = jnp.maximum(r_plane, 2.0)

        # NKG lateral distribution
        Rm = config.MOLIERE_RADIUS_M
        rs = r_plane / Rm

        ln_Cs = (
            gammaln(4.5 - s) - jnp.log(2 * jnp.pi) - 2.0 * jnp.log(Rm)
            - gammaln(s) - gammaln(4.5 - 2.0 * s)
        )

        ln_rho = (
            ln_N_ground + ln_Cs
            + (s - 2.0) * jnp.log(rs)
            + (s - 4.5) * jnp.log(1.0 + rs)
        )

        # Systematics
        bg_counts = jnp.power(10.0, self.particle_bg(x))
        part_cf_grid = jnp.clip(
            self.particle_cf_op(x),
            config.PARTICLE_LOG_MIN, config.PARTICLE_LOG_MAX,
        )
        syst_part_log = self._interpolate_grid(x_coords, y_coords, part_cf_grid)

        if is_detector:
            ln_lambda = (
                ln_rho + jnp.log(self.lora_Aeff * cos_theta)
                + self.log_scale_particle(x) + syst_part_log
            )
            return jnp.exp(jnp.clip(ln_lambda, -30.0, 30.0)) + bg_counts
        else:
            return jnp.exp(ln_rho + syst_part_log) + bg_counts
