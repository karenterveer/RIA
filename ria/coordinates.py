"""
Coordinate transformations for air-shower radio detection.

Provides JAX-differentiable transforms between geographic ground
coordinates, shower-plane coordinates (vxB / vx(vxB)), on-sky
coordinates, and related reference frames.

This module is a JAX-differentiable re-implementation of the coordinate
transformations from the `radiotools <https://github.com/nu-radio/radiotools>`_
package.

Classes
-------
CoordinateTransform
    Full set of transforms for a given shower geometry.

Functions
---------
spherical_to_cartesian
    Convert zenith/azimuth to a unit Cartesian vector.
get_magnetic_field_vector
    Look up the geomagnetic field vector for a given site.
get_angle
    Angle between two 3-vectors.
get_angle_to_magnetic_field
    Angle between shower axis and geomagnetic field.
get_declination
    Magnetic declination from a field vector.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla

from . import config


# ===================================================================
#  Helper geometry functions
# ===================================================================

def spherical_to_cartesian(
    zenith: jax.Array, azimuth: jax.Array
) -> jax.Array:
    """Convert spherical angles to a unit Cartesian 3-vector.

    Parameters
    ----------
    zenith : array_like
        Zenith angle [rad] (0 = straight up).
    azimuth : array_like
        Azimuth angle [rad] (0 = North, pi/2 = East).

    Returns
    -------
    jax.Array
        Cartesian unit vector with shape ``(..., 3)``.
    """
    sin_zen = jnp.sin(zenith)
    x = sin_zen * jnp.cos(azimuth)
    y = sin_zen * jnp.sin(azimuth)
    z = jnp.cos(zenith)
    return jnp.stack([x, y, z], axis=-1)


def get_magnetic_field_vector(site: str = "lofar") -> jax.Array:
    """Return the geomagnetic field vector for a given site [Gauss].

    Parameters
    ----------
    site : str
        Observatory name (case-insensitive).  Available sites are
        ``'lofar'``, ``'auger'``, ``'mooresbay'``, ``'summit'``,
        ``'southpole'``.

    Returns
    -------
    jax.Array
        Field vector ``(Bx_East, By_North, Bz_Up)`` with shape ``(3,)``.

    Notes
    -----
    The *site* argument must be a static Python string when this function
    is called inside JIT-compiled code.
    """
    return jnp.array(config.MAGNETIC_FIELD_VECTORS[site.lower()])


def get_angle(v1: jax.Array, v2: jax.Array) -> jax.Array:
    """Angle [rad] between two 3-vectors.

    Handles batched inputs of shape ``(N, 3)``.  Returns zero when
    either vector has zero norm.

    Parameters
    ----------
    v1, v2 : array_like
        Input vectors with shape ``(3,)`` or ``(N, 3)``.
    """
    norm_v1 = jnp.linalg.norm(v1, axis=-1, keepdims=True)
    norm_v2 = jnp.linalg.norm(v2, axis=-1, keepdims=True)

    safe_v1 = v1 / jnp.where(norm_v1 == 0.0, 1.0, norm_v1)
    safe_v2 = v2 / jnp.where(norm_v2 == 0.0, 1.0, norm_v2)

    dot = jnp.sum(safe_v1 * safe_v2, axis=-1)
    angle = jnp.arccos(jnp.clip(dot, -1.0, 1.0))

    is_zero = jnp.logical_or(
        norm_v1.squeeze(-1) == 0.0, norm_v2.squeeze(-1) == 0.0
    )
    return jnp.where(is_zero, 0.0, angle)


def get_angle_to_magnetic_field(
    zenith: jax.Array,
    azimuth: jax.Array,
    magnetic_field_vector: jax.Array | None = None,
    site: str = "lofar",
) -> jax.Array:
    """Angle [rad] between shower axis and geomagnetic field.

    Parameters
    ----------
    zenith, azimuth : array_like
        Shower direction [rad].
    magnetic_field_vector : array_like or None
        Explicit field vector; if ``None``, looked up via *site*.
    site : str
        Fallback site name when *magnetic_field_vector* is ``None``.
    """
    if magnetic_field_vector is None:
        magnetic_field_vector = get_magnetic_field_vector(site=site)
    v = spherical_to_cartesian(zenith, azimuth)
    return get_angle(magnetic_field_vector, v)


def get_declination(magnetic_field_vector: jax.Array) -> jax.Array:
    """Magnetic declination [rad] from a geomagnetic field vector.

    Parameters
    ----------
    magnetic_field_vector : array_like
        Field vector ``(Bx, By, Bz)`` in geographic coordinates.
    """
    b_horiz = magnetic_field_vector[..., :2]
    norm_horiz = jnp.linalg.norm(b_horiz, axis=-1)

    safe_norm = jnp.where(norm_horiz == 0.0, 1.0, norm_horiz)
    unit_b = b_horiz / safe_norm

    cos_dec = jnp.clip(unit_b[..., 1], -1.0, 1.0)
    declination = jnp.arccos(cos_dec)
    return jnp.where(norm_horiz == 0.0, 0.0, declination)


# ===================================================================
#  Main coordinate-transform class
# ===================================================================

class CoordinateTransform:
    """JAX-differentiable coordinate transforms for an air shower.

    Given the shower arrival direction (zenith, azimuth) and a magnetic
    field vector, this class builds all rotation matrices at construction
    time and exposes named transform methods.

    Parameters
    ----------
    zenith : float or JAX scalar
        Zenith angle [rad].
    azimuth : float or JAX scalar
        Azimuth angle [rad].
    magnetic_field_vector : array_like or None
        Geomagnetic field vector ``(Bx, By, Bz)`` in geographic CS.
        If ``None``, the default for *site* is used.
    site : str or None
        Observatory site for default magnetic field lookup.

    Notes
    -----
    All public methods accept either a single ``(3,)`` vector, a batch
    of ``(N, 3)`` vectors, or a time-series with shape ``(3, N_samples)``.
    """

    def __init__(
        self,
        zenith,
        azimuth,
        magnetic_field_vector=None,
        site=None,
    ):
        zenith = jnp.asarray(zenith)
        azimuth = jnp.asarray(azimuth)

        # Shower axis (points *towards* the origin from the source)
        showeraxis = -1.0 * spherical_to_cartesian(zenith, azimuth)

        if magnetic_field_vector is None:
            magnetic_field_vector = get_magnetic_field_vector(
                site=site or "lofar"
            )
        else:
            magnetic_field_vector = jnp.asarray(magnetic_field_vector)

        b_norm = jla.norm(magnetic_field_vector)
        b_hat = magnetic_field_vector / b_norm

        showeraxis = showeraxis.reshape(3)
        b_hat = b_hat.reshape(3)

        # vxB and vx(vxB) basis
        vxB = jnp.cross(showeraxis, b_hat)
        e1 = vxB / (jla.norm(vxB) + 1e-12)

        vxvxB = jnp.cross(showeraxis, e1)
        e2 = vxvxB / (jla.norm(vxvxB) + 1e-12)

        e3 = showeraxis

        # vxB/vxvxB rotation matrices
        self._mat_vBvvB = jnp.array([e1, e2, e3])
        self._inv_vBvvB = self._mat_vBvvB.T

        # On-sky rotation matrices
        ct, st = jnp.cos(zenith), jnp.sin(zenith)
        cp, sp = jnp.cos(azimuth), jnp.sin(azimuth)

        e1_sky = jnp.array([st * cp, st * sp, ct])
        e2_sky = jnp.array([ct * cp, ct * sp, -st])
        e3_sky = jnp.array([-sp, cp, jnp.zeros_like(sp)])

        self._mat_onsky = jnp.stack([e1_sky, e2_sky, e3_sky], axis=-2)
        self._inv_onsky = self._mat_onsky.T

        # Magnetic-north rotation
        dec = get_declination(magnetic_field_vector)
        c_d, s_d = jnp.cos(dec), jnp.sin(dec)
        self._mat_magnetic = jnp.array([
            [c_d, -s_d, 0.0],
            [s_d, c_d, 0.0],
            [0.0, 0.0, 1.0],
        ])
        self._inv_magnetic = self._mat_magnetic.T

        # Azimuth-aligned rotation
        ang_az = -azimuth
        c_a, s_a = jnp.cos(ang_az), jnp.sin(ang_az)
        z = jnp.zeros_like(s_a)
        o = jnp.ones_like(s_a)
        self._mat_azimuth = jnp.array([
            [c_a, -s_a, z], [s_a, c_a, z], [z, z, o]
        ])
        self._inv_azimuth = self._mat_azimuth.T

        # Early-late (shower-plane) rotation
        ang1 = -azimuth + jnp.pi / 2.0
        c1, s1 = jnp.cos(ang1), jnp.sin(ang1)
        rotZ = jnp.array([
            [c1, -s1, jnp.zeros_like(s1)],
            [s1, c1, jnp.zeros_like(s1)],
            [jnp.zeros_like(s1), jnp.zeros_like(s1), jnp.ones_like(s1)],
        ])

        c2, s2 = jnp.cos(zenith), jnp.sin(zenith)
        rotX = jnp.array([
            [jnp.ones_like(c2), jnp.zeros_like(c2), jnp.zeros_like(c2)],
            [jnp.zeros_like(c2), c2, -s2],
            [jnp.zeros_like(c2), s2, c2],
        ])
        rotX = rotX.reshape(3, 3)
        rotZ = rotZ.reshape(3, 3)

        self._mat_early_late = rotX @ rotZ
        self._inv_early_late = self._mat_early_late.T

    # ------------------------------------------------------------------
    #  Private transform helper
    # ------------------------------------------------------------------

    def _transform(self, positions, matrix):
        """Apply a 3x3 rotation matrix to position vectors.

        Handles shapes ``(3,)``, ``(N, 3)``, and ``(3, N_samples)``.
        """
        positions = jnp.asarray(positions)
        is_single = positions.ndim == 1

        if is_single:
            positions = positions.reshape(1, 3)
        elif positions.ndim == 2 and positions.shape[0] == 3 and positions.shape[1] != 3:
            # Time-series input (3, N_samples)
            return matrix @ positions

        transformed = (matrix @ positions.T).T

        if is_single:
            return transformed.reshape(3)
        return transformed

    # ------------------------------------------------------------------
    #  Public transform methods
    # ------------------------------------------------------------------

    def transform_to_vxB_vxvxB(self, positions, core=None):
        """Geographic CS -> vxB / vx(vxB) shower-plane CS.

        Parameters
        ----------
        positions : array_like
            Positions in geographic coordinates.
        core : array_like or None
            Shower-core position to subtract before rotation.
        """
        positions = jnp.asarray(positions)
        if core is not None:
            core = jnp.asarray(core).reshape(-1)
            if positions.ndim == 2 and positions.shape[0] == core.shape[0]:
                core = core[:, jnp.newaxis]
            positions = positions - core
        return self._transform(positions, self._mat_vBvvB)

    def transform_from_vxB_vxvxB(self, positions, core=None):
        """vxB / vx(vxB) shower-plane CS -> Geographic CS.

        Parameters
        ----------
        positions : array_like
            Positions in shower-plane coordinates.
        core : array_like or None
            Shower-core position to add after rotation.
        """
        positions = jnp.asarray(positions)
        transformed = self._transform(positions, self._inv_vBvvB)

        if core is not None:
            core = jnp.asarray(core).reshape(-1)
            if transformed.ndim == 2 and transformed.shape[0] == core.shape[0]:
                core = core[:, jnp.newaxis]
            transformed = transformed + core
        return transformed

    def transform_from_ground_to_onsky(self, positions):
        """Geographic CS -> On-sky (eR, eTheta, ePhi)."""
        return self._transform(positions, self._mat_onsky)

    def transform_from_onsky_to_ground(self, positions):
        """On-sky (eR, eTheta, ePhi) -> Geographic CS."""
        return self._transform(positions, self._inv_onsky)

    def transform_from_magnetic_to_geographic(self, positions):
        """Magnetic-north CS -> Geographic CS."""
        return self._transform(positions, self._mat_magnetic)

    def transform_from_geographic_to_magnetic(self, positions):
        """Geographic CS -> Magnetic-north CS."""
        return self._transform(positions, self._inv_magnetic)

    def transform_from_azimuth_to_geographic(self, positions):
        """Azimuth-aligned CS -> Geographic CS."""
        return self._transform(positions, self._inv_azimuth)

    def transform_from_geographic_to_azimuth(self, positions):
        """Geographic CS -> Azimuth-aligned CS."""
        return self._transform(positions, self._mat_azimuth)

    def transform_from_early_late(self, positions, core=None):
        """Shower plane (Early-Late) CS -> Geographic CS.

        Parameters
        ----------
        positions : array_like
            Positions in shower-plane coordinates.
        core : array_like or None
            Core position to add after rotation.
        """
        positions = jnp.asarray(positions)
        transformed = self._transform(positions, self._inv_early_late)
        if core is not None:
            core = jnp.asarray(core).reshape(-1)
            if transformed.ndim == 2 and transformed.shape[0] == core.shape[0]:
                core = core[:, jnp.newaxis]
            transformed = transformed + core
        return transformed

    def transform_to_early_late(self, positions, core=None):
        """Geographic CS -> Shower plane (Early-Late) CS.

        Parameters
        ----------
        positions : array_like
            Positions in geographic coordinates.
        core : array_like or None
            Core position to subtract before rotation.
        """
        positions = jnp.asarray(positions)
        if core is not None:
            core = jnp.asarray(core).reshape(-1)
            if positions.ndim == 2 and positions.shape[0] == core.shape[0]:
                core = core[:, jnp.newaxis]
            positions = positions - core
        return self._transform(positions, self._mat_early_late)

    def transform_from_vxB_vxvxB_2D(self, positions_2d, core=None):
        """Transform 2-D shower-plane positions back to 3-D geographic CS.

        Assumes the input points lie in the shower plane (z_sp derived
        from the constraint z_geo = 0).

        Parameters
        ----------
        positions_2d : array_like
            Positions in shower plane, shape ``(N, 2)`` or ``(2,)``.
        core : array_like or None
            Core position to add after rotation.
        """
        positions_2d = jnp.asarray(positions_2d)
        is_single = positions_2d.ndim == 1
        if is_single:
            positions_2d = positions_2d.reshape(1, 2)

        x_sp = positions_2d[:, 0]
        y_sp = positions_2d[:, 1]
        z_sp = self.get_height_in_showerplane(x_sp, y_sp)

        pos_3d = jnp.stack([x_sp, y_sp, z_sp], axis=-1)
        geo = self._transform(pos_3d, self._inv_vBvvB)

        if core is not None:
            geo = geo + jnp.asarray(core)

        if is_single:
            return geo.reshape(3)
        return geo

    def get_height_in_showerplane(self, x_sp, y_sp):
        """Compute z_sp such that z_geo = 0 for a point in the shower plane.

        Parameters
        ----------
        x_sp, y_sp : array_like
            Coordinates in the vxB/vxvxB plane.
        """
        x_sp = jnp.asarray(x_sp)
        y_sp = jnp.asarray(y_sp)
        inv = self._inv_vBvvB
        return -(inv[2, 0] * x_sp + inv[2, 1] * y_sp) / (inv[2, 2] + 1e-12)
